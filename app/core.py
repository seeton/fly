"""Male CNS コネクトームへの問い合わせ層 — UI から切り離してある。

もとは `app/server.py` の FastAPI ハンドラだったもの。デスクトップ版は HTTP を
経由せずこのクラスを直接呼ぶ。JSON に落とす必要がないので、座標は numpy 配列の
まま返して pyqtgraph にそのまま渡せるようにした (ブラウザ版はここで list 化する
のが一番重かった)。

読み込みは3段に分けてある。**画面が使えるようになるまでを短くするため**で、
全部そろうのを待つと最初の絵が出るまで7秒近くかかっていた:

  1. `load()`            注釈 21万行            0.3 秒  ← ここで検索と骨格が動く
  2. `load_connectivity()` 全結合 1億5185万行 (1GB)  3.4 秒  上流/下流の表
  3. `count_synapses()`  シナプス座標の行数        2.9 秒  見出しの数字だけ

シナプス座標 6.8GB そのものは問い合わせのたびにディスクを走査する (1回 2〜3秒)。

例外は `NotFound` / `Unavailable` に統一してある。UI 側はこれを捕まえて
ステータスバーに出すだけでよい。
"""

from __future__ import annotations

import functools
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .paths import DATA, OUT, ROOT, SKEL  # noqa: F401  (他所からも参照する)

SWC_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
           "skeletons-malecns/skeletons-swc/{}.swc")

MOTOR_CLASSES = ["vnc_motor", "cb_motor"]


class NotFound(Exception):
    """細胞型や経路が見つからない。"""


class Unavailable(Exception):
    """データファイルが手元にない (data/ を取得していない)。"""


@dataclass
class Neuron:
    """骨格1本。`verts` は線分の端点を2個ずつ並べたもの (GL に直接渡せる形)。"""

    body: int
    type: str
    verts: np.ndarray  # (2N, 3) float32


@dataclass
class TypeDetail:
    type: str
    n_bodies: int
    superclass: str
    bodies: list[int]
    total_out: int = 0
    total_in: int = 0
    has_connectivity: bool = False   # False なら上流/下流はまだ分からない
    downstream: list[dict] = field(default_factory=list)
    upstream: list[dict] = field(default_factory=list)
    flywire_type: str | None = None
    hemibrain_type: str | None = None
    dimorphism: str | None = None


class Brain:
    """注釈・結合・シナプス・骨格をまとめて持つ。

    `load()` は重いので UI スレッドから呼ばないこと。`progress` に関数を渡すと
    段階ごとに (説明, 0〜1) で呼び返す。
    """

    def __init__(self, data_dir: Path = DATA) -> None:
        self.dir = Path(data_dir)
        self._ready = False
        self.ann: pd.DataFrame | None = None
        self.w: pd.DataFrame | None = None
        self._graph = None
        self._syn = None
        self.types: pd.DataFrame | None = None
        self.n_synapses = 0

    # ---- 読み込み ------------------------------------------------------

    @property
    def ready(self) -> bool:
        """load() が最後まで通ったか。途中の中途半端な状態を ready と呼ばない。"""
        return self._ready

    def load(self, progress: Callable[[str, float], None] | None = None) -> None:
        def tell(msg: str, frac: float) -> None:
            if progress:
                progress(msg, frac)

        ann_p = self.dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        w_p = self.dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        if not ann_p.exists():
            raise Unavailable(
                f"{ann_p.name} がありません。\n"
                "scripts/00_fetch_data.py --malecns で取得してください。")

        tell("注釈を読み込み中 (21万ボディ)", 0.3)
        self.ann = pd.read_feather(ann_p)
        self.type_by_body = self.ann.set_index("bodyId")["type"]
        self.sc_by_body = self.ann.set_index("bodyId")["superclass"]
        self.types = self._type_summary()

        syn_p = self.dir / "syn-partners-male-cns-v1.0-minconf-0.5.feather"
        if syn_p.exists():
            import pyarrow.dataset as ds

            # dataset() は目録を開くだけ。行数を数えると 6.8GB を舐めるので
            # ここではやらない (count_synapses に回す)。
            self._syn = ds.dataset(syn_p, format="feather")
        self._ready = True
        tell("", 1.0)

    def load_connectivity(self, progress: Callable[[str, float], None] | None = None) -> None:
        """全結合 1GB。上流/下流の表とグラフ探索だけがこれを要る。

        素直に `pd.read_feather` すると 1億5185万行 x int64 x 3列 = 3.6GB を抱え、
        読み込みの峰は 7.4GB になる。bodyId は最大 15.7億、weight は最大 2591 で
        どちらも int32 に収まるので、**読みながら 32bit に落とす**。
        Arrow のスキャンに cast を混ぜると中間の int64 が丸ごとは出来ない。

            3.6GB / 峰 7.4GB / 2.9秒   ->   1.8GB / 峰 2.0GB / 1.4秒
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        w_p = self.dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        if not w_p.exists():
            raise Unavailable(f"{w_p.name} がありません。")
        if progress:
            progress("接続を読み込み中 (1GB / 1億5185万接続)", 0.5)

        cols = ("body_pre", "body_post", "weight")
        tbl = ds.dataset(w_p, format="feather").to_table(
            columns={c: pc.field(c).cast("int32") for c in cols})
        self.w = tbl.to_pandas(self_destruct=True, split_blocks=True)
        del tbl
        pa.default_memory_pool().release_unused()

    def count_synapses(self) -> int:
        """シナプス座標の行数。

        見出しに出すだけの数字だが、数えると毎回 1.8 秒かかる (pyarrow は
        覚えていてくれない)。ファイルの大きさと更新時刻をキーにして控えておく。
        """
        if self._syn is None or self.n_synapses:
            return self.n_synapses
        syn_p = self.dir / "syn-partners-male-cns-v1.0-minconf-0.5.feather"
        st = syn_p.stat()
        key = {"size": st.st_size, "mtime": int(st.st_mtime)}
        cache = OUT / "cache" / "syn_count.json"
        try:
            got = json.loads(cache.read_text(encoding="utf-8"))
            if got.get("size") == key["size"] and got.get("mtime") == key["mtime"]:
                self.n_synapses = int(got["rows"])
                return self.n_synapses
        except Exception:
            pass
        self.n_synapses = self._syn.count_rows()
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({**key, "rows": self.n_synapses}),
                             encoding="utf-8")
        except OSError:
            pass
        return self.n_synapses

    @property
    def has_connectivity(self) -> bool:
        return self.w is not None

    def headline(self) -> str:
        """ウィンドウ上部に出す1行。読めたところまでを書く。"""
        if not self.ready:
            return "読み込み前"
        parts = [f"Male CNS v1.0 — {len(self.ann):,} ボディ"]
        parts.append(f"{len(self.w):,} 接続" if self.has_connectivity else "接続を読み込み中…")
        if self._syn is None:
            parts.append("シナプス座標なし")
        elif self.n_synapses:
            parts.append(f"{self.n_synapses:,} シナプス")
        return " / ".join(parts)

    def graph(self):
        """weight>=10 の有向グラフ (280万辺)。初回だけ8秒ほどかかるので遅延構築。

        `nx.from_pandas_edgelist` より、辺を組にして渡すほうが3割速い (11秒 -> 8秒)。
        """
        if not self.has_connectivity:
            raise Unavailable("接続をまだ読み込んでいません")
        if self._graph is None:
            import networkx as nx

            strong = self.w[self.w["weight"] >= 10]
            g = nx.DiGraph()
            g.add_weighted_edges_from(zip(strong["body_pre"].to_numpy().tolist(),
                                          strong["body_post"].to_numpy().tolist(),
                                          strong["weight"].to_numpy().tolist()))
            self._graph = g
        return self._graph

    # ---- 検索と詳細 ----------------------------------------------------

    def _type_summary(self) -> pd.DataFrame:
        """細胞型ごとの (ボディ数, 代表の superclass) を1回だけ作る。

        もとは検索のたびに 21万行を groupby して superclass の最頻値を
        lambda で出していた。1回 1.8 秒かかるので、打鍵のたびに走らせる処理では
        なかった。ここで作っておけば検索は 11,751 行の部分一致だけになる。
        """
        hit = self.ann[self.ann["type"].fillna("") != ""]
        n = hit.groupby("type", observed=True)["bodyId"].size()
        pair = hit.groupby(["type", "superclass"], observed=True).size().rename("c")
        top = (pair.reset_index().sort_values("c", ascending=False)
               .drop_duplicates("type").set_index("type")["superclass"])
        out = pd.DataFrame({"n": n, "superclass": top.reindex(n.index).fillna("")})
        return out.sort_values("n", ascending=False)

    def search(self, q: str = "", limit: int = 60) -> tuple[list[dict], int]:
        hit = self.types
        if q:
            hit = hit[hit.index.str.contains(q, case=False, regex=False)]
        rows = hit.head(limit).reset_index().to_dict("records")
        return rows, int(len(hit))

    def type_detail(self, name: str, top: int = 15) -> TypeDetail:
        """細胞型1つぶんの情報。

        接続 (1GB) をまだ読んでいなければ、注釈から分かるぶんだけ返す。
        骨格を出すのに要るのは bodies だけなので、これで先に絵が出せる。
        """
        sel = self.ann[self.ann["type"] == name]
        if sel.empty:
            raise NotFound(f"{name} が見つかりません")
        row = sel.iloc[0]

        def opt(col: str) -> str | None:
            return None if pd.isna(row[col]) else str(row[col])

        base = dict(
            type=name,
            n_bodies=int(len(sel)),
            superclass=str(row["superclass"]),
            bodies=[int(b) for b in sel["bodyId"].head(24)],
            flywire_type=opt("flywireType"),
            hemibrain_type=opt("hemibrainType"),
            dimorphism=opt("dimorphism"),
        )
        if not self.has_connectivity:
            return TypeDetail(**base)

        ids = set(sel["bodyId"].astype(int))
        down = self.w[self.w["body_pre"].isin(ids)].copy()
        down["partner"] = down["body_post"].map(self.type_by_body)
        up = self.w[self.w["body_post"].isin(ids)].copy()
        up["partner"] = up["body_pre"].map(self.type_by_body)

        def agg(df: pd.DataFrame, col: str) -> list[dict]:
            gg = df.groupby("partner")["weight"].sum().nlargest(top)
            rows = []
            for k, v in gg.items():
                b = df.loc[df["partner"] == k, col].iloc[0]
                rows.append({"type": str(k), "weight": int(v),
                             "superclass": str(self.sc_by_body.get(b, ""))})
            return rows

        return TypeDetail(
            **base,
            total_out=int(down["weight"].sum()),
            total_in=int(up["weight"].sum()),
            has_connectivity=True,
            downstream=agg(down, "body_post"),
            upstream=agg(up, "body_pre"),
        )

    # ---- 骨格 ----------------------------------------------------------

    @staticmethod
    def _fetch_swc(body_id: int) -> Path | None:
        """SWC を手元に用意する (無ければ公開バケットから)。"""
        SKEL.mkdir(parents=True, exist_ok=True)
        p = SKEL / f"{body_id}.swc"
        if p.exists():
            return p
        tmp = p.with_suffix(".part")
        try:
            urllib.request.urlretrieve(SWC_URL.format(body_id), tmp)
            tmp.replace(p)   # 途中で切れたファイルを本番の名前で残さない
        except Exception:
            tmp.unlink(missing_ok=True)
            return None
        return p

    @functools.lru_cache(maxsize=512)
    def _skeleton(self, body_id: int) -> np.ndarray | None:
        """SWC を線分の端点列 (2N, 3) にして返す。

        SWC は「節点とその親」の表なので、親を持つ節点ごとに1本の線分を出せば
        枝分かれもそのまま描ける。GLLinePlotItem(mode='lines') は端点を2個ずつ
        読むので、この並びのまま渡せる。

        navis は使わない。読むのは7列の数表だけで、アプリを固める (PyInstaller)
        ときに navis とその依存を丸ごと抱えたくないため。
        """
        p = self._fetch_swc(body_id)
        if p is None:
            return None
        try:
            raw = pd.read_csv(p, sep=r"\s+", comment="#", header=None,
                              usecols=[0, 2, 3, 4, 6],
                              names=["id", "x", "y", "z", "parent"]).to_numpy()
        except Exception:
            return None
        if len(raw) < 2:
            return None
        ids = raw[:, 0].astype(np.int64)
        xyz = raw[:, 1:4].astype(np.float32)
        parent = raw[:, 4].astype(np.int64)

        order = {int(v): i for i, v in enumerate(ids)}
        child, mother = [], []
        for i, par in enumerate(parent):
            j = order.get(int(par))
            if j is not None:          # 根 (parent=-1) と壊れた参照は飛ばす
                child.append(i)
                mother.append(j)
        if not child:
            return None
        verts = np.empty((2 * len(child), 3), dtype=np.float32)
        verts[0::2] = xyz[child]
        verts[1::2] = xyz[mother]
        return verts

    def skeletons(self, bodies: list[int]) -> list[Neuron]:
        ids = [int(b) for b in list(bodies)[:24]]
        # ダウンロードだけ先に並列で片付ける。1本ずつ順に落とすと、手元に無い
        # 細胞型を開いたとき最初の表示までが目に見えて遅い。
        missing = [b for b in ids if not (SKEL / f"{b}.swc").exists()]
        if missing:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(self._fetch_swc, missing))
        out = []
        for b in ids:
            v = self._skeleton(int(b))
            if v is not None:
                out.append(Neuron(int(b), str(self.type_by_body.get(b)), v))
        return out

    # ---- シナプス ------------------------------------------------------

    def synapses(self, type_name: str, partner: str | None = None,
                 limit: int = 60000) -> dict:
        if self._syn is None:
            raise Unavailable("シナプス座標ファイル (6.8GB) がありません")
        import pyarrow.compute as pc

        ids = [int(b) for b in self.ann[self.ann["type"] == type_name]["bodyId"]]
        if not ids:
            raise NotFound(f"{type_name} が見つかりません")
        df = self._syn.to_table(
            filter=pc.field("body_pre").isin(ids) | pc.field("body_post").isin(ids)
        ).to_pandas()
        is_pre = df["body_pre"].isin(ids)

        def pts(sub: pd.DataFrame, cols: list[str]) -> np.ndarray:
            a = sub[cols].to_numpy(dtype=np.float32)
            if len(a) > limit:
                a = a[np.random.choice(len(a), limit, replace=False)]
            return a

        res = {
            "pre": pts(df[is_pre], ["x_pre", "y_pre", "z_pre"]),
            "post": pts(df[~is_pre], ["x_post", "y_post", "z_post"]),
            "n_pre": int(is_pre.sum()),
            "n_post": int((~is_pre).sum()),
            "rois": {str(k): int(v)
                     for k, v in df["primary_post"].value_counts().head(6).items() if v},
        }
        if partner:
            pids = set(int(b) for b in self.ann[self.ann["type"] == partner]["bodyId"])
            m = is_pre & df["body_post"].isin(pids)
            res["contact"] = pts(df[m], ["x_pre", "y_pre", "z_pre"])
            res["n_contact"] = int(m.sum())
            res["contact_rois"] = {
                str(k): int(v)
                for k, v in df.loc[m, "primary_post"].value_counts().head(3).items() if v}
        return res

    # ---- グラフ探索 ----------------------------------------------------

    def path(self, src: str, dst: str) -> dict:
        import networkx as nx

        G = self.graph()
        s = self.ann[self.ann["type"] == src]["bodyId"].astype(int).tolist()
        d = self.ann[self.ann["type"] == dst]["bodyId"].astype(int).tolist()
        if not s or not d:
            raise NotFound("細胞型が見つかりません")
        best = None
        for a in s[:4]:
            if a not in G:
                continue
            for b in d[:4]:
                if b not in G:
                    continue
                try:
                    p = nx.shortest_path(G, a, b)
                except nx.NetworkXNoPath:
                    continue
                if best is None or len(p) < len(best):
                    best = p
        if best is None:
            raise NotFound("weight>=10 の接続では経路が見つかりません")
        return {
            "path": [{"body": int(b), "type": str(self.type_by_body.get(b)),
                      "superclass": str(self.sc_by_body.get(b))} for b in best],
            "weights": [int(G[a][b]["weight"]) for a, b in zip(best, best[1:])],
        }

    def motor_reach(self, type_name: str, cutoff: int = 4) -> dict:
        import networkx as nx

        G = self.graph()
        ids = self.ann[self.ann["type"] == type_name]["bodyId"].astype(int).tolist()
        motor = set(self.ann[self.ann["superclass"].isin(MOTOR_CLASSES)]["bodyId"].astype(int))
        src = next((b for b in ids if b in G), None)
        if src is None:
            raise NotFound("強い接続 (weight>=10) を持つ細胞がありません")
        dist = nx.single_source_shortest_path_length(G, src, cutoff=cutoff)
        reached = {b: d for b, d in dist.items() if b in motor}
        by_step: dict[int, int] = {}
        for _, d in reached.items():
            by_step[d] = by_step.get(d, 0) + 1
        nearest = sorted(reached.items(), key=lambda kv: kv[1])[:10]
        return {
            "n_motor_total": len(motor),
            "n_reached": len(reached),
            "by_step": dict(sorted(by_step.items())),
            "nearest": [{"body": int(b), "type": str(self.type_by_body.get(b)),
                         "steps": int(d)} for b, d in nearest],
        }
