"""逃避回路 (LC4/LPLC2 → DNp01 Giant Fiber → TTMn) の実測データと、
それを描く 3D パネル。

`23_synapse_movie.py` (脳だけの動画) と `24_escape_full.py` (脳・体・視界を
並べた動画) の両方から使う。

**どこまでが実データか**

    実測 … ニューロンの形 (骨格 SWC)、シナプスの3D座標、接続の有無と本数
    想定 … 伝導速度と各段の遅延。巨大繊維は太いので速い、という文献の
           一般則に基づく模式的なタイミングで、Male CNS から測った値ではない

シナプスの光る順番:

    視覚 → GF のシナプス … 発火の **原因** なので、GF が発火するまでの間に
        集団で次々に光る (LC4/LPLC2 の応答が閾値まで押し上げる過程)。
    GF → TTMn のシナプス … 発火が軸索を降りきった所で光る。
        光る時刻は各シナプスを **GF の軸索経路に射影した位置** で決める。

最初は入力シナプスも経路への射影で光らせていたが、射影すると経路の 0.29 地点に
落ちるため **発火の原因である入力が、発火の後に光る** ことになっていた。

**背景の点群 (中枢神経系ぜんぶ) がなぜ光るか**

    ここは2度やり直している。1度目は各シナプスを Poisson 過程で明滅させた。
    2度目は3次元の平面波を重ねた滑らかな明るさの場を張った。どちらも
    **こちらが作った数字** で、画面に映っていたのは脳ではなく細工だった。

    いまは作らない。**シミュレーションの入力をそのまま配る**:

        視葉 (ME/LO/LOP/LA/AME) … 複眼に映った像。明るさ (光受容器の出力) と
            像が動いた量 (|photo - delayed|、T4/T5 が食べている信号)。
            retinotopy を頼りに個眼の格子から葉の中の位置へ配る。
        脚の神経核 (LegNp)     … その脚の関節が動いた速さ (固有受容)。
        触角葉 (AL)            … 匂いの濃度。逃避の場面には匂い源が無いので
            この動画では暗いままで、それが正しい。

    入力を作っていない領域 (中央脳の大半など) は暗いまま。画面にもそう書く。
    「実際に入力がある所だけが、その入力で光る」が守っている線。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import navis  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "malecns"
SKEL = DATA / "skeletons"
OUT = ROOT / "out"
CACHE = OUT / "escape_circuit.npz"
SYN = DATA / "syn-partners-male-cns-v1.0-minconf-0.5.feather"
SWC_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
           "skeletons-malecns/skeletons-swc/{}.swc")

# 暗いときと光ったときの色
DIM = {"visual": "#17304f", "gf": "#4a1212", "motor": "#123318"}
HOT = {"visual": "#5ab6ff", "gf": "#ff3b30", "motor": "#34e07a"}
SYN_HOT = {"in": "#ffe14d", "out": "#ff7ae0"}

CLOUD_COL = "#cdd6e3"     # 光っているとき。無彩色にして逃避回路の色と混ざらないように
# 光っていないとき。真っ暗にすると「そこに脳が無い」ように見えてしまうので、
# 形だけは分かる程度に残す (点の位置は実測なので、暗い点も情報ではある)
CLOUD_DIM = "#262e3e"

LEGEND = [("中枢神経系ぜんぶ — 45,000 点で 3億1183万シナプスを代表", CLOUD_COL),
          ("LC4 / LPLC2 (視葉)", HOT["visual"]),
          ("DNp01 Giant Fiber", HOT["gf"]),
          ("TTMn (中脚の運動)", HOT["motor"]),
          ("視覚 → GF シナプス", SYN_HOT["in"]),
          ("GF → TTMn シナプス", SYN_HOT["out"])]

# 点群の明るさが何から来ているかの断り書き。呼ぶ側が画面に出す。
CLOUD_NOTE = ("点群の明るさはシミュレーションの入力そのもの — "
              "視葉は複眼に映った像 (明るさと動き)、脚の神経核は関節の動き。"
              "入力を作っていない領域は暗いまま")

JP_FONTS = ("C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/YuGothM.ttc",
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")


def jp_font(size: int):
    """日本語が出るフォントを1つ返す。無ければ既定に落とす。"""
    from PIL import ImageFont

    for p in JP_FONTS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def fetch_swc(body_id: int) -> Path:
    SKEL.mkdir(parents=True, exist_ok=True)
    p = SKEL / f"{body_id}.swc"
    if not p.exists():
        urllib.request.urlretrieve(SWC_URL.format(body_id), p)
    return p


def segments_xyz(n) -> list[np.ndarray]:
    """骨格を線分の集まりに変換 (Line3DCollection 用)。"""
    pos = n.nodes.set_index("node_id")[["x", "y", "z"]]
    out = []
    for seg in n.segments:
        c = pos.loc[seg].to_numpy()
        if len(c) > 1:
            out.append(c)
    return out


def axon_path(n, target_xyz) -> np.ndarray:
    """脳側の末端から target に一番近いノードまでの、骨格上の経路。"""
    pos = n.nodes.set_index("node_id")[["x", "y", "z"]]
    g = n.graph.to_undirected()
    start = int(n.nodes.loc[n.nodes["z"].idxmin(), "node_id"])
    d = np.linalg.norm(pos.to_numpy() - np.asarray(target_xyz), axis=1)
    end = int(pos.index[int(d.argmin())])
    return pos.loc[nx.shortest_path(g, start, end)].to_numpy()


def project_on_path(pts: np.ndarray, path: np.ndarray) -> np.ndarray:
    """各点を軸索経路に射影し、経路上の位置 (0-1) を返す。"""
    if len(pts) == 0:
        return np.zeros(0)
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    arc = arc / max(arc[-1], 1e-9)
    idx = np.empty(len(pts), dtype=int)
    for i in range(0, len(pts), 2048):
        chunk = pts[i:i + 2048]
        dist = np.linalg.norm(chunk[:, None, :] - path[None, :, :], axis=2)
        idx[i:i + 2048] = dist.argmin(axis=1)
    return arc[idx]


def build_circuit(n_visual: int = 8, n_motor: int = 2) -> dict:
    """接続データから逃避回路の body を選び、シナプス座標を抜き出す。"""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    ann = pd.read_feather(DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    ty = ann.set_index("bodyId")["type"]
    print("接続データ読み込み中 (1.05 GB) ...")
    w = pd.read_feather(DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")

    gf_ids = ann[ann["type"] == "DNp01"]["bodyId"].astype(int).tolist()
    up = w[w["body_post"].isin(gf_ids)].copy()
    up["type"] = up["body_pre"].map(ty)
    vis_ids: list[int] = []
    for t in ("LC4", "LPLC2"):
        vis_ids += (up[up["type"] == t].groupby("body_pre")["weight"].sum()
                    .nlargest(n_visual).index.astype(int).tolist())
    down = w[w["body_pre"].isin(gf_ids)].copy()
    down["type"] = down["body_post"].map(ty)
    motor_ids = (down[down["type"] == "TTMn"].groupby("body_post")["weight"].sum()
                 .nlargest(n_motor).index.astype(int).tolist())
    print(f"視覚 {len(vis_ids)} / GF {len(gf_ids)} / 運動 {len(motor_ids)}")

    print(f"シナプス座標を抽出中 ({SYN.stat().st_size/1e9:.1f} GB を絞り込む) ...")
    d = ds.dataset(SYN, format="feather")
    keep = gf_ids + vis_ids + motor_ids
    tbl = d.to_table(filter=pc.field("body_pre").isin(keep)
                     & pc.field("body_post").isin(keep))
    df = tbl.to_pandas()
    m_in = df["body_pre"].isin(vis_ids) & df["body_post"].isin(gf_ids)
    m_out = df["body_pre"].isin(gf_ids) & df["body_post"].isin(motor_ids)
    syn_in = df.loc[m_in, ["x_post", "y_post", "z_post"]].to_numpy(float)
    syn_out = df.loc[m_out, ["x_pre", "y_pre", "z_pre"]].to_numpy(float)
    print(f"  視覚 -> GF  {len(syn_in):,} シナプス")
    print(f"  GF -> TTMn  {len(syn_out):,} シナプス")
    return {"vis_ids": np.array(vis_ids), "gf_ids": np.array(gf_ids),
            "motor_ids": np.array(motor_ids), "syn_in": syn_in, "syn_out": syn_out}


CLOUD = OUT / "cns_synapse_cloud.npz"


def sample_cns_synapses(n_points: int = 45000, seed: int = 0,
                        refresh: bool = False):
    """中枢神経系まるごとのシナプスを、まんべんなく間引いて取る。

    位置に加えて **脳領域 (primary_post)** も持って帰る。どの領域が
    どの感覚の入力を受けているかが分かると、「実際に入力がある所が光る」
    という描き方ができる。

    先頭から N 行取ると body 順に並んでいて空間的にも領域的にも偏るので、
    ストリームで読みながら一定の確率で拾う。
    """
    if CLOUD.exists() and not refresh:
        z = np.load(CLOUD, allow_pickle=True)
        # 点の数もキャッシュの一部。少ない試し取りのファイルが残っていると、
        # 45,000 点を頼んだのに 1,000 点が返ってくる
        if "roi" in z.files and len(z["pts"]) >= n_points:
            return z["pts"][:n_points], z["roi"][:n_points]

    import pyarrow as pa
    import pyarrow.dataset as ds

    print(f"全CNSのシナプスを間引き中 ({SYN.stat().st_size/1e9:.1f} GB を流し読み) ...")
    rng = np.random.default_rng(seed)
    total = 311_830_000
    keep_p = min(1.0, n_points * 3.0 / total)
    d = ds.dataset(SYN, format="feather")
    chunks, rois = [], []
    for b in d.scanner(columns=["x_pre", "y_pre", "z_pre", "primary_post"],
                       batch_size=1 << 20).to_batches():
        take = np.nonzero(rng.random(b.num_rows) < keep_p)[0]
        if not len(take):
            continue
        chunks.append(np.column_stack(
            [b.column(c).to_numpy(zero_copy_only=False)[take]
             for c in ("x_pre", "y_pre", "z_pre")]))
        # **先に間引いてから** Python の文字列に直す。100万行のバッチから残すのは
        # 400行ほどなのに、バッチ全体を to_pylist() すると 3億個の Python 文字列を
        # 作ることになり、そこだけで数十分かかっていた
        rois.append(np.asarray(
            b.column("primary_post").take(pa.array(take)).to_pylist(), dtype=object))
    pts = np.vstack(chunks)
    roi = np.concatenate(rois)
    if len(pts) > n_points:
        sel = rng.choice(len(pts), n_points, replace=False)
        pts, roi = pts[sel], roi[sel]
    print(f"  {len(pts):,} 点。領域の上位: " + ", ".join(
        f"{k}({v})" for k, v in
        sorted(((k, int((roi == k).sum())) for k in set(roi.tolist())),
               key=lambda kv: -kv[1])[:6]))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(CLOUD, pts=pts, roi=roi.astype(str))
    return pts, roi.astype(str)


# 領域 -> 入力の種類。ここに無い領域は「この環境では入力を作っていない」
OPTIC = ("ME", "LO", "LOP", "LA", "AME")      # 視葉。複眼から retinotopic に入る
OLFACTORY = ("AL",)                           # 触角葉。触角から入る
LEG = ("LegNp",)                              # 脚神経核。脚の固有受容が入る

# 外側キアズマ (ラミナ -> 髄質) で視野の前後が入れ替わる領域。
# 内側キアズマ (髄質 -> ロブラ) でもう一度入れ替わるので LO/LOP は元に戻る
CHIASM_FLIP = ("ME", "AME")


def region_kind(roi: str) -> str:
    """領域名 -> optic / olfactory / leg / other。"""
    base = str(roi).split("(")[0]
    if base in OPTIC:
        return "optic"
    if base in OLFACTORY:
        return "olfactory"
    if base in LEG:
        return "leg"
    return "other"


def region_side(roi: str, x: float, mid_x: float) -> str:
    """領域名から左右を取る。名前に無ければ位置で決める (x は右 -> 左)。"""
    s = str(roi)
    if "(R)" in s:
        return "R"
    if "(L)" in s:
        return "L"
    return "L" if x > mid_x else "R"


def load_circuit(n_visual: int = 8, n_motor: int = 2, refresh: bool = False) -> dict:
    OUT.mkdir(exist_ok=True)
    if CACHE.exists() and not refresh:
        z = np.load(CACHE)
        print(f"抽出済みのシナプスを使う ({CACHE.name})。やり直すなら --refresh")
        return {k: z[k] for k in z.files}
    C = build_circuit(n_visual, n_motor)
    np.savez_compressed(CACHE, **C)
    return C


def flash(t, t0, rise: float, decay: float) -> np.ndarray:
    """発火の明るさ。t0 で立ち上がり、decay で減衰する。"""
    dt = np.asarray(t) - np.asarray(t0)
    up = np.clip(dt / max(rise, 1e-9), 0.0, 1.0)
    return up * np.exp(-np.clip(dt - rise, 0.0, None) / decay)


class BrainPanel:
    """逃避回路の 3D パネル。`frame(t)` で任意時刻の絵を返す。

    t_look / t_spike / t_arrive は動画上の時刻 [s]。呼ぶ側が物理シミュレーション
    の事象 (影を検出した時刻、脚を伸ばす時刻) に合わせて渡す。
    """

    def __init__(self, res=(900, 1000), n_visual: int = 8, n_motor: int = 2,
                 refresh: bool = False, t_look: float = 0.7,
                 t_spike: float = 2.5, t_arrive: float = 4.5,
                 spin: float = 50.0, elev: float = 14.0, seed: int = 0,
                 n_cloud: int = 45000, n_omma: int = 28):
        C = load_circuit(n_visual, n_motor, refresh)
        self.C = C
        print("骨格を読み込み中 ...")
        neurons, segs, seg_kind = {}, [], []
        for kind, ids in (("visual", C["vis_ids"]), ("gf", C["gf_ids"]),
                          ("motor", C["motor_ids"])):
            for b in ids.tolist():
                n = navis.read_swc(fetch_swc(int(b)))
                neurons[int(b)] = n
                for c in segments_xyz(n):
                    segs.append(c)
                    seg_kind.append(kind)
        self.segs = segs
        self.seg_kind = np.array(seg_kind)
        print(f"  {len(segs)} セグメント")

        gf = neurons[int(C["gf_ids"][0])]
        motor_center = (neurons[int(C["motor_ids"][0])]
                        .nodes[["x", "y", "z"]].to_numpy().mean(axis=0))
        self.path = axon_path(gf, motor_center)
        s_out = project_on_path(C["syn_out"], self.path)
        seg_mid = np.array([c.mean(axis=0) for c in segs])
        s_seg = project_on_path(seg_mid[self.seg_kind == "gf"], self.path)
        print(f"GF 軸索経路 {len(self.path)} ノード   "
              f"GF->TTMn シナプスの経路位置 中央値 {np.median(s_out):.2f}")

        self.t_look, self.t_spike, self.t_arrive = t_look, t_spike, t_arrive
        rng = np.random.default_rng(seed)
        t_in = t_look + (t_spike - t_look) * rng.random(len(C["syn_in"])) ** 0.55
        self.t_syn = np.concatenate([t_in, t_spike + (t_arrive - t_spike) * s_out])
        self.t_gf_seg = t_spike + (t_arrive - t_spike) * s_seg
        self.syn_pts = (np.vstack([C["syn_in"], C["syn_out"]])
                        if len(C["syn_out"]) else C["syn_in"])
        syn_col = ([SYN_HOT["in"]] * len(C["syn_in"]) +
                   [SYN_HOT["out"]] * len(C["syn_out"]))
        self.fired = np.zeros(len(self.syn_pts), dtype=bool)
        self.spin, self.elev = spin, elev
        # 中枢神経系まるごとの地の活動
        if n_cloud > 0:
            self.cloud, self.cloud_roi = sample_cns_synapses(n_cloud, seed, refresh)
        else:
            self.cloud = np.zeros((0, 3))
            self.cloud_roi = np.zeros(0, dtype=str)
        self._optic = None
        self._leg = None
        self._map_retinotopy(n_omma)

        # --- 描画の準備 ---
        all_xyz = np.vstack([np.vstack(segs), self.syn_pts]
                            + ([self.cloud] if len(self.cloud) else []))
        lo, hi = all_xyz.min(axis=0), all_xyz.max(axis=0)
        ctr, span = (lo + hi) / 2, (hi - lo)
        W, H = res
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100,
                              facecolor="#050508")
        ax = self.fig.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d",
                               facecolor="#050508")
        self.ax = ax
        ax.set_axis_off()
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.fill = False
            a.pane.set_edgecolor((0, 0, 0, 0))
        rxy = max(span[0], span[1]) * 0.56      # 回しても収まるよう横2軸は同幅
        ax.set_xlim(ctr[0] - rxy, ctr[0] + rxy)
        ax.set_ylim(ctr[1] - rxy, ctr[1] + rxy)
        # 脳が上に来るように z を反転 (SWC では脳側が z 小)
        ax.set_zlim(ctr[2] + span[2] * 0.54, ctr[2] - span[2] * 0.54)
        try:
            ax.set_box_aspect((1.0, 1.0, np.clip(span[2] / (2 * rxy), 0.6, 2.6)))
        except Exception:
            pass
        ax.set_position((-0.22, -0.26, 1.44, 1.50))   # 3D 軸まわりの余白を削る

        self.lc = Line3DCollection(segs, linewidths=1.0)
        ax.add_collection3d(self.lc)
        self.front = ax.plot([], [], [], marker="o", ms=13, color="#fff200",
                             alpha=0.0, lw=0)[0]
        # 地の活動の点群を先に置く (逃避回路より奥のレイヤー)
        if len(self.cloud):
            self.cloud_scat = ax.scatter(
                self.cloud[:, 0], self.cloud[:, 1], self.cloud[:, 2],
                s=1.0, c=CLOUD_DIM, depthshade=False, linewidths=0, zorder=2)
        else:
            self.cloud_scat = None
        self.scat = ax.scatter(self.syn_pts[:, 0], self.syn_pts[:, 1],
                               self.syn_pts[:, 2], s=1.0, c="#202430",
                               depthshade=False, linewidths=0, zorder=5)
        # 線は細く薄く。太いとシナプスが骨格に埋もれて見えない
        self.base_w = {"visual": 0.45, "gf": 1.6, "motor": 0.7}
        self.dim_rgb = {k: np.array(matplotlib.colors.to_rgb(v))
                        for k, v in DIM.items()}
        self.hot_rgb = {k: np.array(matplotlib.colors.to_rgb(v))
                        for k, v in HOT.items()}
        self.syn_rgb = np.array([matplotlib.colors.to_rgb(c) for c in syn_col])
        self.syn_dim = np.array(matplotlib.colors.to_rgb("#202430"))

    # ------------------------------------------------------------------
    @staticmethod
    def _grid_index(v: np.ndarray, n: int) -> np.ndarray:
        """座標の並びを 0..n-1 の格子の添字に落とす。

        端の外れ値で潰れないよう 2-98 パーセンタイルで正規化して切り詰める。
        """
        lo, hi = np.percentile(v, [2.0, 98.0])
        u = (v - lo) / max(float(hi - lo), 1e-9)
        return np.clip((u * n).astype(int), 0, n - 1)

    def _map_retinotopy(self, n_omma: int) -> None:
        """点群の各点を「どの入力が届く場所か」に対応づける。

        視葉の点は複眼の個眼格子 (n x n) の 1 マスに落とす。視葉は
        **retinotopic** — 個眼の並びが髄質 (ME)・ロブラ (LO)・ロブラ板 (LOP) の
        柱の並びにそのまま保たれている — ので、葉の中での位置がそのまま
        視野の中での向きにあたる。

        脳の座標系は点群から実際に確かめた:

            x  右 -> 左   ME(R) x~17000 / ME(L) x~80000
            y  背 -> 腹   SMP y~11900 / AL y~27600 / GNG y~40400
            z  前 -> 後   AL z~15300 / 萼 z~33900 / 神経索 z~100000

        使うのは y (仰角) と z (方位) の2軸だけ。x は視葉では **柱の深さ**
        (ラミナ側からロブラ側へ層が重なる向き) にあたり、同じ柱の中では
        どの層も視野の同じ点を見ているので落として構わない。

        画像側の向きは複眼カメラの姿勢から測った。左右どちらの眼も
        画像の上が背側で、列は **左眼では前へ / 右眼では後ろへ** 増える。
        方位はさらに外側キアズマで前後が入れ替わるので、ME と AME だけ
        もう一度ひっくり返す (`CHIASM_FLIP`)。

        **ここは想定**: 葉の中の位置を葉ごとのバウンディングボックスで
        正規化して格子に均等に割っている。実際の柱の並びは一様ではないし、
        葉のどの隅が視野のどの隅かは Male CNS の注釈からは読めない。
        向きの規則 (背腹は保存、前後はキアズマで反転) だけが解剖の事実。

        脚神経核は領域名 `LegNp(T2)(R)` がそのまま脚の名前なので、
        推測なしで関節と結べる。
        """
        n = len(self.cloud)
        self.n_omma = n_omma
        self.kind = np.array([region_kind(r) for r in self.cloud_roi])
        self.optic_eye = np.full(n, -1, dtype=int)
        self.optic_idx = np.zeros(n, dtype=int)
        self.leg_group = np.full(n, -1, dtype=int)
        self.leg_names: list[str] = []
        if n == 0:
            return

        mid_x = float(np.median(self.cloud[:, 0]))
        base = np.array([str(r).split("(")[0] for r in self.cloud_roi])
        side = np.array([region_side(r, x, mid_x)
                         for r, x in zip(self.cloud_roi, self.cloud[:, 0])])

        for b, s in sorted(set(zip(base[self.kind == "optic"].tolist(),
                                   side[self.kind == "optic"].tolist()))):
            m = (self.kind == "optic") & (base == b) & (side == s)
            q = self.cloud[m]
            row = self._grid_index(q[:, 1], n_omma)     # 背 -> 腹 = 行 0 -> n-1
            col = self._grid_index(q[:, 2], n_omma)     # 前 -> 後
            eye = 0 if s == "L" else 1                  # 左の視葉は左眼を見る
            if eye == 0:                                # 左眼の画像は列が前へ増える
                col = n_omma - 1 - col
            if b in CHIASM_FLIP:
                col = n_omma - 1 - col
            self.optic_eye[m] = eye
            self.optic_idx[m] = row * n_omma + col

        # 脚神経核は `LegNp(T2)(R)` のように **脚の番号まで名前に入っている**。
        # 領域名の頭 (LegNp) だけで束ねると T1/T2/T3 が 1 群に潰れるので、
        # 括弧の中まで見て 1 点ずつ振り分ける
        for i in np.nonzero(self.kind == "leg")[0]:
            seg = [q.rstrip(")") for q in str(self.cloud_roi[i]).split("(")
                   if q[:1] == "T" and q[1:2].isdigit()]
            if not seg:
                continue
            key = f"{seg[0]}_{'left' if side[i] == 'L' else 'right'}"
            if key not in self.leg_names:
                self.leg_names.append(key)
            self.leg_group[i] = self.leg_names.index(key)

        counts = {k: int((self.kind == k).sum())
                  for k in ("optic", "olfactory", "leg", "other")}
        print("点群の内訳: " + " / ".join(f"{k} {v:,}" for k, v in counts.items())
              + f"   視葉 -> {n_omma}x{n_omma} の個眼格子、"
              + f"脚 -> {len(self.leg_names)} 群")

    # ------------------------------------------------------------------
    def attach_optic_drive(self, times, photo, motion,
                           w_lum: float = 0.32, w_mot: float = 0.85,
                           gamma: float = 0.80) -> None:
        """複眼で測った入力を視葉に配る。

        times   動画上の時刻 [s] (n_t,)
        photo   個眼の明るさ (n_t, 2, n, n)。目に入っている光そのもの
        motion  |photo - delayed| (n_t, 2, n, n)。像が動いた量

        明るさは 2つの足し合わせ。**光は常に入っている** ので視葉は常に
        駆動されていて (w_lum の項)、そこに像が動いた所だけ強く乗る
        (w_mot の項)。どちらも描画された像から出た量で、作った揺らぎは無い。

        正規化の分母だけは表示の都合。光は 99 パーセンタイル、動きは
        99.5 パーセンタイルを 1 として、暗い側が潰れないよう gamma を掛ける。
        """
        t = np.asarray(times, dtype=float)
        ph = np.asarray(photo, dtype=np.float32).reshape(len(t), 2, -1)
        mo = np.asarray(motion, dtype=np.float32).reshape(len(t), 2, -1)
        p_hi = max(float(np.percentile(ph, 99.0)), 1e-9)
        m_hi = max(float(np.percentile(mo, 99.5)), 1e-9)
        b = (w_lum * np.clip(ph / p_hi, 0.0, 1.0)
             + w_mot * np.clip(mo / m_hi, 0.0, 1.0))
        self._optic = (t, np.clip(b, 0.0, 1.0) ** gamma)
        print(f"視葉の入力: {len(t)} 時刻 x 2 眼 x {ph.shape[2]} 個眼。"
              f"明るさの 99% 点 {p_hi:.3f} / 動きの 99.5% 点 {m_hi:.4f}")

    def attach_leg_drive(self, times, speed: dict, gamma: float = 0.6) -> None:
        """脚の関節が動いた速さを、対応する脚神経核に配る (固有受容の入力)。

        speed は {"T2_right": (n_t,) [rad/s], ...}。99 パーセンタイルで
        正規化する。立っているあいだは中央値 0 rad/s (脚の制御が落ち着いて
        いる) なので暗く、跳ぶ瞬間に光る。95 パーセンタイルだと 0.25 rad/s と
        跳躍の立ち上がりで決まってしまい、跳んでいる間じゅう振り切れて
        中脚と前脚と後脚の区別が消える。
        """
        if not self.leg_names or not speed:
            return
        t = np.asarray(times, dtype=float)
        v = np.stack([np.asarray(speed.get(k, np.zeros(len(t))), dtype=float)
                      for k in self.leg_names], axis=1)
        hi = max(float(np.percentile(v, 99.0)), 1e-9)
        self._leg = (t, np.clip(v / hi, 0.0, 1.0) ** gamma)
        print(f"脚神経核の入力: {v.shape[1]} 群。関節速度の 99% 点 {hi:.2f} rad/s")

    @staticmethod
    def _lerp(times: np.ndarray, arr: np.ndarray, t: float) -> np.ndarray:
        """記録した時系列から時刻 t の1枚を線形補間で取り出す。"""
        f = float(np.interp(t, times, np.arange(len(times), dtype=float)))
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, len(times) - 1)
        w = f - i0
        return arr[i0] * (1.0 - w) + arr[i1] * w

    def _cloud_brightness(self, t: float) -> np.ndarray:
        """時刻 t の点群の明るさ 0..1。

        入力がある領域だけが光る。作った揺らぎは一切入れない — ここを
        乱数や合成の波で埋めると、見えているのは脳ではなくこちらの細工になる。
        """
        b = np.zeros(len(self.cloud))
        if self._optic is not None:
            m = self.kind == "optic"
            a = self._lerp(self._optic[0], self._optic[1], t)   # (2, n*n)
            b[m] = a[self.optic_eye[m], self.optic_idx[m]]
        if self._leg is not None:
            m = self.leg_group >= 0
            v = self._lerp(self._leg[0], self._leg[1], t)       # (n_group,)
            b[m] = v[self.leg_group[m]]
        return b

    def frame(self, t: float, spin_frac: float = 0.0) -> tuple[np.ndarray, int, int]:
        """時刻 t の絵と、(いま光っているシナプス数, 放出済みの数) を返す。"""
        segs, seg_kind = self.segs, self.seg_kind
        cols = np.empty((len(segs), 4))
        lws = np.empty(len(segs))
        for kind, t0, rise, decay in (("visual", self.t_look, 0.25, 2.5),
                                      ("gf", self.t_gf_seg, 0.05, 1.4),
                                      ("motor", self.t_arrive, 0.10, 2.0)):
            m = seg_kind == kind
            x = np.clip(flash(t, t0, rise, decay), 0.0, 1.0)
            x = np.broadcast_to(np.atleast_1d(x), (int(m.sum()),))
            cols[m, :3] = (self.dim_rgb[kind][None, :] * (1 - x[:, None])
                           + self.hot_rgb[kind][None, :] * x[:, None])
            cols[m, 3] = 0.30 + 0.58 * x
            lws[m] = self.base_w[kind] * (1.0 + 1.4 * x)
        self.lc.set_color(cols)
        self.lc.set_linewidth(lws)

        # 逃避の一波だけ。以前はこの下に合成した「普段の弱い光」を敷いていたが、
        # あれは入力から出た量ではないので外した。GF は閾値の高い指令
        # ニューロンで、実物でも逃避のとき以外はここは沈黙している。
        sb = np.clip(flash(t, self.t_syn, 0.035, 0.5), 0.0, 1.0)
        sc = np.empty((len(self.syn_pts), 4))
        sc[:, :3] = (self.syn_dim[None, :] * (1 - sb[:, None])
                     + self.syn_rgb * sb[:, None])
        sc[:, 3] = 0.18 + 0.82 * sb
        self.scat.set_color(sc)
        self.scat.set_sizes(1.2 + 70.0 * sb ** 1.4)
        self.fired |= t >= self.t_syn

        if self.cloud_scat is not None:
            cb = self._cloud_brightness(t)
            cc = np.empty((len(self.cloud), 4))
            base = np.array(matplotlib.colors.to_rgb(CLOUD_DIM))
            live = np.array(matplotlib.colors.to_rgb(CLOUD_COL))
            cc[:, :3] = base[None, :] * (1 - cb[:, None]) + live[None, :] * cb[:, None]
            cc[:, 3] = 0.17 + 0.60 * cb
            self.cloud_scat.set_color(cc)
            self.cloud_scat.set_sizes(0.6 + 3.8 * cb ** 1.5)
            # 0.5 で数えると、景色を見ているだけの視葉 (中央値 0.2 前後) が
            # まるごと「光っていない」ことになる。0.25 で切る
            self.n_cloud_lit = int((cb > 0.25).sum())
        else:
            self.n_cloud_lit = 0

        if self.t_spike <= t <= self.t_arrive + 0.1:
            fr = np.clip((t - self.t_spike) / max(self.t_arrive - self.t_spike, 1e-9),
                         0.0, 1.0)
            pt = self.path[min(int(fr * (len(self.path) - 1)), len(self.path) - 1)]
            self.front.set_data([pt[0]], [pt[1]])
            self.front.set_3d_properties([pt[2]])
            self.front.set_alpha(float(np.clip(1.2 - max(0.0, t - self.t_arrive) * 6,
                                               0, 1)))
        else:
            self.front.set_alpha(0.0)

        self.ax.view_init(elev=self.elev + 5 * np.sin(2 * np.pi * spin_frac),
                          azim=-72 + self.spin * spin_frac)
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
        return img, int((sb > 0.25).sum()), int(self.fired.sum())

    def stage(self, t: float) -> str:
        if t < self.t_look:
            return "静止"
        if t < self.t_spike:
            return "影が迫る — LC4 / LPLC2 が応答"
        if t < self.t_arrive:
            return "Giant Fiber (DNp01) 発火 — 軸索を降りる"
        return "TTMn へ伝達 — 中脚が伸びる"
