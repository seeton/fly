"""脳の点群と、そこへ感覚入力を配る対応づけ。

`escape_circuit.BrainPanel` (動画を書き出す matplotlib のパネル) と
`app/livesim.py` (アプリの中で走らせながら光らせるほう) の **両方** が使う。
navis も matplotlib も import しないのはそのためで、アプリはこの層だけを
取り込めば脳を光らせられる (`CLAUDE.md`: アプリ側で navis を import しない)。

**ここが守っている線**

    光るのは、その場所に **実際に入力が届いているとき** だけ。明るさの元は
    シミュレーションが出した量そのもの (複眼に映った像、関節が動いた速さ、
    匂いの濃度) で、こちらで作った揺らぎは一切入れない。過去に
    「Poisson 過程で明滅させる」「3次元の平面波を重ねる」を2度やって
    どちらも却下されている。画面に映っていたのは脳ではなく細工だった。

    入力を作っていない領域 (中央脳の大半) は暗いまま。それが正しい。

**offline と live の違いは物差しだけ**

    録り終えた記録なら「走り全体の 99 パーセンタイル」で割れる
    (`BrainPanel.attach_optic_drive`)。走らせながら描くときは未来が無いので
    `RunningTop` で追随する値を使う。どちらも **表示のための割り算** で、
    信号そのものには手を入れない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SYN = (ROOT / "data" / "malecns"
       / "syn-partners-male-cns-v1.0-minconf-0.5.feather")
CLOUD = OUT / "cns_synapse_cloud.npz"

# 領域 -> 入力の種類。ここに無い領域は「この環境では入力を作っていない」
OPTIC = ("ME", "LO", "LOP", "LA", "AME")      # 視葉。複眼から retinotopic に入る
OLFACTORY = ("AL",)                           # 触角葉。触角から入る
LEG = ("LegNp",)                              # 脚神経核。脚の固有受容が入る

# 外側キアズマ (ラミナ -> 髄質) で視野の前後が入れ替わる領域。
# 内側キアズマ (髄質 -> ロブラ) でもう一度入れ替わるので LO/LOP は元に戻る
CHIASM_FLIP = ("ME", "AME")

# 学習した価値を扱う場所。このシミュレーションは何も学習していないので
# **光らせない**。名前を持っておくのは「暗いことを示す」ため
MUSHROOM = ("CA", "gL", "aL", "bL", "PED")


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


def sample_cns_synapses(n_points: int = 45000, seed: int = 0,
                        refresh: bool = False, verbose: bool = True):
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

    if verbose:
        print(f"全CNSのシナプスを間引き中 "
              f"({SYN.stat().st_size/1e9:.1f} GB を流し読み) ...")
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
    if verbose:
        print(f"  {len(pts):,} 点。領域の上位: " + ", ".join(
            f"{k}({v})" for k, v in
            sorted(((k, int((roi == k).sum())) for k in set(roi.tolist())),
                   key=lambda kv: -kv[1])[:6]))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(CLOUD, pts=pts, roi=roi.astype(str))
    return pts, roi.astype(str)


def _grid_index(v: np.ndarray, n: int) -> np.ndarray:
    """座標の並びを 0..n-1 の格子の添字に落とす。

    端の外れ値で潰れないよう 2-98 パーセンタイルで正規化して切り詰める。
    """
    lo, hi = np.percentile(v, [2.0, 98.0])
    u = (v - lo) / max(float(hi - lo), 1e-9)
    return np.clip((u * n).astype(int), 0, n - 1)


class CloudMap:
    """点群 (位置と領域名) と、「どの点にどの入力が届くか」の対応。

    視葉の点は複眼の個眼格子 (n_omma x n_omma) の 1 マスに落とす。視葉は
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

    画像側の向きは複眼カメラの姿勢から測った。左右どちらの眼も画像の上が
    背側で、列は **左眼では前へ / 右眼では後ろへ** 増える。方位はさらに
    外側キアズマで前後が入れ替わるので、ME と AME だけもう一度ひっくり返す
    (`CHIASM_FLIP`)。

    **ここは想定**: 葉の中の位置を葉ごとのバウンディングボックスで正規化して
    格子に均等に割っている。実際の柱の並びは一様ではないし、葉のどの隅が
    視野のどの隅かは Male CNS の注釈からは読めない。向きの規則 (背腹は保存、
    前後はキアズマで反転) だけが解剖の事実。

    脚神経核は領域名 `LegNp(T2)(R)` がそのまま脚の名前なので、推測なしで
    関節と結べる。
    """

    def __init__(self, n_points: int = 45000, n_omma: int = 34, seed: int = 0,
                 refresh: bool = False, verbose: bool = True):
        self.pts, self.roi = sample_cns_synapses(n_points, seed, refresh, verbose)
        self.n_omma = int(n_omma)
        self._mask_cache: dict[str, np.ndarray] = {}
        self._build(verbose)

    # ------------------------------------------------------------------
    def _build(self, verbose: bool) -> None:
        n = len(self.pts)
        n_omma = self.n_omma
        self.kind = np.array([region_kind(r) for r in self.roi])
        self.optic_eye = np.full(n, -1, dtype=int)
        self.optic_idx = np.zeros(n, dtype=int)
        self.leg_group = np.full(n, -1, dtype=int)
        self.leg_names: list[str] = []
        if n == 0:
            self.base = np.zeros(0, dtype=object)
            self.side = np.zeros(0, dtype=object)
            return

        mid_x = float(np.median(self.pts[:, 0]))
        base = np.array([str(r).split("(")[0] for r in self.roi])
        side = np.array([region_side(r, x, mid_x)
                         for r, x in zip(self.roi, self.pts[:, 0])])
        self.base, self.side = base, side

        for b, s in sorted(set(zip(base[self.kind == "optic"].tolist(),
                                   side[self.kind == "optic"].tolist()))):
            m = (self.kind == "optic") & (base == b) & (side == s)
            q = self.pts[m]
            row = _grid_index(q[:, 1], n_omma)          # 背 -> 腹 = 行 0 -> n-1
            col = _grid_index(q[:, 2], n_omma)          # 前 -> 後
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
            seg = [q.rstrip(")") for q in str(self.roi[i]).split("(")
                   if q[:1] == "T" and q[1:2].isdigit()]
            if not seg:
                continue
            key = f"{seg[0]}_{'left' if side[i] == 'L' else 'right'}"
            if key not in self.leg_names:
                self.leg_names.append(key)
            self.leg_group[i] = self.leg_names.index(key)

        if verbose:
            counts = {k: int((self.kind == k).sum())
                      for k in ("optic", "olfactory", "leg", "other")}
            print("点群の内訳: " + " / ".join(f"{k} {v:,}" for k, v in counts.items())
                  + f"   視葉 -> {n_omma}x{n_omma} の個眼格子、"
                  + f"脚 -> {len(self.leg_names)} 群")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pts)

    def region_mask(self, key: str) -> np.ndarray:
        """領域名にあたる点の添字。

        `AL` のように頭だけ書けば左右とも、`AL(L)` と書けばその側だけ、
        `LegNp(T2)(R)` のように最後まで書けばその脚だけに当たる。
        """
        if key in self._mask_cache:
            return self._mask_cache[key]
        roi = np.asarray([str(r) for r in self.roi])
        idx = np.nonzero((roi == key) | np.char.startswith(roi, key + "("))[0]
        self._mask_cache[key] = idx
        return idx

    def check_omma(self, n_per_eye: int) -> None:
        """個眼の数が retinotopy を張ったときと合っているか確かめる。

        合っていないと添字が黙って別の個眼を指す (落ちない。**絵が静かに
        嘘になる**)。複眼の視野や個眼間角をいじると n_omma が変わる。
        """
        if n_per_eye != self.n_omma ** 2:
            raise ValueError(
                f"個眼の数が合わない: 入力は 1眼 {n_per_eye} 個 "
                f"({int(round(n_per_eye ** 0.5))}x)、"
                f"retinotopy は {self.n_omma}x{self.n_omma}。"
                f"CloudMap(..., n_omma=eyes.n_omma) を渡すこと")


class RunningTop:
    """走らせながら使う物差し — 大きい側の値に追随する。

    録り終えた記録なら走り全体の 99 パーセンタイルで割れるが、live には
    未来が無い。そこで毎回の上位分位点を取り、**上がるのは速く、下がるのは
    ゆっくり** 追わせる。実物の光受容器の光順応と同じ振る舞い
    (`fly_vision.TAU_ADAPT`)。offline と同じく **表示のための割り算** だけで、
    信号そのものは触らない。
    """

    def __init__(self, q: float = 99.0, tau_up: float = 0.05,
                 tau_down: float = 1.5, floor: float = 1e-9):
        self.q, self.tau_up, self.tau_down, self.floor = q, tau_up, tau_down, floor
        self.v: float | None = None

    def reset(self) -> None:
        self.v = None

    def __call__(self, x, dt: float) -> float:
        a = np.abs(np.asarray(x, dtype=float))
        now = float(a.max() if self.q >= 100.0 else np.percentile(a, self.q))
        if self.v is None:
            self.v = now
        else:
            tau = self.tau_up if now > self.v else self.tau_down
            self.v += (1.0 - np.exp(-dt / max(tau, 1e-9))) * (now - self.v)
        return max(float(self.v), self.floor)


class LiveGlow:
    """いまの感覚入力から、点群の明るさ 0..1 を毎フレーム作る。

    式は `escape_circuit.BrainPanel` と同じ:

        視葉    w_lum * 明るさ + w_mot * 像が動いた量、を gamma 乗
        その他  領域まるごとを 1 本の信号の大きさで光らせる

    違うのは物差しだけ (`RunningTop`)。
    """

    def __init__(self, cloud: CloudMap, w_lum: float = 0.32, w_mot: float = 0.85,
                 gamma_optic: float = 0.80, gamma_region: float = 1.0):
        self.cloud = cloud
        self.w_lum, self.w_mot = w_lum, w_mot
        self.gamma_optic, self.gamma_region = gamma_optic, gamma_region
        self._optic_m = cloud.kind == "optic"
        self._top_photo = RunningTop(99.0)
        self._top_motion = RunningTop(99.5)
        self._top_region: dict[str, RunningTop] = {}
        self.b = np.zeros(len(cloud))

    def reset(self) -> None:
        self._top_photo.reset()
        self._top_motion.reset()
        for t in self._top_region.values():
            t.reset()
        self.b[:] = 0.0

    # ------------------------------------------------------------------
    def update(self, dt: float, photo=None, motion=None,
               regions: dict | None = None,
               scales: dict | None = None) -> np.ndarray:
        """明るさを作り直して返す。

        photo / motion  (2, n, n) — 複眼の光受容器の出力と |photo - delayed|
        regions         {"AL(L)": 値, "WTct": 値, ...} — 領域まるごとの1本の信号。
                        **0..1 に均してから渡すのが既定** (物差し 1.0)。
        scales          キーごとの物差しの決め方。省略すると 1.0 (そのまま)。
                        数値を渡すとそれで割る。
                        文字列を渡すと **その名前の組で追随する物差しを共有**
                        する ({"LegNp(T1)(L)": "leg", ...})。組の中の相対が
                        見たいときだけ使う。

                        1本のスカラーを自分自身の追随最大で割ると、上がって
                        いる間はいつも 1.0 になり、何も分からない絵になる。
                        だから既定は固定の物差しで、0..1 への均し方は
                        **信号を作った側** が持つ (`app/livesim.py`)。
        """
        b = self.b
        b[:] = 0.0
        if photo is not None and motion is not None:
            ph = np.asarray(photo, dtype=np.float32).reshape(2, -1)
            mo = np.asarray(motion, dtype=np.float32).reshape(2, -1)
            self.cloud.check_omma(ph.shape[1])
            p_hi = self._top_photo(ph, dt)
            m_hi = self._top_motion(mo, dt)
            a = (self.w_lum * np.clip(ph / p_hi, 0.0, 1.0)
                 + self.w_mot * np.clip(mo / m_hi, 0.0, 1.0))
            a = np.clip(a, 0.0, 1.0) ** self.gamma_optic
            m = self._optic_m
            b[m] = a[self.cloud.optic_eye[m], self.cloud.optic_idx[m]]

        if regions:
            # 物差しは組ごとに共有する。組を書いていないキーは自分だけの物差し
            scales = scales or {}
            group_v: dict[str, float] = {}
            for key, val in regions.items():
                g = scales.get(key, 1.0)
                if isinstance(g, str):
                    group_v[g] = max(group_v.get(g, 0.0), abs(float(val)))
            hi = {}
            for g, v in group_v.items():
                top = self._top_region.setdefault(g, RunningTop(100.0))
                hi[g] = top([v], dt)
            for key, val in regions.items():
                idx = self.cloud.region_mask(key)
                if not len(idx):
                    continue
                g = scales.get(key, 1.0)
                div = hi[g] if isinstance(g, str) else max(float(g), 1e-12)
                x = np.clip(abs(float(val)) / div, 0.0, 1.0) ** self.gamma_region
                b[idx] = np.maximum(b[idx], x)
        return b

    # ------------------------------------------------------------------
    def level(self, key: str) -> float:
        """いまの絵での、その領域の明るさの平均 0..1。

        領域によっては点が小さく散っていて (外側角は 45,000 点中 810 点)、
        3D の絵だけでは光っているかどうかが読み取れない。同じ数字を文字でも
        出せるようにしておく。
        """
        idx = self.cloud.region_mask(key)
        return float(self.b[idx].mean()) if len(idx) else 0.0

    def n_lit(self, thr: float = 0.25) -> int:
        """光っている点の数。

        0.5 で数えると、景色を見ているだけの視葉 (中央値 0.2 前後) が
        まるごと「光っていない」ことになるので 0.25 で切る。
        """
        return int((self.b > thr).sum())
