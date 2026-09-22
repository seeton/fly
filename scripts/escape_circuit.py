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

# 点群と、そこへ入力を配る対応づけは brain_glow に置いてある。アプリ
# (PySide6) も同じものを使うので、navis / matplotlib の要らない側に分けた。
# 以前ここに定義があった名前は、そのまま引けるように通しておく。
from brain_glow import (CHIASM_FLIP, LEG, OLFACTORY, OPTIC,  # noqa: E402,F401
                        CloudMap, region_kind, region_side,
                        sample_cns_synapses)

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
        # 中枢神経系まるごとの地の活動。点群と「どの点にどの入力が届くか」は
        # brain_glow.CloudMap が持つ (アプリの live 表示と同じもの)
        self.map = CloudMap(max(n_cloud, 0), n_omma=n_omma, seed=seed,
                            refresh=refresh) if n_cloud > 0 else None
        if self.map is not None:
            self.cloud, self.cloud_roi = self.map.pts, self.map.roi
            self.kind = self.map.kind
            self.optic_eye, self.optic_idx = self.map.optic_eye, self.map.optic_idx
            self.leg_group, self.leg_names = self.map.leg_group, self.map.leg_names
        else:
            self.cloud = np.zeros((0, 3))
            self.cloud_roi = np.zeros(0, dtype=str)
            self.kind = np.zeros(0, dtype=object)
            self.optic_eye = self.optic_idx = self.leg_group = np.zeros(0, dtype=int)
            self.leg_names = []
        self.n_omma = n_omma
        self._optic = None
        self._regions = None

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
        # 個眼の数が retinotopy を張ったときと違うと、添字が黙って別の
        # 個眼を指す (落ちない。絵が静かに嘘になる)。複眼の視野や個眼間角を
        # いじると n_omma が変わるので、ここで必ず突き合わせる
        if self.map is not None:
            self.map.check_omma(ph.shape[2])
        p_hi = max(float(np.percentile(ph, 99.0)), 1e-9)
        m_hi = max(float(np.percentile(mo, 99.5)), 1e-9)
        b = (w_lum * np.clip(ph / p_hi, 0.0, 1.0)
             + w_mot * np.clip(mo / m_hi, 0.0, 1.0))
        self._optic = (t, np.clip(b, 0.0, 1.0) ** gamma)
        print(f"視葉の入力: {len(t)} 時刻 x 2 眼 x {ph.shape[2]} 個眼。"
              f"明るさの 99% 点 {p_hi:.3f} / 動きの 99.5% 点 {m_hi:.4f}")

    def region_mask(self, key: str) -> np.ndarray:
        """領域名にあたる点の添字 (`CloudMap.region_mask`)。"""
        if self.map is None:
            return np.zeros(0, dtype=int)
        return self.map.region_mask(key)

    def region_level(self, key: str) -> float:
        """直前に描いた絵での、その領域の明るさの平均 0..1。

        領域によっては点が小さく散っていて (外側角は 45,000 点中 810 点)、
        3D の絵だけでは光っているかどうかが読み取れない。同じ数字を
        文字でも出せるようにしておく。
        """
        cb = getattr(self, "cloud_b", None)
        if cb is None or not len(cb):
            return 0.0
        idx = self.region_mask(key)
        return float(cb[idx].mean()) if len(idx) else 0.0

    def attach_region_drive(self, times, series: dict, scale=None,
                            gamma: float = 0.6, label: str = "領域") -> None:
        """領域ごとの1本の信号で、その領域の点をまとめて光らせる。

        個眼の格子のような場所の対応が無い領域 — 触角葉に届く匂いの濃度、
        脚や翅の神経核に届く関節の動き — はこちらで配る。領域の中の
        どこが光るかは分からないので、領域まるごとを同じ明るさにする。

        series  {"AL(L)": (n_t,), "LH": (n_t,), ...}。キーは領域名。
        scale   表示のための割り算。数値1つなら全キーに同じ値、辞書なら
                キーごと、省略するとキーごとの 99 パーセンタイル。
                **ここだけが見せ方の都合** で、信号そのものは呼ぶ側が作る。
                同じ物理量の領域どうし (脚の6群など) は数値1つを渡して
                共通の物差しにしないと、動いていない脚まで明るくなる。
        """
        t = np.asarray(times, dtype=float)
        if self._regions is None:
            self._regions = (t, [])
        out = self._regions[1]
        for key, arr in series.items():
            idx = self.region_mask(key)
            if not len(idx):
                continue
            v = np.asarray(arr, dtype=float)
            if isinstance(scale, dict):
                hi = scale.get(key)
            else:
                hi = scale
            if hi is None:
                hi = float(np.percentile(np.abs(v), 99.0))
            hi = max(float(hi), 1e-12)
            out.append((idx, np.clip(np.abs(v) / hi, 0.0, 1.0) ** gamma))
            print(f"{label}の入力: {key:14s} {len(idx):5,} 点  物差し {hi:.4g}")

    def attach_leg_drive(self, times, speed: dict, gamma: float = 0.6) -> None:
        """脚の関節が動いた速さを、対応する脚神経核に配る (固有受容の入力)。

        speed は {"T2_right": (n_t,) [rad/s], ...}。6群に共通の物差しとして
        99 パーセンタイルを使う。立っているあいだは中央値 0 rad/s (脚の制御が
        落ち着いている) なので暗く、跳ぶ瞬間に光る。95 パーセンタイルだと
        0.25 rad/s と跳躍の立ち上がりで決まってしまい、跳んでいる間じゅう
        振り切れて中脚と前脚と後脚の区別が消える。
        """
        if not speed:
            return
        v = np.stack([np.asarray(a, dtype=float) for a in speed.values()], axis=1)
        hi = max(float(np.percentile(v, 99.0)), 1e-9)
        series = {}
        for k, arr in speed.items():
            seg, side = k.split("_")
            series[f"LegNp({seg})({'L' if side == 'left' else 'R'})"] = arr
        self.attach_region_drive(times, series, scale=hi, gamma=gamma,
                                 label="脚神経核")

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
        if self._regions is not None:
            times, parts = self._regions
            for idx, vals in parts:
                v = self._lerp(times, vals, t)                  # スカラー1つ
                b[idx] = np.maximum(b[idx], float(v))
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
            self.cloud_b = cb          # region_level() が読む
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
