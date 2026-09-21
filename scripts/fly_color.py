"""ハエの色覚 — 光受容器の分光感度と、場面の色を「ハエのチャンネル」に直す。

これまで花の検出を `(赤 - 緑) > 0.12` と書いていた。**ハエの色覚に赤チャンネルは
無い**ので、これはカメラの RGB という人間側の都合を使っていたことになる。

ショウジョウバエの個眼は光受容器 8 個でできている:

    R1-6   Rh1  約478 nm (広帯域)   明暗と **動き**。ここが主に色盲
    R7 pale   Rh3  約345 nm (UV)
    R8 pale   Rh5  約437 nm (青)
    R7 yellow Rh4  約375 nm (UV)
    R8 yellow Rh6  約508 nm (緑)

個眼は pale 約30% / yellow 約70% がランダムに混ざっている。
**赤はほとんど見えない。かわりに紫外線が見える。**

動きを見る経路 (R1-6 → L1-L3 → T4/T5) は主に色盲なので、
`fly_vision.py` の相関型検出器を輝度だけで回しているのは実物に合っている。
色が要るのは「それが何か」を決める側 (花かどうか) のほう。

**どこまでが実測で、どこからが仮定か**

    実測 … 光受容器のピーク波長。視物質テンプレート (Govardovskii 2000) の形。
    仮定 … **場面の材質の反射スペクトル**。MuJoCo は RGB しか持たないので、
           草・土・空・花びらがどんな分光反射を持つかはここで与えている。
           形は植生や土のよく知られた特徴 (緑のピーク、土の単調増加、
           空の短波長優位、花びらの UV 吸収) に沿わせたが、実測値ではない。

使い方:
    from fly_color import fly_channels
    uv, blue, green, broad = fly_channels("grass")
"""

from __future__ import annotations

import numpy as np

# 波長格子 [nm]。ハエは 300 nm から見えるので、人間より下を含める
LAM = np.arange(300.0, 701.0, 2.0)

# 光受容器のピーク波長 [nm]
PEAKS = {"Rh1": 478.0, "Rh3": 345.0, "Rh4": 375.0, "Rh5": 437.0, "Rh6": 508.0}


def govardovskii(lam: np.ndarray, lmax: float) -> np.ndarray:
    """視物質の分光感度テンプレート (Govardovskii et al. 2000, A1 型)。

    alpha 帯 + beta 帯。ピーク波長を1つ決めれば形が決まる。
    """
    x = lmax / lam
    a = 0.8795 + 0.0459 * np.exp(-((lmax - 300.0) ** 2) / 11940.0)
    alpha = 1.0 / (np.exp(69.7 * (a - x)) + np.exp(28.0 * (0.922 - x))
                   + np.exp(-14.9 * (1.104 - x)) + 0.674)
    lmb = 189.0 + 0.315 * lmax
    b = -40.5 + 0.195 * lmax
    beta = 0.26 * np.exp(-(((lam - lmb) / b) ** 2))
    s = alpha + beta
    return s / s.max()


SENS = {k: govardovskii(LAM, v) for k, v in PEAKS.items()}

# 昼光の相対分光分布。短波長がやや強い平坦な照明として扱う
ILLUM = 1.0 + 0.45 * np.exp(-((LAM - 450.0) ** 2) / (2 * 120.0 ** 2))


def _reflectance(points) -> np.ndarray:
    """(波長, 反射率) の折れ点から反射スペクトルを作る。"""
    lam, r = zip(*points)
    return np.clip(np.interp(LAM, lam, r), 0.0, 1.0)


# --- 場面の材質の反射スペクトル (**仮定値**) ---
# 実測ではない。よく知られた形に沿わせただけ。
MATERIALS = {
    # 葉緑素: UV と青を吸収、550 nm 付近に緑のピーク、赤で再び落ちる
    "foliage": _reflectance([(300, 0.03), (400, 0.04), (470, 0.05), (550, 0.22),
                             (610, 0.08), (700, 0.35)]),
    # 土・枯葉: 短波長を吸収して長波長へ単調に増える
    "soil": _reflectance([(300, 0.04), (400, 0.08), (500, 0.16), (600, 0.28),
                          (700, 0.38)]),
    # 空 (散乱光): 短波長優位。UV が強いのがハエにとって重要
    "sky": _reflectance([(300, 0.85), (400, 0.80), (500, 0.55), (600, 0.35),
                         (700, 0.25)]),
    # 花びら: 可視では明るく、**UV を吸収する** 型。多くの花がこの型で、
    # ハエ・ハチには緑の葉に対して強い対比になる
    "petal": _reflectance([(300, 0.04), (360, 0.05), (410, 0.45), (500, 0.70),
                           (600, 0.72), (700, 0.72)]),
    # 花の中心 (蜜標)。UV をさらに強く吸収する
    "petal_center": _reflectance([(300, 0.03), (400, 0.10), (500, 0.30),
                                  (600, 0.40), (700, 0.45)]),
    # 明るい地面 (アリーナの市松)
    "pale_ground": _reflectance([(300, 0.30), (400, 0.42), (500, 0.55),
                                 (600, 0.62), (700, 0.65)]),
    # 黒い物体 (迫る捕食者)
    "dark": _reflectance([(300, 0.02), (700, 0.03)]),
}


def excitations(refl: np.ndarray) -> dict[str, float]:
    """反射スペクトル -> 各光受容器の励起。照明込みで積分し、白で規格化する。"""
    white = np.ones_like(LAM)
    out = {}
    for k, s in SENS.items():
        num = float(np.trapezoid(s * ILLUM * refl, LAM))
        den = float(np.trapezoid(s * ILLUM * white, LAM))
        out[k] = num / max(den, 1e-12)
    return out


def fly_channels(material: str) -> tuple[float, float, float, float]:
    """材質名 -> (UV, 青, 緑, 広帯域) の励起。

    UV は pale (Rh3) と yellow (Rh4) の混合比 3:7 で平均する。
    広帯域は R1-6 (Rh1) で、動きを見る経路が使うチャンネル。
    """
    e = excitations(MATERIALS[material])
    uv = 0.3 * e["Rh3"] + 0.7 * e["Rh4"]
    return uv, e["Rh5"], e["Rh6"], e["Rh1"]


# 場面のマテリアル名 (world_scene / 24_escape_full) -> 材質の分類
NAME_TO_MATERIAL = {
    "grass": "foliage", "arena_green": "foliage", "leaf": "foliage",
    "ground": "soil", "arena_ground": "pale_ground", "checker": "soil",
    "sky": "sky", "arena_sky": "sky", "wall": "soil",
    "petal": "petal", "flower": "petal", "disc": "petal_center",
    "predator": "dark",
}


def classify(name: str) -> str:
    """マテリアル名から材質の分類を推測する。分からなければ土扱い。"""
    n = (name or "").lower()
    for key, mat in NAME_TO_MATERIAL.items():
        if key in n:
            return mat
    return "soil"


def summary() -> str:
    lines = ["材質            UV      青      緑    広帯域(R1-6)"]
    for m in MATERIALS:
        uv, b, g, br = fly_channels(m)
        lines.append(f"{m:14s} {uv:6.3f}  {b:6.3f}  {g:6.3f}  {br:6.3f}")
    return "\n".join(lines)


# --- 広帯域 R1-6 を UV/青/緑 から復元する係数 ---
# 7 材質で最小二乗を取ると最大誤差 0.0024 (値域 0.024-0.656) で再現できる。
# 描画は RGB の 3 枠しか持てないので、UV/青/緑 を積んで R1-6 はここから作る。
BROAD_W = np.array([-0.099, 0.585, 0.512])

# 花を見分ける指標: (青+緑)/2 と UV の対立。
# 候補を比べたところ、この形がいちばん花と葉を離した (花 +0.571 / 葉 +0.35、
# 土 +0.32、空 -0.10、明るい地面 +0.13、黒 +0.05)。
# 昆虫が花を見つけるのに UV と可視の対立を使うのはよく知られた性質でもある。
PETAL_THRESHOLD = 0.46


def install_fly_colors(model, gain: float | None = None):
    """モデルの色を「ハエのチャンネル」に置き換える一式を作る。

    戻り値は差し替え用の配列の辞書。複眼を描く直前に入れ替え、描いたら戻す。
    人向けのカメラは元の色のままにしておきたいため。

    R,G,B の枠に (UV, 青, 緑) を積む。テクスチャは **模様を残したまま**
    そのマテリアルのハエ色に寄せる (画素の明るさを平均で割った比をかける)。
    模様を消すと視覚の流れが取れなくなるので必須。

    **陰影を切って自己発光にする。** そのままだと場面の照明で画素値が
    0.03-0.05 まで落ち、8bit に量子化した時点でチャンネル比が壊れる
    (花の前でも草むらでも対立指標の中央値が同じ 0.20 になり、判別できなかった)。
    ここで計算している励起はすでに照明を含んだ反射率なので、描画側で
    もう一度陰影をかける必要がない。gain は全材質に共通の倍率で、
    比を変えずに 8bit の範囲を使い切るためのもの。
    """
    if gain is None:
        peak = max(max(fly_channels(m_)[:3]) for m_ in MATERIALS)
        gain = 0.95 / peak

    mat = model.mat_rgba.copy()
    for i in range(model.nmat):
        uv, b, g, _ = fly_channels(classify(model.material(i).name))
        mat[i, :3] = np.clip(np.array([uv, b, g]) * gain, 0.0, 1.0)

    geom = model.geom_rgba.copy()
    for i in range(model.ngeom):
        if model.geom_matid[i] < 0:
            uv, b, g, _ = fly_channels(classify(model.geom(i).name))
            geom[i, :3] = np.clip(np.array([uv, b, g]) * gain, 0.0, 1.0)

    tex = model.tex_data.copy()
    owner = {}
    for i in range(model.nmat):
        for t in np.atleast_1d(model.mat_texid[i]):
            if t >= 0:
                owner.setdefault(int(t), i)
    for t in range(model.ntex):
        nch = int(model.tex_nchannel[t])
        if nch < 3:
            continue
        adr = int(model.tex_adr[t])
        n = int(model.tex_width[t]) * int(model.tex_height[t])
        px = tex[adr:adr + n * nch].reshape(n, nch).astype(np.float32)
        luma = px[:, :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        mean = max(float(luma.mean()), 1e-6)
        mi = owner.get(t)
        name = model.material(mi).name if mi is not None else model.texture(t).name
        uv, b, g, _ = fly_channels(classify(name))
        col = np.clip(np.array([uv, b, g], dtype=np.float32) * gain, 0, 1) * 255.0
        px[:, :3] = np.clip((luma[:, None] / mean) * col[None, :], 0, 255)
        tex[adr:adr + n * nch] = px.reshape(-1).astype(np.uint8)

    return {"mat_rgba": mat, "geom_rgba": geom, "tex_data": tex,
            "mat_emission": np.ones_like(model.mat_emission),
            "mat_specular": np.zeros_like(model.mat_specular),
            "mat_shininess": np.zeros_like(model.mat_shininess),
            "mat_reflectance": np.zeros_like(model.mat_reflectance)}


def decode(rgb: np.ndarray):
    """ハエ色で描いた像 -> (UV, 青, 緑, 広帯域 R1-6)。"""
    uv, b, g = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    broad = BROAD_W[0] * uv + BROAD_W[1] * b + BROAD_W[2] * g
    return uv, b, g, broad


def petal_opponency(rgb: np.ndarray) -> np.ndarray:
    """(青+緑)/2 と UV の対立。花で大きく、葉で小さい。"""
    uv, b, g, _ = decode(rgb)
    vis = 0.5 * (b + g)
    return (vis - uv) / (vis + uv + 1e-6)


if __name__ == "__main__":
    print(__doc__)
    print(summary())
    print()
    uvf, bf, gf, _ = fly_channels("petal")
    uvl, bl, gl, _ = fly_channels("foliage")
    print(f"花びら vs 葉:")
    print(f"  UV 対比 {(uvf-uvl)/(uvf+uvl):+.3f}")
    print(f"  緑 対比 {(gf-gl)/(gf+gl):+.3f}")
    print(f"  UV-緑 の対立 (花) {(uvf-gf)/(uvf+gf):+.3f}   (葉) {(uvl-gl)/(uvl+gl):+.3f}")
