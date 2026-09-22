"""ハエが飛ぶ風景を作る。

スケールの基準 (ショウジョウバエ):
    体長 0.25 cm、翼開長 0.5 cm、飛行速度 15-50 cm/s

  style="meadow" (既定) … 草むら。
      草の茎は半径 0.03-0.09 cm (=0.3-0.9 mm) で、ハエの体より細い。
      隙間は 1-4 cm (翼開長の 2-8 倍) なので、すり抜けるには操縦がいる。
      奥に花を1つ置く。これが目標 (報酬) になる。

  style="posts" … 以前の柱。半径 1.2-3.5 cm あり、ハエから見ると木の幹。
      視覚の流れを作るだけなら十分だが、草むらを縫う話には大きすぎた。

花の位置はサイト `flower` で引ける (data.site_xpos)。
"""

from __future__ import annotations

import numpy as np

GROUND_HALF = 400.0
WALL_DIST = 320.0
WALL_H = 120.0

# 花はここに置く (ハエは原点付近から +x へ飛ぶ)
# 花の位置。視覚で捉えられるのは 8-10 cm 以内なので、ここまで遠いと
# 目だけでは見つけられない。遠距離は嗅覚 (scripts/fly_smell.py) で寄る。
FLOWER_POS = np.array([60.0, 0.0, 9.0])

# 「見た目は同じで匂いだけ違う花」を2つ置く配置。
# 視覚だけでは区別できないので、どちらへ行くかは嗅覚が決める。
# 手前にあるのが好まない花 (おとり)、奥にあるのが好む花。
DECOY_POS = np.array([42.0, -5.0, 9.0])     # 先に出会う。匂いが違う
TARGET_POS = np.array([66.0, 5.0, 9.0])     # 本命


def _mat(spec, name, rgba, shininess=0.1):
    m = spec.add_material()
    m.name = name
    m.rgba = rgba
    m.shininess = shininess
    return m


def _sky_and_ground(spec, square_cm=2.5):
    import mujoco

    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.45, 0.62, 0.85]
    sky.rgb2 = [0.12, 0.18, 0.28]
    sky.width, sky.height = 64, 256

    chk = spec.add_texture()
    chk.name = "checker"
    chk.type = mujoco.mjtTexture.mjTEXTURE_2D
    chk.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    chk.rgb1 = [0.24, 0.20, 0.15]
    chk.rgb2 = [0.34, 0.30, 0.22]
    chk.width, chk.height = 300, 300

    gmat = spec.add_material()
    gmat.name = "ground_mat"
    gmat.textures[1] = "checker"          # 1 = mjTEXROLE_RGB
    # texuniform=True のとき texrepeat は「単位長あたりの繰り返し数」
    gmat.texrepeat = [1.0 / square_cm, 1.0 / square_cm]
    gmat.texuniform = True
    gmat.reflectance = 0.0

    g = spec.worldbody.add_geom()
    g.name = "scene_ground"
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [GROUND_HALF, GROUND_HALF, 1.0]
    g.pos = [0, 0, 0]
    g.material = "ground_mat"
    g.contype, g.conaffinity = 1, 1


def _cameras(spec, close=True):
    import mujoco

    def look_at_origin(pos):
        pos = np.asarray(pos, dtype=float)
        f = -pos / np.linalg.norm(pos)
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(f @ up)) > 0.98:
            up = np.array([0.0, 1.0, 0.0])
        xc = np.cross(f, up)
        xc /= np.linalg.norm(xc)
        yc = np.cross(-f, xc)
        R = np.column_stack([xc, yc, -f])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        return q

    cams = ([("scene_follow", [-5.0, -3.8, 1.8], 55.0),
             ("scene_side", [0.0, -6.0, 0.8], 50.0),
             ("scene_high", [-2.5, 0.0, 7.0], 60.0),
             ("scene_wide", [-22.0, -16.0, 8.0], 55.0)] if close else
            [("scene_follow", [-13.0, -10.0, 5.0], 55.0),
             ("scene_side", [0.0, -14.0, 2.0], 50.0),
             ("scene_high", [-5.0, 0.0, 15.0], 60.0),
             ("scene_wide", [-30.0, -22.0, 12.0], 55.0)])
    for name, pos, fovy in cams:
        c = spec.worldbody.add_camera()
        c.name = name
        c.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
        c.targetbody = "thorax"
        c.pos = pos
        c.fovy = fovy
        c.quat = look_at_origin(pos)


def _lights(spec):
    import mujoco

    lt = spec.worldbody.add_light()
    lt.name = "sun"
    lt.pos = [30, -30, 150]
    lt.dir = [-0.2, 0.2, -1.0]
    lt.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    lt.diffuse = [0.85, 0.85, 0.80]
    lt.specular = [0.25, 0.25, 0.25]


def _flower(spec, pos, n_petals=7, tag="flower", make_materials=True,
            parent=None):
    """花。黄色い中心 + 花びら + 茎。位置はサイト `<tag>` で引ける。

    2つ置くときも **同じ材質** を使う。見た目で区別できてしまうと
    「匂いで選ぶ」話にならないため。

    parent に body を渡すとその中に生やす。mocap の body を渡せば
    **走らせながら花を動かせる** (`add_movable_flower`)。その場合 pos は
    body から見た相対位置なので、ふつうは原点の上に立てる。
    """
    import mujoco

    wb = spec.worldbody if parent is None else parent
    if make_materials:
        _mat(spec, "petal_mat", [0.95, 0.38, 0.58, 1.0], shininess=0.3)
        _mat(spec, "center_mat", [0.98, 0.82, 0.16, 1.0], shininess=0.5)
        _mat(spec, "fstem_mat", [0.25, 0.45, 0.20, 1.0])

    stem = wb.add_geom()
    stem.name = f"{tag}_stem"
    stem.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    stem.size = [0.10, pos[2] / 2.0, 0.0]
    stem.pos = [pos[0], pos[1], pos[2] / 2.0]
    stem.material = "fstem_mat"
    stem.contype = stem.conaffinity = 0

    cen = wb.add_geom()
    cen.name = f"{tag}_center"
    cen.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    cen.size = [0.45, 0.45, 0.18]
    cen.pos = list(pos)
    cen.material = "center_mat"
    cen.contype = cen.conaffinity = 0

    for i in range(n_petals):
        a = 2 * np.pi * i / n_petals
        g = wb.add_geom()
        g.name = f"{tag}_petal{i}"
        g.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        g.size = [0.62, 0.30, 0.07]
        g.pos = [pos[0] + 0.82 * np.cos(a), pos[1] + 0.82 * np.sin(a), pos[2] + 0.02]
        g.quat = [np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]
        g.material = "petal_mat"
        g.contype = g.conaffinity = 0

    s = wb.add_site()
    s.name = tag
    s.pos = list(pos)
    s.size = [0.15, 0.15, 0.15]
    s.rgba = [1.0, 1.0, 1.0, 0.0]


def _meadow(spec, rng, n_blades=260, collide=True, corridor=1.6):
    """草むら。ハエの体より細い茎を散らす。通り道は少しだけ空ける。"""
    import mujoco

    wb = spec.worldbody
    greens = [[0.28, 0.48, 0.22, 1], [0.35, 0.55, 0.26, 1],
              [0.22, 0.40, 0.18, 1], [0.42, 0.58, 0.30, 1],
              [0.50, 0.55, 0.25, 1]]
    for i, c in enumerate(greens):
        _mat(spec, f"grass{i}", c)

    placed, tries = 0, 0
    while placed < n_blades and tries < n_blades * 20:
        tries += 1
        x = rng.uniform(-15.0, 85.0)
        y = rng.uniform(-25.0, 25.0)
        if x < 3.0 and abs(y) < 3.0:                       # 出発点
            continue
        near_flower = any(np.hypot(x - f[0], y - f[1]) < 3.0
                          for f in (FLOWER_POS, DECOY_POS, TARGET_POS))
        if near_flower:                                    # 花のまわり
            continue
        if abs(y) < corridor and rng.random() < 0.75:      # 通り道を少し空ける
            continue
        r = rng.uniform(0.03, 0.09)      # 0.3-0.9 mm。ハエの体(2.5mm)より細い
        h = rng.uniform(3.0, 18.0)
        tilt = rng.uniform(0.0, 0.22)
        ang = rng.uniform(0.0, 2 * np.pi)
        g = wb.add_geom()
        g.name = f"blade{placed}"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [r, h / 2.0, 0.0]
        g.pos = [x, y, h / 2.0]
        g.quat = [np.cos(tilt / 2), np.sin(tilt / 2) * np.cos(ang),
                  np.sin(tilt / 2) * np.sin(ang), 0.0]
        g.material = f"grass{placed % len(greens)}"
        g.contype = g.conaffinity = 1 if collide else 0
        placed += 1

    for i in range(45):
        x = rng.uniform(-10.0, 80.0)
        y = rng.uniform(-22.0, 22.0)
        if x < 4.0 and abs(y) < 3.0:
            continue
        g = wb.add_geom()
        g.name = f"leaf{i}"
        g.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        g.size = [rng.uniform(0.6, 1.6), rng.uniform(0.25, 0.6), 0.04]
        g.pos = [x, y, rng.uniform(1.0, 9.0)]
        a = rng.uniform(0, 2 * np.pi)
        g.quat = [np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]
        g.material = f"grass{i % len(greens)}"
        g.contype = g.conaffinity = 1 if collide else 0


def _posts(spec, rng, n_posts=40):
    import mujoco

    wb = spec.worldbody
    cols = [[0.75, 0.35, 0.25, 1], [0.30, 0.55, 0.35, 1],
            [0.60, 0.60, 0.30, 1], [0.35, 0.40, 0.65, 1]]
    for i in range(n_posts):
        _mat(spec, f"post_mat{i}", cols[i % len(cols)])
        x = rng.uniform(-40.0, 320.0)
        y = rng.uniform(-90.0, 90.0)
        if abs(y) < 6.0 and x < 30.0:
            y += np.sign(y if y else 1.0) * 10.0
        r = rng.uniform(1.2, 3.5)
        h = rng.uniform(8.0, 45.0)
        g = wb.add_geom()
        g.name = f"post{i}"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [r, h, 0.0]
        g.pos = [x, y, h]
        g.material = f"post_mat{i}"
        g.contype, g.conaffinity = 0, 0


def add_movable_flower(spec, pos, tag: str = "flower", n_petals: int = 7,
                       make_materials: bool = True) -> None:
    """動かせる花を足す。mocap の body に載せるので `data.mocap_pos` で移せる。

    アプリの live 画面が使う。報酬 (花) を置き直すたびにモデルを組み直すと
    2.6 秒かかって飛行が途切れるが、mocap なら 1 行で動く。当たり判定は
    元の `_flower` と同じく持たせない。
    """
    b = spec.worldbody.add_body()
    b.name = f"{tag}_body"
    b.mocap = True
    b.pos = [float(v) for v in pos]
    # 茎は body のローカル原点から下へ伸ばす。高さは pos[2] をそのまま使う
    _flower(spec, [0.0, 0.0, float(pos[2])], n_petals=n_petals, tag=tag,
            make_materials=make_materials, parent=b)
    # body 自身が pos[2] の高さにいると花が2倍の高さになるので、
    # body は地面の高さに置き、中身だけ持ち上げる
    b.pos = [float(pos[0]), float(pos[1]), 0.0]


def add_scene(spec, seed: int = 0, style: str = "meadow",
              n_blades: int = 260, collide: bool = True,
              walls: bool = True, flower: bool = True,
              two_flowers: bool = False) -> None:
    """MjSpec に風景を足す (compile 前に呼ぶ)。"""
    rng = np.random.default_rng(seed)
    meadow = (style == "meadow")
    _sky_and_ground(spec, square_cm=2.5 if meadow else 6.7)

    if meadow:
        _meadow(spec, rng, n_blades=n_blades, collide=collide)
        if two_flowers:
            # 見た目は同一。違うのは匂いだけ (fly_smell 側で与える)
            _flower(spec, DECOY_POS, tag="flower_decoy")
            _flower(spec, TARGET_POS, tag="flower_target", make_materials=False)
        elif flower:
            _flower(spec, FLOWER_POS)
    else:
        _posts(spec, rng)

    if walls:
        import mujoco
        for i, (px, py, sx, sy) in enumerate([
                (WALL_DIST, 0, 1.0, GROUND_HALF),
                (-WALL_DIST * 0.4, 0, 1.0, GROUND_HALF),
                (0, WALL_DIST, GROUND_HALF, 1.0),
                (0, -WALL_DIST, GROUND_HALF, 1.0)]):
            _mat(spec, f"wall_mat{i}", [0.30, 0.34, 0.38, 1])
            g = spec.worldbody.add_geom()
            g.name = f"wall{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [sx, sy, WALL_H]
            g.pos = [px, py, WALL_H]
            g.material = f"wall_mat{i}"
            g.contype, g.conaffinity = 0, 0

    _cameras(spec, close=meadow)
    _lights(spec)
