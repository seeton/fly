"""逃避の場面 — 明るい野原・迫る捕食者・立っているハエ、そして第1パスの物理。

`24_escape_full.py` (脳・体・視界の3パネル) と `23_synapse_movie.py` (脳だけ) の
両方が、**同じ1回のシミュレーション** を入力に使う。23 が脳しか映さないからと
いって脳を勝手に光らせてよいことにはならないので、こちらも同じ場面を回して
複眼から入力を採る。結果は `out/escape_sense.npz` に置いて使い回す。

ここで採るもの:

    photo   個眼ごとの明るさ。目に入っている光そのもの (光受容器 + LMC の出力)
    motion  |photo - delayed|。像が動いた量。T4/T5 が食べている信号
    leg     脚の関節が動いた速さ。脚神経核への固有受容の入力

どれもシミュレーションから出てくる量で、脳の絵を光らせるために作った数字では
ない。入力を作っていない領域は暗いままにする。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SENSE = OUT / "escape_sense.npz"

# 接触の解法がこの寸法では敏感で、dt=1e-4 だと脚が地面を突き抜けて機体が
# 2.4 cm まで跳ね上がる (体長 0.25 cm の 10 倍)。2e-5 にすると z=0.132 cm で
# 静かに立つ。羽ばたきは無いので空力の費用はかからず、そのぶんは払える。
DT = 2e-5
PRED_R = 0.55             # 捕食者の半径 [cm] (ハエの体長は 0.25 cm)
LEG_KP, LEG_FORCE = 20.0, 60.0

LEG_SEGMENTS = ("T1", "T2", "T3")
LEG_SIDES = ("left", "right")


def add_arena(spec, start_xyz, seed: int = 3):
    """明るく開けた地面と空、遠くの低い草、そして迫ってくる黒い球。

    最初は `world_scene` の草むらをそのまま使ったが、地面に立ったハエ
    (体長 0.25 cm) から見ると茎 3-18 cm の草は林で、視野の帯がまるごと暗くなる。
    それを「迫る物体」と取り違えた (開始時点で視角 110 度と出た)。

    暗い部分を基準との差分で消そうとしても、ハエが脚の上で少し動くだけで
    背景の暗いマスがずれ、大量の個眼が「新たに暗くなった」ことになってしまう。

    逃避の実験はもともと明るい場所に黒い影を出すものなので、それに合わせる。
    **背景を明るくして、視野で本当に黒いのは捕食者だけ** という状態を作る。
    影も落とさない (地面に黒い領域ができてしまうため)。
    """
    import mujoco

    from world_scene import _cameras, _mat

    sky = spec.add_texture()
    sky.name = "arena_sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.88, 0.93, 1.0]
    sky.rgb2 = [0.62, 0.74, 0.92]
    sky.width, sky.height = 64, 256

    chk = spec.add_texture()
    chk.name = "arena_checker"
    chk.type = mujoco.mjtTexture.mjTEXTURE_2D
    chk.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    chk.rgb1 = [0.58, 0.55, 0.48]
    chk.rgb2 = [0.76, 0.74, 0.66]
    chk.width, chk.height = 128, 128
    gm = spec.add_material()
    gm.name = "arena_ground"
    gm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "arena_checker"
    gm.texuniform = True
    square_cm = 0.6            # ハエの体長 0.25 cm に対して見える細かさ
    gm.texrepeat = [1.0 / square_cm, 1.0 / square_cm]
    gm.reflectance = 0.0
    g = spec.worldbody.add_geom()
    g.name = "floor"
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [200.0, 200.0, 0.1]
    g.material = "arena_ground"

    li = spec.worldbody.add_light()
    li.pos = [0.0, 0.0, 30.0]
    li.dir = [0.0, 0.0, -1.0]
    li.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    li.castshadow = False

    rng = np.random.default_rng(seed)
    greens = [[0.42, 0.60, 0.36, 1], [0.50, 0.68, 0.42, 1], [0.36, 0.54, 0.32, 1]]
    for i, c in enumerate(greens):
        _mat(spec, "arena_green%d" % i, c)
    # 遠くに低い草。視界に手がかりは残すが、迫る影とは重ならない位置に置く
    for i in range(70):
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(6.0, 26.0)
        x, y = r * np.cos(ang), r * np.sin(ang)
        if x > 1.0 and abs(y) < 5.0:        # 捕食者が来る方向は空けておく
            continue
        h = rng.uniform(0.3, 1.1)
        gg = spec.worldbody.add_geom()
        gg.name = "tuft%d" % i
        gg.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        gg.size = [rng.uniform(0.02, 0.05), h / 2, 0.0]
        gg.pos = [x, y, h / 2]
        gg.material = "arena_green%d" % (i % len(greens))
        gg.contype = gg.conaffinity = 0
    _cameras(spec, close=True)
    add_predator(spec, start_xyz)


def add_predator(spec, start_xyz):
    """迫ってくる暗い球を足す。mocap なので位置を直接指定して動かせる。"""
    import mujoco

    b = spec.worldbody.add_body()
    b.name = "predator"
    b.mocap = True
    b.pos = list(start_xyz)
    g = b.add_geom()
    g.name = "predator_geom"
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [PRED_R, 0.0, 0.0]
    g.rgba = [0.03, 0.03, 0.04, 1.0]
    g.group = 0                     # 複眼は group 0 しか描かないので 0 に置く
    g.contype = 0
    g.conaffinity = 0


def set_leg_servos(model, kp=LEG_KP, force=LEG_FORCE):
    """脚の位置サーボを立たせられる強さにする。

    XML の既定は kp=0.1 / forcerange=±0.1 で、体重 0.98 mg を支えると
    脚が潰れる。翅と頭は別で制御するのでここでは触らない。
    """
    n = 0
    for i in range(model.nu):
        name = model.actuator(i).name
        if "wing" in name or "head" in name or "adhere" in name:
            continue
        model.actuator_gainprm[i][:3] = [kp, 0, 0]
        model.actuator_biasprm[i][:3] = [0, -kp, -2 * np.sqrt(kp * 1e-6)]
        model.actuator_forcerange[i] = [-force, force]
        model.actuator_forcelimited[i] = 1
        n += 1
    return n


def leg_dof_groups(model) -> dict:
    """脚神経核ごとの自由度の添字。

    関節名は `<部位>_T<番号>_<左右>` なので、脳側の領域名 `LegNp(T2)(R)` と
    そのまま対応が付く。脚神経核は同じ側の同じ脚から固有受容の入力を受ける
    ので、ここの対応づけには推測が要らない。
    """
    groups: dict = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        parts = name.rsplit("_", 2)
        if len(parts) != 3 or parts[1] not in LEG_SEGMENTS:
            continue
        if parts[2] not in LEG_SIDES:
            continue
        groups.setdefault(f"{parts[1]}_{parts[2]}", []).append(
            int(model.jnt_dofadr[j]))
    return {k: np.array(v) for k, v in sorted(groups.items())}


def dark_mask(eyes, dark: float = 0.12):
    """暗い個眼のマスク (左右)。"""
    return [f < dark for f in eyes.frames]


def visual_angle(eyes, base) -> float:
    """複眼の像から、迫ってくる暗い物体の見かけの直径 [deg] を求める。

    **静止した暗さではなく、増えた暗さ** を数える。背景 (市松の暗いマス、
    遠くの草) も暗い個眼を作るので、絶対値でしきい値を切ると 7 cm 離れた
    捕食者 (幾何的には視角 9 度) が 37 度と出てしまった。
    開始時に暗かった個眼を基準から外す。

    実物の LC4 / LPLC2 が応答するのも「拡大する縁」であって静止した暗さでは
    ないので、差分を見るほうが実物に近い。

    個眼1つが受け持つ立体角はおよそ dphi^2。新たに暗くなった個眼の数から
    同じ面積の円に直した直径を返す。
    """
    from fly_vision import DPHI_DEG

    n = 0
    for now, b in zip(dark_mask(eyes), base):
        n += int((now & ~b).sum())
    if n == 0:
        return 0.0
    return float(2.0 * np.sqrt(n * DPHI_DEG ** 2 / np.pi))


def signature(p: dict) -> str:
    """キャッシュが今の設定で作られたものかを見るための文字列。"""
    return "|".join(f"{k}={p[k]}" for k in sorted(p))


def run_approach(approach: float = 30.0, start_cm: float = 7.0,
                 theta_trigger: float = 45.0, gf_delay_ms: float = 1.1,
                 rise_ms: float = 5.0, amp: float = 1.9,
                 seconds: float = 0.26, settle: float = 0.15,
                 jump_force: float = 3.0, vis_hz: float = 400.0,
                 keep_state: bool = True, verbose: bool = True) -> dict:
    """第1パス — 物理を回し、複眼から視角を測って発火時刻を決める。

    返すもの (すべてシミュレーションから出た量):

        t_trigger  視角がしきい値を超えた時刻 [s]
        t_extend   そこから GF の伝導遅れぶん後 = 中脚を伸ばし始める時刻
        t_vis      視覚を更新した時刻 [s] (n_t,)
        photo      個眼の明るさ (n_t, 2, n, n)
        motion     |photo - delayed| (n_t, 2, n, n)
        leg_speed  脚神経核ごとの |関節速度| の平均 {"T2_right": (n_t,), ...}
        qpos_log / mocap_log / model / data / eyes  (keep_state のときだけ)

    `keep_state=False` なら姿勢のログを捨てる (脳だけ描く 23 用。
    13000 ステップ x 110 自由度を持ち回らずに済む)。
    """
    import mujoco

    from fly_vision import FlyEyes
    from flybody_model import load_model

    start = np.array([start_cm, 0.0, 0.9])
    m, _ = load_model(scene=False, add_floor=False,
                      extra=lambda sp: add_arena(sp, start))
    m.opt.timestep = DT
    n_leg = set_leg_servos(m)
    if verbose:
        print(f"脚のサーボ {n_leg} 個を kp={LEG_KP} force={LEG_FORCE} にした")

    eyes = FlyEyes(m, rate_hz=vis_hz)
    femur = [m.actuator(f"femur_T2_{s}").id for s in LEG_SIDES]
    tibia = [m.actuator(f"tibia_T2_{s}").id for s in LEG_SIDES]
    # 跳ぶ関節だけ力の上限を別に決める。立たせるための 60 dyn cm のままだと
    # 体重 (0.96 dyn) の 60 倍で蹴ることになり、体長の 14 倍も跳んでしまった。
    for a_ in femur + tibia:
        m.actuator_forcerange[a_] = [-jump_force, jump_force]
    dofs = leg_dof_groups(m)

    d = mujoco.MjData(m)
    d.qpos[:3] = [0.0, 0.0, 0.245]
    # 捕食者を遠くに置いたまま、まず脚で立たせて落ち着かせる。
    # これをやらずに初期姿勢で基準を取ると、沈み込みで視界が動いたぶんを
    # 「迫る影」と取り違える (捕食者が 6.5 cm 先にいる時点で視角 72 度と出た)。
    d.mocap_pos[0] = [200.0, 0.0, 40.0]
    for _ in range(int(settle / DT)):
        mujoco.mj_step(m, d)
    if verbose:
        print(f"静置後の胸部の高さ {d.qpos[2]:.3f} cm")
    eyes.maybe_update(m, d, 0.0)
    base = dark_mask(eyes)
    if verbose:
        print(f"基準 (何も迫っていない) の暗い個眼 "
              f"{sum(int(b_.sum()) for b_ in base)} / {2*eyes.n_omma**2}")
    d.mocap_pos[0] = start
    mujoco.mj_forward(m, d)

    n_steps = int(seconds / DT)
    qpos_log = np.zeros((n_steps, m.nq)) if keep_state else None
    mocap_log = np.zeros((n_steps, 3)) if keep_state else None
    theta_log = np.zeros(n_steps)
    t_vis, photo_log, motion_log = [], [], []
    leg_log = {k: [] for k in dofs}
    eyes.reset()
    theta = 0.0
    t_trigger = t_extend = None
    femur0 = float(d.qpos[m.jnt_qposadr[m.actuator_trnid[femur[0], 0]]])
    if verbose:
        print("第1パス: 物理と複眼 ...")
    for s in range(n_steps):
        t = s * DT
        # 捕食者はハエへ一直線に近づく
        dir_ = np.array([-1.0, 0.0, -0.09])
        dir_ /= np.linalg.norm(dir_)
        d.mocap_pos[0] = start + dir_ * approach * t

        if eyes.maybe_update(m, d, t):
            theta = visual_angle(eyes, base)
            if t_trigger is None and theta >= theta_trigger:
                t_trigger = t
                t_extend = t + gf_delay_ms * 1e-3
                if verbose:
                    print(f"  視角 {theta:.1f} deg でしきい値到達 t={t*1e3:.1f} ms "
                          f"(捕食者まで "
                          f"{np.linalg.norm(d.mocap_pos[0]-d.qpos[:3]):.2f} cm)")
            # 視葉への入力をそのまま控えておく。脳の絵はここから光る
            t_vis.append(t)
            photo_log.append(np.stack(eyes.photo).astype(np.float32))
            motion_log.append(np.stack(eyes.activity).astype(np.float32))
            for k, ix in dofs.items():
                leg_log[k].append(float(np.abs(d.qvel[ix]).mean()))
        theta_log[s] = theta

        if t_extend is not None and t >= t_extend:
            fr = min((t - t_extend) / (rise_ms * 1e-3), 1.0)
            for a in femur:
                d.ctrl[a] = femur0 + amp * fr
            for a in tibia:
                d.ctrl[a] = amp * 0.5 * fr
        if keep_state:
            qpos_log[s] = d.qpos
            mocap_log[s] = d.mocap_pos[0]
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            if verbose:
                print(f"  発散 t={t*1e3:.1f} ms")
            n_steps = s + 1
            theta_log = theta_log[:n_steps]
            if keep_state:
                qpos_log, mocap_log = qpos_log[:n_steps], mocap_log[:n_steps]
            break

    R = {"t_trigger": t_trigger, "t_extend": t_extend,
         "t_end": (n_steps - 1) * DT, "n_steps": n_steps,
         "n_omma": eyes.n_omma, "theta_log": theta_log,
         "t_vis": np.array(t_vis),
         "photo": np.array(photo_log), "motion": np.array(motion_log),
         "leg_speed": {k: np.array(v) for k, v in leg_log.items()}}
    if keep_state:
        R["qpos_log"], R["mocap_log"] = qpos_log, mocap_log
        R["model"], R["data"], R["eyes"] = m, d, eyes
    return R


def save_sense(R: dict, params: dict) -> None:
    """感覚の入力だけを保存する (姿勢は入れない)。"""
    if R["t_trigger"] is None:
        return
    OUT.mkdir(exist_ok=True)
    legs = {f"leg_{k}": v for k, v in R["leg_speed"].items()}
    np.savez_compressed(
        SENSE, sig=np.array(signature(params)), t_vis=R["t_vis"],
        photo=R["photo"], motion=R["motion"],
        t_trigger=np.array(R["t_trigger"], dtype=float),
        t_extend=np.array(R["t_extend"], dtype=float),
        t_end=np.array(R["t_end"], dtype=float),
        n_omma=np.array(R["n_omma"]), **legs)


def load_sense(params: dict, refresh: bool = False):
    """保存した感覚の入力を読む。設定が違えば None。"""
    if refresh or not SENSE.exists():
        return None
    z = np.load(SENSE, allow_pickle=False)
    if str(z["sig"]) != signature(params):
        return None
    legs = {k[4:]: z[k] for k in z.files if k.startswith("leg_")}
    return {"t_vis": z["t_vis"], "photo": z["photo"], "motion": z["motion"],
            "t_trigger": float(z["t_trigger"]), "t_extend": float(z["t_extend"]),
            "t_end": float(z["t_end"]), "n_omma": int(z["n_omma"]),
            "leg_speed": legs}
