"""飛行の評価環境 — 羽ばたき + 姿勢フィードバックのパラメトリック制御器。

本物のハエは平均棍(haltere)で機体の回転を検出して羽ばたきを補正している。
ここでも同じ構造にする:

  ピッチ <- ストローク中心 (roll_mean) を前後にずらす
  ロール <- 左右のストローク振幅に差をつける
  ヨー   <- 左右の迎角に差をつける
  高度   <- ストローク振幅を左右同時に増減

パラメータ (16個) は scripts/13_train_flight.py が探索する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DT = 2e-5
WING_KP = 50.0
N_ELEM = 6   # 翼素の分割数

# (旧) MuJoCo 内蔵の楕円体流体の係数。いまは insect_aero を使うので未使用。
# 空力係数 [有効, blunt抗力, slender抗力, 角度抗力, Kutta揚力, Magnus揚力]
# MuJoCo の既定値を一様に2倍したもの。楕円体流体モデルは昆虫の前縁渦(LEV)を
# 表現しないので、実測の昆虫の力係数に合わせて底上げする。
# 以前は lift だけ3倍・blunt抗力だけ3倍という非対称な改変をしていたが、
# それは力の「向き」まで歪めるので一様スケールに改めた。
AERO = [1.0, 1.0, 0.5, 3.0, 2.0, 2.0]

WINGS = ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
         "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]

# name, low, high
PARAMS = [
    ("freq",       150.0, 260.0),
    ("yaw",         -1.4,   0.2),
    ("roll_mean",   -0.3,   0.8),
    ("roll_amp",     0.5,   1.25),
    ("pitch_down",  -1.2,   2.9),
    ("pitch_up",    -1.2,   2.9),
    ("sharp",        1.5,   9.0),
    ("phase",    -np.pi, np.pi),
    # 展開角(deviation)も周期内で振る。これが無いとストローク面の向きが
    # 固定され、正味の力が体軸から34度ずれたままになる → 機体が機首下げ
    # 50度で飛ぶ解しか残らない。振れるようにすると力を体軸の真上に向けられる。
    ("yaw_amp",      0.0,   1.4),
    ("yaw_phase", -np.pi, np.pi),
    # 制御ゲインの範囲は、ピッチの制御感度を実測して決めた。
    #   d(ピッチ角加速度)/d(roll_mean) = 14959 rad/s^2 per rad
    # 固有振動数 ω を 30〜120 rad/s に取ると kp = ω^2/S = 0.06〜0.96。
    # 最初は [0,8] にしていて、これは2桁大きすぎた (制御が発散した)。
    ("kp_pitch",     0.0,   1.2),
    ("kd_pitch",     0.0,   0.02),
    ("kp_roll",      0.0,   1.2),
    ("kd_roll",      0.0,   0.02),
    ("kp_yaw",       0.0,   1.5),
    ("kd_yaw",       0.0,   0.04),
    ("kp_alt",       0.0,   0.5),
    ("kd_alt",       0.0,   0.05),
]
LO = np.array([p[1] for p in PARAMS])
HI = np.array([p[2] for p in PARAMS])
NAMES = [p[0] for p in PARAMS]

_MODEL = None
_AERO = None
_LEG_TARGETS = None


def _has_actuator(model, name):
    try:
        model.actuator(name)
        return True
    except Exception:
        return False


def get_aero():
    get_model()
    return _AERO


def get_model():
    """ワーカープロセスごとに1回だけモデルを組む。"""
    global _MODEL, _AERO
    if _MODEL is None:
        import mujoco
        from flybody_model import load_model
        from insect_aero import WingAero
        m, _ = load_model(wing_kp=WING_KP, wing_kv=2 * np.sqrt(WING_KP * 1e-6), add_floor=True)
        m.opt.timestep = DT
        # 翅は準定常翼素理論 (insect_aero) が担当。胴体の空気抵抗は MuJoCo に残す。
        aero = WingAero(m, n_elem=N_ELEM)
        aero.route_wings_to_this_model(m)
        # 付加質量は明示的な力ではなく慣性として入れる (エネルギー保存)
        i_add = aero.added_mass_inertia(m)
        for jn in ("wing_roll_left", "wing_roll_right",
                   "wing_yaw_left", "wing_yaw_right"):
            adr = m.jnt_dofadr[m.joint(jn).id]
            m.dof_armature[adr] += i_add
        _MODEL = m
        _AERO = aero
    return _MODEL


def decode(x):
    """[0,1]^16 -> 物理パラメータ。"""
    return dict(zip(NAMES, LO + np.clip(x, 0.0, 1.0) * (HI - LO)))


def encode(d):
    return (np.array([d[n] for n in NAMES]) - LO) / (HI - LO)


def euler(q):
    w, x, y, z = q
    return (np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rollout(x, target=(0.0, 0.0, 10.0), seconds=0.5, z0=10.0, record=False):
    """1エピソード回して報酬と軌跡を返す。

    報酬は毎ステップ、**必ず正**になるように作る:

        + 生存            1.0
        + 目標高度ボーナス 2.0 / (1 + (dz/5)^2)
        + 的へのボーナス   1.0 / (1 + (dxy/5)^2)
        + 機首方向ボーナス 1.0 / (1 + (yaw/0.6)^2)
        + 省エネボーナス   1.0 / (1 + (P_air/800)^2)
        - 横倒れ          0.4 * |roll|
        - 回転            0.005 * min(spin, 100)

    P_air は **翅が空気に渡したパワー** [erg/s] を1周期ぶん平滑化した値。

    最初はアクチュエータの正味パワー sum(F v) を使ったが、これは
    サーボ内部の速度フィードバック (ダンパー) の散逸を負の値として
    含んでしまう。翅は 1400 rad/s で振れるので kv*qd*qd だけで
    -27,000 erg/s になり、報酬が max(P,0) だったため
    **「ダンパーを強く回すほど省エネ扱い」** という抜け穴になった。
    実際 CMA-ES はそこを突いて -31,126 erg/s の解を見つけた。
    エネルギーを生んでいたわけではなく、指標の欠陥。
    本物の昆虫は飛翔筋のパワーに強く制約されていて、効率の悪い羽ばたきは
    選べない。姿勢を報酬で指定する代わりに、この物理的な制約を入れる。
    基準の 800 erg/s は実測のショウジョウバエの上限 (300-800 erg/s)。

    体の角度そのものには報酬を与えない。一度は「機首上げ34度」を狙う
    ボーナスを入れたが、姿勢は物理の釣り合いで決まるものなので報酬で
    捻じ曲げるのは筋が悪く、実際まったく効かなかった (-57度のまま)。
    原因は羽ばたきの側で、ストローク面が固定されていたことだった。

    機首方向を入れないと、水平は保つが垂直軸まわりにコマのように
    回り続ける解になる (実際そうなった: 1秒で -242 度)。
    差動迎角によるヨー制御はロールへの干渉が大きく (±37,000 deg/s^2)、
    ゲインを独立に上げると墜落するので、探索に同時最適化させる。

    ここが最重要。penalty を線形で引くと「すぐ死んだ方が高得点」になり、
    探索は飛べる解を捨てて即墜落する解を選ぶ。実際にそうなった:

        飛べる種    : 0.50s 完走・高度33cm  → 報酬 -4.386
        学習後      : 0.03s で墜落・高度10cm → 報酬 +0.003

    ボーナスを有界な正の項にして、生存が常に得になるようにした。
    傾きが 80 度を超えるか、高度が 0.5 cm を切ったら打ち切り。
    """
    import mujoco

    m = get_model()
    aero = get_aero()
    aero.reset()
    p = decode(x)
    A = {n: m.actuator(n).id for n in WINGS}
    # 実物のハエは飛ぶとき脚をたたむ。指令を 0 のままにすると脚を広げた
    # 着地姿勢で飛ぶことになり、余計な抗力で揚力が約10%落ちる。
    from flight_posture import leg_targets
    global _LEG_TARGETS
    if _LEG_TARGETS is None:
        _LEG_TARGETS = {m.actuator(k).id: v for k, v in leg_targets(m).items()
                        if _has_actuator(m, k)}
    body_id = m.body("thorax").id
    d = mujoco.MjData(m)
    d.qpos[2] = z0
    for aid, val in _LEG_TARGETS.items():
        d.ctrl[aid] = val
        d.qpos[m.jnt_qposadr[m.actuator_trnid[aid, 0]]] = val
    tx, ty, tz = target

    mid = (p["pitch_down"] + p["pitch_up"]) / 2
    amp_p = (p["pitch_down"] - p["pitch_up"]) / 2
    n_steps = int(seconds / DT)
    reward = 0.0
    alive = 0
    rec = []
    act_ids = [A[n] for n in WINGS]
    power_ema = 0.0
    ema_alpha = DT / 5e-3          # 時定数 5 ms = 羽ばたき約1周期
    power_sum, power_n = 0.0, 0

    for s in range(n_steps):
        t = s * DT
        th = 2 * np.pi * p["freq"] * t
        roll_b, pitch_b, yaw_b = euler(d.qpos[3:7])
        wx, wy, wz = d.qvel[3], d.qvel[4], d.qvel[5]

        u_pitch = np.clip(-p["kp_pitch"] * pitch_b - p["kd_pitch"] * wy, -0.8, 0.8)
        u_roll = np.clip(-p["kp_roll"] * roll_b - p["kd_roll"] * wx, -0.5, 0.5)
        u_yaw = np.clip(-p["kp_yaw"] * yaw_b - p["kd_yaw"] * wz, -0.6, 0.6)
        u_alt = np.clip(p["kp_alt"] * (tz - d.qpos[2]) - p["kd_alt"] * d.qvel[2], -0.4, 0.4)

        mean = p["roll_mean"] + u_pitch
        base = np.clip(p["roll_amp"] + u_alt, 0.35, 1.25)
        feather = mid + amp_p * np.tanh(p["sharp"] * np.sin(th + p["phase"]))

        dev = p["yaw"] + p["yaw_amp"] * np.cos(th + p["yaw_phase"])
        for side, sgn in (("left", +1.0), ("right", -1.0)):
            d.ctrl[A[f"wing_yaw_{side}"]] = np.clip(dev, -1.5, 1.5)
            d.ctrl[A[f"wing_roll_{side}"]] = np.clip(mean + (base + sgn * u_roll) * np.cos(th), -1.0, 1.5)
            d.ctrl[A[f"wing_pitch_{side}"]] = np.clip(feather + sgn * u_yaw, -1.27, 2.92)

        aero.apply(m, d)
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            break

        tilt = max(abs(roll_b), abs(pitch_b))
        # 機体の前方軸(+x)が水平面となす角。正なら機首上げ。
        nose = float(np.arcsin(np.clip(d.xmat[body_id].reshape(3, 3)[2, 0], -1, 1)))
        dz = abs(d.qpos[2] - tz)
        dxy = np.hypot(d.qpos[0] - tx, d.qpos[1] - ty)
        spin = np.linalg.norm(d.qvel[3:6])
        p_now = float(aero.air_power)          # 翅が空気に渡したパワー
        power_ema += ema_alpha * (p_now - power_ema)
        power_sum += p_now
        power_n += 1
        reward += DT * (1.0
                        + 2.0 / (1.0 + (dz / 5.0) ** 2)
                        + 1.0 / (1.0 + (dxy / 5.0) ** 2)
                        + 1.0 / (1.0 + (yaw_b / 0.6) ** 2)
                        + 1.0 / (1.0 + (abs(power_ema) / 800.0) ** 2)
                        - 0.4 * abs(roll_b)
                        - 0.005 * min(spin, 100.0))
        alive += 1

        if tilt > 1.4 or d.qpos[2] < 0.5:
            break
        if record and s % 250 == 0:
            rec.append((t, *map(float, d.qpos[:3]), *np.degrees(euler(d.qpos[3:7]))))

    out = dict(reward=float(reward), alive_s=alive * DT,
               z=float(d.qpos[2]), pos=[float(v) for v in d.qpos[:3]],
               tilt_deg=float(np.degrees(max(abs(euler(d.qpos[3:7])[0]),
                                             abs(euler(d.qpos[3:7])[1])))),
               spin_dps=float(np.degrees(np.linalg.norm(d.qvel[3:6]))),
               nose_deg=float(np.degrees(np.arcsin(
                   np.clip(d.xmat[body_id].reshape(3, 3)[2, 0], -1, 1)))),
               dist=float(np.linalg.norm(np.array(d.qpos[:3]) - np.array(target))),
               power=float(power_sum / power_n) if power_n else float("nan"))
    if record:
        out["rec"] = rec
    return out


def splice(base, sub, idx):
    """base の idx 番目だけを sub で置き換えた完全なパラメータ列を作る。

    「羽ばたきは固定して制御ゲインだけ学習する」のように、
    一部だけを探索するために使う。
    """
    x = np.array(base, dtype=float)
    x[list(idx)] = sub
    return x


def evaluate(args):
    """マルチプロセス用のラッパ。"""
    sub, base, idx, target, seconds, z0 = args
    try:
        x = splice(base, sub, idx)
        return rollout(x, target=target, seconds=seconds, z0=z0)["reward"]
    except Exception:
        return -1e3
