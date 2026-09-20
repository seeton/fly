"""見た目が同じ2つの花から、匂いで正しい方を選ぶ。

前の版 (20_forage.py) は「遠いと嗅覚、近いと視覚」と **切り替えて** いた。
これは実物と違う。ハエの目は常に開いているし、近づいても匂いは消えない。
役割が違うだけで、同時に働いている:

    視覚 … そこに「何か」があること、その方向。形と位置は分かるが
           それが自分の好む花かどうかは分からない。
    嗅覚 … 朧げな方向 (風上) と、**それが好む花かどうかの判定**。

なのでここでは両方を常に計算し、**嗅覚が視覚の的への接近を許可する**形にする。

    好み pref = (好む匂い - それ以外) / (合計)      -1 .. +1
    風上へ寄る  = k_odor * max(pref,0) * (風上の向き)
    的へ寄る    = k_see  * max(pref,0) * (視覚で見た的の向き)

おとりの花 (見た目は同じ、匂いが違う) の近くでは pref が負になるので、
**目に見えていても寄らない**。

配置:
    出発  (15, 0, 9)
    おとり (42, -5, 9)   先に出会う。匂いが違う
    本命   (66, +5, 9)

使い方:
  python scripts/21_choose.py                 # 嗅覚で選ぶ
  python scripts/21_choose.py --no-smell      # 匂いを無視 -> おとりに寄る
  python scripts/21_choose.py --no-video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flight_posture import leg_targets  # noqa: E402
from fly_smell import WIND, FlyNose, OdorPlume  # noqa: E402
from fly_vision import FlyEyes  # noqa: E402
from flybody_model import load_model  # noqa: E402
from insect_aero import WingAero  # noqa: E402
from world_scene import DECOY_POS, TARGET_POS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
POLICY = OUT / "flight_policy.json"
WINGS = ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
         "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]
DT = 2e-5


def euler(q):
    w, x, y, z = q
    return (np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--x0", type=float, default=15.0)
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--z0", type=float, default=9.0)
    ap.add_argument("--camera", default="scene_follow")
    ap.add_argument("--slow", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, nargs=2, default=[540, 860])
    ap.add_argument("--k-flow", type=float, default=0.14,
                    help="視運動反応のゲイン。旧実装の 0.01 は角速度推定値 [rad/s] に"
                         "かけていたもの。新しい検出器の出力は任意単位 (1 rad/s で約 0.07) "
                         "なので同じ効きになる値にしてある。これも学習対象にすべき値")
    ap.add_argument("--k-odor", type=float, default=0.22)
    ap.add_argument("--k-see", type=float, default=0.22)
    ap.add_argument("--see-on", type=float, default=8e-5)
    ap.add_argument("--see-off", type=float, default=2e-5)
    ap.add_argument("--no-smell", action="store_true",
                    help="匂いを無視して視覚だけで寄る (おとりに引っかかるはず)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = json.loads(POLICY.read_text(encoding="utf-8"))["params"]
    m, _ = load_model(wing_kp=50.0, wing_kv=2 * np.sqrt(50.0 * 1e-6),
                      scene=True, two_flowers=True)
    m.opt.timestep = DT
    aero = WingAero(m, n_elem=6)
    aero.route_wings_to_this_model(m)
    i_add = aero.added_mass_inertia(m)
    for jn in ("wing_roll_left", "wing_roll_right", "wing_yaw_left", "wing_yaw_right"):
        m.dof_armature[m.jnt_dofadr[m.joint(jn).id]] += i_add

    wind = np.array(WIND, dtype=float)
    aero.wind = wind
    m.opt.wind[:] = wind

    eyes = FlyEyes(m, rate_hz=200.0)
    plumes = {"target": OdorPlume(TARGET_POS), "decoy": OdorPlume(DECOY_POS)}
    nose = FlyNose(m, plumes)

    A = {n: m.actuator(n).id for n in WINGS}
    legs = {m.actuator(k).id: v for k, v in leg_targets(m).items()}
    head_ids = {h: m.actuator(h).id for h in ("head_abduct", "head")}
    thorax = m.body("thorax").id
    if not args.no_video:
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.res[1])
        m.vis.global_.offheight = max(m.vis.global_.offheight, args.res[0])
        renderer = mujoco.Renderer(m, height=args.res[0], width=args.res[1])

    d = mujoco.MjData(m)
    d.qpos[0], d.qpos[1], d.qpos[2] = args.x0, args.y0, args.z0
    for aid, val in legs.items():
        d.ctrl[aid] = val
        d.qpos[m.jnt_qposadr[m.actuator_trnid[aid, 0]]] = val
    aero.reset()
    eyes.reset()
    nose.reset()

    mid = (p["pitch_down"] + p["pitch_up"]) / 2
    amp_p = (p["pitch_down"] - p["pitch_up"]) / 2
    a_slow, a_vis = DT / 0.05, DT / 0.03
    yaw_vis = yaw_vis_f = yaw_slow = roll_slow = 0.0
    see_f = 0.0
    seeing = False
    frames, traj = [], []
    frame_every = max(int(1.0 / (args.fps * args.slow) / DT), 1)
    best = {"target": 1e9, "decoy": 1e9}

    print(f"おとり {DECOY_POS}  本命 {TARGET_POS}  風 {wind} cm/s")
    print(f"出発 ({args.x0}, {args.y0}, {args.z0})   "
          f"嗅覚 {'無視' if args.no_smell else '使う'}")

    for s in range(int(args.seconds / DT)):
        t = s * DT
        th = 2 * np.pi * p["freq"] * t
        roll_b, pitch_b, yaw_b = euler(d.qpos[3:7])
        wx, wy, wz = d.qvel[3], d.qvel[4], d.qvel[5]

        if eyes.maybe_update(m, d, t):
            yaw_vis = eyes.rotation_signal
        yaw_vis_f += a_vis * (yaw_vis - yaw_vis_f)
        nose.update(m, d, DT)

        yaw_slow += a_slow * (yaw_b - yaw_slow)
        roll_slow += a_slow * (roll_b - roll_slow)
        d.ctrl[head_ids["head_abduct"]] = float(np.clip(-(yaw_b - yaw_slow), -0.2, 0.2))
        d.ctrl[head_ids["head"]] = float(np.clip(-(roll_b - roll_slow), -0.5, 0.3))

        # --- 視覚と嗅覚を同時に使う ---
        if eyes.front_size > args.see_on:
            seeing = True
        elif eyes.front_size < args.see_off:
            seeing = False
        if eyes.front_size > args.see_off:
            see_f += (DT / 0.05) * (eyes.front_bearing(args.see_off) - see_f)

        pref = 1.0 if args.no_smell else nose.preference("target")
        gate = max(pref, 0.0)          # 好む匂いのときだけ寄る
        left_up, _ = nose.body_frame_upwind(d, thorax)
        steer = args.k_odor * gate * left_up
        if seeing:
            steer += -args.k_see * gate * see_f

        u_p = np.clip(-p["kp_pitch"] * pitch_b - p["kd_pitch"] * wy, -0.8, 0.8)
        u_r = np.clip(-p["kp_roll"] * roll_b - p["kd_roll"] * wx, -0.5, 0.5)
        u_y = np.clip(-args.k_flow * yaw_vis_f + steer, -0.6, 0.6)
        u_a = np.clip(p["kp_alt"] * (args.z0 - d.qpos[2]) - p["kd_alt"] * d.qvel[2],
                      -0.4, 0.4)

        mean = p["roll_mean"] + u_p
        base = np.clip(p["roll_amp"] + u_a, 0.35, 1.25)
        feather = mid + amp_p * np.tanh(p["sharp"] * np.sin(th + p["phase"]))
        dev = p["yaw"] + p["yaw_amp"] * np.cos(th + p["yaw_phase"])
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            d.ctrl[A[f"wing_yaw_{side}"]] = np.clip(dev, -1.5, 1.5)
            d.ctrl[A[f"wing_roll_{side}"]] = np.clip(
                mean + (base + sgn * u_r) * np.cos(th), -1.0, 1.5)
            d.ctrl[A[f"wing_pitch_{side}"]] = np.clip(feather + sgn * u_y, -1.27, 2.92)

        aero.apply(m, d)
        mujoco.mj_step(m, d)
        best["target"] = min(best["target"], float(np.linalg.norm(d.qpos[:3] - TARGET_POS)))
        best["decoy"] = min(best["decoy"], float(np.linalg.norm(d.qpos[:3] - DECOY_POS)))
        if not np.isfinite(d.qpos).all() or d.qpos[2] < 0.5:
            print(f"  t={t:.2f}s で終了 (高度 {d.qpos[2]:.2f}cm)")
            break
        if s % 20000 == 0:
            traj.append((t, float(d.qpos[0]), float(d.qpos[1]),
                         nose.channels["target"], nose.channels["decoy"], pref,
                         eyes.front_size, "見えている" if seeing else "-"))
        if (not args.no_video) and s % frame_every == 0:
            renderer.update_scene(d, camera=args.camera)
            img = renderer.render()
            eye = eyes.side_by_side(scale=2)
            h, w = eye.shape[:2]
            img[6:6 + h, 6:6 + w] = eye
            frames.append(img)

    print(f"\n{'t[s]':>6} {'x':>7} {'y':>7} {'本命の匂い':>11} {'おとりの匂い':>12} "
          f"{'好み':>6} {'花の写り':>9}  視覚")
    for r in traj:
        print(f"{r[0]:6.2f} {r[1]:7.2f} {r[2]:7.2f} {r[3]:11.4f} {r[4]:12.4f} "
              f"{r[5]:+6.2f} {r[6]*100:8.3f}%  {r[7]}")
    print(f"\n本命への最接近  : {best['target']:.2f} cm")
    print(f"おとりへの最接近: {best['decoy']:.2f} cm   (花の直径は 2.5 cm)")
    REACH = 4.0      # 花の直径2.5cmなので、4cm以内なら「その花に来た」とみなす
    hit_t, hit_d = best["target"] < REACH, best["decoy"] < REACH
    if hit_t and not hit_d:
        verdict = "本命に到達 (おとりには寄らなかった)"
    elif hit_d and not hit_t:
        verdict = "おとりに引っかかった"
    elif hit_t and hit_d:
        verdict = "両方に寄ってしまった"
    else:
        verdict = f"どちらにも届かず (最接近 本命 {best['target']:.1f} / おとり {best['decoy']:.1f} cm)"
    print(f"判定: {verdict}")

    if not args.no_video and frames:
        OUT.mkdir(exist_ok=True)
        path = OUT / (args.out or "choose.mp4")
        imageio.mimsave(path, frames, fps=args.fps, macro_block_size=None)
        print(f"書き出し: {path} ({len(frames)} フレーム, "
              f"{path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
