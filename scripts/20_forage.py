"""匂いで寄って、目で仕上げる — 花を探しに行く。

視覚で直径2.5cmの花を捉えられるのは 8-10 cm 以内だった。花は 60 cm 先なので、
目だけでは見つけられない。実物のハエと同じ段取りにする:

  1. **嗅覚** (遠距離)  触角で匂いを感じたら風上へ向かう
                        (odor-gated anemotaxis。実物のハエの定石)
  2. **視覚** (近距離)  花が複眼に写ったらそちらへ機首を向ける
  3. **視運動反応**     どちらの段階でも、複眼の流れで機首のふらつきを抑える

風は匂いを運ぶだけでなく、ハエ自身も流す (翅の空力にも胴体の抗力にも入れる)。

使い方:
  python scripts/20_forage.py                      # 匂い+視覚
  python scripts/20_forage.py --no-smell           # 視覚だけ (花を見つけられない)
  python scripts/20_forage.py --no-vision-homing   # 匂いだけ (花の手前で行き止まる)
  python scripts/20_forage.py --no-video           # 速い

出力: out/forage.mp4
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
from world_scene import FLOWER_POS  # noqa: E402

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
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--z0", type=float, default=9.0)
    ap.add_argument("--x0", type=float, default=20.0,
                    help="出発時の前後位置 (花は x=60)")
    ap.add_argument("--y0", type=float, default=-6.0,
                    help="出発時の横位置 (プルームの外から始める)")
    ap.add_argument("--camera", default="scene_follow")
    ap.add_argument("--slow", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, nargs=2, default=[540, 860])
    ap.add_argument("--k-flow", type=float, default=0.14,
                    help="視運動反応のゲイン (新しい検出器は任意単位、1 rad/s で約 0.07)")
    ap.add_argument("--k-odor", type=float, default=0.22, help="風上へ向かうゲイン")
    ap.add_argument("--k-see", type=float, default=0.30, help="花へ向かうゲイン")
    ap.add_argument("--odor-th", type=float, default=0.002, help="匂いの検出しきい値")
    ap.add_argument("--steer-sign", type=float, default=1.0)
    ap.add_argument("--see-on", type=float, default=8e-5,
                    help="視覚ホーミングに入るしきい値 (視野に占める割合)")
    ap.add_argument("--see-off", type=float, default=2e-5,
                    help="視覚ホーミングを抜けるしきい値 (ヒステリシス)")
    ap.add_argument("--no-smell", action="store_true")
    ap.add_argument("--no-vision-homing", action="store_true")
    ap.add_argument("--no-wind", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = json.loads(POLICY.read_text(encoding="utf-8"))["params"]
    m, _ = load_model(wing_kp=50.0, wing_kv=2 * np.sqrt(50.0 * 1e-6), scene=True)
    m.opt.timestep = DT
    aero = WingAero(m, n_elem=6)
    aero.route_wings_to_this_model(m)
    i_add = aero.added_mass_inertia(m)
    for jn in ("wing_roll_left", "wing_roll_right", "wing_yaw_left", "wing_yaw_right"):
        m.dof_armature[m.jnt_dofadr[m.joint(jn).id]] += i_add

    wind = np.zeros(3) if args.no_wind else np.array(WIND, dtype=float)
    aero.wind = wind
    m.opt.wind[:] = wind          # 胴体の抗力にも風を効かせる

    eyes = FlyEyes(m, rate_hz=200.0)
    plume = OdorPlume(FLOWER_POS, wind=WIND)
    nose = FlyNose(m, plume)

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
    see_f = 0.0          # 花の方向 (平滑化)
    seeing = False       # 視覚ホーミング中か (ヒステリシスで保持)
    frames, traj = [], []
    frame_every = max(int(1.0 / (args.fps * args.slow) / DT), 1)
    best_dist = 1e9
    mode = "探索"

    print(f"花 {FLOWER_POS}  風 {wind} cm/s  出発 ({args.x0}, {args.y0}, {args.z0})")
    print(f"嗅覚 {'切' if args.no_smell else '入'}  "
          f"視覚ホーミング {'切' if args.no_vision_homing else '入'}")

    for s in range(int(args.seconds / DT)):
        t = s * DT
        th = 2 * np.pi * p["freq"] * t
        roll_b, pitch_b, yaw_b = euler(d.qpos[3:7])
        wx, wy, wz = d.qvel[3], d.qvel[4], d.qvel[5]

        if eyes.maybe_update(m, d, t):
            yaw_vis = eyes.rotation_signal
        yaw_vis_f += a_vis * (yaw_vis - yaw_vis_f)
        nose.update(m, d, DT)

        # 頭部の姿勢安定化 (速い揺れだけ打ち消す)
        yaw_slow += a_slow * (yaw_b - yaw_slow)
        roll_slow += a_slow * (roll_b - roll_slow)
        d.ctrl[head_ids["head_abduct"]] = float(np.clip(-(yaw_b - yaw_slow), -0.2, 0.2))
        d.ctrl[head_ids["head"]] = float(np.clip(-(roll_b - roll_slow), -0.5, 0.3))

        # --- 操舵: 近ければ目、遠ければ匂い ---
        # 花の写りは小さいので、しきい値をまたぐたびに状態が飛ぶ。
        # 入るしきい値と抜けるしきい値を分けて (ヒステリシス)、
        # 方向自体も平滑化してから使う。
        if not args.no_vision_homing:
            if eyes.front_size > args.see_on:
                seeing = True
            elif eyes.front_size < args.see_off:
                seeing = False
            if eyes.front_size > args.see_off:
                see_f += (DT / 0.05) * (eyes.front_bearing(args.see_off) - see_f)
        else:
            seeing = False
        if seeing:
            steer = -args.k_see * see_f      # 花が左(負)なら左へ
            mode = "視覚"
        elif (not args.no_smell) and nose.strength > args.odor_th:
            left_up, _ = nose.body_frame_upwind(d, thorax)
            steer = args.k_odor * left_up    # 風上が左なら左へ
            mode = "嗅覚"
        else:
            steer = 0.0
            mode = "探索"
        steer *= args.steer_sign

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
        dist = float(np.linalg.norm(d.qpos[:3] - FLOWER_POS))
        best_dist = min(best_dist, dist)
        if not np.isfinite(d.qpos).all() or d.qpos[2] < 0.5:
            print(f"  t={t:.2f}s で終了 (高度 {d.qpos[2]:.2f}cm)")
            break
        if s % 12500 == 0:
            traj.append((t, float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2]),
                         dist, nose.strength, eyes.front_size, mode))
        if (not args.no_video) and s % frame_every == 0:
            renderer.update_scene(d, camera=args.camera)
            img = renderer.render()
            eye = eyes.side_by_side(scale=2)
            h, w = eye.shape[:2]
            img[6:6 + h, 6:6 + w] = eye
            frames.append(img)

    print(f"\n{'t[s]':>6} {'x':>7} {'y':>7} {'z':>6} {'花まで':>8} "
          f"{'匂い':>8} {'花の写り':>9}  状態")
    for r in traj:
        print(f"{r[0]:6.2f} {r[1]:7.2f} {r[2]:7.2f} {r[3]:6.2f} {r[4]:8.2f} "
              f"{r[5]:8.4f} {r[6]*100:8.3f}%  {r[7]}")
    print(f"\n花への最接近: {best_dist:.2f} cm  (花の直径は 2.5 cm)")

    if not args.no_video and frames:
        OUT.mkdir(exist_ok=True)
        path = OUT / (args.out or "forage.mp4")
        imageio.mimsave(path, frames, fps=args.fps, macro_block_size=None)
        print(f"書き出し: {path} ({len(frames)} フレーム, "
              f"{path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
