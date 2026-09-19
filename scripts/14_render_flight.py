"""学習した飛び方を動画にする。

scripts/13_train_flight.py が出した out/flight_policy.json を読んで、
そのまま飛ばして録画する。制御則は scripts/flight_env.py と同じもの。

使い方:
  python scripts/14_render_flight.py                     # 学習後の飛行
  python scripts/14_render_flight.py --seconds 2.0
  python scripts/14_render_flight.py --camera track1
  python scripts/14_render_flight.py --no-control        # 制御なし(宙返りする)

出力: out/flight_learned.mp4  と軌跡の数値
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
from flight_env import DT, WINGS, decode, euler, get_aero, get_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
POLICY = OUT / "flight_policy.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=1.2)
    ap.add_argument("--camera", default="track3",
                    help="track1 / track2 / track3 / back / side / hero")
    ap.add_argument("--slow", type=float, default=25.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, nargs=2, default=[540, 720])
    ap.add_argument("--z0", type=float, default=10.0)
    ap.add_argument("--target", type=float, nargs=3, default=[8.0, 0.0, 10.0])
    ap.add_argument("--no-control", action="store_true", help="姿勢制御を切る")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not POLICY.exists():
        raise SystemExit(f"{POLICY} がありません。先に scripts/13_train_flight.py を実行してください。")
    pol = json.loads(POLICY.read_text(encoding="utf-8"))
    p = dict(pol["params"])
    print(f"方策: stage {pol['stage']} {pol['label']}  羽ばたき {p['freq']:.0f} Hz")
    if args.no_control:
        for k in ("kp_pitch", "kd_pitch", "kp_roll", "kd_roll",
                  "kp_yaw", "kd_yaw", "kp_alt", "kd_alt"):
            p[k] = 0.0
        print("  姿勢制御: 切")

    m = get_model()
    aero = get_aero()      # 翼素理論の空力 (胴体の抗力は MuJoCo 側)
    aero.reset()
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.res[1])
    m.vis.global_.offheight = max(m.vis.global_.offheight, args.res[0])
    A = {n: m.actuator(n).id for n in WINGS}
    d = mujoco.MjData(m)
    d.qpos[2] = args.z0
    tx, ty, tz = args.target

    renderer = mujoco.Renderer(m, height=args.res[0], width=args.res[1])
    frames = []
    frame_every = max(int(1.0 / (args.fps * args.slow) / DT), 1)

    mid = (p["pitch_down"] + p["pitch_up"]) / 2
    amp_p = (p["pitch_down"] - p["pitch_up"]) / 2
    n_steps = int(args.seconds / DT)
    traj = []

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

        dev = p["yaw"] + p.get("yaw_amp", 0.0) * np.cos(th + p.get("yaw_phase", 0.0))
        for side, sgn in (("left", +1.0), ("right", -1.0)):
            d.ctrl[A[f"wing_yaw_{side}"]] = np.clip(dev, -1.5, 1.5)
            d.ctrl[A[f"wing_roll_{side}"]] = np.clip(mean + (base + sgn * u_roll) * np.cos(th), -1.0, 1.5)
            d.ctrl[A[f"wing_pitch_{side}"]] = np.clip(feather + sgn * u_yaw, -1.27, 2.92)

        aero.apply(m, d)
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            print(f"t={t:.3f}s で発散")
            break
        if d.qpos[2] < 0.5:
            print(f"t={t:.3f}s で接地")
            break
        if s % 2500 == 0:
            r_, p_, y_ = np.degrees(euler(d.qpos[3:7]))
            nose = np.degrees(np.arcsin(np.clip(
                d.xmat[m.body("thorax").id].reshape(3, 3)[2, 0], -1, 1)))
            traj.append((t, *map(float, d.qpos[:3]), r_, float(nose), y_))
        if s % frame_every == 0:
            renderer.update_scene(d, camera=args.camera)
            frames.append(renderer.render())

    print(f"\n{'t[s]':>6} {'x':>7} {'y':>7} {'z':>7} {'roll':>7} {'体軸角':>7} {'yaw':>7}  [cm, deg]")
    for r in traj:
        print(f"{r[0]:6.2f} {r[1]:7.2f} {r[2]:7.2f} {r[3]:7.2f} {r[4]:7.1f} {r[5]:7.1f} {r[6]:7.1f}")

    OUT.mkdir(exist_ok=True)
    path = OUT / (args.out or ("flight_nocontrol.mp4" if args.no_control else "flight_learned.mp4"))
    imageio.mimsave(path, frames, fps=args.fps, macro_block_size=None)
    print(f"\n書き出し: {path} ({len(frames)} フレーム, {path.stat().st_size/1e6:.1f} MB, "
          f"{args.slow:.0f}倍スロー)")


if __name__ == "__main__":
    main()
