"""視覚フィードバックの有無を並べて比べる動画を作る。

同じ外乱 (t=0.5s にヨーへ 6 rad/s) を与えて、3つの条件を横に並べる:

  1. 複眼 + 頭部の姿勢安定化   <- 実物のハエに近い形
  2. 複眼のみ (頭を安定させない)
  3. ヨーのフィードバックなし

使い方:
  python scripts/19_compare_vision.py
  python scripts/19_compare_vision.py --seconds 2.0

出力: out/vision_compare.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flight_posture import leg_targets  # noqa: E402
from fly_vision import FlyEyes  # noqa: E402
from flybody_model import load_model  # noqa: E402
from insect_aero import WingAero  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
POLICY = OUT / "flight_policy.json"
WINGS = ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
         "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]
DT = 2e-5
FONT = r"C:\Windows\Fonts\meiryo.ttc"

CASES = [
    ("複眼 + 頭部安定化", "vision", True),
    ("複眼のみ (頭を固定)", "vision", False),
    ("ヨーの手がかり無し", "none", True),
]


def euler(q):
    w, x, y, z = q
    return (np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run_case(yaw_source, head_stab, args, p):
    m, _ = load_model(wing_kp=50.0, wing_kv=2 * np.sqrt(50.0 * 1e-6), scene=True)
    m.opt.timestep = DT
    aero = WingAero(m, n_elem=6)
    aero.route_wings_to_this_model(m)
    i_add = aero.added_mass_inertia(m)
    for jn in ("wing_roll_left", "wing_roll_right", "wing_yaw_left", "wing_yaw_right"):
        m.dof_armature[m.jnt_dofadr[m.joint(jn).id]] += i_add
    eyes = FlyEyes(m, rate_hz=200.0)

    A = {n: m.actuator(n).id for n in WINGS}
    legs = {m.actuator(k).id: v for k, v in leg_targets(m).items()}
    head_ids = {h: m.actuator(h).id for h in ("head_abduct", "head")}
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.panel[1])
    m.vis.global_.offheight = max(m.vis.global_.offheight, args.panel[0])
    renderer = mujoco.Renderer(m, height=args.panel[0], width=args.panel[1])

    d = mujoco.MjData(m)
    d.qpos[2] = args.z0
    for aid, val in legs.items():
        d.ctrl[aid] = val
        d.qpos[m.jnt_qposadr[m.actuator_trnid[aid, 0]]] = val
    aero.reset()
    eyes.reset()

    mid = (p["pitch_down"] + p["pitch_up"]) / 2
    amp_p = (p["pitch_down"] - p["pitch_up"]) / 2
    a_slow, a_vis = DT / 0.05, DT / 0.03
    yaw_vis = yaw_vis_f = yaw_slow = roll_slow = 0.0
    frames, heads = [], []
    frame_every = max(int(1.0 / (args.fps * args.slow) / DT), 1)

    for s in range(int(args.seconds / DT)):
        t = s * DT
        th = 2 * np.pi * p["freq"] * t
        roll_b, pitch_b, yaw_b = euler(d.qpos[3:7])
        wx, wy, wz = d.qvel[3], d.qvel[4], d.qvel[5]

        if eyes.maybe_update(m, d, t):
            yaw_vis = eyes.rotation_signal
        yaw_vis_f += a_vis * (yaw_vis - yaw_vis_f)

        yaw_slow += a_slow * (yaw_b - yaw_slow)
        roll_slow += a_slow * (roll_b - roll_slow)
        if head_stab:
            d.ctrl[head_ids["head_abduct"]] = float(np.clip(-(yaw_b - yaw_slow), -0.2, 0.2))
            d.ctrl[head_ids["head"]] = float(np.clip(-(roll_b - roll_slow), -0.5, 0.3))

        wz_src = yaw_vis_f if yaw_source == "vision" else 0.0
        u_p = np.clip(-p["kp_pitch"] * pitch_b - p["kd_pitch"] * wy, -0.8, 0.8)
        u_r = np.clip(-p["kp_roll"] * roll_b - p["kd_roll"] * wx, -0.5, 0.5)
        u_y = np.clip(-args.k_vis * wz_src, -0.6, 0.6)
        u_a = np.clip(p["kp_alt"] * (args.z0 - d.qpos[2]) - p["kd_alt"] * d.qvel[2], -0.4, 0.4)

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
        if args.disturb and abs(t - 0.5) < DT / 2:
            d.qvel[5] += args.disturb
        crashed = (not np.isfinite(d.qpos).all()) or d.qpos[2] < 0.5
        if s % frame_every == 0:
            renderer.update_scene(d, camera=args.camera)
            img = renderer.render()
            eye = eyes.side_by_side(scale=2)
            h, w = eye.shape[:2]
            img[6:6 + h, 6:6 + w] = eye
            frames.append(img)
            heads.append(float(np.degrees(yaw_b)))
        if crashed:
            break
    last = frames[-1] if frames else np.zeros((args.panel[0], args.panel[1], 3), np.uint8)
    n_want = int(args.seconds * args.fps * args.slow)
    while len(frames) < n_want:
        frames.append(last)                      # 墜落後はその場面を保持
        heads.append(heads[-1] if heads else 0.0)
    stats = dict(x=float(d.qpos[0]), y=float(d.qpos[1]), z=float(d.qpos[2]),
                 head_std=float(np.std(heads)))
    return frames[:n_want], stats


def label(img, text, sub):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    f1 = ImageFont.truetype(FONT, 20)
    f2 = ImageFont.truetype(FONT, 15)
    h = im.size[1]
    dr.rectangle([0, h - 52, im.size[0], h], fill=(0, 0, 0))
    dr.text((10, h - 48), text, font=f1, fill=(255, 255, 255))
    dr.text((10, h - 23), sub, font=f2, fill=(180, 190, 200))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=1.6)
    ap.add_argument("--slow", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--panel", type=int, nargs=2, default=[420, 500])
    ap.add_argument("--z0", type=float, default=10.0)
    ap.add_argument("--disturb", type=float, default=6.0)
    ap.add_argument("--k-vis", type=float, default=0.01)
    ap.add_argument("--camera", default="scene_follow")
    args = ap.parse_args()

    p = json.loads(POLICY.read_text(encoding="utf-8"))["params"]
    panels, stats = [], []
    for name, src, hs in CASES:
        print(f"計算中: {name} ...")
        fr, st = run_case(src, hs, args, p)
        panels.append((name, fr, st))
        stats.append(st)
        print(f"   到達 x={st['x']:+.1f}cm  高度 {st['z']:.2f}cm  "
              f"方位のばらつき {st['head_std']:.1f}deg")

    n = min(len(f) for _, f, _ in panels)
    out = []
    for i in range(n):
        row = []
        for name, fr, st in panels:
            sub = f"到達 {st['x']:+.1f}cm / 方位のばらつき {st['head_std']:.0f}deg"
            row.append(label(fr[i], name, sub))
        out.append(np.concatenate(row, axis=1))

    OUT.mkdir(exist_ok=True)
    path = OUT / "vision_compare.mp4"
    imageio.mimsave(path, out, fps=args.fps, macro_block_size=None)
    print(f"\n書き出し: {path} ({len(out)} フレーム, {path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
