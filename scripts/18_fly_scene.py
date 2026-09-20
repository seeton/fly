"""風景の中を、複眼を使って飛ばす。

これまでの飛行は機体の姿勢を「正確に知っている」前提だった (物理エンジンから
真値を読んでいた)。実物のハエは目と平均棍でそれを推定している。
ここではヨーの角速度を **複眼のオプティックフローだけ** から推定して、
針路の維持に使う (視運動反応)。

較正結果 (scripts の較正で実測):
    ヨー角速度 ≒ -8.6 * (左の流れ + 右の流れ)
    左右の和にすると並進の流れが打ち消し合い、回転だけが残る。
    前進速度を 0〜40 cm/s まで変えても **符号は100%一致** する。

使い方:
  python scripts/18_fly_scene.py                     # 視覚で針路維持
  python scripts/18_fly_scene.py --no-vision         # 視覚を切る (比較用)
  python scripts/18_fly_scene.py --disturb 8.0       # 途中で横風のような外乱
  python scripts/18_fly_scene.py --camera scene_side

出力: out/fly_scene.mp4
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
from fly_vision import FlyEyes  # noqa: E402
from flybody_model import load_model  # noqa: E402
from insect_aero import WingAero  # noqa: E402

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


def build(vis_hz=200.0, seed=0):
    m, _ = load_model(wing_kp=50.0, wing_kv=2 * np.sqrt(50.0 * 1e-6),
                      scene=True, scene_seed=seed)
    m.opt.timestep = DT
    aero = WingAero(m, n_elem=6)
    aero.route_wings_to_this_model(m)
    i_add = aero.added_mass_inertia(m)
    for jn in ("wing_roll_left", "wing_roll_right", "wing_yaw_left", "wing_yaw_right"):
        m.dof_armature[m.jnt_dofadr[m.joint(jn).id]] += i_add
    eyes = FlyEyes(m, rate_hz=vis_hz)
    return m, aero, eyes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--camera", default="scene_follow")
    ap.add_argument("--slow", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, nargs=2, default=[540, 860])
    ap.add_argument("--z0", type=float, default=12.0)
    ap.add_argument("--yaw-source", choices=["vision", "true", "none"],
                    default="vision",
                    help="ヨー角速度をどこから取るか: 複眼 / 物理エンジンの真値 / 使わない")
    ap.add_argument("--disturb", type=float, default=0.0,
                    help="t=0.5s に与えるヨー方向の外乱 [rad/s]")
    ap.add_argument("--k-vis", type=float, default=0.02,
                    help="ヨー角速度フィードバックのゲイン [制御量 / (rad/s)]")
    ap.add_argument("--no-heading", action="store_true", default=True,
                    help="方位の比例項を切る (既定。実物のハエに絶対方位センサは無い)")
    ap.add_argument("--with-heading", dest="no_heading", action="store_false")
    ap.add_argument("--no-head-stab", dest="head_stab", action="store_false",
                    default=True, help="頭部の姿勢安定化を切る (比較用)")
    ap.add_argument("--vis-tau", type=float, default=0.03,
                    help="視覚推定の平滑化の時定数 [s]")
    ap.add_argument("--seek", type=float, default=0.0,
                    help="花へ向かう操舵のゲイン (0で無効)")
    ap.add_argument("--seek-sign", type=float, default=1.0)
    ap.add_argument("--no-video", action="store_true", help="動画を作らない (速い)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = json.loads(POLICY.read_text(encoding="utf-8"))["params"]
    m, aero, eyes = build()
    A = {n: m.actuator(n).id for n in WINGS}
    legs = {m.actuator(k).id: v for k, v in leg_targets(m).items()}
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.res[1])
    m.vis.global_.offheight = max(m.vis.global_.offheight, args.res[0])
    renderer = mujoco.Renderer(m, height=args.res[0], width=args.res[1])

    try:
        flower_sid = m.site("flower").id
    except Exception:
        flower_sid = None

    d = mujoco.MjData(m)
    d.qpos[2] = args.z0
    for aid, val in legs.items():
        d.ctrl[aid] = val
        d.qpos[m.jnt_qposadr[m.actuator_trnid[aid, 0]]] = val
    aero.reset()
    eyes.reset()

    mid = (p["pitch_down"] + p["pitch_up"]) / 2
    amp_p = (p["pitch_down"] - p["pitch_up"]) / 2
    frames, traj = [], []
    frame_every = max(int(1.0 / (args.fps * args.slow) / DT), 1)
    n_steps = int(args.seconds / DT)
    yaw_vis = 0.0
    yaw_vis_f = 0.0
    # 頭部の姿勢安定化に使う、ゆっくり動く基準 (これとの差 = 速い揺れ)
    yaw_slow, roll_slow = 0.0, 0.0
    a_slow = DT / 0.05
    a_vis = DT / max(args.vis_tau, 1e-4)
    head_ids = {}
    for hn in ("head_abduct", "head"):
        try:
            head_ids[hn] = m.actuator(hn).id
        except Exception:
            pass

    src = {"vision": "複眼から推定", "true": "物理エンジンの真値",
           "none": "使わない"}[args.yaw_source]
    print(f"ヨー角速度の取得元: {src}   外乱 {args.disturb} rad/s   {args.seconds}s")
    for s in range(n_steps):
        t = s * DT
        th = 2 * np.pi * p["freq"] * t
        roll_b, pitch_b, yaw_b = euler(d.qpos[3:7])
        wx, wy, wz = d.qvel[3], d.qvel[4], d.qvel[5]

        if eyes.maybe_update(m, d, t):
            yaw_vis = eyes.rotation_signal
        # 羽ばたき(250Hz)の揺れが視覚の更新(200Hz)にエイリアシングで乗るので平滑化
        yaw_vis_f += a_vis * (yaw_vis - yaw_vis_f)

        # --- 頭部の姿勢安定化 ---
        # 実物のハエは首で頭を逆に振り、視線を安定させる。
        # ゆっくりした成分(=本来の針路)は残し、速い揺れだけ打ち消す。
        yaw_slow += a_slow * (yaw_b - yaw_slow)
        roll_slow += a_slow * (roll_b - roll_slow)
        if args.head_stab:
            if "head_abduct" in head_ids:
                d.ctrl[head_ids["head_abduct"]] = float(
                    np.clip(-(yaw_b - yaw_slow), -0.2, 0.2))
            if "head" in head_ids:
                d.ctrl[head_ids["head"]] = float(
                    np.clip(-(roll_b - roll_slow), -0.5, 0.3))

        # ヨーの減衰項をどこから取るか。
        #   vision: 複眼のオプティックフローから推定した角速度 [rad/s]
        #   true  : 物理エンジンの真値 (これまでの飛行はこれ)
        #   none  : 減衰なし (外乱を入れると回り続ける)
        if args.yaw_source == "vision":
            wz_src = yaw_vis_f
        elif args.yaw_source == "true":
            wz_src = wz
        else:
            wz_src = 0.0
        # 実物のハエに「絶対方位」を知る手段は無い。既定では方位の比例項を切り、
        # 角速度のフィードバックだけで針路を保つ (視運動反応と同じ形)。
        kp_yaw = 0.0 if args.no_heading else p["kp_yaw"]
        u_p = np.clip(-p["kp_pitch"] * pitch_b - p["kd_pitch"] * wy, -0.8, 0.8)
        u_r = np.clip(-p["kp_roll"] * roll_b - p["kd_roll"] * wx, -0.5, 0.5)
        # 花へ向かう操舵。複眼で見えたピンクの方向へ機首を振る。
        seek = 0.0
        if args.seek:
            seek = args.seek_sign * args.seek * eyes.front_bearing()
        u_y = np.clip(-kp_yaw * yaw_b - args.k_vis * wz_src + seek, -0.6, 0.6)
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
        if args.disturb and abs(t - 0.5) < DT / 2:
            d.qvel[5] += args.disturb        # 横風のようにヨーを蹴る
            print(f"  t=0.50s: ヨーに {args.disturb:+.1f} rad/s の外乱")
        if not np.isfinite(d.qpos).all() or d.qpos[2] < 0.5:
            print(f"  t={t:.2f}s で終了 (高度 {d.qpos[2]:.2f}cm)")
            break
        if s % 5000 == 0:
            if flower_sid is not None:
                dist_f = float(np.linalg.norm(d.site_xpos[flower_sid] - d.qpos[:3]))
            else:
                dist_f = float("nan")
            traj.append((t, *map(float, d.qpos[:3]),
                         float(np.degrees(euler(d.qpos[3:7])[2])), dist_f))
        if (not args.no_video) and s % frame_every == 0:
            renderer.update_scene(d, camera=args.camera)
            img = renderer.render()
            eye = eyes.side_by_side(scale=2)
            h, w = eye.shape[:2]
            img[8:8 + h, 8:8 + w] = eye          # 左上に複眼の像を貼る
            frames.append(img)

    print(f"\n{'t[s]':>6} {'x':>8} {'y':>8} {'z':>7} {'方位[deg]':>10} {'視覚のヨー推定':>14}")
    for r in traj:
        print(f"{r[0]:6.2f} {r[1]:8.2f} {r[2]:8.2f} {r[3]:7.2f} {r[4]:10.1f} {r[5]:12.2f}")
    if traj:
        head = [r[4] for r in traj]
        print(f"\n方位のばらつき: {np.std(head):.1f} deg  "
              f"最終の横ずれ: {traj[-1][2]:+.2f} cm")

    OUT.mkdir(exist_ok=True)
    name = args.out or f"fly_scene_{args.yaw_source}.mp4"
    path = OUT / name
    imageio.mimsave(path, frames, fps=args.fps, macro_block_size=None)
    print(f"書き出し: {path} ({len(frames)} フレーム, {path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
