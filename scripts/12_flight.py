"""ハエを飛ばす — 翅の空力を有効にして羽ばたかせる。

使っているのは Janelia flybody (fruitfly.xml) の翅つき全身モデル。
翅の付け根は3自由度:

    roll  = ストローク (前→後ろに扇ぐ)      ← 主動力
    pitch = 迎角 (翅の長軸まわりのひねり)   ← 打ち返しで反転
    yaw   = 展開 (畳む↔広げる)

この3つを 200 Hz で動かすと、MuJoCo の楕円体流体モデルが翅にかかる
揚力・抗力を解いて、機体が浮く。

パラメータは「正味空力の大きさ」を目的関数にしたランダム探索で決めた
(scripts/README 参照)。見つかった最適値は羽ばたき周波数 200 Hz、
ストローク振幅ほぼ最大 — 実際のショウジョウバエとほぼ同じ。

空力係数について:
  MuJoCo の標準の楕円体流体モデルは、昆虫が使う **前縁渦(LEV)** を
  表現しない。実測の昆虫の揚力係数は定常翼理論の2〜3倍あるので、
  --aero real ではその効果を Kutta 揚力係数で粗く代表させている。
    --aero plain  : MuJoCo 既定の係数 → 体重の 0.73 倍しか出ず、落ちる
    --aero real   : 前縁渦を反映   → 体重の 1.43 倍、浮いて上昇する

使い方:
  python scripts/12_flight.py                    # 飛ぶ
  python scripts/12_flight.py --aero plain       # 既定係数だと落ちる様子
  python scripts/12_flight.py --camera track3 --seconds 0.4

出力: out/flight.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flybody_model import load_model  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
WINGS = ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
         "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]

# ランダム探索で得た羽ばたきパラメータ
GAIT = dict(freq=199.85, yaw=-0.048, roll_mean=0.263, roll_amp=1.250,
            pitch_down=2.900, pitch_up=1.627, sharp=7.894, phase=-2.969)
GAIT_PLAIN = dict(freq=245.52, yaw=-0.109, roll_mean=-0.300, roll_amp=1.225,
                  pitch_down=-0.130, pitch_up=1.606, sharp=3.176, phase=3.121)

AERO = {
    # [enable, blunt drag, slender drag, angular drag, Kutta lift, Magnus lift]
    "plain": [1.0, 0.5, 0.25, 1.5, 1.0, 1.0],
    "real": [1.0, 1.5, 0.25, 1.5, 3.0, 1.0],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aero", choices=["real", "plain"], default="real")
    ap.add_argument("--seconds", type=float, default=0.3)
    ap.add_argument("--start-height", type=float, default=1.0, help="開始高度 cm")
    ap.add_argument("--camera", default="track3",
                    help="track1 / track2 / track3 / back / side / hero")
    ap.add_argument("--slow", type=float, default=60.0, help="何倍スローにするか")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, nargs=2, default=[540, 720])
    args = ap.parse_args()

    dt = 2e-5
    kp = 50.0
    model, _ = load_model(wing_kp=kp, wing_kv=2 * np.sqrt(kp * 1e-6), add_floor=True)
    model.opt.timestep = dt
    for i in range(model.ngeom):
        n = model.geom(i).name
        if "wing" in n and "fluid" in n:
            model.geom_fluid[i][:6] = AERO[args.aero]

    gait = GAIT if args.aero == "real" else GAIT_PLAIN
    A = {n: model.actuator(n).id for n in WINGS}
    mass = model.body_subtreemass[model.body("thorax").id]
    g = abs(model.opt.gravity[2])
    print(f"体重 {mass*1e3:.3f} mg   羽ばたき {gait['freq']:.0f} Hz   空力: {args.aero}")

    data = mujoco.MjData(model)
    data.qpos[2] = args.start_height

    # オフスクリーンバッファは既定 640x480 なので広げる
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.res[1])
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.res[0])
    renderer = mujoco.Renderer(model, height=args.res[0], width=args.res[1])
    frames = []
    frame_every = max(int(1.0 / (args.fps * args.slow) / dt), 1)

    n_steps = int(args.seconds / dt)
    mid = (gait["pitch_down"] + gait["pitch_up"]) / 2
    amp = (gait["pitch_down"] - gait["pitch_up"]) / 2
    z0 = float(data.qpos[2])
    z_track = []

    for s in range(n_steps):
        t = s * dt
        th = 2 * np.pi * gait["freq"] * t
        roll = gait["roll_mean"] + gait["roll_amp"] * np.cos(th)
        pitch = mid + amp * np.tanh(gait["sharp"] * np.sin(th + gait["phase"]))
        for side in ("left", "right"):
            data.ctrl[A[f"wing_yaw_{side}"]] = gait["yaw"]
            data.ctrl[A[f"wing_roll_{side}"]] = roll
            data.ctrl[A[f"wing_pitch_{side}"]] = pitch
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            print("発散しました")
            break
        z_track.append(float(data.qpos[2]))
        if s % frame_every == 0:
            renderer.update_scene(data, camera=args.camera)
            frames.append(renderer.render())

    z1 = z_track[-1] if z_track else z0
    vz = float(data.qvel[2])
    lift = vz / (g * args.seconds) + 1.0
    print(f"高度 {z0:.2f} → {z1:.2f} cm  ({z1-z0:+.2f} cm)   上昇速度 {vz:.1f} cm/s")
    print(f"正味空力 ≒ 体重の {lift:.2f} 倍  ({'浮く' if lift > 1 else '落ちる'})")

    OUT.mkdir(exist_ok=True)
    path = OUT / ("flight.mp4" if args.aero == "real" else "flight_plain.mp4")
    imageio.mimsave(path, frames, fps=args.fps, macro_block_size=None)
    print(f"書き出し: {path} ({len(frames)} フレーム, {path.stat().st_size/1e6:.1f} MB, "
          f"{args.slow:.0f}倍スロー)")


if __name__ == "__main__":
    main()
