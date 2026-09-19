"""コネクトームが指した動作を、物理エンジンの体にやらせる。

回路の追跡結果 (scripts/05, 06, 08):

    LC4 / LPLC2 (迫る影)  →  DNp01 Giant Fiber  →  TTMn  →  中脚の伸展  →  跳躍
                                              90シナプス, LTct 内

TTMn (tergotrochanteral motor neuron) が動かすのは **中脚** の
転節-腿節関節。ここを一気に伸ばすと体が浮く。それを NeuroMechFly v2 の
実物大の体 (MuJoCo) で実行する。

中脚だけ伸ばした方が、後脚も一緒に伸ばすより高く跳ぶ。これは
TTM が中脚の筋肉であるという解剖と一致する (--with-hind で比較できる)。

使い方:
  python scripts/11_escape_takeoff.py
  python scripts/11_escape_takeoff.py --amp 2.5 --with-hind
  python scripts/11_escape_takeoff.py --speed 0.02      # さらにスロー

出力: out/escape_takeoff.mp4
"""

import argparse
from pathlib import Path

import numpy as np
from flygym import Simulation
from flygym.anatomy import ActuatedDOFPreset, Skeleton
from flygym.compose import (ActuatorType, FlatGroundWorld, KinematicPosePreset,
                            NeuroMechFly)
from flygym.utils.math import Rotation3D

OUT = Path(__file__).resolve().parent.parent / "out"
LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]


def build(kp: float, force: float):
    fly = NeuroMechFly()
    sk = Skeleton(axis_order=list(NeuroMechFly.AXIS_ORDER_CLASS)[0], joint_preset="legs_only")
    fly.add_joints(sk, KinematicPosePreset.NEUTRAL, stiffness=10.0, damping=0.5)
    dofs = sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ONLY)
    fly.add_actuators(dofs, ActuatorType.POSITION, KinematicPosePreset.NEUTRAL,
                      forcerange=(-force, force), kp=kp)
    fly.add_tracking_camera()
    fly.colorize()
    world = FlatGroundWorld()
    world.add_fly(fly, spawn_position=np.array([0.0, 0.0, 0.5]),
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=False)
    return fly, world


def dofmap(fly):
    order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    idx = {l: {} for l in LEGS}
    for i, d in enumerate(order):
        leg, _, seg = d.child.name.partition("_")
        if leg in idx:
            idx[leg][(seg, d.axis.value)] = i
    return idx, len(order)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amp", type=float, default=2.5, help="伸展の大きさ rad")
    ap.add_argument("--trigger", type=float, default=0.25, help="DNp01 が発火する時刻 s")
    ap.add_argument("--seconds", type=float, default=0.8)
    ap.add_argument("--rise-ms", type=float, default=4.0, help="伸展にかける時間 ms")
    ap.add_argument("--with-hind", action="store_true", help="後脚も一緒に伸ばす(比較用)")
    ap.add_argument("--kp", type=float, default=45.0)
    ap.add_argument("--force", type=float, default=500.0)
    ap.add_argument("--speed", type=float, default=0.05, help="再生速度(0.05=20倍スロー)")
    ap.add_argument("--res", type=int, nargs=2, default=[540, 720])
    args = ap.parse_args()

    legs = ("lm", "rm", "lh", "rh") if args.with_hind else ("lm", "rm")
    print(f"蹴る脚: {legs}  (TTMn は中脚の運動ニューロン)")

    fly, world = build(args.kp, args.force)
    sim = Simulation(world)
    idx, n_act = dofmap(fly)
    sim.set_renderer("nmf/trackcam", camera_res=tuple(args.res),
                     playback_speed=args.speed, output_fps=30)
    sim.reset()
    sim.warmup(0.15)
    neutral = sim.mj_data.ctrl[:n_act].copy()

    z0 = float(sim.mj_data.qpos[2])
    z_max = z0
    fired = False
    for step in range(int(args.seconds / sim.timestep)):
        t = step * sim.timestep
        ctrl = neutral.copy()
        if t >= args.trigger:
            if not fired:
                print(f"t={t*1000:.0f} ms : DNp01 発火 → TTMn → 中脚伸展")
                fired = True
            k = min((t - args.trigger) / (args.rise_ms / 1000.0), 1.0)
            for leg in legs:
                ctrl[idx[leg][("trochanterfemur", "pitch")]] += args.amp * k
                ctrl[idx[leg][("tibia", "pitch")]] += args.amp * k
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, ctrl)
        sim.step()
        sim.render_as_needed()
        z_max = max(z_max, float(sim.mj_data.qpos[2]))

    print(f"胸部の高さ: {z0:.2f} mm → 最高 {z_max:.2f} mm  (上昇 {z_max-z0:+.2f} mm)")
    print(f"体長がおよそ 2.5 mm なので、体長の {(z_max-z0)/2.5*100:.0f}% ぶん浮いた")

    OUT.mkdir(exist_ok=True)
    path = OUT / "escape_takeoff.mp4"
    sim.renderer.save_video(path)
    print(f"書き出し: {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
