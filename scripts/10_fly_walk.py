"""物理エンジンの中でハエを歩かせる (NeuroMechFly v2 / MuJoCo)。

体はマイクロCTスキャンから作られた実物大の3Dモデル。6本脚 66関節を
三脚歩行(tripod gait)のパターンで動かして、MuJoCo が接触・摩擦・重力を解く。

コネクトーム側の話ではなく「体」の側。ここに脳を繋ぐのが今の流行り (README参照)。

使い方:
  python scripts/10_fly_walk.py                       # 1.5秒ぶんを 0.1倍速で録画
  python scripts/10_fly_walk.py --seconds 3 --freq 10
  python scripts/10_fly_walk.py --adhesion            # 脚の吸着をONに

出力: out/fly_walk.mp4
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

# 三脚歩行: 左前・右中・左後 が一組、残りが逆位相
TRIPOD_A = {"lf", "rm", "lh"}
LEG_ORDER = ["lf", "lm", "lh", "rf", "rm", "rh"]


def build(adhesion: bool, kp: float = 45.0, force: float = 100.0):
    fly = NeuroMechFly()
    sk = Skeleton(axis_order=list(NeuroMechFly.AXIS_ORDER_CLASS)[0], joint_preset="legs_only")
    fly.add_joints(sk, KinematicPosePreset.NEUTRAL, stiffness=10.0, damping=0.5)
    dofs = sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ONLY)
    # kp を上げないと関節が指令に追従せず、その場で震えるだけになる
    fly.add_actuators(dofs, ActuatorType.POSITION, KinematicPosePreset.NEUTRAL,
                      forcerange=(-force, force), kp=kp)
    if adhesion:
        fly.add_leg_adhesion()
    fly.add_tracking_camera()
    fly.colorize()

    world = FlatGroundWorld()
    world.add_fly(fly, spawn_position=np.array([0.0, 0.0, 0.5]),
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=False)
    return fly, world


def dof_index(fly):
    """各脚の coxa pitch / femur pitch / tibia pitch の通し番号を拾う。"""
    order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    idx = {leg: {} for leg in LEG_ORDER}
    for i, d in enumerate(order):
        child = d.child.name           # 例 'lf_coxa'
        leg, _, seg = child.partition("_")
        if leg not in idx:
            continue
        axis = d.axis.value
        if seg == "coxa" and axis == "pitch":
            idx[leg]["coxa"] = i
        elif seg == "trochanterfemur" and axis == "pitch":
            idx[leg]["femur"] = i
        elif seg == "tibia" and axis == "pitch":
            idx[leg]["tibia"] = i
    return idx, len(order)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--freq", type=float, default=9.0, help="歩行周波数 Hz")
    ap.add_argument("--swing", type=float, default=0.45, help="前後振り幅 rad")
    ap.add_argument("--lift", type=float, default=0.35, help="持ち上げ幅 rad")
    ap.add_argument("--speed", type=float, default=0.1, help="再生速度(0.1=10倍スロー)")
    ap.add_argument("--res", type=int, nargs=2, default=[480, 640])
    ap.add_argument("--adhesion", action="store_true")
    ap.add_argument("--kp", type=float, default=45.0)
    args = ap.parse_args()

    print("ハエの体を組み立て中 ...")
    fly, world = build(args.adhesion, kp=args.kp)
    sim = Simulation(world)
    idx, n_act = dof_index(fly)
    legs_order = [str(l.value if hasattr(l, "value") else l) for l in fly.get_legs_order()]
    print(f"関節アクチュエータ {n_act} 個 / timestep {sim.timestep*1e3:.2f} ms")

    sim.set_renderer("nmf/trackcam", camera_res=tuple(args.res),
                     playback_speed=args.speed, output_fps=30)
    sim.reset()
    sim.warmup(0.1)
    neutral = sim.mj_data.ctrl[:n_act].copy()
    start = sim.mj_data.qpos[:3].copy()

    n_steps = int(args.seconds / sim.timestep)
    print(f"{n_steps:,} ステップ回します ...")
    for step in range(n_steps):
        t = step * sim.timestep
        ctrl = neutral.copy()
        for leg in LEG_ORDER:
            phase = 0.0 if leg in TRIPOD_A else np.pi
            th = 2 * np.pi * args.freq * t + phase
            if "coxa" in idx[leg]:
                ctrl[idx[leg]["coxa"]] -= args.swing * np.cos(th)
            if "femur" in idx[leg]:
                ctrl[idx[leg]["femur"]] += args.lift * max(0.0, np.sin(th))
            if "tibia" in idx[leg]:
                ctrl[idx[leg]["tibia"]] += 0.15 * max(0.0, np.sin(th))
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, ctrl)
        if args.adhesion:
            # 遊脚(持ち上げ中)は吸着を切り、接地脚は吸着させる
            states = np.array([
                0.0 if np.sin(2 * np.pi * args.freq * t + (0.0 if leg in TRIPOD_A else np.pi)) > 0 else 1.0
                for leg in legs_order
            ])
            sim.set_leg_adhesion_states(fly.name, states)
        sim.step()
        sim.render_as_needed()

    disp = sim.mj_data.qpos[:3] - start
    print(f"移動: dx={disp[0]:+.2f} dy={disp[1]:+.2f} mm  ({np.hypot(disp[0],disp[1])/args.seconds:.1f} mm/s)")
    print("注意: 素朴なサイン波の歩容なので、実際のハエのようには歩かない(よろけて進む)。")

    OUT.mkdir(exist_ok=True)
    path = OUT / "fly_walk.mp4"
    sim.renderer.save_video(path)
    print(f"書き出し: {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
