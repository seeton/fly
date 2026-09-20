"""実装した翼素理論の空力を検証する。

合格基準 (実物のショウジョウバエ):
    ホバリング相当の羽ばたき (約200 Hz, ストローク振幅 約2.4 rad p-p,
    迎角 約45度) で
      揚力  約 1.0 体重ぶん
      パワー 約 300-800 erg/s

MuJoCo 内蔵の楕円体流体では「実物の4倍のパワーで6割の揚力」だった。

使い方:
  python scripts/16_validate_aero.py
  python scripts/16_validate_aero.py --sweep      # 迎角と振幅を振る
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flybody_model import load_model  # noqa: E402
from insect_aero import C_D, C_L, WingAero  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
WINGS = ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
         "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]


def build(dt=2e-5, kp=50.0, n_elem=6):
    m, _ = load_model(wing_kp=kp, wing_kv=2 * np.sqrt(kp * 1e-6), add_floor=False)
    m.opt.timestep = dt
    aero = WingAero(m, n_elem=n_elem)
    aero.route_wings_to_this_model(m)
    return m, aero


def tethered(m, aero, freq=200.0, roll_mean=0.25, roll_amp=1.2,
             feather=0.79, sharp=3.0, yaw_mean=0.0, yaw_amp=0.0, yaw_phase=0.0,
             pitch_deg=47.0, seconds=0.10):
    """機体を機首上げ pitch_deg に固定して、平均の空気力 [体重] と
    正味パワー [erg/s] を返す。

    水平に固定して測ると揚力が過小に出る。実物のハエはホバリング時に
    機首を30-45度上げていて、そこが力の釣り合う姿勢だから。
    """
    A = {n: m.actuator(n).id for n in WINGS}
    ids = [A[n] for n in WINGS]
    mass = m.body_subtreemass[m.body("thorax").id]
    W = mass * abs(m.opt.gravity[2])
    half = np.radians(pitch_deg) / 2.0
    quat = np.array([np.cos(half), 0.0, -np.sin(half), 0.0])
    d = mujoco.MjData(m)
    d.qpos[2] = 30.0
    d.qpos[3:7] = quat
    aero.reset()

    n = int(seconds / m.opt.timestep)
    settle = n // 3
    pw, imp = [], []
    for s in range(n):
        t = s * m.opt.timestep
        th = 2 * np.pi * freq * t
        # 迎角は打ち返しで符号が反転する (feather は片側の振幅 [rad])
        ft = feather * np.tanh(sharp * np.sin(th))
        dev = yaw_mean + yaw_amp * np.cos(th + yaw_phase)
        for side in ("left", "right"):
            d.ctrl[A[f"wing_yaw_{side}"]] = np.clip(dev, -1.5, 1.5)
            d.ctrl[A[f"wing_roll_{side}"]] = np.clip(roll_mean + roll_amp * np.cos(th), -1.0, 1.5)
            d.ctrl[A[f"wing_pitch_{side}"]] = np.clip(ft, -1.27, 2.92)
        aero.apply(m, d)
        mujoco.mj_step(m, d)
        d.qpos[3:7] = quat
        d.qvel[3:6] = 0
        if not np.isfinite(d.qvel).all():
            return None
        if s >= settle:
            # 正味の仕事率。|F v| の和にすると共振で戻るぶんまで費用に数えてしまう
            pw.append(float(np.sum(d.actuator_force[ids] * d.actuator_velocity[ids])))
            imp.append(mass * d.qvel[:3] / m.opt.timestep)
        d.qpos[:3] = [0, 0, 30.0]
        d.qvel[:3] = 0
    F = np.mean(imp, axis=0)
    F[2] += mass * abs(m.opt.gravity[2])
    return F / W, float(np.mean(pw))


def compare_implementations(n_elem: int = 6, steps: int = 20000) -> None:
    """空力の3実装が同じ力を出すことを確かめ、速度を測る。

    速度は学習で回せる試行数に直結する。翼素は片翼6個しかないので
    計算量は無いに等しく、費用はほぼ NumPy の呼び出し回数で決まっていた。
    """
    import time

    import mujoco

    m, aero = build(n_elem=n_elem)
    aero.wind = np.array([-8.0, 0.0, 0.0])
    p = json.loads((OUT / "flight_policy.json").read_text(encoding="utf-8"))["params"]
    A = {n: m.actuator(n).id for n in WINGS}
    mid = (p["pitch_down"] + p["pitch_up"]) / 2
    amp = (p["pitch_down"] - p["pitch_up"]) / 2
    d = mujoco.MjData(m)
    d.qpos[2] = 9.0
    aero.reset()
    aero.apply(m, d)                                   # numba の初回コンパイル

    print("=== 3実装の一致 ===")
    print(f"numba: {'使う' if aero._use_kernel else '無し (NumPy で動作)'}")
    worst_f = worst_m = worst_p = scale = 0.0
    for s in range(steps):
        t = s * m.opt.timestep
        th = 2 * np.pi * p["freq"] * t
        feather = mid + amp * np.tanh(p["sharp"] * np.sin(th + p["phase"]))
        dev = p["yaw"] + p["yaw_amp"] * np.cos(th + p["yaw_phase"])
        for side in ("left", "right"):
            d.ctrl[A[f"wing_yaw_{side}"]] = np.clip(dev, -1.5, 1.5)
            d.ctrl[A[f"wing_roll_{side}"]] = np.clip(
                p["roll_mean"] + p["roll_amp"] * np.cos(th), -1.0, 1.5)
            d.ctrl[A[f"wing_pitch_{side}"]] = np.clip(feather, -1.27, 2.92)
        aero.apply(m, d)
        f1, p1 = d.xfrc_applied.copy(), aero.air_power
        aero._apply_numpy(m, d)
        f2, p2 = d.xfrc_applied.copy(), aero.air_power
        aero._apply_loop(m, d)
        f3, p3 = d.xfrc_applied.copy(), aero.air_power
        worst_f = max(worst_f, float(np.abs(f1[:, :3] - f3[:, :3]).max()),
                      float(np.abs(f2[:, :3] - f3[:, :3]).max()))
        worst_m = max(worst_m, float(np.abs(f1[:, 3:] - f3[:, 3:]).max()))
        worst_p = max(worst_p, abs(p1 - p3), abs(p2 - p3))
        scale = max(scale, float(np.abs(f3).max()))
        mujoco.mj_step(m, d)
    print(f"  {steps} ステップで、素の loop 版との最大の差")
    print(f"    力     {worst_f:.3e}  (力の大きさ {scale:.3e} -> 相対 {worst_f/max(scale,1e-30):.2e})")
    print(f"    トルク {worst_m:.3e}")
    print(f"    パワー {worst_p:.3e} erg/s")

    print("\n=== 1 ステップの費用 ===")
    N = 6000
    times = {}
    for nm, fn in (("numba", aero.apply), ("NumPy", aero._apply_numpy),
                   ("loop ", aero._apply_loop)):
        t0 = time.perf_counter()
        for _ in range(N):
            fn(m, d)
        times[nm] = (time.perf_counter() - t0) / N
    t0 = time.perf_counter()
    for _ in range(N):
        mujoco.mj_step(m, d)
    t_step = (time.perf_counter() - t0) / N
    for nm, v in times.items():
        print(f"  空力 {nm}  {v*1e6:7.1f} us   (mj_step {t_step*1e6:.1f} us に対し "
              f"{v/t_step*100:5.1f}%)")
    print(f"\n  1 秒の飛行 (50000 ステップ) あたり")
    for nm, v in times.items():
        print(f"    {nm}: {(v+t_step)*50000:6.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-elem", type=int, default=6)
    ap.add_argument("--compare", action="store_true",
                    help="空力の3実装 (numba / NumPy / 素の loop) の一致と速度を見る")
    args = ap.parse_args()

    if args.compare:
        compare_implementations(args.n_elem)
        return

    print("=== 力係数の確認 (Dickinson et al. 1999) ===")
    for a in (0, 15, 30, 45, 60, 90):
        print(f"  迎角 {a:2d} deg :  C_L = {C_L(a):+.3f}   C_D = {C_D(a):.3f}")
    print("  実測のショウジョウバエ翅: C_L は迎角45度付近で最大 約1.8")
    print("  (定常翼理論なら約0.9。差が前縁渦のぶん)\n")

    m, aero = build(n_elem=args.n_elem)
    mass = m.body_subtreemass[m.body("thorax").id]
    print(f"体重 {mass*1e3:.3f} mg = {mass*981:.3f} dyn\n")

    print("=== ホバリング相当の羽ばたき ===")
    r = tethered(m, aero, freq=240.0, roll_amp=1.25, feather=0.87, pitch_deg=50.0)
    if r is None:
        print("  発散しました")
        return
    F, P = r
    tilt = np.degrees(np.arctan2(F[0], max(F[2], 1e-9)))
    print(f"  240 Hz, ストローク振幅 1.25 rad, 迎角 50 deg, 機首上げ 50 deg")
    print(f"    揚力   {F[2]:+.3f} 体重ぶん   (目標 約 1.0)")
    print(f"    前後力 {F[0]:+.3f} 体重ぶん   体軸zからの傾き {tilt:+.0f} deg")
    print(f"    パワー {P:.0f} erg/s          (実物 300-800)")

    if args.sweep:
        print("\n=== 迎角と振幅を振る ===")
        print(f"{'迎角[deg]':>9} {'振幅[rad]':>9} {'揚力':>8} {'傾き':>7} {'パワー':>10}")
        for feather in (0.70, 0.79, 0.87):
            for ra in (1.0, 1.25):
                out = tethered(m, aero, freq=240.0, roll_amp=ra, feather=feather)
                if out is None:
                    print(f"{np.degrees(feather):9.0f} {ra:9.2f}   発散")
                    continue
                F, P = out
                tilt = np.degrees(np.arctan2(F[0], max(F[2], 1e-9)))
                print(f"{np.degrees(feather):9.0f} {ra:9.2f} {F[2]:+8.3f} "
                      f"{tilt:+6.0f}d {P:9.0f}")


if __name__ == "__main__":
    main()
