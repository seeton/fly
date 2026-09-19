"""検証済みの羽ばたきを固定して、制御ゲインだけ先に探す。

空力を翼素理論に入れ替えたので、以前のゲインは全く合わない。
16_validate_aero.py で「揚力0.951体重・機首上げ50度で釣り合う」ことを
確認した羽ばたきをそのまま使い、姿勢を保てるゲイン8個をランダム探索する。
ここで生き残る個体を見つけてから 13_train_flight.py に渡す。

使い方:
  python scripts/17_seed_gains.py --n 900

出力: out/flight_policy.json  (13_train_flight.py --resume がこれを読む)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flight_env import NAMES, decode, encode, evaluate, rollout  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
POLICY = OUT / "flight_policy.json"
GAIN_IDX = list(range(10, 18))
Z0 = 10.0
TARGET = (0.0, 0.0, Z0)

# 16_validate_aero.py で検証した羽ばたき
# 240 Hz, ストローク振幅 1.25 rad, 迎角 ±50 deg, 機首上げ50度で Fx≒0, 揚力0.951
GAIT = dict(freq=240.0, yaw=0.0, roll_mean=0.25, roll_amp=1.25,
            pitch_down=0.87, pitch_up=-0.87, sharp=3.0, phase=0.0,
            yaw_amp=0.0, yaw_phase=0.0,
            kp_pitch=0.0, kd_pitch=0.0, kp_roll=0.0, kd_roll=0.0,
            kp_yaw=0.0, kd_yaw=0.0, kp_alt=0.0, kd_alt=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=0.4)
    ap.add_argument("--cma", type=int, default=30)
    args = ap.parse_args()

    base = np.clip(encode(GAIT), 0.0, 1.0)
    r0 = rollout(base, target=TARGET, seconds=args.seconds, z0=Z0)
    print(f"ゲインなし: 生存 {r0['alive_s']:.3f}s  体軸角 {r0['nose_deg']:+.0f}deg  "
          f"報酬 {r0['reward']:+.3f}")

    rng = np.random.default_rng(0)
    # ゲインは小さめを厚くサンプルする
    samples = rng.uniform(0.0, 1.0, size=(args.n, 8)) ** 2

    ctx = mp.get_context("spawn")
    t0 = time.time()
    print(f"\nゲイン8個を {args.n} 通り、{args.workers} 並列で試します ...")
    with ctx.Pool(args.workers) as pool:
        jobs = [(s, base, GAIN_IDX, TARGET, args.seconds, Z0) for s in samples]
        rew = pool.map(evaluate, jobs)
        order = np.argsort(rew)[::-1]
        print(f"\n上位5件 ({(time.time()-t0)/60:.1f} 分):")
        for i in order[:5]:
            x = base.copy()
            x[GAIN_IDX] = samples[i]
            r = rollout(x, target=TARGET, seconds=args.seconds, z0=Z0)
            print(f"  報酬 {r['reward']:+.3f}  生存 {r['alive_s']:.3f}s  "
                  f"高度 {r['z']:5.2f}cm  体軸角 {r['nose_deg']:+.0f}deg")

        best_sub = samples[order[0]]
        if args.cma:
            import cma
            print(f"\n上位から CMA-ES を {args.cma} 世代 ...")
            es = cma.CMAEvolutionStrategy(
                list(best_sub), 0.25,
                {"popsize": 30, "bounds": [0, 1], "verbose": -9, "maxiter": args.cma})
            br = max(rew)
            for gen in range(args.cma):
                sols = es.ask()
                rr = pool.map(evaluate, [(s, base, GAIN_IDX, TARGET, args.seconds, Z0)
                                         for s in sols])
                es.tell(sols, [-v for v in rr])
                i = int(np.argmax(rr))
                if rr[i] > br:
                    br, best_sub = rr[i], np.array(sols[i])
                if gen % 10 == 0 or gen == args.cma - 1:
                    x = base.copy()
                    x[GAIN_IDX] = best_sub
                    r = rollout(x, target=TARGET, seconds=args.seconds, z0=Z0)
                    print(f"  gen {gen:3d}  報酬 {br:+.3f}  生存 {r['alive_s']:.3f}s  "
                          f"高度 {r['z']:5.2f}cm  体軸角 {r['nose_deg']:+.0f}deg  "
                          f"[{(time.time()-t0)/60:.1f} 分]")

    x = base.copy()
    x[GAIN_IDX] = best_sub
    r = rollout(x, target=TARGET, seconds=args.seconds, z0=Z0)
    OUT.mkdir(exist_ok=True)
    POLICY.write_text(json.dumps(
        {"x": [float(v) for v in x], "params": decode(x), "stage": -1,
         "label": "翼素理論の空力でゲインを探した種", "target": list(TARGET),
         "seconds": args.seconds, "z0": Z0, "reward": r["reward"],
         "alive_s": r["alive_s"], "nose_deg": r["nose_deg"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {POLICY}")
    print(f"  生存 {r['alive_s']:.3f}s  高度 {r['z']:.2f}cm  体軸角 {r['nose_deg']:+.0f}deg")
    p = decode(x)
    for n in NAMES[10:]:
        print(f"  {n:11} = {p[n]:.5f}")


if __name__ == "__main__":
    main()
