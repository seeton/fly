"""学習の出発点を探す — 18パラメータをランダムに振って、飛べる個体を拾う。

展開角も振る新しい制御構造にしたので、手計算のトリム点は当てにならない
(ストローク指令が関節の可動域でクリップされる等、相互作用が強い)。
まず「とにかく落ちない個体」を総当りで見つけ、それを 13_train_flight.py の
出発点にする。

使い方:
  python scripts/15_bootstrap_flight.py            # 600個ためす
  python scripts/15_bootstrap_flight.py --n 2000

出力: out/flight_seed.json
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
from flight_env import NAMES, decode, evaluate, rollout  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
SEED_FILE = OUT / "flight_seed.json"
Z0 = 10.0
TARGET = (0.0, 0.0, Z0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cma", type=int, default=0,
                    help="ランダム探索のあと、上位から CMA-ES を何世代か回す")
    ap.add_argument("--restarts", type=int, default=3, help="CMA-ES の多点開始")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_par = len(NAMES)
    idx = list(range(n_par))
    base = np.full(n_par, 0.5)

    # 羽ばたきは広く、制御ゲインは実測から見て小さめを厚くサンプルする
    samples = rng.uniform(0.0, 1.0, size=(args.n, n_par))
    samples[:, 10:] *= rng.uniform(0.05, 1.0, size=(args.n, 1))

    ctx = mp.get_context("spawn")
    t0 = time.time()
    print(f"{args.n} 個体を {args.workers} 並列で評価します ...")
    with ctx.Pool(args.workers) as pool:
        jobs = [(s, base, idx, TARGET, args.seconds, Z0) for s in samples]
        rewards = pool.map(evaluate, jobs)

    order = np.argsort(rewards)[::-1]
    print(f"\n上位10個体 ({(time.time()-t0)/60:.1f} 分):")
    print(f"{'報酬':>8} {'生存':>8} {'高度':>8} {'傾き':>7} {'体軸角':>8}")
    for i in order[:10]:
        r = rollout(samples[i], target=TARGET, seconds=args.seconds, z0=Z0)
        print(f"{r['reward']:+8.3f} {r['alive_s']:7.3f}s {r['z']:7.2f}cm "
              f"{r['tilt_deg']:6.0f}d {r['nose_deg']:+7.0f}d")

    best = samples[order[0]]

    if args.cma:
        import cma
        print(f"\n上位 {args.restarts} 点から CMA-ES を {args.cma} 世代ずつ回します ...")
        with ctx.Pool(args.workers) as pool:
            for k in range(args.restarts):
                x0 = samples[order[k]]
                es = cma.CMAEvolutionStrategy(
                    list(x0), 0.30,
                    {"popsize": 30, "bounds": [0, 1], "verbose": -9, "maxiter": args.cma})
                br, bx = -1e9, x0
                for gen in range(args.cma):
                    sols = es.ask()
                    rew = pool.map(evaluate, [(s_, base, idx, TARGET, args.seconds, Z0)
                                              for s_ in sols])
                    es.tell(sols, [-v for v in rew])
                    i = int(np.argmax(rew))
                    if rew[i] > br:
                        br, bx = rew[i], np.array(sols[i])
                    if gen % 10 == 0 or gen == args.cma - 1:
                        rr = rollout(bx, target=TARGET, seconds=args.seconds, z0=Z0)
                        print(f"  restart {k} gen {gen:3d}  報酬 {br:+.3f}  "
                              f"生存 {rr['alive_s']:.3f}s  高度 {rr['z']:.2f}cm  "
                              f"体軸角 {rr['nose_deg']:+.0f}deg  [{(time.time()-t0)/60:.1f} 分]")
                rr = rollout(bx, target=TARGET, seconds=args.seconds, z0=Z0)
                if rr["reward"] > rollout(best, target=TARGET, seconds=args.seconds,
                                          z0=Z0)["reward"]:
                    best = bx

    r = rollout(best, target=TARGET, seconds=args.seconds, z0=Z0)
    OUT.mkdir(exist_ok=True)
    SEED_FILE.write_text(json.dumps(
        {"x": [float(v) for v in best], "params": decode(best),
         "reward": r["reward"], "alive_s": r["alive_s"],
         "nose_deg": r["nose_deg"], "tilt_deg": r["tilt_deg"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {SEED_FILE}")
    p = decode(best)
    for n in NAMES[:10]:
        print(f"  {n:11} = {p[n]:.4f}")


if __name__ == "__main__":
    main()
