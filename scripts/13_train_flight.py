"""飛び方を学習させる — 報酬の的を少しずつ前に置いていくカリキュラム。

手でPDゲインを合わせるかわりに、CMA-ES で制御器のパラメータを探索する。
肝は報酬の出し方で、いきなり「前に飛べ」と言っても宙返りして終わる。

失敗から学んだ2点 (最初の版はどちらも踏んだ):

  1. 目標高度からのずれを報酬に入れないと、探索は「姿勢は完璧だが
     揚力を捨てて真っ直ぐ落ちる」解に落ちる。実際そうなった
     (係留揚力 1.08 → 0.44 体重ぶん に低下した)。
  2. 羽ばたきと制御ゲインを同時に動かすと、揚力を生む羽ばたきが壊れる。
     まず羽ばたきを固定して制御ゲインだけ学ばせ、その後で解放する。

ステージ:
  A 制御ゲインだけ学ぶ (羽ばたきは固定) — その場でホバリング
  B 羽ばたきも解放して微調整        — その場でホバリング
  C 的を 3 cm 前へ
  D 的を 8 cm 前へ、時間も延ばす

報酬は毎ステップ  生存 - 傾き - 回転 - 目標高度からのずれ - 的までの水平距離。
傾きが80度を超えるか高度0.5cmを切ったら打ち切り。

32コアで並列評価する。

使い方:
  python scripts/13_train_flight.py
  python scripts/13_train_flight.py --gens 20
  python scripts/13_train_flight.py --resume

出力: out/flight_policy.json
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
from flight_env import NAMES, decode, encode, evaluate, rollout, splice  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
POLICY = OUT / "flight_policy.json"

GAIT_IDX = list(range(0, 10))    # freq..yaw_phase
GAIN_IDX = list(range(10, 18))   # kp_pitch..kd_alt

# 出発点:
#   展開角(yaw)も周期内で振ってストローク面を水平にした羽ばたき。
#   係留測定で、正味空力が体軸の真上を向き (傾き +1 deg)、
#   体重の 1.0 倍前後が出ることを確認した値。
#   ゲインは制御感度 S=14959 rad/s^2/rad から ω=80 rad/s, ζ=0.7 で算出
SEED = dict(freq=240.0, yaw=0.0, roll_mean=-0.10, roll_amp=1.10,
            pitch_down=2.900, pitch_up=1.627, sharp=7.894, phase=-2.969,
            # 展開角も振ってストローク面を水平にする (係留測定で決めた)
            yaw_amp=1.0, yaw_phase=0.0,
            kp_pitch=0.428, kd_pitch=0.0075, kp_roll=0.428, kd_roll=0.0075,
            kp_yaw=0.05, kd_yaw=0.002, kp_alt=0.05, kd_alt=0.01)

Z0 = 10.0
STAGES = [
    ("A 制御ゲインだけ学ぶ",  GAIN_IDX,             (0.0, 0.0, Z0), 0.5, 0.25),
    ("B 羽ばたきも解放",      GAIT_IDX + GAIN_IDX,  (0.0, 0.0, Z0), 0.5, 0.10),
    ("C 的を 3 cm 前へ",      GAIT_IDX + GAIN_IDX,  (3.0, 0.0, Z0), 0.6, 0.10),
    ("D 的を 8 cm 前へ",      GAIT_IDX + GAIN_IDX,  (8.0, 0.0, Z0), 0.7, 0.10),
]


def report(x, target, seconds):
    r = rollout(np.asarray(x), target=target, seconds=seconds, z0=Z0)
    return (f"生存 {r['alive_s']:.2f}/{seconds:.2f}s  高度 {r['z']:5.2f}cm  "
            f"傾き {r['tilt_deg']:3.0f}deg  回転 {r['spin_dps']:5.0f}deg/s  "
            f"的まで {r['dist']:.2f}cm  報酬 {r['reward']:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=30)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stages", type=int, default=len(STAGES))
    args = ap.parse_args()

    import cma

    base = np.clip(encode(SEED), 0.01, 0.99)
    if args.resume and POLICY.exists():
        base = np.array(json.loads(POLICY.read_text(encoding="utf-8"))["x"])
        print("保存済みの解から再開します")

    OUT.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()

    with ctx.Pool(args.workers) as pool:
        for si, (label, idx, target, seconds, sigma) in enumerate(STAGES[:args.stages]):
            print(f"\n=== stage {si}: {label}   的={target} 時間={seconds}s "
                  f"探索={len(idx)}個 ===")
            print(f"  開始: {report(base, target, seconds)}")
            es = cma.CMAEvolutionStrategy(
                list(base[idx]), sigma,
                {"popsize": args.popsize, "bounds": [0, 1], "verbose": -9,
                 "maxiter": args.gens})
            best_r, best_sub = -1e9, base[idx].copy()
            for gen in range(args.gens):
                sols = es.ask()
                jobs = [(s, base, idx, target, seconds, Z0) for s in sols]
                rewards = pool.map(evaluate, jobs)
                es.tell(sols, [-r for r in rewards])
                i = int(np.argmax(rewards))
                if rewards[i] > best_r:
                    best_r, best_sub = rewards[i], np.array(sols[i])
                if gen % 5 == 0 or gen == args.gens - 1:
                    print(f"  gen {gen:3d}  最良 {best_r:+.3f}  今世代 {max(rewards):+.3f}"
                          f"   [{(time.time()-t0)/60:.1f} 分]")
            base = splice(base, best_sub, idx)
            print(f"  結果: {report(base, target, seconds)}")
            POLICY.write_text(json.dumps(
                {"x": [float(v) for v in base], "params": decode(base),
                 "stage": si, "label": label, "target": list(target),
                 "seconds": seconds, "z0": Z0, "reward": float(best_r)},
                ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完了 ({(time.time()-t0)/60:.1f} 分)  → {POLICY}")
    p = decode(base)
    for n in NAMES:
        print(f"  {n:11} = {p[n]:.4f}")


if __name__ == "__main__":
    main()
