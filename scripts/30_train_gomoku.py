"""五目並べを覚えさせる — 報酬は4つだけ、打ち方は教えない。

`13_train_flight.py` / `26_train_escape.py` と同じ CMA-ES + カリキュラム。
違うのは物理を回さないこと (盤の上の話なので MuJoCo は要らない。1局 25 ms)。

ステージ (`CLAUDE.md`: 羽ばたきと制御ゲインを同時に動かさない、と同じ理由で
「打てる」「並べる」「止める」「競る」を一度に求めない):

  0 打てるようになる  相手ランダム。報酬は**合法だった割合だけ**。
                      空きマスに限定していないので、石のある所を選べば空振り。
                      お手つきが 100 手に1回を切ったらそこで切り上げる
                      (飽和したまま回すと残りの次元が暴れるだけ)
  1 並べる            相手ランダム。合法 + 最長連 + 勝ち
  2 止める            相手は連を数える素朴な打ち手。相手の最長連を伸ばさせない
                      加点を足す (**罰では引かない**)
  3 自分と打つ        相手は stage 2 の自分。何世代かごとに相手を今の自分へ
                      入れ替える

**探索範囲の端に張り付いた重みを数えて出す。** 逃避の学習で「端に来た値は答え
ではなく箱の壁」を踏んだので (`CLAUDE.md`)、毎ステージ何個が端にいるか出す。
重みは ±3 に収めてあり、tanh の飽和で効くので端に居ること自体は異常ではないが、
**大半が端に行ったら報酬に足りないものがある** と読む。

**最後のステージを無条件に採用しない。** 自己対戦は自分のコピーに勝つのが目的で、
素朴な相手への強さはむしろ落ちることがある (この実装で 11.7% → 0% になった)。
終わりに全ステージの解を同じ物差しで測り直し、いちばん強いものを
`out/gomoku_policy.json` に書く (`pick_best`)。

使い方:
  python scripts/30_train_gomoku.py                  # 全ステージ
  python scripts/30_train_gomoku.py --stages 2       # stage 0,1 だけ
  python scripts/30_train_gomoku.py --resume
  python scripts/30_train_gomoku.py --report         # 保存済みの解を測るだけ
  python scripts/30_train_gomoku.py --pick           # ステージから選び直すだけ

出力: out/gomoku_policy.json  (ステージごとの控えも gomoku_policy_stageN.json)
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
from gomoku_env import FlyPlayer, STAGES, evaluate, measure  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
POLICY = OUT / "gomoku_policy.json"

# ステージごとの (世代数, 1評価あたりの対局数, CMA-ES の初期 sigma)
PLAN = [(60, 12, 0.30), (150, 16, 0.25), (200, 16, 0.20), (150, 16, 0.15)]

# 最良が何世代伸びなかったら次のステージへ移るか。**報酬が飽和した先は
# 探索ではなく徘徊**になるので、上限の世代数まで回しきらない
STALL = 60


def report(z, K: int, ref_z=None, games: int = 60) -> str:
    """人が読む形で測る。**相手を変えて2通り**測らないと強さが分からない。"""
    vs_rand = measure(z, 1, games=games, ref_z=None, K=K)
    vs_greedy = measure(z, 2, games=games, ref_z=None, K=K)
    edge = int(((z < 0.02) | (z > 0.98)).sum())
    lines = [
        f"  合法手の割合   {vs_rand['legal']*100:5.1f} %   "
        f"(石のあるマスを選んだら空振り)",
        f"  対ランダム     勝ち {vs_rand['win']*100:5.1f} %  "
        f"負け {vs_rand['loss']*100:5.1f} %   自分の最長連 {vs_rand['run']:.2f}",
        f"  対 素朴な相手  勝ち {vs_greedy['win']*100:5.1f} %  "
        f"負け {vs_greedy['loss']*100:5.1f} %   "
        f"相手の最長連 {vs_greedy['opp_run']:.2f}",
        f"  範囲の端にいる重み {edge} / {len(z)}",
    ]
    if ref_z is not None:
        vs_self = measure(z, 3, games=games, ref_z=ref_z, K=K)
        lines.insert(3, f"  対 前の自分    勝ち {vs_self['win']*100:5.1f} %  "
                        f"負け {vs_self['loss']*100:5.1f} %")
    return "\n".join(lines)


def pick_best(stages: int, K: int, games: int = 120) -> None:
    """**最後のステージを無条件に採用しない。** 同じ物差しで全部測って選ぶ。

    自己対戦 (stage 3) は自分のコピーに勝つのが目的なので、素朴な相手への強さは
    むしろ落ちることがある (実際に 11.7% → 0% になった)。逃避で「ステージごとの
    解も残す」を学んだのと同じ話で、残すだけでなく**測って選ぶ**ところまでやる。

    物差しは 素朴な相手への勝率 + 0.1 x ランダムへの勝率。合法手が 99% 未満の
    解は外す (お手つきする打ち手は相手にならない)。
    """
    print("\n=== どのステージの解を使うか (同じ物差しで測って選ぶ) ===")
    best, best_score, rows = None, -1.0, []
    for si in range(stages):
        path = OUT / f"gomoku_policy_stage{si}.json"
        if not path.exists():
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        z = np.array(saved["x"])
        r = measure(z, 1, games=games, K=K)
        g = measure(z, 2, games=games, K=K)
        score = g["win"] + 0.1 * r["win"] if r["legal"] >= 0.99 else -1.0
        rows.append((si, saved["label"], r["legal"], r["win"], g["win"], score))
        # 測った値は解と一緒に残す。アプリの画面にはこれをそのまま出す
        # (強さを文章で書くと、打ち手を学習し直したときに嘘になる)
        saved["scoreboard"] = {"games": games, "legal": r["legal"],
                               "win_random": r["win"], "win_greedy": g["win"]}
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        if score > best_score:
            best, best_score = saved, score
    for si, label, legal, wr, wg, score in rows:
        mark = "  <= これを使う" if best and best["stage"] == si else ""
        print(f"  stage {label:16} 合法 {legal*100:5.1f} %  "
              f"対ランダム {wr*100:5.1f} %  対 素朴 {wg*100:5.1f} %{mark}")
    if best is not None:
        POLICY.write_text(json.dumps(best, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"→ {POLICY} = stage {best['label']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=6, help="受容野の隠れ素子の数")
    ap.add_argument("--popsize", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0, help="0 なら CPU 数から決める")
    ap.add_argument("--stages", type=int, default=len(STAGES))
    ap.add_argument("--gens", type=int, default=0, help="全ステージの世代数を上書き")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--pick", action="store_true",
                    help="学習せず、保存済みのステージから選び直すだけ")
    args = ap.parse_args()

    K = args.K
    dim = FlyPlayer(K).dim

    if args.pick:
        pick_best(args.stages, K)
        return

    if args.report:
        saved = json.loads(POLICY.read_text(encoding="utf-8"))
        z = np.array(saved["x"])
        print(f"保存済みの解 ({saved['label']}, {len(z)} 個の重み):")
        print(report(z, saved.get("K", K)))
        return

    import cma

    rng = np.random.default_rng(args.seed)
    base = np.clip(0.5 + 0.05 * rng.standard_normal(dim), 0.01, 0.99)
    if args.resume and POLICY.exists():
        saved = json.loads(POLICY.read_text(encoding="utf-8"))
        base, K = np.array(saved["x"]), saved.get("K", K)
        print(f"保存済みの解から再開します ({saved['label']})")

    workers = args.workers or min(mp.cpu_count(), args.popsize)
    OUT.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    ref_z = None        # 自己対戦の相手 (stage 3 で使う)

    with ctx.Pool(workers) as pool:
        print(f"重み {dim} 個 / 個体数 {args.popsize} / 並列 {workers}")
        print("\n=== 出発点 (まだ何も覚えていない) ===")
        print(report(base, K))
        for si in range(args.stages):
            gens, games, sigma = PLAN[si]
            if args.gens:
                gens = args.gens
            print(f"\n=== stage {STAGES[si]}   {gens} 世代 x {games} 局 ===")
            if si == 3:
                ref_z = base.copy()          # 相手は「いまの自分」から始める
            es = cma.CMAEvolutionStrategy(
                list(base), sigma,
                {"popsize": args.popsize, "bounds": [0, 1], "verbose": -9,
                 "maxiter": gens, "seed": args.seed + si})
            best_r, best_z, stale = -1e9, base.copy(), 0
            for gen in range(gens):
                sols = es.ask()
                # **同じ世代は同じ対局で比べる**。世代ごとには変える
                # (固定すると、その棋譜にだけ強い解が残る)
                seed = args.seed * 1000 + gen * 97
                jobs = [(s, si, seed, games, ref_z, K) for s in sols]
                rewards = pool.map(evaluate, jobs)
                es.tell(sols, [-r for r in rewards])
                i = int(np.argmax(rewards))
                if rewards[i] > best_r:
                    best_r, best_z, stale = rewards[i], np.array(sols[i]), 0
                else:
                    stale += 1
                if stale >= STALL:
                    # **報酬が伸びなくなったら切り上げる**。stage 1 は相手が
                    # ランダムだと 30 世代ほどで 1.000 に張り付き、そこから先は
                    # 何をしても同点なので残りの次元が暴れるだけだった
                    print(f"  gen {gen:3d}  {STALL} 世代伸びない。次へ")
                    break
                if si == 3 and gen and gen % 40 == 0:
                    # 相手を今の自分へ入れ替える。据え置くと、その一人にだけ
                    # 勝つ手を覚えて止まる
                    ref_z = best_z.copy()
                    best_r, stale = -1e9, 0
                    print(f"  gen {gen:3d}  相手を今の自分に入れ替え")
                if gen % 10 == 0 or gen == gens - 1:
                    print(f"  gen {gen:3d}  最良 {best_r:+.3f}  "
                          f"今世代 {max(rewards):+.3f}"
                          f"   [{(time.time()-t0)/60:.1f} 分]")
                # stage 0 は「打てるようになったら」そこで切り上げる。
                # 合法手だけを報酬にしたまま回し続けても、残りの次元は
                # 報酬が飽和した先で暴れるだけ (`CLAUDE.md`)
                # しきい値は**学習に使っていない 20 局**で測る。学習側の報酬は
                # 12 局で 1.000 に届いてもそこは覚えた棋譜で、未知の局面では
                # 99% 前後にとどまる (実測)。ここを 0.999 にすると永久に
                # 抜けられず、飽和したまま回り続ける
                if si == 0 and gen % 10 == 9:
                    if measure(best_z, 0, games=20, K=K)["legal"] >= 0.99:
                        print(f"  gen {gen:3d}  お手つきが 100 手に1回を切った。次へ")
                        break
            base = best_z
            print(report(base, K, ref_z))
            body = json.dumps(
                {"x": [float(v) for v in base], "K": K, "stage": si,
                 "label": STAGES[si], "reward": float(best_r),
                 "dim": dim, "popsize": args.popsize, "gens": gens,
                 "games": games, "seed": args.seed},
                ensure_ascii=False, indent=2)
            POLICY.write_text(body, encoding="utf-8")
            # ステージごとの控えも残す (逃避のとき上書きで良い解を取り逸した)
            (OUT / f"gomoku_policy_stage{si}.json").write_text(
                body, encoding="utf-8")

    pick_best(args.stages, K)
    print(f"\n完了 ({(time.time()-t0)/60:.1f} 分)  → {POLICY}")


if __name__ == "__main__":
    main()
