"""五目並べを覚えさせる — 報酬は4つだけ、打ち方は教えない。

`13_train_flight.py` / `26_train_escape.py` と同じ CMA-ES + カリキュラム。
違うのは物理を回さないこと (盤の上の話なので MuJoCo は要らない)。

盤は 15x15 の自由形。9x9 は先手必勝ではなかったので移した (`gomoku_env.N`)。

段 (`gomoku_env.STAGES`。羽ばたきと制御ゲインを同時に動かさないのと同じ理由で
一度に求めない):

  0 打てるようになる   相手ランダム。報酬は**合法だった割合だけ**。石のある所を
                       選べばその場で負け。お手つきが 100 手に1回を切ったら切り上げる
  1 並べる             相手ランダム。合法 + 最長連 + 決めた + 勝ち
  2-3 止める           相手は連を数える素朴な打ち手 (手書きの知識は相手側だけ)。
                       ただし手の5割 → 2割をランダムにして弱めてある。蝿は読み1
  4 読みながら止める   弱めていない素朴な相手に、**蝿自身が2手先まで読んで**打つ
  5-7 Rapfi 深さ1〜3   外のエンジン Rapfi の読みを浅く制限した梯子を登る

**読みは蝿の中にある** (`FlyPlayer.move`)。読みの途中の局面を評価するのも、
読む候補を選ぶのも蝿自身の点数で、「四は止めろ」のような打ち方は書いていない。
**学習も打つときも同じ深さで同じ関数を通す** — 深さは解の json に入れ、
アプリもそれを読む。

報酬は 2 段目から「合法 / 決めた / 止めた / 生き延びた手数 / 勝ち」の有界な正の
項の和。**決めた・止めた は1手ごとの割合**で、審判の側が数える (蝿の打ち方には
入らない)。勝ち負けだけにしていたら、15x15 の素朴な相手に毎局 11〜14 手で負けて
全員同点になり、2〜3段目が 0% のまま止まった。生き延びた手数は**勝ったら満点**で
数える (負けるまでの手数だけを数えると、早く勝つほど損になる)。

**入ってきた解を CMA-ES に混ぜる** (`es.inject(..., force=True)`)。混ぜずに
-1e9 から数えると、勝てない段では負けた候補が解を上書きする (疑似農業で踏んだ)。

**最後の段を無条件に採用しない。** 9x9 で自己対戦の段を通したら素朴な相手への
勝率が 12.5% → 0.8% に落ちた。終わりに全段の解を同じ物差し (素朴な相手と
Rapfi 深さ1・2 に対する成績) で測り直し、いちばん強いものを
`out/gomoku_policy.json` に書く (`pick_best`)。測った値も json に入れ、アプリの
画面にはそれをそのまま出す。

**探索範囲の端に張り付いた重みを数えて出す。** 大半が端に行ったら、答えではなく
箱の壁 — 報酬に足りないものがある (`CLAUDE.md`)。

使い方:
  python scripts/30_train_gomoku.py                         # 全段
  python scripts/30_train_gomoku.py --init out/gomoku_policy_9x9.json
  python scripts/30_train_gomoku.py --start-stage 4         # 保存済みの解から4段目以降
  python scripts/30_train_gomoku.py --report                # 保存済みの解を測るだけ
  python scripts/30_train_gomoku.py --pick                  # 段の解から選び直すだけ

出力: out/gomoku_policy.json  (段ごとの控えも gomoku_policy_stageN.json)
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
from gomoku_env import FlyPlayer, N, STAGES, evaluate, measure  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
POLICY = OUT / "gomoku_policy.json"

# 段ごとの (世代数の上限, 1評価あたりの対局数, CMA-ES の初期 sigma)。
# 読みの入る段は1局が重い (1手 24 ms / 読み2 で 58 ms) ので局数を減らす
PLAN = [(60, 12, 0.30), (150, 12, 0.25), (200, 12, 0.20), (200, 12, 0.15),
        (120, 8, 0.12), (120, 8, 0.12), (120, 8, 0.10), (120, 8, 0.10)]

# 最良が何世代伸びなかったら次の段へ移るか。**報酬が飽和した先は探索ではなく
# 徘徊**になるので、上限の世代数まで回しきらない
STALL = 50

# 解を選ぶ物差し。自分で書いた相手 (素朴な相手) だけでなく外のエンジンを混ぜる
YARDSTICK = ("greedy", ("rapfi", 1), ("rapfi", 2))


def label(spec) -> str:
    if spec == "random":
        return "ランダム"
    if spec == "greedy" or spec == ("greedy", 0.0):
        return "素朴な相手"
    if isinstance(spec, tuple) and spec[0] == "greedy":
        return f"素朴な相手 ({spec[1]*10:.0f}割気まぐれ)"
    if isinstance(spec, tuple) and spec[0] == "rapfi":
        return f"Rapfi 深さ{spec[1]}"
    return str(spec)


def points(m: dict) -> float:
    """勝ち 1、引き分け 0.5 の成績。"""
    return m["win"] + 0.5 * m["draw"]


def report(z, K: int, depth: int, games: int = 20) -> str:
    """人が読む形で測る。**相手を変えて並べる**と強さの位置が分かる。"""
    lines = []
    r = measure(z, "random", games=games, K=K, depth=depth)
    lines.append(f"  合法手の割合   {r['legal']*100:5.1f} %   (読み {depth})"
                 f"   対ランダム 勝ち {r['win']*100:5.1f} %")
    for spec in YARDSTICK:
        m = measure(z, spec, games=games, K=K, depth=depth)
        lines.append(f"  対 {label(spec):10} 勝ち {m['win']*100:5.1f} %  "
                     f"引き分け {m['draw']*100:5.1f} %  負け {m['loss']*100:5.1f} %"
                     f"   平均 {m['plies']:.0f} 手   決めた {m['finish']*100:5.1f} %"
                     f"  止めた {m['block']*100:5.1f} %")
    edge = int(((z < 0.02) | (z > 0.98)).sum())
    lines.append(f"  範囲の端にいる重み {edge} / {len(z)}")
    return "\n".join(lines)


def pick_best(K: int, games: int = 20) -> None:
    """**最後の段を無条件に採用しない。** 同じ物差しで全部測って選ぶ。

    物差しは素朴な相手・Rapfi 深さ1・深さ2 に対する成績 (勝ち 1、引き分け 0.5)
    の平均。合法手が 99% 未満の解は外す (お手つきする打ち手は相手にならない)。
    """
    print("\n=== どの段の解を使うか (同じ物差しで測って選ぶ) ===", flush=True)
    best, best_score = None, -1.0
    for si in range(len(STAGES)):
        path = OUT / f"gomoku_policy_stage{si}.json"
        if not path.exists():
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        z, depth = np.array(saved["x"]), saved.get("depth", 0)
        legal = measure(z, "random", games=games, K=K, depth=depth)["legal"]
        board = {"games": games, "legal": legal, "depth": depth}
        for spec in YARDSTICK:
            m = measure(z, spec, games=games, K=K, depth=depth)
            board[label(spec)] = {"win": m["win"], "draw": m["draw"], "loss": m["loss"]}
        score = (np.mean([points(board[label(s)]) for s in YARDSTICK])
                 if legal >= 0.99 else -1.0)
        board["score"] = float(score)
        # 測った値は解と一緒に残す。アプリの画面にはこれをそのまま出す
        saved["scoreboard"] = board
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        cols = "  ".join(f"{label(s)} {points(board[label(s)])*100:5.1f}%"
                         for s in YARDSTICK)
        print(f"  {saved['label']:16} 合法 {legal*100:5.1f}%  {cols}", flush=True)
        if score > best_score:
            best, best_score = saved, score
    if best is not None:
        POLICY.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {POLICY} = {best['label']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=6, help="受容野の隠れ素子の数")
    ap.add_argument("--popsize", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0, help="0 なら CPU 数から決める")
    ap.add_argument("--start-stage", type=int, default=0)
    ap.add_argument("--stages", type=int, default=len(STAGES), help="何段目まで回すか")
    ap.add_argument("--gens", type=int, default=0, help="全段の世代数を上書き")
    ap.add_argument("--init", default=None, help="出発点にする解の json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--pick", action="store_true",
                    help="学習せず、保存済みの段から選び直すだけ")
    args = ap.parse_args()

    K = args.K
    dim = FlyPlayer(K).dim

    if args.pick:
        pick_best(K)
        return
    if args.report:
        saved = json.loads(POLICY.read_text(encoding="utf-8"))
        print(f"保存済みの解 ({saved['label']}, {len(saved['x'])} 個の重み, "
              f"{saved.get('board', '?')} 盤):")
        print(report(np.array(saved["x"]), saved.get("K", K), saved.get("depth", 0)))
        return

    import cma

    rng = np.random.default_rng(args.seed)
    base = np.clip(0.5 + 0.05 * rng.standard_normal(dim), 0.01, 0.99)
    init = args.init or (str(POLICY) if args.start_stage > 0 else None)
    if init:
        saved = json.loads(Path(init).read_text(encoding="utf-8"))
        base, K = np.array(saved["x"]), saved.get("K", K)
        print(f"出発点: {init} ({saved.get('label', '?')})")

    workers = args.workers or min(mp.cpu_count(), args.popsize)
    OUT.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()

    def save(si: int, z, r: float, partial: bool = False) -> str:
        name, spec, depth = STAGES[si]
        body = json.dumps(
            {"x": [float(v) for v in z], "K": K, "depth": depth, "board": N,
             "stage": si, "label": name, "opponent": label(spec),
             "reward": float(r), "dim": dim, "popsize": args.popsize,
             "games": PLAN[si][1], "seed": args.seed, "partial": partial},
            ensure_ascii=False, indent=2)
        (OUT / f"gomoku_policy_stage{si}.json").write_text(body, encoding="utf-8")
        return body

    with ctx.Pool(workers) as pool:
        print(f"盤 {N}x{N} / 重み {dim} 個 / 個体数 {args.popsize} / 並列 {workers}",
              flush=True)
        for si in range(args.start_stage, min(args.stages, len(STAGES))):
            name, spec, depth = STAGES[si]
            gens, games, sigma = PLAN[si]
            if args.gens:
                gens = args.gens
            print(f"\n=== {name}   相手 {label(spec)} / 読み {depth} / "
                  f"{gens} 世代 x {games} 局 ===", flush=True)
            es = cma.CMAEvolutionStrategy(
                list(base), sigma,
                {"popsize": args.popsize, "bounds": [0, 1], "verbose": -9,
                 "maxiter": gens, "seed": args.seed + si})
            # 入ってきた解を最初の世代に混ぜる。**同じ対局で比べた上で**しか
            # 上書きされない (混ぜずに -1e9 から数えると、勝てない段では
            # 負けた候補が解を上書きする)
            es.inject([list(base)], force=True)
            best_r, best_z, stale = -1e9, base.copy(), 0
            for gen in range(gens):
                sols = es.ask()
                # **同じ世代は同じ対局で比べる**。世代ごとには変える
                seed = args.seed * 1000 + si * 100000 + gen * 97
                rewards = pool.map(evaluate, [(s, si, seed, games, K) for s in sols])
                es.tell(sols, [-r for r in rewards])
                i = int(np.argmax(rewards))
                if rewards[i] > best_r:
                    best_r, best_z, stale = rewards[i], np.array(sols[i]), 0
                else:
                    stale += 1
                if gen % 10 == 0 or gen == gens - 1:
                    print(f"  gen {gen:3d}  最良 {best_r:+.3f}  今世代 {max(rewards):+.3f}"
                          f"   [{(time.time()-t0)/60:.1f} 分]", flush=True)
                    # 段の途中でも控えを書く。**外から落とされても途中から再開できる**
                    # ように (別の作業が Python をまとめて落とし、段の終わりまで
                    # 書かない作りでは 20 分ぶんを失った)。再開は
                    # --start-stage N --init out/gomoku_policy_stageN.json
                    save(si, best_z, best_r, partial=True)
                if stale >= STALL:
                    print(f"  gen {gen:3d}  {STALL} 世代伸びない。次へ", flush=True)
                    break
                # 0 段目はお手つきが消えたら切り上げる (学習に使っていない 20 局で測る。
                # 学習側の 12 局で 1.000 に届いても未知の局面では 99% 前後にとどまる)
                if si == 0 and gen % 10 == 9:
                    if measure(best_z, "random", games=20, K=K)["legal"] >= 0.99:
                        print(f"  gen {gen:3d}  お手つきが 100 手に1回を切った。次へ",
                              flush=True)
                        break
            base = best_z
            print(report(base, K, depth), flush=True)
            POLICY.write_text(save(si, base, best_r), encoding="utf-8")

    pick_best(K)
    print(f"\n完了 ({(time.time()-t0)/60:.1f} 分)  → {POLICY}")


if __name__ == "__main__":
    main()
