"""外のエンジン (Rapfi) を物差しにして、蝿の五目並べを測る。

求めているのは2つ:

  A 先手なら勝ち切る       自由形の五目並べは先手必勝 (15x15 は Allis 1994)。
                           空盤から蝿が先手で打ち、梯子のどの段まで勝ち切れるか
  B 後手でも1ミスを咎める   先手 (全力の Rapfi) が1手だけランダムな手を打つ。
                           そこから後手の蝿が勝ち切れるか。同じ条件で後手も
                           Rapfi にしたものを「届きうる上限」として並べる

梯子は「素朴な相手 → Rapfi 読みの深さ上限 1 / 2 / 3 → 全力」。素朴な相手は
私が書いた相手なので、それに勝っても自分の書いた相手に勝っただけになる。
その上に外で鍛えられた Rapfi を置く (窓口は `gomoku_pbrain.py`)。

**物差しの確認を先にやる。** 9x9 のとき、Rapfi どうしでも毎回引き分けで、
先手必勝も「1ミスで勝ち切る」も最強のエンジンですら見せなかった。
15x15 では Rapfi 先手が勝ち切ることを、A の最初の行で毎回確かめる。

**お手つきはその場で負け** (学習とアプリと同じ規則)。

**深さを制限した Rapfi は決定的、全力の Rapfi は揺れる。** 深さの段と蝿は毎回
同じ手を打つので、空盤から始める A は1局で足りる。全力の Rapfi は持ち時間で
読むので、計算機の混み具合で手が変わる (Rapfi 同士が 33 手で先手勝ち・41 手で
先手勝ち・225 手で引き分け、と揺れた)。全力相手は --full-games 局打つ。
B は誤る手番と誤り方を乱数で変えて数を取る。

蝿は2通り測れる: 網 (`out/gomoku_net.pt`、`32_train_gomoku_net.py`) と、前の打ち手
(`out/gomoku_policy.json`、`30_train_gomoku.py`)。既定は網があれば網。

使い方:
  python scripts/31_gomoku_match.py                  # A と B を全部
  python scripts/31_gomoku_match.py --player old     # 前の打ち手を測る
  python scripts/31_gomoku_match.py --sims 400       # 網の読みの回数
  python scripts/31_gomoku_match.py --games 20 --turn-ms 300
  python scripts/31_gomoku_match.py --policy out/gomoku_policy_stage4.json

出力は画面と out/logs/gomoku_match.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gomoku_env import (N, FlyPlayer, empty_board, five_through,  # noqa: E402
                        greedy_move)
from gomoku_pbrain import PbrainEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


def duel(first, second, rng=None, blunder_ply: int = -1) -> dict:
    """first (先手) と second (後手) に一局打たせる。

    first / second は「自分を +1 として見た盤」を受けて手を返す関数。
    blunder_ply の手番だけは、その側の代わりにランダムな空きマスへ打つ。
    お手つき (石のあるマスを選ぶ) はその場で負け。
    """
    board = empty_board()
    for ply in range(N * N):
        me = 1 if ply % 2 == 0 else -1
        if ply == blunder_ply:
            mv = int(rng.choice(np.flatnonzero(board.ravel() == 0)))
        else:
            mv = (first if me == 1 else second)(board * me)
        y, x = divmod(mv, N)
        if board[y, x] != 0:
            return {"winner": -me, "plies": ply + 1, "forfeit": True}
        board[y, x] = me
        if five_through(board, y, x, me):
            return {"winner": me, "plies": ply + 1}
    return {"winner": 0, "plies": N * N}


def load_fly(path: Path) -> tuple[FlyPlayer, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    fly = FlyPlayer(d.get("K", 6), depth=d.get("depth", 0)).set_flat(np.array(d["x"]))
    return fly, d


def load_net(sims: int | None):
    """網の打ち手。1局面ずつ読むので torch のスレッドは 1 (混んだ CPU で速い)。"""
    import torch
    import gomoku_net as M

    torch.set_num_threads(1)
    ck = torch.load(OUT / "gomoku_net.pt", weights_only=False)
    net = M.FlyNet(**ck["config"]).eval()
    net.load_state_dict(ck["net"])
    meta_p = OUT / "gomoku_net.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    player = M.Player(net, sims=sims or meta.get("eval_sims", 200))
    meta["label"] = f"網 {meta.get('games_total', '?')} 局ぶん / 読み {player.sims} 回"
    return player, meta


def verdict(r: dict) -> str:
    w = {1: "先手の勝ち", -1: "後手の勝ち", 0: "引き分け"}[r["winner"]]
    return f"{w} ({r['plies']} 手{'、お手つき' if r.get('forfeit') else ''})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", choices=("net", "old"), default=None,
                    help="net = 畳み込みの網 / old = 重み159個の打ち手。既定は網があれば網")
    ap.add_argument("--sims", type=int, default=None, help="網の読みの回数")
    ap.add_argument("--policy", default=str(OUT / "gomoku_policy.json"))
    ap.add_argument("--turn-ms", type=int, default=300, help="全力 Rapfi の1手の持ち時間")
    ap.add_argument("--games", type=int, default=20, help="B の局数")
    ap.add_argument("--full-games", type=int, default=3,
                    help="A で全力 Rapfi と打つ局数 (全力は持ち時間で読むので揺れる)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    kind = args.player or ("net" if (OUT / "gomoku_net.pt").exists() else "old")
    if kind == "net":
        fly, meta = load_net(args.sims)
        fly.depth = f"木 {fly.sims} 回"
    else:
        fly, meta = load_fly(Path(args.policy))
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    say(f"蝿: {meta.get('label', '?')}  読み {fly.depth}")
    say(f"盤 {N}x{N} 自由形 / 全力 Rapfi は1手 {args.turn_ms} ms")
    t0 = time.time()

    def fly_move(b):
        return fly.move(b)

    rng0 = np.random.default_rng(0)

    def greedy(b):
        return greedy_move(b, rng0)

    engines = {d: PbrainEngine(turn_ms=args.turn_ms, max_depth=d)
               for d in (1, 2, 3)}
    full = PbrainEngine(turn_ms=args.turn_ms)
    ladder = [("素朴な相手", greedy)] + \
        [(f"Rapfi 深さ{d}", e.move) for d, e in engines.items()]
    try:
        # --- A: 空盤から、先手で勝ち切れるか ---
        say("\n=== A 空盤から蝿が先手 (深さの段は各1局、全力は揺れるので複数局) ===")
        say(f"  (物差しの確認) Rapfi 全力 同士        {verdict(duel(full.move, full.move))}")
        for name, opp in ladder:
            say(f"  蝿 先手 vs {name:10} 後手      {verdict(duel(fly_move, opp))}")
        for k in range(args.full_games):
            say(f"  蝿 先手 vs Rapfi 全力 後手 ({k+1})  {verdict(duel(fly_move, full.move))}")

        # --- B: 先手が1手だけ誤ったら、後手は咎められるか ---
        say(f"\n=== B 全力 Rapfi が先手で1手だけランダムに誤る ({args.games} 局) ===")
        rng = np.random.default_rng(args.seed)
        plan = [(int(2 * rng.integers(1, 8)), int(rng.integers(1 << 30)))
                for _ in range(args.games)]      # 誤る手番は先手の 2..14 手目
        for name, second in (("後手 Rapfi 全力 (上限)", full.move), ("後手 蝿", fly_move)):
            won = lost = 0
            for ply, s in plan:
                r = duel(full.move, second, np.random.default_rng(s), blunder_ply=ply)
                won += r["winner"] == -1
                lost += r["winner"] == 1
            say(f"  {name:22} 後手の勝ち {won:3d} / {args.games}   負け {lost:3d}"
                f"   引き分け {args.games - won - lost}")
    finally:
        for e in [full, *engines.values()]:
            e.close()

    say(f"\n({(time.time() - t0) / 60:.1f} 分)")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / "gomoku_match.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
