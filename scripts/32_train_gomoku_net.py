"""蝿の目 (畳み込みの網) を、自分と打って勝ち負けだけから覚えさせる。

`30_train_gomoku.py` (重み 159 個を CMA-ES で探す) の後継。あちらは 15x15 で
素朴な相手に 15%、外のエンジン Rapfi の一番浅い段にも全敗で、全敗が続くと
報酬が動かず学ぶ手がかりが無かった。ここでは AlphaZero の型で学ぶ
(網・読み・自己対戦は `gomoku_net.py`):

    1. いまの網で自分自身と打つ (G 局を並べて、1手ごとに読み sims 回)
    2. 各局面を「読んだ末の訪問の割合」と「その局を最後に勝ったか」の組にして貯める
    3. 貯めた局面から網を学習し直す。1 に戻る

**報酬は勝ち負けの一つだけ。** 手作りの加点 (決めた・止めた・生き延びた) は無い。

**9x9 で立ち上げてから 15x15 に移す。** 網は畳み込みと盤全体の平均/最大だけで
できていて盤の大きさに依らない。9x9 は1局が短く (約30手) 1時間に数千局回るので、
五を作る・四を止めるといった近い所の戦いを先に覚えさせる。9x9 の自由形は
先手必勝ではない (Rapfi どうしでも引き分け) が、覚えさせたいのはそこではない。

**30 分ごとに外の物差しで測る** (素朴な相手と Rapfi 深さ1、15x15)。あわせて
**自己対戦で先手が勝った割合**を出す — 15x15 は先手必勝なので、上手くなるほど
ここが上がっていくはず。

**落とされても再開できるように**、学習の区切りごとに網と最適化の状態を書く
(`out/gomoku_net.pt`)。局面の貯め (最大数十万局面) は書かない — 再開すると
貯め直しになるが、網が覚えたことは残る。`--resume` で続きから。

使い方:
  python scripts/32_train_gomoku_net.py                    # 9x9 で 60 分、あとは 15x15
  python scripts/32_train_gomoku_net.py --resume --hours 8
  python scripts/32_train_gomoku_net.py --eval-only         # 保存した網を測るだけ

出力: out/gomoku_net.pt (網) / out/gomoku_net.json (設定と測った値)
      ログは out/logs/gomoku_net.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gomoku_net as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
NET = OUT / "gomoku_net.pt"
META = OUT / "gomoku_net.json"


def measure(net, games: int, sims: int, opponents) -> dict:
    """15x15 で外の物差しに当てる。先後は半分ずつ。お手つきはこちらの読みでは起きない。

    **1局面ずつ網に通すので、その間だけ torch のスレッドを 1 にする。** CPU が
    埋まっているときに何本も立てると取り合いになり、1局面 4 ms が 104 ms になった
    (物差しだけで2時間止まる計算だった)。
    """
    import gomoku_env as G

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _measure(net, games, sims, opponents, G)
    finally:
        torch.set_num_threads(threads)


def _measure(net, games, sims, opponents, G) -> dict:
    out = {}
    player = M.Player(net, sims=sims)
    for name, opp in opponents:
        win = draw = 0
        plies = []
        for g in range(games):
            rng = np.random.default_rng(1000 + g)
            t = G.play_game(player, opp, rng, fly_first=(g % 2 == 0))
            win += t["win"]
            draw += t["draw"]
            plies.append(t["plies"])
        out[name] = {"win": win / games, "draw": draw / games,
                     "loss": 1 - (win + draw) / games, "plies": float(np.mean(plies))}
    return out


def selfplay_worker(q, stop, cfg: dict) -> None:
    """自己対戦だけをする手伝い (別プロセス)。終わった局の局面を q に送る。

    網は学習する側が書き出す `out/gomoku_net.pt` を、書き換わるたびに読み直す。
    盤の大きさも `out/gomoku_net.json` から取る (9x9 → 15x15 の切り替えに付いていく)。
    """
    torch.set_num_threads(cfg["threads"])
    net_path, meta_path = Path(cfg["net"]), Path(cfg["meta"])
    rng = np.random.default_rng(cfg["seed"])
    net = sp = None
    mtime, size = 0.0, None
    while not stop.is_set():
        try:
            m = os.path.getmtime(net_path)
        except OSError:
            time.sleep(1.0)
            continue
        if m != mtime:
            try:
                ck = torch.load(net_path, weights_only=False)
                new_size = json.loads(meta_path.read_text(encoding="utf-8")).get("size", 15)
            except Exception:
                time.sleep(0.5)                 # 書き替えの途中に当たった
                continue
            if net is None:
                net = M.FlyNet(**ck["config"]).eval()
            net.load_state_dict(ck["net"])
            mtime = m
            if sp is None or new_size != size:
                size = new_size
                sp = M.SelfPlay(net, size, cfg["games"], cfg["sims"], rng, gamma=cfg["gamma"])
        out = sp.step()
        if out:
            q.put((size, out))


def opponents_for_eval():
    import gomoku_env as G

    opps = [("素朴な相手", G.greedy_move)]
    try:
        from gomoku_pbrain import rapfi_opponent
        opps.append(("Rapfi 深さ1", rapfi_opponent(1)))
    except Exception as e:                           # Rapfi が無い環境でも学習は回す
        print(f"  (Rapfi は使えない: {e})", flush=True)
    return opps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=6.0, help="全体で回す時間")
    ap.add_argument("--warmup-min", type=float, default=60.0, help="9x9 で立ち上げる時間")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--games", type=int, default=64, help="同時に進める自己対戦の局数")
    ap.add_argument("--sims", type=int, default=64, help="自己対戦で1手ごとに読む回数")
    ap.add_argument("--chunk", type=int, default=128, help="何局終わるごとに学習するか")
    ap.add_argument("--buffer", type=int, default=250_000, help="貯める局面の上限")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reuse", type=float, default=4.0,
                    help="新しく貯めた局面1つを何回学習に使うか (目安)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.95,
                    help="勝ち負けを決着までの手数で割り引く率 (gomoku_net.SelfPlay)")
    # 64 局まとめて網に通すときのスレッド数。CPU を別の学習と分け合っている状態で
    # 測ると 4 が最速だった (25 ms。8 で 53 ms、16 で 430 ms — 取り合いで遅くなる)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0,
                    help="自己対戦だけをする手伝いのプロセス数 (学習は元の1本が受け持つ)")
    ap.add_argument("--worker-threads", type=int, default=2)
    ap.add_argument("--eval-every-min", type=float, default=30.0)
    ap.add_argument("--eval-games", type=int, default=10)
    ap.add_argument("--eval-sims", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="gomoku_net",
                    help="書き出す名前 (out/<name>.pt / .json)。試運転で本番を上書きしないため")
    args = ap.parse_args()

    global NET, META
    NET, META = OUT / f"{args.name}.pt", OUT / f"{args.name}.json"

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    net = M.FlyNet(args.ch, args.blocks).eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    meta = {"games_total": 0, "positions_total": 0, "hours": 0.0, "history": []}
    if args.resume or args.eval_only:
        ck = torch.load(NET, weights_only=False)
        net = M.FlyNet(**ck["config"]).eval()
        net.load_state_dict(ck["net"])
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if META.exists():
            meta = json.loads(META.read_text(encoding="utf-8"))
        print(f"続きから: {meta['games_total']} 局 / {meta['hours']:.1f} 時間ぶん学習済み",
              flush=True)

    if args.eval_only:
        m = measure(net, args.eval_games, args.eval_sims, opponents_for_eval())
        for k, v in m.items():
            print(f"  対 {k}: 勝ち {v['win']*100:.0f}% 引き分け {v['draw']*100:.0f}%"
                  f" 平均 {v['plies']:.0f} 手")
        return

    print(f"網 {args.ch}ch x {args.blocks} ブロック "
          f"(重み {sum(p.numel() for p in net.parameters())} 個) / 自己対戦 {args.games} 局並列 "
          f"/ 読み {args.sims} 回 / スレッド {args.threads}", flush=True)

    buffers: dict = {}                     # 盤の大きさ -> deque[(x, pi, z)]
    t0 = time.time()
    t_eval = t0
    hours0 = meta["hours"]
    size = 15 if (args.resume and meta.get("size") == 15) else 9
    sp = M.SelfPlay(net, size, args.games, args.sims, rng, gamma=args.gamma)
    stats = {"games": 0, "first": 0, "draw": 0, "plies": 0}
    fresh = 0

    def save():
        meta.update({"config": net.config(), "size": size, "gamma": args.gamma,
                     "hours": hours0 + (time.time() - t0) / 3600,
                     "sims": args.sims, "eval_sims": args.eval_sims})
        # 手伝いのプロセスが読みに来るので、書きかけを見せないよう置き換えで書く
        tmp = NET.with_suffix(".tmp")
        torch.save({"config": net.config(), "net": net.state_dict(),
                    "opt": opt.state_dict()}, tmp)
        os.replace(tmp, NET)
        tmpm = META.with_suffix(".tmp")
        tmpm.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmpm, META)

    def take(recs, rsize: int) -> int:
        """自己対戦の局面を貯めに入れ、入れた局面の数を返す。

        盤の大きさが今と違うもの (9x9 → 15x15 の切り替えの前に打たれた局) は捨てる。
        """
        n = 0
        for rec in recs:
            if rsize != size:
                continue
            if isinstance(rec[0], str):                  # 局の終わりの印
                _, plies, result = rec
                stats["games"] += 1
                stats["plies"] += plies
                stats["first"] += result == 1
                stats["draw"] += result == 0
                meta["games_total"] += 1
            else:
                buffers.setdefault(size, deque(maxlen=args.buffer)).append(rec)
                meta["positions_total"] += 1
                n += 1
        return n

    # --- 自己対戦だけをする手伝いのプロセス ---
    procs, q, stop = [], None, None
    if args.workers:
        import multiprocessing as mp
        save()                                   # 手伝いが最初に読む網と盤の大きさ
        ctx = mp.get_context("spawn")
        q, stop = ctx.Queue(maxsize=64), ctx.Event()
        cfg = {"threads": args.worker_threads, "games": args.games, "sims": args.sims,
               "gamma": args.gamma, "net": str(NET), "meta": str(META)}
        for w in range(args.workers):
            p = ctx.Process(target=selfplay_worker,
                            args=(q, stop, {**cfg, "seed": args.seed * 100 + 1000 + w}),
                            daemon=True)
            p.start()
            procs.append(p)
        print(f"自己対戦の手伝い {args.workers} プロセス (各スレッド {args.worker_threads})",
              flush=True)

    try:
        while time.time() - t0 < args.hours * 3600:
            # 9x9 の立ち上げが済んだら 15x15 へ
            if size == 9 and time.time() - t0 > args.warmup_min * 60:
                size = 15
                sp = M.SelfPlay(net, size, args.games, args.sims, rng, gamma=args.gamma)
                save()                           # 手伝いにも切り替えを知らせる
                print(f"\n=== 15x15 に移る ({(time.time()-t0)/60:.0f} 分) ===", flush=True)

            fresh += take(sp.step(), size)
            while q is not None:                 # 手伝いが送ってきた局面
                try:
                    rsize, recs = q.get_nowait()
                except Exception:
                    break
                fresh += take(recs, rsize)

            if stats["games"] < args.chunk:
                continue

            # --- 学習 ---
            buf = buffers[size]
            steps = max(1, int(fresh * args.reuse / args.batch))
            lp = lv = 0.0
            for _ in range(steps):
                idx = rng.integers(0, len(buf), args.batch)
                xs, pis, zs = [], [], []
                for i in idx:
                    x, pi, z = buf[i]
                    x, pi = M.symmetries(x, pi, size, int(rng.integers(8)))
                    xs.append(x)
                    pis.append(pi)
                    zs.append(z)
                a, b = M.train_step(net, opt, np.stack(xs),
                                    np.stack(pis).astype(np.float32),
                                    np.array(zs, dtype=np.float32))
                lp += a / steps
                lv += b / steps
            g = stats["games"]
            print(f"[{(time.time()-t0)/60:6.1f} 分] {size}x{size} 累計 {meta['games_total']:6d} 局"
                  f"  先手の勝ち {stats['first']/g*100:4.0f}% 引き分け {stats['draw']/g*100:3.0f}%"
                  f"  平均 {stats['plies']/g:4.1f} 手  損失 手 {lp:.3f} 見込み {lv:.3f}"
                  f"  (学習 {steps} 回, 貯め {len(buf)})", flush=True)
            meta["history"].append({"min": round((time.time() - t0) / 60, 1), "size": size,
                                    "games_total": meta["games_total"],
                                    "first_win": stats["first"] / g, "draw": stats["draw"] / g,
                                    "plies": stats["plies"] / g, "loss_p": lp, "loss_v": lv})
            for k in stats:
                stats[k] = 0
            fresh = 0
            save()

            # --- 外の物差し ---
            if time.time() - t_eval > args.eval_every_min * 60:
                t_eval = time.time()
                m = measure(net, args.eval_games, args.eval_sims, opponents_for_eval())
                meta["scoreboard"] = {"games": args.eval_games, "sims": args.eval_sims,
                                      "board": 15, **m}
                meta["history"][-1]["eval"] = m
                print("  物差し (15x15): " + "  ".join(
                    f"対 {k} 勝ち {v['win']*100:.0f}% 引分 {v['draw']*100:.0f}% ({v['plies']:.0f}手)"
                    for k, v in m.items()), flush=True)
                save()
    finally:
        if stop is not None:
            stop.set()
            for p in procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()

    save()
    print(f"\n完了 ({(time.time()-t0)/60:.0f} 分, 累計 {meta['games_total']} 局) → {NET}")


if __name__ == "__main__":
    main()
