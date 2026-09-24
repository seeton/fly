"""五目並べの網を、ブラウザだけで打てる形に書き出す (`web/gomoku/`)。

アプリ (PySide6) の五目並べタブと同じ蝿を、Python もサーバも無しで打てるように
する。網の計算 (畳み込み7層) と読み (木探索) は `web/gomoku/engine.js` に
JavaScript で書き直してあり、ここでは重みだけを渡す:

    net.bin    重みを float32 で並べたもの (約 230 KB)
    net.json   並び順と形、学習の来歴 (何局ぶん・物差しで測った値)
    golden.json  答え合わせ用。いくつかの盤について torch が出した
               手の logits と見込み、それに読んだ末の手。ページを ?test で開くと
               JavaScript の計算がこれと一致するかを確かめて表示する

**外の部品を使わない。** ONNX Runtime Web なども使えるが、置く物が増え (数 MB の
WebAssembly)、書き出しに onnx も要る。網は重み 5.7 万個と小さく、JavaScript で
直に計算しても1回 10〜20 ms で済む。

Cloudflare Pages などに `web/gomoku/` をそのまま置けば公開できる。

使い方:
  python scripts/33_export_gomoku_web.py                    # out/gomoku_net.pt から
  python scripts/33_export_gomoku_web.py --net out/gomoku_net_snap.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gomoku_net as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
WEB = ROOT / "web" / "gomoku"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default=str(OUT / "gomoku_net.pt"))
    ap.add_argument("--golden-sims", type=int, default=200,
                    help="答え合わせで読む回数 (ブラウザ側も同じ回数で読んで手を比べる)")
    args = ap.parse_args()

    torch.set_num_threads(1)
    ck = torch.load(args.net, weights_only=False)
    net = M.FlyNet(**ck["config"]).eval()
    net.load_state_dict(ck["net"])
    meta_p = Path(args.net).with_suffix(".json")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}

    # --- 重み ---
    WEB.mkdir(parents=True, exist_ok=True)
    tensors, chunks, off = [], [], 0
    for name, t in net.state_dict().items():
        a = t.detach().numpy().astype(np.float32).ravel()
        tensors.append({"name": name, "shape": list(t.shape), "offset": off, "size": a.size})
        chunks.append(a)
        off += a.size
    (WEB / "net.bin").write_bytes(np.concatenate(chunks).astype("<f4").tobytes())
    info = {
        "config": net.config(), "board": 15, "tensors": tensors,
        "games_total": meta.get("games_total"), "hours": meta.get("hours"),
        "sims": meta.get("eval_sims", 200), "scoreboard": meta.get("scoreboard"),
        "gamma": meta.get("gamma"),
    }
    (WEB / "net.json").write_text(json.dumps(info, ensure_ascii=False, indent=1),
                                  encoding="utf-8")

    # --- 答え合わせ ---
    rng = np.random.default_rng(0)
    boards = []
    for k in range(6):
        b = np.zeros((15, 15), np.int8)
        stones = int(rng.integers(3, 30))
        cells = rng.choice(225, stones, replace=False)
        for i, c in enumerate(cells):
            b.flat[c] = 1 if i % 2 == 0 else -1
        boards.append(b)
    # 読みの答え合わせは、手番の側 (+1) に四がある盤・相手に四がある盤も入れる
    b = np.zeros((15, 15), np.int8)
    b[7, 3:7] = 1
    b[0, 0] = b[0, 2] = b[14, 14] = b[14, 12] = -1
    boards.append(b)
    b = np.zeros((15, 15), np.int8)
    b[7, 3:7] = -1
    b[7, 2] = 1
    b[0, 0] = b[14, 14] = b[0, 14] = 1
    boards.append(b)
    golden = []
    for b in boards:
        lg, v = M.evaluate(net, [b], [1])
        player = M.Player(net, sims=args.golden_sims)
        mv = player.move(b)
        golden.append({"board": b.ravel().tolist(), "logits": lg[0].tolist(),
                       "value": float(v[0]), "move": int(mv),
                       "visits": player.last_tree.visits().tolist()})
    (WEB / "golden.json").write_text(json.dumps({"sims": args.golden_sims, "cases": golden}),
                                     encoding="utf-8")
    # 版の印を書き換える (古い engine.js や網がブラウザのキャッシュに残らないように)
    import re
    import time
    stamp = time.strftime("%Y%m%d%H%M%S")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"\?v=[0-9A-Za-z]+", f"?v={stamp}", html)
    (WEB / "index.html").write_text(html, encoding="utf-8", newline="\n")
    size = (WEB / "net.bin").stat().st_size
    print(f"書き出し: {WEB} (net.bin {size/1024:.0f} KB, 重み {off} 個, "
          f"{meta.get('games_total', '?')} 局ぶん)")


if __name__ == "__main__":
    main()
