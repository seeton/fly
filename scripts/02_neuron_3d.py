"""ハエのニューロンを 3D でぐりぐり回して見る。

navis には hemibrain の実物のニューロン骨格(SWC)と脳のメッシュが同梱されている
ので、ネット接続もログインも無しでいきなり 3D 表示できる。

使い方:
  python scripts/02_neuron_3d.py              # 同梱の 5 本 + 脳メッシュ
  python scripts/02_neuron_3d.py --n 3        # 本数を変える
  python scripts/02_neuron_3d.py --no-open    # ブラウザを開かない

出力: out/neurons_3d.html (ブラウザで開くと回せる/拡大できる)
"""

import argparse
import webbrowser
from pathlib import Path

import navis

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5, help="表示するニューロン数 (最大5)")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)

    neurons = navis.example_neurons(min(args.n, 5))
    print("ニューロン:")
    for n in neurons:
        print(f"  id={n.id}  ノード数={n.n_nodes:,}  全長={n.cable_length/1000:.0f} um")

    to_plot = [neurons]
    try:
        brain = navis.example_volume("neuropil")
        brain.color = (250, 250, 250, 0.08)
        to_plot.append(brain)
        print("脳のメッシュ(hemibrain neuropil)も一緒に表示します")
    except Exception as exc:  # pragma: no cover
        print(f"脳メッシュはスキップ: {exc}")

    fig = navis.plot3d(to_plot, backend="plotly", inline=False)
    path = OUT / "neurons_3d.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"\n書き出し: {path}")

    if not args.no_open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
