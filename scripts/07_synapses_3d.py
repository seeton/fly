"""シナプスそのものを 3D で見る。

ニューロンの骨格に、実測されたシナプスの点を重ねる。

  水色 = post (入力シナプス。このニューロンが「受け取る」場所)
  赤   = pre  (出力シナプス。このニューロンが「送る」場所)

入力は樹状突起にびっしり集まり、出力は軸索の終末に固まる。
「どっちが入口でどっちが出口か」が形から読める。

データは navis 同梱の hemibrain 嗅覚投射ニューロン (実測値)。

使い方:
  python scripts/07_synapses_3d.py            # 1本 + 全シナプス
  python scripts/07_synapses_3d.py --n 3      # 3本重ねる
  python scripts/07_synapses_3d.py --no-open

出力: out/synapses_3d.html
"""

import argparse
import webbrowser
from pathlib import Path

import navis
import numpy as np
import plotly.graph_objects as go

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1, help="ニューロン数 (1-5)")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    nl = navis.example_neurons(min(max(args.n, 1), 5))
    if not isinstance(nl, navis.NeuronList):
        nl = navis.NeuronList([nl])

    total_pre = total_post = 0
    for n in nl:
        c = n.connectors
        total_pre += int((c["type"] == "pre").sum())
        total_post += int((c["type"] == "post").sum())
        print(f"{n.id}: ノード{n.n_nodes:,}  シナプス{len(c):,} (pre {int((c['type']=='pre').sum()):,} / post {int((c['type']=='post').sum()):,})")
    print(f"合計: pre {total_pre:,} / post {total_post:,}")

    fig = navis.plot3d(nl, backend="plotly", inline=False, color=(170, 170, 170), linewidth=1.5)

    try:
        brain = navis.example_volume("neuropil")
        brain.color = (255, 255, 255, 0.05)
        fig = navis.plot3d(brain, backend="plotly", fig=fig, inline=False)
    except Exception as exc:
        print(f"脳メッシュはスキップ: {exc}")

    for kind, color, label in [("post", "#22d3ee", "post (入力)"), ("pre", "#ff3b30", "pre (出力)")]:
        pts = np.vstack([n.connectors.loc[n.connectors["type"] == kind, ["x", "y", "z"]].values for n in nl])
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=2.2, color=color, opacity=0.75),
                name=f"{label}  {len(pts):,}",
            )
        )

    fig.update_layout(
        title="hemibrain 嗅覚投射ニューロンの実測シナプス — 水色=入力 / 赤=出力",
        paper_bgcolor="#0b0b0b",
        font=dict(color="#eee"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
    )

    OUT.mkdir(exist_ok=True)
    path = OUT / "synapses_3d.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"書き出し: {path}")
    if not args.no_open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
