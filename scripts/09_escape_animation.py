"""逃避回路を「動かす」— 実際の3D骨格の上を信号が伝わるアニメーション。

再生ボタンを押すと:
  1. 影が迫る → LC4 / LPLC2 (目) が光る
  2. Giant Fiber (DNp01) が発火 → 実際の軸索の形に沿って信号が脳から神経索へ降りる
  3. TTMn (中脚の運動ニューロン) が光る → 脚が伸びて飛び立つ

形・経路・接触点はすべて Male CNS の実データ。伝導速度だけは文献値の想定
(巨大繊維は太いので速い) を使った模式的なタイミング。

使い方:
  python scripts/09_escape_animation.py
  python scripts/09_escape_animation.py --frames 90 --no-open

出力: out/escape_animation.html
"""

import argparse
import urllib.request
import webbrowser
from pathlib import Path

import navis
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "malecns"
SKEL = DATA / "skeletons"
OUT = ROOT / "out"
SWC_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{}.swc"

DIM = {"visual": "#1b3a6b", "gf": "#5a1414", "motor": "#14401f"}
BRIGHT = {"visual": "#4da3ff", "gf": "#ff2d2d", "motor": "#2be06a"}


def fetch_swc(body_id: int) -> Path:
    SKEL.mkdir(parents=True, exist_ok=True)
    p = SKEL / f"{body_id}.swc"
    if not p.exists():
        urllib.request.urlretrieve(SWC_URL.format(body_id), p)
    return p


def line_coords(n):
    """骨格を1本の折れ線(セグメント間は None 区切り)に変換。"""
    xs, ys, zs = [], [], []
    pos = n.nodes.set_index("node_id")[["x", "y", "z"]]
    for seg in n.segments:
        c = pos.loc[seg].to_numpy()
        xs += list(c[:, 0]) + [None]
        ys += list(c[:, 1]) + [None]
        zs += list(c[:, 2]) + [None]
    return xs, ys, zs


def axon_path(n, target_xyz):
    """脳側の末端から、target に一番近いノードまでの骨格上の経路の座標列。"""
    pos = n.nodes.set_index("node_id")[["x", "y", "z"]]
    g = n.graph.to_undirected()
    start = int(n.nodes.loc[n.nodes["z"].idxmin(), "node_id"])
    d = np.linalg.norm(pos.to_numpy() - np.asarray(target_xyz), axis=1)
    end = int(pos.index[int(d.argmin())])
    path = nx.shortest_path(g, start, end)
    return pos.loc[path].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--n-visual", type=int, default=6)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    ann = pd.read_feather(DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    print("接続データ読み込み中 ...")
    w = pd.read_feather(DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    ty = ann.set_index("bodyId")["type"]

    gf_ids = ann[ann["type"] == "DNp01"]["bodyId"].astype(int).tolist()
    up = w[w["body_post"].isin(gf_ids)].copy()
    up["type"] = up["body_pre"].map(ty)
    vis_ids = []
    for t in ["LC4", "LPLC2"]:
        vis_ids += up[up["type"] == t].groupby("body_pre")["weight"].sum().nlargest(args.n_visual).index.astype(int).tolist()
    down = w[w["body_pre"].isin(gf_ids)].copy()
    down["type"] = down["body_post"].map(ty)
    motor_ids = down[down["type"] == "TTMn"].groupby("body_post")["weight"].sum().nlargest(2).index.astype(int).tolist()
    print(f"視覚 {len(vis_ids)} / GF {len(gf_ids)} / 運動 {len(motor_ids)}")

    groups = [("visual", vis_ids), ("gf", gf_ids), ("motor", motor_ids)]
    traces, kinds = [], []
    neurons = {}
    print("骨格を読み込み中 ...")
    for kind, ids in groups:
        for b in ids:
            n = navis.read_swc(fetch_swc(b))
            neurons[b] = n
            xs, ys, zs = line_coords(n)
            traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color=DIM[kind], width=4 if kind == "gf" else 2),
                                       name=f"{ty.get(b)} {b}", showlegend=False, hoverinfo="skip"))
            kinds.append(kind)

    # Giant Fiber の軸索に沿った経路(脳 → TTMn 接触点付近)
    gf = neurons[gf_ids[0]]
    motor_center = neurons[motor_ids[0]].nodes[["x", "y", "z"]].to_numpy().mean(axis=0)
    path = axon_path(gf, motor_center)
    print(f"Giant Fiber 軸索経路: {len(path)} ノード, z {path[0][2]:.0f} → {path[-1][2]:.0f}")

    spike = go.Scatter3d(x=[path[0][0]], y=[path[0][1]], z=[path[0][2]], mode="markers",
                         marker=dict(size=11, color="#fff200", opacity=0.0,
                                     line=dict(width=0)), name="信号", showlegend=False)
    traces.append(spike)
    spike_idx = len(traces) - 1

    fig = go.Figure(data=traces)

    # --- フレーム ---
    F = args.frames
    p1, p2 = int(F * 0.2), int(F * 0.75)   # 視覚→GF発火 / GF→運動
    frames = []
    for f in range(F):
        colors, widths = [], []
        for kind in kinds:
            if kind == "visual":
                on = f >= int(F * 0.05)
            elif kind == "gf":
                on = f >= p1
            else:
                on = f >= p2
            colors.append(BRIGHT[kind] if on else DIM[kind])
            widths.append((6 if kind == "gf" else 3) if on else (4 if kind == "gf" else 2))

        if p1 <= f <= p2:
            frac = (f - p1) / max(p2 - p1, 1)
            pt = path[min(int(frac * (len(path) - 1)), len(path) - 1)]
            sx, sy, sz, op = [pt[0]], [pt[1]], [pt[2]], 1.0
        else:
            sx, sy, sz, op = [path[0][0]], [path[0][1]], [path[0][2]], 0.0

        data = [go.Scatter3d(line=dict(color=c, width=wd)) for c, wd in zip(colors, widths)]
        data.append(go.Scatter3d(x=sx, y=sy, z=sz, marker=dict(size=13, color="#fff200", opacity=op)))
        label = "① 影が迫る（LC4/LPLC2）" if f < p1 else ("② Giant Fiber 発火・神経索へ" if f <= p2 else "③ TTMn 発火 → 跳躍")
        frames.append(go.Frame(data=data, traces=list(range(len(traces))), name=f"{f}",
                               layout=go.Layout(title=f"逃避回路  {label}")))
    fig.frames = frames

    fig.update_layout(
        title="逃避回路  ① 影が迫る（LC4/LPLC2）",
        paper_bgcolor="#080808", font=dict(color="#eee"),
        scene=dict(bgcolor="#080808", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.05,
                          buttons=[
                              dict(label="▶ 再生", method="animate",
                                   args=[None, dict(frame=dict(duration=45, redraw=True), fromcurrent=True,
                                                    transition=dict(duration=0))]),
                              dict(label="■ 停止", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                          ])],
    )

    OUT.mkdir(exist_ok=True)
    path_out = OUT / "escape_animation.html"
    fig.write_html(str(path_out), include_plotlyjs="cdn", auto_play=False)
    print(f"書き出し: {path_out}  ({path_out.stat().st_size/1e6:.1f} MB)")
    if not args.no_open:
        webbrowser.open(path_out.as_uri())


if __name__ == "__main__":
    main()
