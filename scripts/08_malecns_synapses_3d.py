"""Male CNS の実測シナプスを 3D で見る。

3億1183万個のシナプス座標 (syn-partners) から、指定したニューロンの分を抜き出して
骨格の上に重ねる。さらに特定の相手との接触点だけを光らせられる。

  水色 = post (このニューロンが受け取る)
  橙   = pre  (このニューロンが送る)
  黄   = --partner で指定した相手との接触点

使い方:
  python scripts/08_malecns_synapses_3d.py                            # DNp01 の全シナプス
  python scripts/08_malecns_synapses_3d.py --partner TTMn             # GF→TTMn の接触点を強調
  python scripts/08_malecns_synapses_3d.py --type DNp02 --no-open

出力: out/malecns_synapses_3d.html
"""

import argparse
import urllib.request
import webbrowser
from pathlib import Path

import navis
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "malecns"
SKEL = DATA / "skeletons"
OUT = ROOT / "out"
SYN = DATA / "syn-partners-male-cns-v1.0-minconf-0.5.feather"
SWC_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{}.swc"


def fetch_swc(body_id: int) -> Path:
    SKEL.mkdir(parents=True, exist_ok=True)
    path = SKEL / f"{body_id}.swc"
    if not path.exists():
        urllib.request.urlretrieve(SWC_URL.format(body_id), path)
    return path


def skeletons(body_ids):
    out = []
    for b in body_ids:
        try:
            n = navis.read_swc(fetch_swc(int(b)))
            n.id, n.name = int(b), str(b)
            out.append(n)
        except Exception as exc:
            print(f"  骨格が取れず {b}: {exc}")
    return navis.NeuronList(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", default="DNp01", help="主役の細胞型")
    ap.add_argument("--partner", help="接触点を強調したい相手の細胞型")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if not SYN.exists():
        raise SystemExit(f"シナプス座標がありません: {SYN}\n  6.8GB: 00_fetch_data.py の MCNS_FILES を参照")

    ann = pd.read_feather(DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    main_ids = ann[ann["type"] == args.type]["bodyId"].astype(int).tolist()
    if not main_ids:
        raise SystemExit(f"'{args.type}' が見つかりません")
    print(f"{args.type}: {main_ids}")

    partner_ids = []
    if args.partner:
        partner_ids = ann[ann["type"] == args.partner]["bodyId"].astype(int).tolist()
        print(f"{args.partner}: {len(partner_ids)} ボディ")

    print("シナプスを検索中 (6.8GB を走査・数秒) ...")
    d = ds.dataset(SYN, format="feather")
    tbl = d.to_table(filter=pc.field("body_pre").isin(main_ids) | pc.field("body_post").isin(main_ids))
    df = tbl.to_pandas()
    print(f"  {len(df):,} シナプス")

    is_pre = df["body_pre"].isin(main_ids)
    pre_pts = df.loc[is_pre, ["x_pre", "y_pre", "z_pre"]].to_numpy()
    post_pts = df.loc[~is_pre, ["x_post", "y_post", "z_post"]].to_numpy()
    print(f"  出力(pre) {len(pre_pts):,} / 入力(post) {len(post_pts):,}")
    print("  主な領域:", ", ".join(df["primary_post"].value_counts().head(4).index.astype(str)))

    contact = None
    if partner_ids:
        m = is_pre & df["body_post"].isin(partner_ids)
        contact = df.loc[m, ["x_pre", "y_pre", "z_pre"]].to_numpy()
        rois = df.loc[m, "primary_post"].value_counts()
        print(f"  {args.type} -> {args.partner} の接触: {len(contact)} シナプス  領域: {dict(rois.head(3))}")

    print("骨格を取得中 ...")
    nl_main = skeletons(main_ids)
    nl_partner = skeletons(partner_ids[:2]) if partner_ids else navis.NeuronList([])

    fig = navis.plot3d(nl_main, backend="plotly", inline=False, color=(255, 70, 70), linewidth=3)
    if len(nl_partner):
        fig = navis.plot3d(nl_partner, backend="plotly", fig=fig, inline=False, color=(60, 230, 120), linewidth=3)

    for pts, color, size, label in [
        (post_pts, "#22d3ee", 1.6, f"post 入力 {len(post_pts):,}"),
        (pre_pts, "#ff9f0a", 1.6, f"pre 出力 {len(pre_pts):,}"),
    ]:
        if len(pts):
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                                       marker=dict(size=size, color=color, opacity=0.5), name=label))
    if contact is not None and len(contact):
        fig.add_trace(go.Scatter3d(x=contact[:, 0], y=contact[:, 1], z=contact[:, 2], mode="markers",
                                   marker=dict(size=7, color="#ffe600", opacity=1.0,
                                               line=dict(width=1, color="#000")),
                                   name=f"{args.type}→{args.partner} 接触 {len(contact)}"))

    title = f"Male CNS 実測シナプス — {args.type}"
    if args.partner:
        title += f" と {args.partner} の接触点(黄)"
    fig.update_layout(title=title, paper_bgcolor="#0b0b0b", font=dict(color="#eee"),
                      legend=dict(bgcolor="rgba(0,0,0,0.5)"))

    OUT.mkdir(exist_ok=True)
    path = OUT / "malecns_synapses_3d.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"書き出し: {path}")
    if not args.no_open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
