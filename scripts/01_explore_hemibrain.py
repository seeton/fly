"""hemibrain コネクトームを触ってみる最初のスクリプト。

データ: Janelia hemibrain v1.2 の "traced adjacencies"(公開・ログイン不要)
  - traced-neurons.csv          : 21,739 個のニューロン (bodyId, type, instance)
  - traced-total-connections.csv: 355 万本の接続 (pre -> post, weight=シナプス数)
  - traced-roi-connections.csv  : 上を脳領域(ROI)ごとに分けたもの

使い方:
  python scripts/01_explore_hemibrain.py                 # 全体サマリ
  python scripts/01_explore_hemibrain.py --type MBON01   # 特定の細胞型の上流/下流
  python scripts/01_explore_hemibrain.py --search MBON   # 型名を検索
"""

import argparse
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "hemibrain" / "exported-traced-adjacencies-v1.2"


def load():
    neurons = pd.read_csv(DATA / "traced-neurons.csv")
    conns = pd.read_csv(DATA / "traced-total-connections.csv")
    return neurons, conns


def summary(neurons: pd.DataFrame, conns: pd.DataFrame) -> None:
    print(f"ニューロン数      : {len(neurons):,}")
    print(f"型が付いたニューロン: {neurons['type'].notna().sum():,}")
    print(f"ユニークな細胞型   : {neurons['type'].nunique():,}")
    print(f"接続(エッジ)数     : {len(conns):,}")
    print(f"シナプス総数       : {conns['weight'].sum():,}")
    print()

    print("== 個体数の多い細胞型 TOP15 ==")
    print(neurons["type"].value_counts().head(15).to_string())
    print()

    out_deg = conns.groupby("bodyId_pre")["weight"].sum()
    in_deg = conns.groupby("bodyId_post")["weight"].sum()
    name = neurons.set_index("bodyId")["type"]

    print("== 出力シナプスが多いニューロン TOP10 ==")
    for body_id, w in out_deg.nlargest(10).items():
        print(f"  {body_id:>12}  {str(name.get(body_id)):<20} {w:>8,} synapses out")
    print()
    print("== 入力シナプスが多いニューロン TOP10 ==")
    for body_id, w in in_deg.nlargest(10).items():
        print(f"  {body_id:>12}  {str(name.get(body_id)):<20} {w:>8,} synapses in")


def partners(neurons: pd.DataFrame, conns: pd.DataFrame, cell_type: str, top: int = 15) -> None:
    sel = neurons[neurons["type"] == cell_type]
    if sel.empty:
        sel = neurons[neurons["type"].fillna("").str.contains(cell_type, case=False)]
        if sel.empty:
            print(f"'{cell_type}' に一致する細胞型が見つかりません (--search で検索できます)")
            return
        print(f"完全一致なし → 部分一致 {sel['type'].nunique()} 型 / {len(sel)} 細胞を使います")

    ids = set(sel["bodyId"])
    name = neurons.set_index("bodyId")["type"]
    print(f"\n== {cell_type} : {len(ids)} 細胞 ==")
    print(sel[["bodyId", "type", "instance"]].head(10).to_string(index=False))

    downstream = conns[conns["bodyId_pre"].isin(ids)].copy()
    downstream["partner_type"] = downstream["bodyId_post"].map(name)
    upstream = conns[conns["bodyId_post"].isin(ids)].copy()
    upstream["partner_type"] = upstream["bodyId_pre"].map(name)

    print(f"\n-- 下流 (このニューロンが出力する先) TOP{top} --")
    print(downstream.groupby("partner_type")["weight"].sum().nlargest(top).to_string())
    print(f"\n-- 上流 (このニューロンへ入力してくる元) TOP{top} --")
    print(upstream.groupby("partner_type")["weight"].sum().nlargest(top).to_string())


def search(neurons: pd.DataFrame, query: str) -> None:
    hit = neurons[neurons["type"].fillna("").str.contains(query, case=False)]
    counts = hit["type"].value_counts()
    print(f"'{query}' を含む細胞型: {len(counts)} 種 / {len(hit)} 細胞")
    print(counts.head(40).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", help="調べたい細胞型 (例: MBON01, ER4d, DA1_lPN)")
    ap.add_argument("--search", help="細胞型名の部分一致検索")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(f"データが見つかりません: {DATA}\nREADME の取得手順を見てください。")

    print("読み込み中 ... (数秒かかります)")
    neurons, conns = load()

    if args.search:
        search(neurons, args.search)
    elif args.type:
        partners(neurons, conns, args.type, args.top)
    else:
        summary(neurons, conns)


if __name__ == "__main__":
    main()
