"""Male CNS v1.0 — オスの中枢神経系まるごと(脳＋視葉＋腹部神経索)を触る。

hemibrain/FlyWire との違いは **脳から神経索(＝脚や翅の運動)まで繋がっている** こと。
「脳の判断が筋肉にどう届くか」を端から端まで辿れる。

データ (data/malecns/, 公開バケットより・ログイン不要):
  body-annotations-male-cns-v1.0-minconf-0.5.feather  21万ボディの注釈
  connectome-weights-male-cns-v1.0-minconf-0.5.feather 全結合 (weight=シナプス数)
  body-neurotransmitters-male-cns-v1.0.feather        神経伝達物質の推定

使い方:
  python scripts/05_explore_malecns.py                    # 全体サマリ
  python scripts/05_explore_malecns.py --search DNp01     # 細胞型を検索
  python scripts/05_explore_malecns.py --type DNp01       # 上流/下流
  python scripts/05_explore_malecns.py --descending DNp01 # 脳→運動ニューロンの経路
  python scripts/05_explore_malecns.py --dimorphic        # 性的二型のニューロン
"""

import argparse
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "malecns"
ANN = DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


def load_annotations() -> pd.DataFrame:
    if not ANN.exists():
        raise SystemExit(f"データがありません: {ANN}\n  python scripts/00_fetch_data.py --malecns")
    return pd.read_feather(ANN)


def load_weights() -> pd.DataFrame:
    if not WEIGHTS.exists():
        raise SystemExit(f"データがありません: {WEIGHTS}\n  python scripts/00_fetch_data.py --malecns")
    print("接続データを読み込み中 ... (1GB・20秒ほど)")
    return pd.read_feather(WEIGHTS)


def summary(ann: pd.DataFrame) -> None:
    print(f"ボディ総数        : {len(ann):,}")
    print(f"型が付いたもの     : {ann['type'].notna().sum():,}")
    print(f"ユニークな細胞型   : {ann['type'].nunique():,}")
    print()
    print("== superclass (大分類) ==")
    print(ann["superclass"].value_counts(dropna=False).head(20).to_string())
    print()
    print("== 他データセットとの対応付け ==")
    for col, label in [("flywireType", "FlyWire (メス脳)"), ("hemibrainType", "hemibrain"), ("mancType", "MANC (オスVNC)")]:
        print(f"  {label:<20} {ann[col].notna().sum():>7,} 件")
    print()
    print("== 性的二型 ==")
    print(ann["dimorphism"].value_counts().to_string())


def search(ann: pd.DataFrame, query: str) -> None:
    cols = ["type", "flywireType", "hemibrainType", "mancType"]
    mask = False
    for c in cols:
        mask = mask | ann[c].fillna("").str.contains(query, case=False)
    hit = ann[mask]
    print(f"'{query}' にヒット: {len(hit):,} ボディ / {hit['type'].nunique()} 型")
    show = ["bodyId", "type", "instance", "superclass", "somaSide", "flywireType", "hemibrainType"]
    print(hit[show].head(30).to_string(index=False))


def partners(ann: pd.DataFrame, cell_type: str, top: int) -> None:
    w = load_weights()
    pre_col, post_col, wt_col = w.columns[0], w.columns[1], w.columns[2]

    sel = ann[ann["type"] == cell_type]
    if sel.empty:
        sel = ann[ann["type"].fillna("").str.contains(cell_type, case=False)]
        if sel.empty:
            print(f"'{cell_type}' が見つかりません (--search で検索)")
            return
        print(f"完全一致なし → 部分一致 {sel['type'].nunique()} 型 / {len(sel)} ボディ")

    ids = set(sel["bodyId"])
    name = ann.set_index("bodyId")["type"]
    sc = ann.set_index("bodyId")["superclass"]
    print(f"\n== {cell_type} : {len(ids)} ボディ ==")
    print(sel[["bodyId", "type", "instance", "superclass"]].head(10).to_string(index=False))

    down = w[w[pre_col].isin(ids)].copy()
    down["partner"] = down[post_col].map(name)
    down["sc"] = down[post_col].map(sc)
    up = w[w[post_col].isin(ids)].copy()
    up["partner"] = up[pre_col].map(name)
    up["sc"] = up[pre_col].map(sc)

    print(f"\n-- 下流 TOP{top} --")
    print(down.groupby(["partner", "sc"])[wt_col].sum().nlargest(top).to_string())
    print(f"\n-- 上流 TOP{top} --")
    print(up.groupby(["partner", "sc"])[wt_col].sum().nlargest(top).to_string())


def descending(ann: pd.DataFrame, cell_type: str, min_weight: int = 10) -> None:
    """脳の下行ニューロンから運動ニューロンまでの経路を探す。"""
    import networkx as nx

    w = load_weights()
    pre_col, post_col, wt_col = w.columns[0], w.columns[1], w.columns[2]

    name = ann.set_index("bodyId")["type"]
    sc = ann.set_index("bodyId")["superclass"]

    src_rows = ann[ann["type"] == cell_type]
    if src_rows.empty:
        print(f"'{cell_type}' が見つかりません")
        return
    motor = set(ann[ann["superclass"].isin(["vnc_motor", "cb_motor"])]["bodyId"])
    print(f"出発点: {cell_type} ({len(src_rows)} ボディ) / 運動ニューロン {len(motor):,} 個")

    strong = w[w[wt_col] >= min_weight]
    print(f"weight>={min_weight} の接続で探索: {len(strong):,} エッジ")
    G = nx.from_pandas_edgelist(strong, pre_col, post_col, edge_attr=wt_col, create_using=nx.DiGraph)

    src = int(src_rows["bodyId"].iloc[0])
    if src not in G:
        print("この細胞は強い接続を持たないので min_weight を下げてください")
        return

    lengths = nx.single_source_shortest_path_length(G, src, cutoff=4)
    reached = {b: d for b, d in lengths.items() if b in motor}
    print(f"\n{cell_type} から 4 ステップ以内に届く運動ニューロン: {len(reached):,} 個")

    if reached:
        target = min(reached, key=reached.get)
        path = nx.shortest_path(G, src, target)
        print("\n最短経路の例:")
        for i, b in enumerate(path):
            arrow = "    " if i == 0 else " -> "
            print(f"{arrow}{b}  {str(name.get(b)):<14} [{sc.get(b)}]")


def dimorphic(ann: pd.DataFrame) -> None:
    d = ann[ann["dimorphism"].notna()]
    print(f"性的二型のラベルが付いたボディ: {len(d):,}")
    print(d["dimorphism"].value_counts().to_string())
    print("\n== 型ごと (オス特異的) TOP25 ==")
    ms = d[d["dimorphism"].str.contains("male-specific")]
    print(ms["type"].value_counts().head(25).to_string())
    print("\nfru/dsx 発現ラベル:")
    print(ann["fruDsx"].value_counts().head(10).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search")
    ap.add_argument("--type")
    ap.add_argument("--descending", metavar="TYPE", help="脳→運動ニューロンの経路を探す")
    ap.add_argument("--dimorphic", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-weight", type=int, default=10)
    args = ap.parse_args()

    ann = load_annotations()

    if args.search:
        search(ann, args.search)
    elif args.type:
        partners(ann, args.type, args.top)
    elif args.descending:
        descending(ann, args.descending, args.min_weight)
    elif args.dimorphic:
        dimorphic(ann)
    else:
        summary(ann)


if __name__ == "__main__":
    main()
