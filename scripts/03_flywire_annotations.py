"""FlyWire (FAFB) 成虫全脳コネクトームのニューロン注釈 13.9 万個を眺める。

hemibrain が「脳の一部を高解像度で」なのに対し、FlyWire は
「成虫メス 1 匹の脳まるごと」。ここでは公開されている注釈テーブル
(Schlegel et al. 2024 / snapshot 783) をローカルで集計する。
接続(シナプス)データ自体は FlyWire のアカウントが必要 → README 参照。

使い方:
  python scripts/03_flywire_annotations.py                  # 全体サマリ
  python scripts/03_flywire_annotations.py --search MBON    # 細胞型を検索
  python scripts/03_flywire_annotations.py --nt             # 神経伝達物質の内訳
"""

import argparse
from pathlib import Path

import pandas as pd

TSV = Path(__file__).resolve().parent.parent / "data" / "flywire" / "flywire_783_annotations.tsv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", help="cell_type / hemibrain_type の部分一致検索")
    ap.add_argument("--nt", action="store_true", help="神経伝達物質の内訳を出す")
    args = ap.parse_args()

    if not TSV.exists():
        raise SystemExit(f"データが見つかりません: {TSV}\nREADME の取得手順を見てください。")

    df = pd.read_csv(TSV, sep="\t", low_memory=False)

    if args.search:
        q = args.search
        hit = df[
            df["cell_type"].fillna("").str.contains(q, case=False)
            | df["hemibrain_type"].fillna("").str.contains(q, case=False)
        ]
        print(f"'{q}' にヒット: {len(hit):,} ニューロン")
        cols = ["root_id", "cell_type", "hemibrain_type", "super_class", "side", "top_nt"]
        print(hit[cols].head(30).to_string(index=False))
        print("\nroot_id を https://codex.flywire.ai/app/cell_details?root_id=... で見られます")
        return

    if args.nt:
        print("== 推定神経伝達物質 ==")
        print(df["top_nt"].value_counts(dropna=False).to_string())
        return

    print(f"ニューロン総数   : {len(df):,}")
    print(f"細胞型 (cell_type): {df['cell_type'].nunique():,} 種")
    print(f"hemibrain と対応付いた: {df['hemibrain_type'].notna().sum():,}")
    print()
    print("== super_class (大分類) ==")
    print(df["super_class"].value_counts(dropna=False).to_string())
    print()
    print("== 左右 ==")
    print(df["side"].value_counts(dropna=False).to_string())
    print()
    print("== 個体数の多い細胞型 TOP15 ==")
    print(df["cell_type"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
