"""FlyWire のサーバから「生の」データ(接続・骨格)を取ってくる。

これだけは無料アカウント登録が必要(数分):
  1. https://join.flywire.ai/ でアカウントを作る
  2. https://global.daf-apis.com/auth/api/v1/user/token でトークンを表示
  3. 一度だけ:  python scripts/04_flywire_live.py --token <トークン>
  4. 以後:      python scripts/04_flywire_live.py --root-id 720575940628857210

root_id は data/flywire/flywire_783_annotations.tsv や
https://codex.flywire.ai から探せる。
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", help="初回のみ: CAVE のトークンを保存する")
    ap.add_argument("--root-id", type=int, help="接続を調べたいニューロンの root_id")
    ap.add_argument("--skeleton", action="store_true", help="骨格を取って out/ に 3D HTML を書き出す")
    args = ap.parse_args()

    from fafbseg import flywire

    if args.token:
        flywire.set_chunkedgraph_secret(args.token)
        print("トークンを保存しました (~/.cloudvolume/secrets/)")
        return

    try:
        flywire.get_chunkedgraph_secret()
    except Exception:
        raise SystemExit("トークンが未設定です。上の説明の手順 1-3 を実行してください。")

    flywire.set_default_dataset("public")  # 公開スナップショット(783)

    if not args.root_id:
        raise SystemExit("--root-id を指定してください")

    print(f"root_id={args.root_id} の接続を取得中 ...")
    cn = flywire.get_connectivity(args.root_id, upstream=True, downstream=True)
    print(f"パートナー: {len(cn):,} 件")
    print(cn.sort_values("weight", ascending=False).head(25).to_string(index=False))

    if args.skeleton:
        import webbrowser
        from pathlib import Path

        import navis

        out = Path(__file__).resolve().parent.parent / "out"
        out.mkdir(exist_ok=True)
        sk = flywire.get_skeletons(args.root_id)
        fig = navis.plot3d(sk, backend="plotly", inline=False)
        path = out / f"flywire_{args.root_id}.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        print(f"書き出し: {path}")
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
