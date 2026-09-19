"""逃避回路を 3D で組み立てる — 見る → 判断する → 脚を蹴る、を1枚の絵に。

Male CNS の骨格(SWC)を公開バケットから取ってきて重ねる:

  LC4 / LPLC2 (視覚・迫ってくる影の検出器)   青
      ↓
  DNp01 = Giant Fiber (脳で一番太い軸索。首を貫いてVNCへ)  赤
      ↓
  TTMn (中脚を蹴り出す運動ニューロン)  緑

使い方:
  python scripts/06_escape_circuit_3d.py
  python scripts/06_escape_circuit_3d.py --n-visual 12 --no-open

出力: out/escape_circuit_3d.html
"""

import argparse
import urllib.request
import webbrowser
from pathlib import Path

import navis
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "malecns"
SKEL = DATA / "skeletons"
OUT = ROOT / "out"
SWC_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{}.swc"


def fetch_swc(body_id: int) -> Path:
    SKEL.mkdir(parents=True, exist_ok=True)
    path = SKEL / f"{body_id}.swc"
    if not path.exists():
        urllib.request.urlretrieve(SWC_URL.format(body_id), path)
    return path


def load(body_ids, color, label):
    neurons = []
    for i, b in enumerate(body_ids, 1):
        try:
            n = navis.read_swc(fetch_swc(int(b)))
        except Exception as exc:
            print(f"    {b}: 取得できず ({exc})")
            continue
        n.id = int(b)
        n.name = f"{label}_{b}"
        neurons.append(n)
        print(f"  [{i}/{len(body_ids)}] {label} {b}  ノード{n.n_nodes:,}")
    return navis.NeuronList(neurons), color


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-visual", type=int, default=8, help="LC4/LPLC2 を何本描くか(各)")
    ap.add_argument("--n-motor", type=int, default=2)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    ann = pd.read_feather(DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    print("接続データ読み込み中 ...")
    w = pd.read_feather(DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")

    dnp01 = ann[ann["type"] == "DNp01"]["bodyId"].astype(int).tolist()
    print(f"DNp01 (Giant Fiber): {dnp01}")

    # DNp01 に強く入力している LC4 / LPLC2 を選ぶ
    up = w[w["body_post"].isin(dnp01)].copy()
    up["type"] = up["body_pre"].map(ann.set_index("bodyId")["type"])
    visual = {}
    for t in ["LC4", "LPLC2"]:
        rows = up[up["type"] == t].groupby("body_pre")["weight"].sum().nlargest(args.n_visual)
        visual[t] = rows.index.tolist()
        print(f"{t}: {len(rows)} 本 (最大 {rows.max()} シナプス)")

    # DNp01 の下流の運動ニューロン
    down = w[w["body_pre"].isin(dnp01)].copy()
    down["type"] = down["body_post"].map(ann.set_index("bodyId")["type"])
    down["sc"] = down["body_post"].map(ann.set_index("bodyId")["superclass"])
    motor_rows = down[down["sc"] == "vnc_motor"].groupby(["body_post", "type"])["weight"].sum().nlargest(args.n_motor)
    print("運動ニューロン:")
    print(motor_rows.to_string())
    motor_ids = [b for b, _ in motor_rows.index]

    print("\n骨格をダウンロード中 (初回のみ) ...")
    layers = []
    for t, col in [("LC4", (60, 130, 255)), ("LPLC2", (0, 200, 230))]:
        nl, c = load(visual[t], col, t)
        if len(nl):
            layers.append((nl, c))
    nl, c = load(dnp01, (255, 40, 40), "DNp01")
    layers.append((nl, c))
    nl, c = load(motor_ids, (40, 220, 90), "TTMn")
    layers.append((nl, c))

    print("\n描画中 ...")
    all_neurons = navis.NeuronList([n for nl, _ in layers for n in nl])
    colors = [c for nl, c in layers for _ in nl]
    fig = navis.plot3d(
        all_neurons,
        color=colors,
        backend="plotly",
        inline=False,
        linewidth=2,
    )
    fig.update_layout(
        title="逃避回路 (Male CNS): 青=LC4/水色=LPLC2 視覚 → 赤=DNp01 Giant Fiber → 緑=TTMn 運動",
        paper_bgcolor="#111",
        font=dict(color="#eee"),
    )

    OUT.mkdir(exist_ok=True)
    path = OUT / "escape_circuit_3d.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"書き出し: {path}")
    if not args.no_open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
