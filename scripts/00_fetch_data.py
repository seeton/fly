"""データを取得/再取得するスクリプト(全部ログイン不要の公開データ)。

  python scripts/00_fetch_data.py            # hemibrain + FlyWire注釈 (78 MB)
  python scripts/00_fetch_data.py --malecns  # 上 + Male CNS 全結合 (+1.1 GB)
  python scripts/00_fetch_data.py --all
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEMI = ROOT / "data" / "hemibrain"
FLYWIRE = ROOT / "data" / "flywire"
MALECNS = ROOT / "data" / "malecns"

HEMI_URL = "https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz"
FLYWIRE_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
MCNS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
MCNS_FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",       #   14 MB
    "body-neurotransmitters-male-cns-v1.0.feather",             #   43 MB
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",     # 1051 MB
]


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"既にあります: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"取得中: {dest.name}")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def fetch_base() -> None:
    tar_path = HEMI / "exported-traced-adjacencies-v1.2.tar.gz"
    download(HEMI_URL, tar_path)
    if not (HEMI / "exported-traced-adjacencies-v1.2").exists():
        print("展開中 ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(HEMI)
    download(FLYWIRE_URL, FLYWIRE / "flywire_783_annotations.tsv")


def fetch_malecns() -> None:
    print("\n-- Male CNS v1.0 (合計 1.1 GB, 数分かかります) --")
    for fn in MCNS_FILES:
        download(MCNS_BASE + fn, MALECNS / fn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--malecns", action="store_true", help="Male CNS も取得する")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    fetch_base()
    if args.malecns or args.all:
        fetch_malecns()

    print("\n完了。")


if __name__ == "__main__":
    main()
