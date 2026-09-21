"""data/ と out/ の在りかを決める。

ソースから走らせるときはリポジトリのルートでよい。PyInstaller で固めると
`__file__` が展開先の一時ディレクトリを指してしまうので、実行ファイルの
置き場から上へ辿って `data/` か `out/` を持つ場所を探す。

環境変数 `FLY_ROOT` があればそれを最優先する。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MARKERS = ("data", "out", "scripts")


def _looks_like_root(p: Path) -> bool:
    return any((p / m).is_dir() for m in MARKERS)


def find_root() -> Path:
    env = os.environ.get("FLY_ROOT")
    if env and Path(env).is_dir():
        return Path(env).resolve()

    if getattr(sys, "frozen", False):
        here = Path(sys.executable).resolve().parent
        for cand in (here, *here.parents):
            if _looks_like_root(cand):
                return cand
        return here

    return Path(__file__).resolve().parent.parent


ROOT = find_root()
DATA = ROOT / "data" / "malecns"
SKEL = DATA / "skeletons"
OUT = ROOT / "out"
SCRIPTS = ROOT / "scripts"
ARCHIVE = OUT / "archive"
PYEXE = ROOT / ".venv" / "Scripts" / "python.exe"
