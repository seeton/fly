"""デスクトップアプリを実行ファイルに固める (PyInstaller)。

    .\\.venv\\Scripts\\python.exe build_app.py

`dist/ハエ脳ビューア/ハエ脳ビューア.exe` ができる。Python を入れていない
Windows でもこれだけで起動する。

同梱するのは UI と問い合わせ層だけ。**data/ (7.7GB) と out/ は入れない**。
起動した exe は自分の置き場から上へ辿って `data/` `out/` を持つ場所を探す
(`app/paths.py`)。見つからないときは環境変数 `FLY_ROOT` で教える:

    set FLY_ROOT=C:\\Users\\<you>\\fly

MuJoCo・numba・navis などシミュレーション側は入っていないので、動画タブの
「今のコードで作り直す」は **リポジトリの .venv がある環境でだけ** 働く。
固めた exe を配って回すなら、そのボタンは使えないと思っておくこと。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "ハエ脳ビューア"

# 重いだけで使わないものは落とす。アプリが要るのは
# PySide6 / pyqtgraph / numpy / pandas / pyarrow / networkx だけ。
EXCLUDE = [
    "mujoco", "flygym", "numba", "llvmlite", "navis", "trimesh", "skeletor",
    "matplotlib", "seaborn", "scipy", "IPython", "jupyter", "jupyterlab",
    "notebook", "nbconvert", "nbformat", "ipykernel", "ipywidgets",
    "cloudvolume", "caveclient", "fafbseg", "flybrains", "neuprint",
    "imageio", "imageio_ffmpeg", "PIL", "boto3", "botocore", "google",
    "tkinter", "PyQt5", "PyQt6", "PySide2", "fastapi", "uvicorn", "starlette",
    "xarray", "h5py", "zarr", "sklearn", "torch",
]

HIDDEN = ["pyqtgraph.opengl", "OpenGL.platform.win32", "pandas._libs.tslibs"]


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller が要ります:\n"
              "  .\\.venv\\Scripts\\python.exe -m pip install pyinstaller")
        return 1

    for d in (ROOT / "build", ROOT / "dist"):
        shutil.rmtree(d, ignore_errors=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--windowed", "--onedir",
        "--name", NAME,
        "--icon", str(ROOT / "app" / "icon.ico"),
        "--add-data", f"{ROOT / 'app' / 'icon.png'}{';'}app",
        "--collect-submodules", "pyqtgraph",
    ]
    for m in HIDDEN:
        cmd += ["--hidden-import", m]
    for m in EXCLUDE:
        cmd += ["--exclude-module", m]
    cmd.append(str(ROOT / "run_app.py"))

    print(" ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc == 0:
        exe = ROOT / "dist" / NAME / f"{NAME}.exe"
        size = sum(f.stat().st_size for f in (ROOT / "dist" / NAME).rglob("*")
                   if f.is_file())
        print(f"\nできました: {exe}  (フォルダ全体で {size / 1e6:.0f} MB)")
        print("data/ out/ は同梱していない。別の場所に置くなら FLY_ROOT を設定すること。")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
