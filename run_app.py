"""ハエ脳ビューアの入口。

ソースから起動するとき:
    .\.venv\Scripts\python.exe run_app.py
PyInstaller で固めるときの起点もここ (build_app.py が指す)。
"""

from app.desktop import main

raise SystemExit(main())
