"""アプリのアイコンを描く — app/icon.png と app/icon.ico を作る。

絵の中身は Giant Fiber (DNp01)。脳の樹状突起から太い軸索が神経索へ降り、
中脚の運動ニューロンに渡る、という このリポジトリの一番の見どころを
そのまま記号にしてある。

    python scripts/../app/make_icon.py     (画像を作り直すときだけ)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QGuiApplication, QImage, QPainter,
                           QPainterPath, QPen)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import theme  # noqa: E402

HERE = Path(__file__).resolve().parent
SIZE = 512


def draw(size: int = SIZE) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = size / 512.0

    # 下地 — 角を丸めた暗い板
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(theme.PANEL)))
    p.drawRoundedRect(QRectF(0, 0, size, size), 96 * s, 96 * s)
    p.setPen(QPen(QColor(theme.LINE), 6 * s))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3 * s, 3 * s, size - 6 * s, size - 6 * s), 94 * s, 94 * s)

    # 軸索 — 脳 (上) から神経索 (下) へ降りる
    path = QPainterPath(QPointF(196 * s, 128 * s))
    path.cubicTo(QPointF(268 * s, 210 * s), QPointF(236 * s, 300 * s),
                 QPointF(300 * s, 384 * s))
    p.setPen(QPen(QColor(theme.ACCENT), 26 * s, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)

    # 上の樹状突起 (視葉からの入力)
    p.setPen(QPen(QColor(theme.ACCENT), 16 * s, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap))
    for dx, dy in ((-64, -38), (-18, -60), (34, -44)):
        p.drawLine(QPointF(196 * s, 128 * s),
                   QPointF((196 + dx) * s, (128 + dy) * s))

    # 下の終末 — ここで運動ニューロンに渡す
    p.setPen(QPen(QColor(theme.YELLOW), 14 * s, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap))
    for dx, dy in ((-52, 44), (8, 58), (58, 30)):
        p.drawLine(QPointF(300 * s, 384 * s),
                   QPointF((300 + dx) * s, (384 + dy) * s))

    # シナプス
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(theme.YELLOW)))
    for x, y in ((248, 428), (308, 442), (358, 414)):
        p.drawEllipse(QPointF(x * s, y * s), 17 * s, 17 * s)
    p.end()
    return img


def main() -> None:
    QGuiApplication(sys.argv[:1])
    img = draw()
    png = HERE / "icon.png"
    img.save(str(png))
    try:
        from PIL import Image

        Image.open(png).save(HERE / "icon.ico",
                             sizes=[(s, s) for s in (16, 32, 48, 64, 128, 256)])
    except Exception as exc:
        print(f"ico は作れなかった: {exc}")
    print(f"書き出し: {png}")


if __name__ == "__main__":
    main()
