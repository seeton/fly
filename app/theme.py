"""見た目 — 暗い配色と Qt スタイルシート。

色は旧ブラウザ版 (app/static/index.html) と同じものを引き継いでいる。
凡例の色 (赤=選択, 緑=相手, 水色=入力, 橙=出力, 黄=接触点) は
09/23 の動画スクリプトとも揃えてあるので、動かすときは両方直すこと。
"""

from __future__ import annotations

BG = "#0b0d10"
PANEL = "#14171c"
PANEL2 = "#1b1f26"
LINE = "#262b33"
FG = "#e7e9ec"
MUTED = "#878e99"

ACCENT = "#ff5a4d"   # 選択した細胞
GREEN = "#2be06a"    # 相手 / 経路
CYAN = "#22d3ee"     # 入力シナプス
AMBER = "#ff9f0a"    # 出力シナプス
YELLOW = "#ffe600"   # 接触点
BLUE = "#4da3ff"

PATH_COLORS = ("#4da3ff", "#ff5a4d", "#ffb648", "#2be06a", "#22d3ee", "#c58bff")

FONTS = '"Segoe UI Variable Text","Segoe UI","Yu Gothic UI","Meiryo",sans-serif'


def palette():
    """Fusion スタイルに渡す暗いパレット。

    スタイルシートだけだと、名前を付けていないウィジェット (スクロール領域の
    中身など) が既定の明るい色のまま残る。土台はパレットで塗っておく。
    """
    from PySide6.QtGui import QColor, QPalette

    c = QPalette()
    g = QPalette.ColorRole
    c.setColor(g.Window, QColor(BG))
    c.setColor(g.WindowText, QColor(FG))
    c.setColor(g.Base, QColor(PANEL))
    c.setColor(g.AlternateBase, QColor(PANEL2))
    c.setColor(g.Text, QColor(FG))
    c.setColor(g.Button, QColor(PANEL2))
    c.setColor(g.ButtonText, QColor(FG))
    c.setColor(g.ToolTipBase, QColor(PANEL2))
    c.setColor(g.ToolTipText, QColor(FG))
    c.setColor(g.Highlight, QColor(ACCENT))
    c.setColor(g.HighlightedText, QColor("#ffffff"))
    c.setColor(g.PlaceholderText, QColor(MUTED))
    c.setColor(g.Link, QColor(BLUE))
    dis = QPalette.ColorGroup.Disabled
    c.setColor(dis, g.WindowText, QColor(MUTED))
    c.setColor(dis, g.Text, QColor(MUTED))
    c.setColor(dis, g.ButtonText, QColor(MUTED))
    return c


def rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """#rrggbb を OpenGL に渡す 0〜1 の4つ組にする。"""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
            int(h[4:6], 16) / 255, alpha)


QSS = f"""
* {{
  font-family: {FONTS};
  font-size: 13px;
  color: {FG};
}}
QMainWindow, QWidget#root {{ background: {BG}; }}

QWidget#header {{ background: {BG}; border-bottom: 1px solid {LINE}; }}
QLabel#appTitle {{ font-size: 15px; font-weight: 600; letter-spacing: .3px; }}
QLabel#headline {{ color: {MUTED}; font-size: 12px; }}
QLabel#muted, QLabel#hint {{ color: {MUTED}; font-size: 12px; }}
QLabel#big {{ font-size: 21px; font-weight: 600; }}
QLabel#sectionTitle {{
  color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1.2px;
}}
QLabel#count {{ color: {MUTED}; }}

QWidget#sidebar {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
QWidget#toolbar {{ background: {PANEL2}; border-bottom: 1px solid {LINE}; }}
QWidget#infobar {{ background: {PANEL2}; border-top: 1px solid {LINE}; }}
QWidget#card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; }}

QLineEdit {{
  background: {PANEL2}; border: 1px solid {LINE}; border-radius: 8px;
  padding: 7px 10px; selection-background-color: {ACCENT};
}}
QLineEdit:focus {{ border: 1px solid {ACCENT}; }}

QPushButton {{
  background: {PANEL2}; border: 1px solid {LINE}; border-radius: 8px;
  padding: 6px 12px;
}}
QPushButton:hover {{ border-color: #3a414d; background: #222831; }}
QPushButton:pressed {{ background: #2a313b; }}
QPushButton:disabled {{ color: {MUTED}; background: {PANEL}; }}
QPushButton:checked {{
  background: {ACCENT}; border-color: {ACCENT}; color: #ffffff; font-weight: 600;
}}
QPushButton#ghost {{ background: transparent; border-color: transparent; color: {MUTED}; }}
QPushButton#ghost:hover {{ color: {FG}; background: {PANEL2}; }}

QPushButton#segment {{
  background: transparent; border: none; border-radius: 7px;
  padding: 6px 18px; color: {MUTED}; font-weight: 600;
}}
QPushButton#segment:hover {{ color: {FG}; }}
QPushButton#segment:checked {{ background: {PANEL2}; color: {FG}; }}
QWidget#segmentBar {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 9px; }}

QListWidget {{
  background: transparent; border: none; outline: none;
}}
QListWidget::item {{ border-radius: 6px; padding: 0px; }}
QListWidget::item:hover {{ background: {PANEL2}; }}
QListWidget::item:selected {{ background: #2a2320; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #333a44; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #454e5c; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #333a44; border-radius: 5px; min-width: 30px; }}

QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
  background: {FG}; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
}}

QComboBox {{
  background: {PANEL2}; border: 1px solid {LINE}; border-radius: 8px; padding: 5px 10px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
  background: {PANEL}; border: 1px solid {LINE}; selection-background-color: {PANEL2};
  outline: none;
}}

QProgressBar {{
  background: {PANEL2}; border: none; border-radius: 3px; height: 6px; text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QPlainTextEdit {{
  background: {BG}; border: 1px solid {LINE}; border-radius: 8px;
  font-family: "Cascadia Mono","Consolas",monospace; font-size: 12px; color: {MUTED};
}}
QToolTip {{
  background: {PANEL2}; color: {FG}; border: 1px solid {LINE};
  border-radius: 6px; padding: 5px 8px;
}}
QSplitter::handle {{ background: {LINE}; }}
QDialog {{ background: {BG}; }}
"""
