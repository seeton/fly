"""五目並べのタブ — 報酬だけで覚えた蝿と打つ。

盤は 9x9、五目で勝ち。蝿は `scripts/gomoku_env.py` の打ち手で、重み 159 個を
`scripts/30_train_gomoku.py` が CMA-ES で決めたもの。**定石は一つも入っていない**。
渡した報酬は「打てた」「並べた」「止めた」「勝った」の4つだけで、どこが良い手か
は一度も教えていない。

**盤の点数をそのまま出す。** 右に出しているのは蝿が 81 マス全部につけた点数で、
こちらで作った色付けではない (脳の点群を実入力だけから光らせているのと同じ)。
石のあるマスにも点数は出る — そこを選ばない理由も報酬から覚えたものなので、
隠さずに見せる。

**物差しはこの局面の中の相対**。点数に単位はなく、翅や脚の速さのように測って
決めた固定の物差しが無い。局面ごとに最小〜最大で割っていることを画面にも書く。

**コネクトームとは別物**。ここで動いているのは学習した重みで、ハエの神経回路
そのものではない。回路タブや飛翔タブと混ぜて読まないよう、画面にも断っておく。
"""

from __future__ import annotations

import json
import sys

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from . import theme
from .paths import OUT, SCRIPTS

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

FLY = 1          # 盤の値。蝿は +1、人は -1 (蝿の打ち手から見た符号に合わせる)
HUMAN = -1
FLY_COLOR = theme.AMBER
HUMAN_COLOR = theme.CYAN


def available() -> tuple[bool, str]:
    """この環境で遊べるか。無理なら理由をそのまま画面に出す。"""
    try:
        import gomoku_env  # noqa: F401
    except Exception as exc:
        return False, f"scripts/gomoku_env.py を読み込めません ({exc})。"
    if not (OUT / "gomoku_policy.json").exists():
        return False, ("out/gomoku_policy.json がありません (学習した打ち方)。\n"
                       "    .\\.venv\\Scripts\\python.exe scripts\\30_train_gomoku.py")
    return True, ""


# ---------------------------------------------------------------- 盤

class BoardView(QWidget):
    """盤そのもの。石と、蝿がつけた点数を重ねて描く。"""

    clicked = Signal(int)

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.board = np.zeros((n, n), dtype=np.int8)
        self.scores = None            # (n*n,) 蝿の点数。None なら出さない
        self.last = -1                # 最後に打たれたマス
        self.best = -1                # 蝿がいま打つなら選ぶマス
        self.line = ()                # 勝った五目の並び
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    # --- 座標 ---
    def _geom(self):
        side = min(self.width(), self.height())
        cell = side / (self.n + 1.0)
        ox = (self.width() - cell * self.n) / 2 + cell / 2
        oy = (self.height() - cell * self.n) / 2 + cell / 2
        return cell, ox, oy

    def _center(self, y: int, x: int) -> QPointF:
        cell, ox, oy = self._geom()
        return QPointF(ox + x * cell, oy + y * cell)

    def mousePressEvent(self, ev) -> None:
        cell, ox, oy = self._geom()
        x = int(round((ev.position().x() - ox) / cell))
        y = int(round((ev.position().y() - oy) / cell))
        if 0 <= x < self.n and 0 <= y < self.n:
            self.clicked.emit(y * self.n + x)

    # --- 描画 ---
    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cell, ox, oy = self._geom()
        n = self.n

        p.fillRect(self.rect(), QColor(theme.BG))
        board_rect = QRectF(ox - cell * 0.7, oy - cell * 0.7,
                            cell * (n - 1 + 1.4), cell * (n - 1 + 1.4))
        p.setBrush(QColor(theme.PANEL))
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.drawRoundedRect(board_rect, 8, 8)

        # 蝿の点数。**上位だけ**塗る。全部を最小〜最大に伸ばすと 81 マスが
        # 軒並み色づいて、どこを高く見ているのか読めなくなる (一度そうなった)。
        # 下限は中央値より上の 70 パーセンタイルに取る
        if self.scores is not None:
            s = np.asarray(self.scores, dtype=float)
            lo, hi = float(np.percentile(s, 70)), float(s.max())
            v = np.clip((s - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            warm = QColor(FLY_COLOR)
            p.setClipRect(board_rect)     # 端のマスが盤からはみ出さないように
            for i, val in enumerate(v):
                y, x = divmod(i, n)
                c = self._center(y, x)
                col = QColor(warm)
                col.setAlpha(int(140 * val ** 1.6))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col)
                p.drawRect(QRectF(c.x() - cell / 2, c.y() - cell / 2, cell, cell))
            p.setClipping(False)

        # 格子
        p.setPen(QPen(QColor(theme.LINE), 1))
        for i in range(n):
            a, b = self._center(i, 0), self._center(i, n - 1)
            p.drawLine(a, b)
            a, b = self._center(0, i), self._center(n - 1, i)
            p.drawLine(a, b)

        # 蝿がいま打つなら選ぶマス (打つ前から出す)
        if self.best >= 0 and self.board.ravel()[self.best] == 0:
            y, x = divmod(self.best, n)
            c = self._center(y, x)
            pen = QPen(QColor(FLY_COLOR), 1.4, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, cell * 0.40, cell * 0.40)

        # 石
        r = cell * 0.40
        for i, val in enumerate(self.board.ravel()):
            if val == 0:
                continue
            y, x = divmod(i, n)
            c = self._center(y, x)
            col = QColor(FLY_COLOR if val == FLY else HUMAN_COLOR)
            p.setBrush(col)
            p.setPen(QPen(col.darker(160), 1.2))
            p.drawEllipse(c, r, r)
            if i == self.last:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(QColor(theme.FG), 1.6))
                p.drawEllipse(c, r * 0.45, r * 0.45)

        # 勝った並び
        if self.line:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(theme.FG), 2.2))
            for y, x in self.line:
                p.drawEllipse(self._center(y, x), r * 1.15, r * 1.15)
        p.end()


# ---------------------------------------------------------------- ページ

class GomokuPage(QWidget):
    status = Signal(str, bool)

    def __init__(self) -> None:
        super().__init__()
        self.ok, msg = available()
        self.fly = None
        self.meta = {}
        if self.ok:
            import gomoku_env as G

            self.G = G
            saved = json.loads((OUT / "gomoku_policy.json").read_text("utf-8"))
            self.meta = saved
            self.fly = G.FlyPlayer(saved.get("K", 6)).set_flat(np.array(saved["x"]))
            self.n = G.N
        else:
            self.G = None
            self.n = 9

        self.board = np.zeros((self.n, self.n), dtype=np.int8)
        self.rng = np.random.default_rng()
        self.turn = HUMAN
        self.over = True
        self.fumbles = 0
        self.human_first = True

        self.view = BoardView(self.n)
        self.view.clicked.connect(self._human_move)

        # --- 右の帯 ---
        side = QWidget()
        side.setFixedWidth(330)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(16, 14, 16, 14)
        sl.setSpacing(10)

        head = QLabel("蝿と五目並べ")
        head.setStyleSheet("font-size:17px;font-weight:700;")
        sl.addWidget(head)

        self.verdict = QLabel("")
        self.verdict.setStyleSheet("font-size:14px;font-weight:600;")
        self.verdict.setWordWrap(True)
        sl.addWidget(self.verdict)

        btns = QHBoxLayout()
        self.b_new = QPushButton("新しい対局")
        self.b_new.clicked.connect(lambda: self.new_game(self.human_first))
        btns.addWidget(self.b_new)
        self.b_first = QPushButton("先手")
        self.b_second = QPushButton("後手")
        grp = QButtonGroup(self)
        for b, first in ((self.b_first, True), (self.b_second, False)):
            b.setCheckable(True)
            b.setObjectName("segment")
            grp.addButton(b)
            b.clicked.connect(lambda _=False, f=first: self.new_game(f))
            btns.addWidget(b)
        self.b_first.setChecked(True)
        sl.addLayout(btns)

        # 点数を出すと、蝿が次にどこへ打つかが打つ前から分かってしまう。
        # このアプリは中を見せる側なので既定は入、対等に打ちたい人は切る
        self.heat = QCheckBox("蝿がつけた点数を見せる (次の手も分かる)")
        self.heat.setChecked(True)
        self.heat.stateChanged.connect(self._refresh)
        sl.addWidget(self.heat)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(f"color:{theme.FG};font-size:12px;")
        sl.addWidget(self.detail)

        note = QLabel(self._note())
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{theme.MUTED};font-size:11px;")
        sl.addWidget(note)
        sl.addStretch(1)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.view, 1)
        lay.addWidget(side)

        if not self.ok:
            self.verdict.setText("遊べません")
            self.detail.setText(msg)
            self.view.setEnabled(False)
        else:
            self.new_game(True)

    # --- 説明 ---
    def _note(self) -> str:
        if not self.ok:
            return ""
        m = self.meta
        sb = m.get("scoreboard")
        # 強さは文章で書かず、学習のときに測った値をそのまま出す
        # (打ち手を学習し直したら文章の方が先に古くなる)
        strength = ""
        if sb:
            strength = (
                f"実測 ({sb['games']} 局): 合法手 {sb['legal']*100:.1f} %、"
                f"ランダムな相手に {sb['win_random']*100:.0f} % 勝ち、"
                f"連を数える素朴な相手には {sb['win_greedy']*100:.1f} % しか勝てない。"
                "先読みを一切していないので、人が相手だとまず勝てない。\n\n")
        return (
            f"覚えたもの: 重み {m.get('dim', '?')} 個を CMA-ES で決めた "
            f"(採用したのは stage「{m.get('label', '?')}」の解)。"
            "定石は一つも入っていない。\n\n"
            + strength +
            "渡した報酬は「打てた」「並べた」「止めた」「勝った」の4つだけで、"
            "どこが良い手かは教えていない。合法手すら教えていないので、"
            "石のあるマスを選べば空振りする (お手つき)。\n\n"
            "盤の見方は、そのマスを通る4本の線の前後4マスを見る受容野。"
            "重みは4方向で共有してある — 縦の三連も斜めの三連も同じ形なので、"
            "個眼が視野のどこでも同じ受容野を持っているのと同じ作りにした。\n\n"
            "色の濃さは蝿がつけた点数そのもの。単位が無いので、"
            "この局面の上位3割 (70 パーセンタイル〜最大) を伸ばした相対値。\n\n"
            "これは学習した重みで、コネクトームの回路ではない。")

    # --- 対局 ---
    def new_game(self, human_first: bool) -> None:
        if not self.ok:
            return
        self.human_first = human_first
        self.b_first.setChecked(human_first)
        self.b_second.setChecked(not human_first)
        self.board[:] = 0
        self.rng = np.random.default_rng()
        self.over = False
        self.fumbles = 0
        self.view.last = -1
        self.view.line = ()
        self.turn = HUMAN if human_first else FLY
        self.verdict.setText("あなたの番" if human_first else "蝿の番")
        self._refresh()
        if not human_first:
            QTimer.singleShot(350, self._fly_move)

    def _human_move(self, idx: int) -> None:
        if self.over or self.turn != HUMAN:
            return
        y, x = divmod(idx, self.n)
        if self.board[y, x] != 0:
            return
        self.board[y, x] = HUMAN
        self.view.last = idx
        if self._check():
            return
        self.turn = FLY
        self.verdict.setText("蝿の番")
        self._refresh()
        QTimer.singleShot(350, self._fly_move)

    def _fly_move(self) -> None:
        if self.over or self.turn != FLY:
            return
        s = self.fly.scores(self.board)
        idx = int(np.argmax(s + self.rng.normal(0, 1e-6, s.size)))
        y, x = divmod(idx, self.n)
        if self.board[y, x] != 0:
            # お手つき。合法手を教えていないので起こりうる。
            # 盤は変わらないまま手番だけ移る (学習のときと同じ扱い)
            self.fumbles += 1
            self.turn = HUMAN
            self.verdict.setText("蝿がお手つき — あなたの番")
            self._refresh()
            return
        self.board[y, x] = FLY
        self.view.last = idx
        if self._check():
            return
        self.turn = HUMAN
        self.verdict.setText("あなたの番")
        self._refresh()

    def _check(self) -> bool:
        """勝敗がついたか。ついたら画面を締める。"""
        w = self.G.winner(self.board)
        if w != 0:
            self.over = True
            self.view.line = self._line(w)
            self.verdict.setText("蝿の勝ち" if w == FLY else "あなたの勝ち")
            self.status.emit("対局が終わりました", False)
            self._refresh()
            return True
        if (self.board == 0).sum() == 0:
            self.over = True
            self.verdict.setText("引き分け (盤が埋まった)")
            self._refresh()
            return True
        return False

    def _line(self, who: int) -> tuple:
        """勝った五目の並びを探す (印をつけるため)。"""
        n, W = self.n, self.G.WIN
        for y in range(n):
            for x in range(n):
                for dy, dx in self.G.DIRS:
                    ys, xs = y + dy * (W - 1), x + dx * (W - 1)
                    if not (0 <= ys < n and 0 <= xs < n):
                        continue
                    cells = [(y + dy * k, x + dx * k) for k in range(W)]
                    if all(self.board[a, b] == who for a, b in cells):
                        return tuple(cells)
        return ()

    # --- 画面 ---
    def _refresh(self) -> None:
        if not self.ok:
            return
        s = self.fly.scores(self.board)
        show = self.heat.isChecked()
        self.view.board = self.board
        self.view.scores = s if show else None
        self.view.best = int(np.argmax(s)) if (show and not self.over) else -1
        self.view.update()

        if not show:
            self.detail.setText(f"この対局のお手つき {self.fumbles} 回")
            return
        order = np.argsort(s)[::-1]
        top = order[0]
        gap = float(s[order[0]] - s[order[1]])
        y, x = divmod(int(top), self.n)
        legal = "空いている" if self.board[y, x] == 0 else "石がある (お手つきになる)"
        self.detail.setText(
            f"蝿がいちばん高く見ているマス: {y+1} 段 {x+1} 列 — {legal}\n"
            f"点数 {s[top]:+.3f}  (2位との差 {gap:.3f})\n"
            f"この対局のお手つき {self.fumbles} 回")
