"""五目並べのタブ — 勝ち負けだけで覚えた蝿と打つ。

盤は 15x15、五目で勝ち (自由形)。打ち手は2通りあり、ある方を使う:

  網 (`out/gomoku_net.pt`)      3x3 の畳み込みの網を、自分と打って勝ち負けだけから
                                覚えさせたもの (`scripts/32_train_gomoku_net.py`)。
                                その目で候補を選び、局面を見込みで評価しながら読む
  前の打ち手 (`gomoku_policy.json`)  重み 159 個を CMA-ES で探したもの
                                (`scripts/30_train_gomoku.py`)。torch が無いときはこちら

どちらも**定石は一つも入っていない**。読みに入る規則は「五で勝ち」「石のある所には
打てない」だけ。

**盤の色は蝿の目そのもの。** 網なら「読む前に目だけで見た手の確からしさ」、
前の打ち手なら全マスにつけた点数。こちらで作った色付けではない (脳の点群を実入力
だけから光らせているのと同じ)。単位が無いので、局面ごとの上位3割を伸ばして塗る。

**読みは別のスレッドで回す。** 網は1手に 0.5〜1 秒読むので、画面のスレッドで
回すとその間固まる (`CLAUDE.md`: 重い処理は UI スレッドで呼ばない)。

**コネクトームとは別物**。ここで動いているのは学習した重みで、ハエの神経回路
そのものではない。回路タブや飛翔タブと混ぜて読まないよう、画面にも断っておく。
"""

from __future__ import annotations

import json
import sys

import numpy as np
from PySide6.QtCore import (QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool,
                            QTimer, Signal, Slot)
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
N = 15


# ---------------------------------------------------------------- 打ち手

class NetBrain:
    """畳み込みの網 + 読み。学習と同じ `gomoku_net.Player` で打つ。"""

    kind = "網"

    def __init__(self) -> None:
        import torch
        import gomoku_net as M

        # 読みは1局面ずつ網に通す。スレッドを何本も立てると、CPU が混んでいるとき
        # 取り合いで 25 倍遅くなった (1局面 4 ms → 104 ms)
        torch.set_num_threads(1)
        self._torch, self._M = torch, M
        self._mtime = 0.0
        self.net = self.player = None
        self.refresh()

    def refresh(self) -> bool:
        """網が書き換わっていたら読み直す。新しい対局のたびに呼ぶ。

        学習 (`32_train_gomoku_net.py`) を裏で回したまま遊べるように。学習側は
        置き換えで書くので、書きかけを読むことはない。
        """
        path = OUT / "gomoku_net.pt"
        m = path.stat().st_mtime
        if m == self._mtime:
            return False
        ck = self._torch.load(path, weights_only=False)
        net = self._M.FlyNet(**ck["config"]).eval()
        net.load_state_dict(ck["net"])
        meta_p = OUT / "gomoku_net.json"
        self.meta = json.loads(meta_p.read_text("utf-8")) if meta_p.exists() else {}
        self.net = net
        # 読む回数は、物差しで測ったときと同じにする (強さの数字と合わせるため)
        self.player = self._M.Player(net, sims=self.meta.get("eval_sims", 200))
        self._mtime = m
        return True

    def heat(self, board):
        return self.player.prior(board)

    def move(self, board, rng):
        return self.player.move(board, rng)

    def plan(self, board) -> int:
        return -1          # 読むのは重いので、打つ前の「ここに打つ」は出さない

    def after(self, board) -> list[str]:
        t = self.player.last_tree
        if t is None:
            return []
        vis = t.visits()
        # 同点の並びは argmax (打った手) と同じく、先に出てくるマスを上にする
        order = np.argsort(-vis, kind="stable")[:3]
        r = t.root
        v = float(r.W.sum() / max(r.N.sum(), 1))
        rows = [f"{int(i) // N + 1} 段 {int(i) % N + 1} 列 ({int(vis[i])} 回)"
                for i in order if vis[i] > 0]
        return [f"読んだ回数の上位: " + " / ".join(rows),
                f"打つ前の見込み (蝿から見て): {v:+.2f}  (+1 勝ち / -1 負け)"]

    def note(self) -> str:
        m = self.meta
        cfg = m.get("config", {})
        nw = sum(p.numel() for p in self.net.parameters())
        head = (f"打ち手: 3x3 の畳み込み {1 + 2 * cfg.get('blocks', 0)} 層 "
                f"({cfg.get('ch', '?')} ch、重み {nw} 個) を、自分と {m.get('games_total', 0)} 局"
                f"打って覚えさせたもの ({m.get('hours', 0):.1f} 時間)。"
                f"1手ごとに {self.player.sims} 回読む。\n\n")
        return head + _scoreboard(m.get("scoreboard")) + (
            "教えたのは勝ったか負けたかだけ。定石も、形の良し悪しも、「四は止めろ」も"
            "書いていない。読みに入る規則は「五で勝ち」「石のある所には打てない」だけ。\n\n"
            "盤の色は、網が読む前に目だけで見た手の確からしさ。打ったあとに、読んだ末に"
            "どこを何回読んだかを出す。\n\n")


class OldBrain:
    """前の打ち手 (重み 159 個)。torch が無いときはこちらで打つ。"""

    kind = "前の打ち手"

    def __init__(self) -> None:
        import gomoku_env as G

        saved = json.loads((OUT / "gomoku_policy.json").read_text("utf-8"))
        self.meta = saved
        # 読みの深さも解の json から取る。**学習したときと同じ深さで打つ**
        self.fly = G.FlyPlayer(saved.get("K", 6), depth=saved.get("depth", 0)
                               ).set_flat(np.array(saved["x"]))

    def heat(self, board):
        return self.fly.scores(board)

    def move(self, board, rng):
        return self.fly.move(board, rng)

    def plan(self, board) -> int:
        return self.fly.move(board)

    def refresh(self) -> bool:
        return False

    def after(self, board) -> list[str]:
        return []

    def note(self) -> str:
        m = self.meta
        return (f"打ち手: 重み {m.get('dim', '?')} 個を CMA-ES で決めたもの "
                f"(「{m.get('label', '?')}」の段)。読み {m.get('depth', 0)} 手。\n\n"
                + _scoreboard(m.get("scoreboard")) +
                "渡した報酬は「打てた」「止めた」「生き延びた」「勝った」だけで、"
                "どこが良い手かは教えていない。\n\n")


def _scoreboard(sb) -> str:
    """学習のときに測った強さをそのまま出す (文章で書くと学習し直したとき先に古くなる)。"""
    if not sb:
        return ""
    rows = [f"{k}: 勝ち {v['win']*100:.0f} % / 引き分け {v['draw']*100:.0f} %"
            for k, v in sb.items() if isinstance(v, dict)]
    if not rows:
        return ""
    return f"実測 (各 {sb.get('games', '?')} 局、先後半々):\n  " + "\n  ".join(rows) + "\n\n"


def load_brain():
    """使える打ち手を返す。網を優先し、無ければ前の打ち手。どちらも無ければ理由。"""
    try:
        import gomoku_env  # noqa: F401
    except Exception as exc:
        return None, f"scripts/gomoku_env.py を読み込めません ({exc})。"
    if (OUT / "gomoku_net.pt").exists():
        try:
            return NetBrain(), ""
        except Exception as exc:          # torch が無い (固めた exe など)
            fallback = f"(網は使えない: {exc})"
        else:
            fallback = ""
    else:
        fallback = ""
    if (OUT / "gomoku_policy.json").exists():
        return OldBrain(), fallback
    return None, ("学習した打ち手がありません。\n"
                  "    .\\.venv\\Scripts\\python.exe scripts\\32_train_gomoku_net.py")


class _MoveSignals(QObject):
    done = Signal(int, int)        # (対局の通し番号, 打つマス)


class _MoveJob(QRunnable):
    """蝿の読みを別スレッドで回す。

    `desktop.Job` と同じ理由で、配り終えるまで Python 側で握っておく
    (握らないと signals ごと回収され、結果が黙って捨てられる)。
    """

    def __init__(self, brain, board, rng, game_id: int):
        super().__init__()
        self.signals = _MoveSignals()
        self.brain, self.board, self.rng, self.game_id = brain, board, rng, game_id

    @Slot()
    def run(self) -> None:
        mv = int(self.brain.move(self.board, self.rng))
        try:
            self.signals.done.emit(self.game_id, mv)
        except RuntimeError:
            pass                       # 窓を閉じたあと


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
        self.setMinimumSize(560, 560)
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

        # 蝿の点数。**上位だけ**塗る。全部を最小〜最大に伸ばすと盤全体が
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
        self.brain, msg = load_brain()
        self.ok = self.brain is not None
        import gomoku_env as G
        self.G = G
        self.n = N

        self.board = np.zeros((self.n, self.n), dtype=np.int8)
        self.rng = np.random.default_rng()
        self.turn = HUMAN
        self.over = True
        self.human_first = True
        self.game_id = 0
        self._job = None                # 読んでいる最中の仕事 (握っておく)
        self.pool = QThreadPool.globalInstance()

        self.view = BoardView(self.n)
        self.view.clicked.connect(self._human_move)

        # --- 右の帯 ---
        side = QWidget()
        side.setFixedWidth(340)
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

        # 目の色を出すと、蝿がどこを良いと見ているかが打つ前から分かってしまう。
        # このアプリは中を見せる側なので既定は入、対等に打ちたい人は切る
        self.heat = QCheckBox("蝿の目に見えているものを重ねる")
        self.heat.setChecked(True)
        self.heat.stateChanged.connect(self._refresh)
        sl.addWidget(self.heat)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(f"color:{theme.FG};font-size:12px;")
        sl.addWidget(self.detail)

        self.note = QLabel(self._note())
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color:{theme.MUTED};font-size:11px;")
        sl.addWidget(self.note)
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
            if msg:
                self.status.emit(msg, False)
            self.new_game(True)

    # --- 説明 ---
    def _note(self) -> str:
        if not self.ok:
            return ""
        return (self.brain.note() +
                "色の濃さは局面ごとの上位3割を伸ばした相対値 (単位が無いので)。\n\n"
                "これは学習した重みで、コネクトームの回路ではない。")

    # --- 対局 ---
    def new_game(self, human_first: bool) -> None:
        if not self.ok:
            return
        self.game_id += 1              # 読んでいる途中の古い対局の答えは捨てる
        # 学習を裏で回していれば網は書き換わっている。対局の頭で最新を読み直す
        if self.brain.refresh():
            self.note.setText(self._note())
        self.human_first = human_first
        self.b_first.setChecked(human_first)
        self.b_second.setChecked(not human_first)
        self.board[:] = 0
        self.rng = np.random.default_rng()
        self.over = False
        self.view.last = -1
        self.view.line = ()
        self.turn = HUMAN if human_first else FLY
        self.verdict.setText("あなたの番" if human_first else "蝿の番")
        self.detail.setText("")
        self._refresh()
        if not human_first:
            QTimer.singleShot(200, self._fly_move)

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
        self._refresh()
        QTimer.singleShot(150, self._fly_move)

    def _fly_move(self) -> None:
        if self.over or self.turn != FLY:
            return
        self.verdict.setText("蝿が読んでいる…")
        job = _MoveJob(self.brain, self.board.copy(), self.rng, self.game_id)
        job.signals.done.connect(self._fly_done)
        self._job = job
        self.pool.start(job)

    def _fly_done(self, game_id: int, idx: int) -> None:
        self._job = None
        if game_id != self.game_id or self.over or self.turn != FLY:
            return                     # その間に新しい対局が始まった
        y, x = divmod(idx, self.n)
        if self.board[y, x] != 0:
            # お手つき。前の打ち手は読みなしだと起こりうる。学習と同じくその場で負け
            self.over = True
            self.verdict.setText("蝿のお手つき (石のあるマスを選んだ) — あなたの勝ち")
            self.status.emit("対局が終わりました", False)
            self._refresh()
            return
        self.board[y, x] = FLY
        self.view.last = idx
        after = self.brain.after(self.board)
        if self._check():
            return
        self.turn = HUMAN
        self.verdict.setText("あなたの番")
        self._refresh(after)

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
    def _refresh(self, after: list | None = None) -> None:
        if not self.ok:
            return
        show = self.heat.isChecked()
        s = self.brain.heat(self.board) if show else None
        self.view.board = self.board
        self.view.scores = s
        # 点線の丸は「読んだ末に打つマス」。網は読みが重いので出さない
        self.view.best = (self.brain.plan(self.board)
                          if (show and not self.over and self.turn == HUMAN) else -1)
        self.view.update()
        if not show:
            self.detail.setText("")
            return
        top = int(np.argmax(s))
        lines = [f"目がいちばん良いと見ているマス: {top // self.n + 1} 段 {top % self.n + 1} 列"]
        if after:
            lines += after
        self.detail.setText("\n".join(lines))
