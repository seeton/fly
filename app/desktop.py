"""ハエ脳ビューア — デスクトップアプリ本体 (PySide6)。

もとはローカルに uvicorn を立ててブラウザで見る形だった。HTTP と Plotly を
やめて Qt のウィンドウにしてある。利点は3つ:

  * 21万ボディ / 1億5185万接続 をプロセス内に持ち、問い合わせが関数呼び出しになる
  * シナプス点群を OpenGL に直接渡せる (数万点でも回る)
  * 動画が QtMultimedia でそのまま再生できる (ループ・こま送り・速度)

重い処理 (起動時の読み込み 20秒、シナプス走査 2〜3秒) は QThreadPool に逃がし、
ウィンドウは最初から出して進捗を見せる。

起動:
  .venv\\Scripts\\python.exe -m app
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import (QObject, QProcess, QProcessEnvironment, QRunnable,
                            Qt, QThreadPool, QTimer, Signal, Slot)
from PySide6.QtGui import (QFont, QFontMetrics, QGuiApplication, QIcon,
                           QPixmap)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy,
                               QSplitter, QStackedWidget, QVBoxLayout, QWidget)

from . import theme, videos
from .core import Brain, NotFound
from .gomoku import GomokuPage
from .livepage import LivePage
from .player import VideoPlayer
from .view3d import Brain3DView, Layer

from .paths import PYEXE, ROOT


# ---------------------------------------------------------------- 下働き

class JobSignals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str, float)


class Job(QRunnable):
    """関数1つをスレッドプールで回して結果をシグナルで返す。

    **必ず `job.start(pool)` で投げること。** `QThreadPool.start()` に渡しただけだと
    Python 側の Job が回収され、道連れに `signals` (QObject) も消える。別スレッドから
    のシグナルはキューに積まれてから配達されるので、その前に受け皿が消えると
    結果が黙って捨てられ、画面が「読み込み中」のまま止まる。実行中は `_alive` で
    握っておき、結果を配り終えてから放す。
    """

    _alive: set["Job"] = set()

    def __init__(self, fn, *args, with_progress: bool = False, **kwargs) -> None:
        super().__init__()
        self.signals = JobSignals()
        if with_progress:
            # 進捗はシグナル経由で返す。ワーカースレッドから UI を直接触らないため。
            kwargs["progress"] = self.signals.progress.emit
        self._fn, self._args, self._kwargs = fn, args, kwargs

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # UI を落とさず、そのままステータスへ出す
            self._emit(self.signals.failed, str(exc))
        else:
            self._emit(self.signals.done, result)

    def start(self, pool: QThreadPool) -> None:
        Job._alive.add(self)
        # 呼び出し側のハンドラを繋いだ後にここを繋ぐので、解放は配達の後になる。
        self.signals.done.connect(lambda *_: Job._alive.discard(self))
        self.signals.failed.connect(lambda *_: Job._alive.discard(self))
        pool.start(self)

    @staticmethod
    def _emit(signal, payload) -> None:
        try:
            signal.emit(payload)
        except RuntimeError:
            pass  # 読み込み中に窓を閉じた場合。受け手はもう居ない。


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{theme.LINE};background:{theme.LINE};max-height:1px;")
    return f


def label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


def tag(text: str, color: str = theme.MUTED) -> QLabel:
    lb = QLabel(text)
    lb.setStyleSheet(
        f"color:{color};background:{theme.PANEL2};border:1px solid {theme.LINE};"
        "border-radius:4px;padding:1px 7px;font-size:11px;")
    lb.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return lb


def row(left: str, right: str = "", sub: str = "", color: str = theme.FG,
        right_color: str = theme.MUTED) -> QWidget:
    """一覧の1行 — 左に名前、右に数。`sub` があれば下に小さく添える。"""
    w = QWidget()
    outer = QVBoxLayout(w)
    outer.setContentsMargins(8, 5, 8, 5)
    outer.setSpacing(1)
    top = QHBoxLayout()
    top.setContentsMargins(0, 0, 0, 0)
    lb = QLabel(left)
    lb.setStyleSheet(f"color:{color};")
    top.addWidget(lb)
    top.addStretch(1)
    if right:
        rb = QLabel(right)
        rb.setStyleSheet(f"color:{right_color};font-size:12px;")
        top.addWidget(rb)
    outer.addLayout(top)
    if sub:
        sb = QLabel(sub)          # 色を混ぜたいときは sub に HTML を渡す
        sb.setStyleSheet(f"color:{theme.MUTED};font-size:11px;")
        outer.addWidget(sb)
    return w


def kv(key: str, value: str) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 1, 0, 1)
    k = QLabel(key)
    k.setStyleSheet(f"color:{theme.MUTED};font-size:12px;")
    v = QLabel(value)
    v.setStyleSheet("font-weight:600;font-size:12px;")
    lay.addWidget(k)
    lay.addStretch(1)
    lay.addWidget(v)
    return w


def clear_layout(lay) -> None:
    while lay.count():
        item = lay.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


# ---------------------------------------------------------------- 回路ページ

class CircuitPage(QWidget):
    """細胞型を探して、骨格とシナプスを 3D で見る。"""

    status = Signal(str, bool)   # (文言, 処理中か)

    def __init__(self, brain: Brain, pool: QThreadPool) -> None:
        super().__init__()
        self.brain, self.pool = brain, pool
        self.type: str | None = None
        self.detail = None
        self._token = 0

        # --- 左: 検索と詳細 ---
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("細胞型を検索  (DNp01, TTMn, LC4, pC1 …)")
        self.search_box.setClearButtonEnabled(True)
        self.results = QListWidget()
        self.results.setMinimumHeight(120)
        self.results.itemClicked.connect(
            lambda it: self.select(it.data(Qt.ItemDataRole.UserRole)))

        self.detail_box = QVBoxLayout()
        self.detail_box.setSpacing(6)
        self.detail_box.addStretch(1)
        holder = QWidget()
        holder.setLayout(self.detail_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        scroll.setViewportMargins(0, 0, 8, 0)   # 右端の数がスクロールバーに隠れないように

        top_box = QWidget()
        tl = QVBoxLayout(top_box)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(6)
        tl.addWidget(label("検索結果", "sectionTitle"))
        tl.addWidget(self.results, 1)

        inner = QSplitter(Qt.Orientation.Vertical)
        inner.addWidget(top_box)
        inner.addWidget(scroll)
        inner.setSizes([250, 520])
        inner.setChildrenCollapsible(False)

        side = QWidget()
        side.setObjectName("sidebar")
        sl = QVBoxLayout(side)
        sl.setContentsMargins(12, 12, 8, 12)
        sl.setSpacing(8)
        sl.addWidget(self.search_box)
        sl.addWidget(inner, 1)
        side.setMinimumWidth(310)
        side.setMaximumWidth(400)

        # --- 右上: 操作 ---
        self.b_skel = self._toggle("骨格", True)
        self.b_syn = self._toggle("シナプス", False)
        self.b_contact = self._toggle("接触点", False)
        self.partner = QLineEdit()
        self.partner.setPlaceholderText("接触の相手 (例 TTMn)")
        self.partner.setFixedWidth(170)
        self.partner.returnPressed.connect(self.redraw)

        self.b_motor = QPushButton("運動ニューロンまで何段？")
        self.b_motor.clicked.connect(self.motor_reach)
        self.dst = QLineEdit()
        self.dst.setPlaceholderText("経路の行き先")
        self.dst.setFixedWidth(140)
        self.dst.returnPressed.connect(self.find_path)
        self.b_path = QPushButton("経路探索")
        self.b_path.clicked.connect(self.find_path)
        b_home = QPushButton("視点を戻す")
        b_home.setObjectName("ghost")

        bar = QWidget()
        bar.setObjectName("toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(12, 8, 12, 8)
        bl.setSpacing(7)
        for w in (self.b_skel, self.b_syn, self.b_contact, self.partner):
            bl.addWidget(w)
        bl.addSpacing(10)
        bl.addWidget(self.b_motor)
        bl.addWidget(self.dst)
        bl.addWidget(self.b_path)
        bl.addStretch(1)
        bl.addWidget(b_home)

        # --- 凡例 ---
        legend = QWidget()
        ll = QHBoxLayout(legend)
        ll.setContentsMargins(14, 6, 14, 6)
        ll.setSpacing(16)
        for color, text in ((theme.ACCENT, "選択した細胞"), (theme.GREEN, "相手 / 経路"),
                            (theme.CYAN, "入力シナプス"), (theme.AMBER, "出力シナプス"),
                            (theme.YELLOW, "接触点")):
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color};font-size:10px;")
            txt = QLabel(text)
            txt.setStyleSheet(f"color:{theme.MUTED};font-size:11px;")
            ll.addWidget(dot)
            ll.addWidget(txt)
        ll.addStretch(1)

        self.view = Brain3DView()
        b_home.clicked.connect(self.view.reset_camera)

        self.info = label("細胞型を選ぶと 3D で表示します。", "muted", wrap=True)
        info_box = QWidget()
        info_box.setObjectName("infobar")
        il = QVBoxLayout(info_box)
        il.setContentsMargins(14, 8, 14, 8)
        il.addWidget(self.info)
        info_box.setMinimumHeight(52)
        info_box.setMaximumHeight(110)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(bar)
        rl.addWidget(legend)
        rl.addWidget(self.view, 1)
        rl.addWidget(info_box)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(side)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        for b in (self.b_motor, self.b_path):
            b.setEnabled(False)          # 接続 (1GB) が来てから使えるようになる
            b.setToolTip("接続を読み込み中…")

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(180)
        self._timer.timeout.connect(lambda: self.do_search(self.search_box.text()))
        self.search_box.textChanged.connect(lambda _: self._timer.start())

        # 描き直しはまとめる。ボタンを続けて押されたぶんだけ 6.8GB を走査すると、
        # 捨てるだけの結果のために何十秒も食う。
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(140)
        self._redraw_timer.timeout.connect(self._do_redraw)
        self._pending_reset = False

    def _toggle(self, text: str, on: bool) -> QPushButton:
        b = QPushButton(text)
        b.setCheckable(True)
        b.setChecked(on)
        b.toggled.connect(lambda _: self.redraw())
        return b

    # ---- 起動後のはじめの表示 ----

    def start(self) -> None:
        self.do_search("")
        self.select("DNp01")

    # ---- 検索 ----

    def do_search(self, q: str) -> None:
        if not self.brain.ready:
            return
        job = Job(self.brain.search, q)
        job.signals.done.connect(self._fill_results)
        job.signals.failed.connect(lambda m: self.status.emit(m, False))
        job.start(self.pool)

    def _fill_results(self, res) -> None:
        rows, total = res
        self.results.clear()
        for r in rows:
            it = QListWidgetItem()
            w = row(str(r["type"]), f"{r['n']}", str(r["superclass"] or ""))
            it.setSizeHint(w.sizeHint())
            it.setData(Qt.ItemDataRole.UserRole, str(r["type"]))
            self.results.addItem(it)
            self.results.setItemWidget(it, w)
        if total > len(rows):
            self.status.emit(f"{total} 型中 {len(rows)} 件を表示", False)

    # ---- 細胞型を選ぶ ----

    def select(self, type_name: str) -> None:
        if not type_name or not self.brain.ready:
            return
        self.type = type_name
        self.status.emit(f"{type_name} を読み込み中", True)
        job = Job(self.brain.type_detail, type_name)
        job.signals.done.connect(self._on_detail)
        job.signals.failed.connect(lambda m: self.status.emit(m, False))
        job.start(self.pool)

    def _on_detail(self, d) -> None:
        self.detail = d
        self._render_detail(d)
        self.redraw(reset_view=True)

    def connectivity_ready(self) -> None:
        """接続 (1GB) が届いたので、上流/下流とグラフ探索を有効にする。"""
        for b in (self.b_motor, self.b_path):
            b.setEnabled(True)
            b.setToolTip("")
        if self.type:
            job = Job(self.brain.type_detail, self.type)
            job.signals.done.connect(self._on_detail_only)
            job.signals.failed.connect(lambda m: self.status.emit(m, False))
            job.start(self.pool)

    def _on_detail_only(self, d) -> None:
        """3D はそのままに、横の表と下の1行だけ差し替える。"""
        self.detail = d
        self._render_detail(d)
        if not (self.b_syn.isChecked() or self.b_contact.isChecked()):
            self.info.setText(f"{d.type} — {d.n_bodies} ボディ。"
                              f" 出力 {d.total_out:,} / 入力 {d.total_in:,} シナプス。")

    def _render_detail(self, d) -> None:
        clear_layout(self.detail_box)
        box = self.detail_box
        box.addWidget(label(d.type, "big"))

        tags = QHBoxLayout()
        tags.setSpacing(4)
        if d.superclass:
            tags.addWidget(tag(d.superclass))
        if d.dimorphism:
            tags.addWidget(tag(d.dimorphism, theme.ACCENT))
        tags.addStretch(1)
        box.addLayout(tags)

        box.addWidget(kv("ボディ数", f"{d.n_bodies}"))
        if d.has_connectivity:
            box.addWidget(kv("出力シナプス", f"{d.total_out:,}"))
            box.addWidget(kv("入力シナプス", f"{d.total_in:,}"))
        if d.flywire_type:
            box.addWidget(kv("FlyWire (メス)", d.flywire_type))
        if d.hemibrain_type:
            box.addWidget(kv("hemibrain", d.hemibrain_type))

        if not d.has_connectivity:
            box.addWidget(hline())
            box.addWidget(label("上流 / 下流は接続 (1GB) を読み終えてから出ます。",
                                "muted", wrap=True))
            box.addStretch(1)
            return

        for title, rows in (("下流 — 出力先", d.downstream), ("上流 — 入力元", d.upstream)):
            box.addWidget(hline())
            box.addWidget(label(title, "sectionTitle"))
            lst = QListWidget()
            lst.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            lst.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            height = 6
            for r in rows:
                it = QListWidgetItem()
                w = row(r["type"], f"{r['weight']:,}", r["superclass"])
                it.setSizeHint(w.sizeHint())
                it.setData(Qt.ItemDataRole.UserRole, r["type"])
                lst.addItem(it)
                lst.setItemWidget(it, w)
                height += it.sizeHint().height()
            lst.setFixedHeight(height)
            lst.itemClicked.connect(
                lambda it: self.select(it.data(Qt.ItemDataRole.UserRole)))
            box.addWidget(lst)
        box.addStretch(1)

    # ---- 3D ----

    def redraw(self, reset_view: bool = False) -> None:
        self._pending_reset = self._pending_reset or bool(reset_view)
        self._redraw_timer.start()

    def _do_redraw(self) -> None:
        reset_view, self._pending_reset = self._pending_reset, False
        if not self.detail or not self.brain.ready:
            return
        self._token += 1
        token = self._token
        want_skel = self.b_skel.isChecked()
        want_syn = self.b_syn.isChecked()
        partner = self.partner.text().strip()
        want_contact = self.b_contact.isChecked() and bool(partner)
        d = self.detail

        if want_syn or want_contact:
            self.status.emit("シナプス座標 6.8GB を走査中", True)
        else:
            self.status.emit("骨格を読み込み中", True)

        def work():
            layers: list[Layer] = []
            note = ""
            if want_skel:
                for n in self.brain.skeletons(d.bodies[:8]):
                    layers.append(Layer("lines", n.verts, theme.ACCENT, 1.0, 1.6))
            if want_contact:
                try:
                    p = self.brain.type_detail(partner)
                    for n in self.brain.skeletons(p.bodies[:4]):
                        layers.append(Layer("lines", n.verts, theme.GREEN, 1.0, 1.6))
                except NotFound:
                    pass
            if want_syn or want_contact:
                sy = self.brain.synapses(d.type, partner if want_contact else None)
                if want_syn:
                    layers.append(Layer("points", sy["post"], theme.CYAN, 0.5, 2.0))
                    layers.append(Layer("points", sy["pre"], theme.AMBER, 0.5, 2.0))
                if want_contact and "contact" in sy:
                    layers.append(Layer("points", sy["contact"], theme.YELLOW, 1.0, 7.0))
                    rois = "、".join(f"{k} ({v})"
                                    for k, v in (sy.get("contact_rois") or {}).items())
                    note = (f"{d.type} → {partner} の接触は {sy['n_contact']} シナプス。"
                            f" 領域: {rois or '—'}")
                elif want_syn:
                    rois = "、".join(f"{k} ({v})" for k, v in sy["rois"].items())
                    note = (f"{d.type} のシナプス {sy['n_pre'] + sy['n_post']:,} 個 — "
                            f"入力 {sy['n_post']:,} / 出力 {sy['n_pre']:,}。主な領域: {rois}")
            if not note:
                note = f"{d.type} — {d.n_bodies} ボディ。"
                if d.has_connectivity:
                    note += (f" 出力 {d.total_out:,} / 入力 {d.total_in:,} シナプス。")
            return layers, note

        job = Job(work)
        job.signals.done.connect(
            lambda res, t=token, rv=reset_view: self._on_layers(res, t, rv))
        job.signals.failed.connect(self._on_fail)
        job.start(self.pool)

    def _on_layers(self, res, token: int, reset_view: bool) -> None:
        if token != self._token:
            return  # 追い越された古い結果は捨てる
        layers, note = res
        self.view.set_scene(layers, reset_view=reset_view)
        self.info.setText(note)
        self.status.emit("", False)

    def _on_fail(self, msg: str) -> None:
        self.status.emit(msg, False)
        self.info.setText(msg)

    # ---- グラフ探索 ----

    def motor_reach(self) -> None:
        if not self.type:
            return
        self.status.emit("グラフを探索中 (初回は構築に時間がかかる)", True)
        job = Job(self.brain.motor_reach, self.type)
        job.signals.done.connect(self._on_motor)
        job.signals.failed.connect(self._on_fail)
        job.start(self.pool)

    def _on_motor(self, d) -> None:
        steps = " / ".join(f"{k}段 {v}" for k, v in d["by_step"].items())
        near = "、".join(f"{n['type']} ({n['steps']}段)" for n in d["nearest"][:6])
        self.info.setText(
            f"{self.type} から weight≥10 の接続で 4 段以内に届く運動ニューロン: "
            f"{d['n_reached']:,} / {d['n_motor_total']:,} 個。 {steps}\n最短: {near}")
        self.status.emit("", False)

    def find_path(self) -> None:
        dst = self.dst.text().strip()
        if not self.type or not dst:
            return
        self.status.emit(f"{self.type} → {dst} の経路を探索中", True)
        self._token += 1
        token = self._token

        def work():
            p = self.brain.path(self.type, dst)
            layers = []
            for i, node in enumerate(p["path"]):
                for n in self.brain.skeletons([node["body"]]):
                    color = theme.PATH_COLORS[i % len(theme.PATH_COLORS)]
                    layers.append(Layer("lines", n.verts, color, 1.0, 2.2))
            chain = ""
            for i, node in enumerate(p["path"]):
                chain += node["type"]
                if i < len(p["weights"]):
                    chain += f"  —{p['weights'][i]}シナプス→  "
            return layers, f"経路 ({len(p['path']) - 1} ステップ): {chain}"

        job = Job(work)
        job.signals.done.connect(lambda res, t=token: self._on_layers(res, t, True))
        job.signals.failed.connect(self._on_fail)
        job.start(self.pool)


# ---------------------------------------------------------------- 動画ページ

class VideoPage(QWidget):
    """out/ の動画を、説明と「いつ作ったか」つきで再生する。"""

    status = Signal(str, bool)

    def __init__(self) -> None:
        super().__init__()
        self.states: list[videos.ClipState] = []
        self.current: videos.ClipState | None = None
        self.proc: QProcess | None = None
        self._log_buf: list[str] = []
        self._loaded: tuple[str, float] | None = None   # 再生中のファイルと更新時刻

        self.list = QListWidget()
        self.list.setMinimumWidth(300)
        self.list.setMaximumWidth(360)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.itemClicked.connect(self._on_pick)

        side = QWidget()
        side.setObjectName("sidebar")
        sl = QVBoxLayout(side)
        sl.setContentsMargins(10, 12, 6, 12)
        sl.setSpacing(8)
        sl.addWidget(label("out/ の動画", "sectionTitle"))
        sl.addWidget(self.list, 1)
        self.b_folder = QPushButton("out フォルダを開く")
        self.b_folder.setObjectName("ghost")
        self.b_folder.clicked.connect(lambda: os.startfile(str(videos.OUT)))
        sl.addWidget(self.b_folder)

        self.player = VideoPlayer()
        self.title = label("", "big")
        self.title.setWordWrap(True)
        self.desc = label("左から動画を選んでください。", "muted", wrap=True)
        self.meta = QHBoxLayout()
        self.meta.setSpacing(6)
        self.archive_row = QHBoxLayout()
        self.archive_row.setSpacing(6)

        self.b_render = QPushButton("今のコードで作り直す")
        self.b_render.clicked.connect(self.rerender)
        self.b_render.setEnabled(False)
        self.cmd = label("", "hint", wrap=True)
        self.cmd.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log_title = label("作ったときの出力", "sectionTitle")
        self.log.hide()
        self.log_title.hide()

        head = QHBoxLayout()
        head.addWidget(self.title, 1)
        head.addWidget(self.b_render)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(14, 12, 14, 12)
        rl.setSpacing(8)
        rl.addLayout(head)
        rl.addLayout(self.meta)
        rl.addLayout(self.archive_row)
        rl.addWidget(self.desc)
        rl.addWidget(self.player, 1)
        rl.addWidget(self.cmd)
        rl.addWidget(self.log_title)
        rl.addWidget(self.log)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(side)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        self.refresh()

    # ---- 一覧 ----

    def refresh(self, keep: str | None = None) -> None:
        self.states = videos.scan()
        self.list.clear()
        for group in videos.GROUPS:
            members = [s for s in self.states if s.clip.group == group]
            if not members:
                continue
            head = QListWidgetItem()
            w = label(group, "sectionTitle")
            w.setContentsMargins(8, 10, 8, 2)
            head.setSizeHint(w.sizeHint())
            head.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(head)
            self.list.setItemWidget(head, w)
            metrics = QFontMetrics(self.list.font())
            for st in members:
                it = QListWidgetItem()
                color = {"最新": theme.GREEN, "古い": theme.AMBER,
                         "未生成": theme.MUTED}[st.badge]
                title = metrics.elidedText(st.clip.title,
                                           Qt.TextElideMode.ElideRight, 252)
                detail = "" if not st.exists else f"  ·  {st.when}  ·  {st.size_mb:.1f} MB"
                w = row(title, sub=f'<span style="color:{color}">{st.badge}</span>{detail}')
                it.setSizeHint(w.sizeHint())
                it.setData(Qt.ItemDataRole.UserRole, st.clip.name)
                self.list.addItem(it)
                self.list.setItemWidget(it, w)
        target = keep or (self.current.clip.name if self.current else None)
        if target:
            self.show_clip(target, autoplay=False)

    def _on_pick(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
        if name:
            self.show_clip(name)

    def show_clip(self, name: str, autoplay: bool = True) -> None:
        st = next((s for s in self.states if s.clip.name == name), None)
        if st is None:
            return
        self.current = st
        self.title.setText(st.clip.title)
        self.desc.setText(st.clip.desc)
        self.cmd.setText(st.command)
        can_render = self.proc is None and PYEXE.exists()
        self.b_render.setEnabled(can_render)
        self.b_render.setToolTip(
            "" if can_render else
            f"{PYEXE} が要ります (固めた exe だけを配った環境では作り直せません)")

        clear_layout(self.meta)
        self.meta.addWidget(tag(st.clip.name))
        if st.exists:
            self.meta.addWidget(tag(f"作成 {st.when}"))
            if st.stale:
                self.meta.addWidget(
                    tag(f"{st.newest_source} の方が新しい — 今のコードの姿ではない",
                        theme.AMBER))
            else:
                self.meta.addWidget(tag("今のコードで作られている", theme.GREEN))
        else:
            self.meta.addWidget(tag("まだ作っていない", theme.MUTED))
        self.meta.addStretch(1)

        # 過去の版 — 並べて見比べられるようにしておく
        clear_layout(self.archive_row)
        if st.archives:
            self.archive_row.addWidget(label("過去の版:", "hint"))
            for a in st.archives:
                b = QPushButton(a.name.split("_", 1)[0])   # 先頭の MMDD
                b.setObjectName("ghost")
                b.setToolTip(str(a))
                b.clicked.connect(lambda _=False, path=a: self._play_archive(path))
                self.archive_row.addWidget(b)
            back = QPushButton("今の版に戻す")
            back.setObjectName("ghost")
            back.clicked.connect(
                lambda _=False, n=st.clip.name: self.show_clip(n))
            self.archive_row.addWidget(back)
        self.archive_row.addStretch(1)

        self._show_log(st.log)
        if not st.exists:
            self._loaded = None
            self.player.stop()
        elif self._loaded != (st.clip.name, st.mtime):
            # 同じファイルのままなら読み直さない (タブを戻すたびに再生が飛ぶのを防ぐ)
            self._loaded = (st.clip.name, st.mtime)
            self.player.load(st.path, autoplay=autoplay)

    def _show_log(self, text: str) -> None:
        """作ったときのスクリプトの出力。揚力や方位の実測値はここにある。"""
        if self.proc is not None:
            return                      # 実行中は流れているログを消さない
        self.log.setPlainText(text.strip())
        self.log.setVisible(bool(text.strip()))
        self.log_title.setVisible(bool(text.strip()))
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    def _play_archive(self, path: Path) -> None:
        """退避してある過去の版を再生する (今の版との見比べ用)。"""
        self._loaded = None
        self.player.load(path, autoplay=True)
        self.status.emit(f"過去の版を再生中: {path.name}", False)

    # ---- 作り直す ----

    def rerender(self) -> None:
        if self.current is None or self.proc is not None:
            return
        st = self.current
        self.player.stop()          # 上書きするのでファイルを離させる
        self._loaded = None
        self._log_buf = []
        self.log.clear()
        self.log.show()
        self.log_title.show()
        self.b_render.setEnabled(False)
        self.status.emit(f"{st.clip.name} を作り直しています…", True)

        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_out)
        self.proc.finished.connect(lambda code, _s: self._on_done(code))
        self.proc.start(str(PYEXE), list(st.clip.cmd))

    def _on_out(self) -> None:
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._log_buf.append(text)
        self.log.appendPlainText(text.rstrip())

    def _on_done(self, code: int) -> None:
        name = self.current.clip.name if self.current else ""
        self.proc = None
        if code == 0 and name:
            # 次に開いたときも数字が読めるように、出力を動画の横に残す。
            videos.save_log(name, "".join(self._log_buf))
        self.b_render.setEnabled(True)
        self.status.emit("できました" if code == 0 else f"失敗しました (終了コード {code})",
                         False)
        self.refresh(keep=name)


# ---------------------------------------------------------------- 本体

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ハエ脳ビューア")
        self.resize(1480, 940)
        self.setMinimumSize(1050, 700)
        icon = ROOT / "app" / "icon.png"
        if icon.exists():
            self.setWindowIcon(QIcon(QPixmap(str(icon))))

        self.brain = Brain()
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(max(2, (os.cpu_count() or 4) // 2))

        # --- 上の帯 ---
        title = label("ハエ脳ビューア", "appTitle")
        self.headline = label("読み込み中…", "headline")

        # 「飛ばす」が最初。脳をぐるぐる回して眺めるのではなく、
        # 脳・体・視界を並べて動かすのがこのアプリの主目的
        self.seg_live = QPushButton("飛ばす")
        self.seg_circuit = QPushButton("回路")
        self.seg_video = QPushButton("動画")
        self.seg_gomoku = QPushButton("五目並べ")
        seg_bar = QWidget()
        seg_bar.setObjectName("segmentBar")
        sb = QHBoxLayout(seg_bar)
        sb.setContentsMargins(3, 3, 3, 3)
        sb.setSpacing(2)
        for i, b in enumerate((self.seg_live, self.seg_circuit, self.seg_video,
                               self.seg_gomoku)):
            b.setObjectName("segment")
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.clicked.connect(lambda _=False, i=i: self.stack.setCurrentIndex(i))
            sb.addWidget(b)
        self.seg_live.setChecked(True)

        header = QWidget()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 10, 16, 10)
        hl.setSpacing(14)
        hl.addWidget(title)
        hl.addWidget(self.headline)
        hl.addStretch(1)
        hl.addWidget(seg_bar)

        # --- 中身 ---
        self.live = LivePage()
        self.circuit = CircuitPage(self.brain, self.pool)
        self.video = VideoPage()
        self.gomoku = GomokuPage()
        self.live.status.connect(self.set_status)
        self.circuit.status.connect(self.set_status)
        self.video.status.connect(self.set_status)
        self.gomoku.status.connect(self.set_status)
        self.stack = QStackedWidget()
        for page in (self.live, self.circuit, self.video, self.gomoku):
            self.stack.addWidget(page)
        self.stack.currentChanged.connect(self._on_page)

        # --- 下の帯 ---
        self._conn_msg = ""
        self.status_label = label("", "muted")
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setFixedWidth(90)
        self.bar.hide()
        foot = QWidget()
        foot.setObjectName("infobar")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(14, 5, 14, 5)
        fl.addWidget(self.status_label)
        fl.addStretch(1)
        fl.addWidget(self.bar)

        root = QWidget()
        root.setObjectName("root")
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(header)
        rl.addWidget(self.stack, 1)
        rl.addWidget(foot)
        self.setCentralWidget(root)

        self.circuit.setEnabled(False)
        QTimer.singleShot(60, self.load_brain)
        # 画面が出てから始める。モデルの組み立てに 2.6 秒かかるので、
        # 注釈の読み込み (0.3 秒) より後ろに置いて窓を先に見せる
        QTimer.singleShot(200, self.live.start)

    # ---- 起動時の読み込み ----

    def load_brain(self) -> None:
        """注釈だけ先に読んで画面を開ける。重い2つは後から追いかける。"""
        job = Job(self.brain.load, with_progress=True)
        job.signals.progress.connect(lambda msg, _frac: self.set_status(msg, True))
        job.signals.done.connect(self._loaded)
        job.signals.failed.connect(self._load_failed)
        self.set_status("注釈を読み込み中", True)
        job.start(self.pool)

    def _loaded(self, _=None) -> None:
        self.headline.setText(self.brain.headline())
        self.circuit.setEnabled(True)
        self.set_status("", False)
        self.circuit.start()

        conn = Job(self.brain.load_connectivity, with_progress=True)
        self._conn_msg = "接続を読み込み中 (1GB / 1億5185万接続)"
        conn.signals.progress.connect(lambda msg, _f: self.set_status(msg, True))
        conn.signals.done.connect(self._connectivity_loaded)
        conn.signals.failed.connect(lambda m: self.set_status(m, False))
        conn.start(self.pool)

        count = Job(self.brain.count_synapses)
        count.signals.done.connect(lambda _: self.headline.setText(self.brain.headline()))
        count.start(self.pool)

    def _connectivity_loaded(self, _=None) -> None:
        self.headline.setText(self.brain.headline())
        self.circuit.connectivity_ready()
        # 裏で走っている別の処理がステータスを使っていたら奪わない。
        if self.status_label.text() == self._conn_msg:
            self.set_status("", False)

    def _load_failed(self, msg: str) -> None:
        self.headline.setText("データを読み込めませんでした")
        self.set_status(msg, False)
        self.stack.setCurrentIndex(2)
        self.seg_video.setChecked(True)
        self.circuit.info.setText(
            msg + "\n動画タブは data/ が無くても見られます。")

    # ---- ステータス ----

    def set_status(self, text: str, busy: bool = False) -> None:
        self.status_label.setText(text)
        self.bar.setVisible(busy)

    def _on_page(self, idx: int) -> None:
        # 見ていないページは止める。live は回しっぱなしだと CPU を丸ごと使う
        self.live.set_visible(idx == 0)
        if idx == 2:
            self.video.refresh()
        else:
            self.video.player.player.pause()

    def closeEvent(self, event) -> None:
        self.video.player.stop()
        self.live.stop()
        super().closeEvent(event)


def main() -> int:
    QGuiApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setApplicationName("ハエ脳ビューア")
    app.setOrganizationName("fly")
    app.setStyle("Fusion")
    app.setPalette(theme.palette())
    app.setFont(QFont("Segoe UI Variable Text", 10))
    app.setStyleSheet(theme.QSS)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
