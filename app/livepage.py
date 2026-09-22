"""live 画面 — 脳・体・視界を並べて、飛ばしながら見る。

    ┌─ 脳 ──────────┬─ 体 ──────────┬─ 視界 ─────────┐
    │ 点群 (OpenGL)  │ MuJoCo の描画  │ 複眼の像        │
    │ 領域ごとの明るさ│ 位置・姿勢・速さ│ 報酬 (花) の操作 │
    └───────────────┴───────────────┴────────────────┘

3つのパネルは **同じ1本のシミュレーション** の同じ時刻から出ている。
脳が光るのはその場所に入力が届いているときだけで、こちらで作った揺らぎは
入っていない (`scripts/brain_glow.py`)。

**重い処理は全部ワーカースレッドで回す。** UI スレッドがやるのは、届いた
配列を QImage にして貼ることと、点の色を GL に流すことだけ。
シミュレーションは止めずに回り続け、操作 (花を置く、好みを変える) は
キューに積まれてステップの合間に届く。

固めた exe には mujoco が入っていないので (`build_app.py` の EXCLUDE)、
その場合はここに理由を出すだけで、ほかのタブは普通に使える。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import (Q_ARG, QMetaObject, QObject, Qt, QThread, QTimer,
                            Signal, Slot)
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout,
                               QLabel, QPushButton, QSizePolicy, QSlider,
                               QSplitter, QVBoxLayout, QWidget)

from . import theme
from .view3d import LiveBrainView

# 報酬の各項の振れ幅 (バーの長さを決めるだけ)。flight_env.REWARD_TERMS と同じ順
REWARD_SPAN = (1.0, 2.0, 1.0, 1.0, 1.0, 0.6, 0.6)

# 領域名 -> 点の色分け (LiveBrainView.TINTS の添字)
TINT_OF = {"ME": 1, "LO": 1, "LOP": 1, "LA": 1, "AME": 1,
           "AL": 2, "LH": 3,
           "WTct": 4, "LegNp": 4, "HTct": 4, "IntTct": 4, "ANm": 4}


def tint_array(cloud) -> np.ndarray:
    """点ごとの色分け。領域名の頭で決める。"""
    base = np.array([str(r).split("(")[0] for r in cloud.roi])
    t = np.zeros(len(base), dtype=int)
    for name, idx in TINT_OF.items():
        t[base == name] = idx
    return t


# ---------------------------------------------------------------- 部品

class LevelBar(QWidget):
    """名前・値・横棒の 1 行。負の値も出せる (報酬の罰の項)。"""

    def __init__(self, name: str, span: float = 1.0, color: str = theme.CYAN,
                 signed: bool = False, hint: str = "") -> None:
        super().__init__()
        self.name, self.span, self.color, self.signed = name, span, color, signed
        self.value = 0.0
        self.text = ""
        self.setFixedHeight(20)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if hint:
            self.setToolTip(hint)

    def set_value(self, v: float, text: str = "") -> None:
        self.value = float(v)
        self.text = text
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()
        name_w = 74
        val_w = 54
        bar_x = name_w
        bar_w = max(w - name_w - val_w, 10)
        y = h // 2 - 3

        p.setPen(QColor(theme.MUTED if self.value == 0 else theme.FG))
        p.drawText(0, 0, name_w - 6, h, Qt.AlignmentFlag.AlignVCenter
                   | Qt.AlignmentFlag.AlignLeft, self.name)

        p.fillRect(bar_x, y, bar_w, 6, QColor(theme.PANEL2))
        frac = max(min(self.value / max(self.span, 1e-9), 1.0), -1.0)
        if self.signed:
            mid = bar_x + bar_w // 2
            n = int(abs(frac) * bar_w / 2)
            if frac >= 0:
                p.fillRect(mid, y, n, 6, QColor(self.color))
            else:
                p.fillRect(mid - n, y, n, 6, QColor(theme.ACCENT))
            p.fillRect(mid, y - 2, 1, 10, QColor(theme.LINE))
        elif frac > 0:
            p.fillRect(bar_x, y, int(frac * bar_w), 6, QColor(self.color))

        p.setPen(QColor(theme.MUTED))
        p.drawText(w - val_w, 0, val_w - 2, h, Qt.AlignmentFlag.AlignVCenter
                   | Qt.AlignmentFlag.AlignRight,
                   self.text or f"{self.value:.2f}")
        p.end()


class ImagePanel(QLabel):
    """numpy の画像を貼るだけのラベル。拡大の仕方を選べる。"""

    def __init__(self, smooth: bool = True, min_h: int = 200) -> None:
        super().__init__()
        self.smooth = smooth
        self.setMinimumHeight(min_h)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background:{theme.BG};border-radius:8px;")
        self._pix: QPixmap | None = None

    def set_image(self, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype=np.uint8)
        h, w = a.shape[:2]
        img = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(img)
        self._rescale()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is None:
            return
        mode = (Qt.TransformationMode.SmoothTransformation if self.smooth
                else Qt.TransformationMode.FastTransformation)
        self.setPixmap(self._pix.scaled(self.size(),
                                        Qt.AspectRatioMode.KeepAspectRatio, mode))


def _title(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setStyleSheet(f"color:{theme.FG};font-size:14px;font-weight:600;")
    return lb


def _note(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet(f"color:{theme.MUTED};font-size:11px;")
    return lb


def _column(title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout, QLabel]:
    w = QWidget()
    w.setObjectName("card")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(6)
    lay.addWidget(_title(title))
    sub = _note(subtitle)
    lay.addWidget(sub)
    return w, lay, sub


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{theme.LINE};background:{theme.LINE};max-height:1px;")
    return f


# ------------------------------------------------------------ ワーカー

class SimWorker(QObject):
    """ワーカースレッド側 — シミュレーションを持ち、フレームを配る。

    0 ミリ秒の QTimer で回す。0 でも Qt はイベントを処理するので、
    **ステップの合間にキューの操作 (花を置く等) が届く**。ループを
    while で回すとそれが届かない。
    """

    cloud_ready = Signal(object, object)     # 点の位置, 色分け
    scene_ready = Signal(str, str, float, float)  # 名前, 副題, 視点の距離と方位
    frame_ready = Signal(object)
    note = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.sim = None
        self.cloud = None
        self.timer: QTimer | None = None
        self.playing = False
        self.budget = 0.05        # 1 コマぶんの物理に使う実時間 [s]
        # None は「まだ何も作っていない」。場面が **変わった** ときだけ視点を
        # その場面の既定へ戻したいので、名前で比べられるようにしておく
        self._scene = None
        self._quality = "標準"
        self._shadows = False
        # 画質を変えるとモデルを組み直す。**そこで設定が消えないよう、
        # 状態はワーカーが持っておいて組み直した側へ入れ直す。**
        # (以前は花の位置・好み・おとり・視点が既定に戻っていた)
        self._like = "target"
        self._decoy = True
        self._flowers: dict = {}
        self._view: tuple[float, float] | None = None

    # -- 起動 ----------------------------------------------------------
    @Slot()
    def boot(self) -> None:
        try:
            from . import livesim
            ok, msg = livesim.available()
            if not ok:
                self.failed.emit(msg)
                return
            self.note.emit("シナプスの点群を読み込み中 (45,000 点)")
            from brain_glow import CloudMap
            self.cloud = CloudMap(45000, n_omma=34, verbose=False)
            self.cloud_ready.emit(self.cloud.pts, tint_array(self.cloud))
        except Exception as exc:
            self.failed.emit(f"点群を読めませんでした: {exc}")
            return
        self.timer = QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self._tick)
        self.set_scene("flight")

    # -- 操作 ----------------------------------------------------------
    @Slot(str)
    def set_scene(self, name: str) -> None:
        from . import livesim

        self.playing = False
        changed = (name != self._scene)
        if self.sim is not None:
            if hasattr(self.sim, "flower_pos"):
                self._flowers = {k: self.sim.flower_pos(k).copy()
                                 for k in ("target", "decoy")}
            self.sim.close()
            self.sim = None
        cls = livesim.FlightSim if name == "flight" else livesim.EscapeSim
        self._scene = name
        self.note.emit(f"{cls.NAME} を組み立て中 (モデル・複眼・空力)")
        try:
            self.sim = cls(self.cloud, quality=self._quality,
                           shadows=self._shadows)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"場面を作れませんでした: {exc}")
            return
        self._restore()
        if changed or self._view is None:
            self._view = (float(self.sim.cam_dist), float(self.sim.cam_az))
        self.sim.set_view(dist=self._view[0], azimuth=self._view[1])
        self.scene_ready.emit(cls.NAME, cls.SUBTITLE, *self._view)
        self.note.emit("")
        self._emit_frame()
        self.set_playing(True)

    def _restore(self) -> None:
        """組み直した場面に、いまの設定を入れ直す。"""
        sim = self.sim
        if sim is None or not hasattr(sim, "set_like"):
            return
        for which, pos in self._flowers.items():
            sim.move_flower(which, pos)
        sim.set_like(self._like)
        sim.set_decoy(self._decoy)

    @Slot(str, bool)
    def set_quality(self, quality: str, shadows: bool) -> None:
        if quality == self._quality and shadows == self._shadows:
            return
        self._quality, self._shadows = quality, shadows
        self.set_scene(self._scene)

    @Slot(bool)
    def set_playing(self, on: bool) -> None:
        self.playing = bool(on) and self.sim is not None
        if self.timer is None:
            return
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    @Slot()
    def restart(self) -> None:
        if self.sim is not None:
            self.sim.reset()
            self._emit_frame()

    @Slot(float, float, float)
    def place_reward(self, dist: float, side: float, height: float) -> None:
        if self.sim is not None and hasattr(self.sim, "place_reward"):
            self.sim.place_reward(dist, side, height)
            self._flowers[self._like] = self.sim.flower_pos(self._like).copy()
            if not self.playing:
                self._emit_frame()

    @Slot(str)
    def set_like(self, which: str) -> None:
        self._like = which
        if self.sim is not None and hasattr(self.sim, "set_like"):
            self.sim.set_like(which)

    @Slot(bool)
    def set_decoy(self, on: bool) -> None:
        self._decoy = bool(on)
        if self.sim is not None and hasattr(self.sim, "set_decoy"):
            self.sim.set_decoy(on)

    @Slot(float, float)
    def set_view(self, dist: float, azimuth: float) -> None:
        self._view = (float(dist), float(azimuth))
        if self.sim is not None:
            self.sim.set_view(dist=dist, azimuth=azimuth)
            if not self.playing:
                self._emit_frame()      # 止めていても視点は追う

    @Slot()
    def shutdown(self) -> None:
        self.playing = False
        if self.timer is not None:
            self.timer.stop()
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    # -- ループ --------------------------------------------------------
    def _tick(self) -> None:
        if self.sim is None or not self.playing:
            return
        try:
            self.sim.step_for(self.budget)
            self._emit_frame()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.playing = False
            if self.timer is not None:
                self.timer.stop()
            self.failed.emit(f"シミュレーションが止まりました: {exc}")

    def _emit_frame(self) -> None:
        if self.sim is not None:
            self.frame_ready.emit(self.sim.frame())


# ---------------------------------------------------------------- 画面

class LivePage(QWidget):
    """脳・体・視界の3枚と、飛ばす / 報酬を置く操作。"""

    status = Signal(str, bool)

    # 画質の選択肢 -> (livesim の画質, 複眼にも影を落とすか)
    QUALITY = (("軽い (速い)", "軽い", False),
               ("標準", "標準", False),
               ("精細", "精細", False),
               ("精細 + 影 (動画と同じ / とても遅い)", "精細", True))

    def __init__(self) -> None:
        super().__init__()
        self._booted = False
        self._n_regions = 0
        self._build()

        self.thread = QThread()
        self.worker = SimWorker()
        self.worker.moveToThread(self.thread)
        self.worker.cloud_ready.connect(self._on_cloud)
        self.worker.scene_ready.connect(self._on_scene)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.note.connect(lambda m: self.status.emit(m, bool(m)))
        self.worker.failed.connect(self._on_failed)
        self.thread.start()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(10)
        root.addWidget(self._toolbar())

        self.banner = _note("")
        self.banner.setStyleSheet(
            f"color:{theme.ACCENT};font-size:12px;padding:6px 2px;")
        self.banner.hide()
        root.addWidget(self.banner)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._brain_column())
        split.addWidget(self._body_column())
        split.addWidget(self._eye_column())
        split.setSizes([420, 520, 380])
        root.addWidget(split, 1)

    # -- 上の帯 --------------------------------------------------------
    def _toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("toolbar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)

        self.scene_box = QComboBox()
        self.scene_box.addItem("飛ぶ — 匂いのする花に寄る", "flight")
        self.scene_box.addItem("逃げる — ハエ叩きが迫る", "escape")
        self.scene_box.currentIndexChanged.connect(
            lambda _: self._call("set_scene", self.scene_box.currentData()))
        self.scene_box.setMinimumWidth(210)
        lay.addWidget(QLabel("場面"))
        lay.addWidget(self.scene_box)

        self.play_btn = QPushButton("一時停止")
        self.play_btn.setCheckable(True)
        self.play_btn.setChecked(True)
        self.play_btn.clicked.connect(self._on_play)
        lay.addWidget(self.play_btn)

        restart = QPushButton("最初から")
        restart.clicked.connect(lambda: self._call("restart"))
        lay.addWidget(restart)

        lay.addSpacing(8)
        self.quality_box = QComboBox()
        for label, q, sh in self.QUALITY:
            self.quality_box.addItem(label, (q, sh))
        self.quality_box.setCurrentIndex(1)
        self.quality_box.currentIndexChanged.connect(self._on_quality)
        self.quality_box.setMinimumWidth(240)
        self.quality_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        lay.addWidget(QLabel("画質"))
        lay.addWidget(self.quality_box)

        lay.addStretch(1)
        self.clock = QLabel("—")
        self.clock.setStyleSheet(
            f"color:{theme.FG};font-size:12px;font-family:'Cascadia Mono',"
            "'Consolas',monospace;")
        lay.addWidget(self.clock)
        return bar

    # -- 脳 ------------------------------------------------------------
    def _brain_column(self) -> QWidget:
        col, lay, _ = _column("脳 — Male CNS v1.0",
                              "45,000 点で 3億1183万シナプスを代表する。"
                              "点の位置と領域は実測")
        self.brain_view = LiveBrainView()
        self.brain_view.setMinimumHeight(260)
        lay.addWidget(self.brain_view, 1)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.lit_label = _note("")
        foot.addWidget(self.lit_label, 1)
        self.spin_chk = QCheckBox("回す")
        self.spin_chk.setChecked(True)
        self.spin_chk.setToolTip("自動で回す。ドラッグすると自分で回せる (自動は止まる)")
        self.spin_chk.toggled.connect(self._on_spin)
        foot.addWidget(self.spin_chk)
        reset_cam = QPushButton("視点を戻す")
        reset_cam.setObjectName("ghost")
        reset_cam.clicked.connect(self.brain_view.reset_camera)
        foot.addWidget(reset_cam)
        lay.addLayout(foot)
        lay.addWidget(_hline())

        self.region_box = QWidget()
        self.region_lay = QVBoxLayout(self.region_box)
        self.region_lay.setContentsMargins(0, 2, 0, 2)
        self.region_lay.setSpacing(2)
        self.region_bars: dict[str, LevelBar] = {}
        lay.addWidget(self.region_box)

        from .livesim import GLOW_NOTE
        lay.addWidget(_note(GLOW_NOTE))
        return col

    # -- 体 ------------------------------------------------------------
    def _body_column(self) -> QWidget:
        col, lay, sub = _column("体 — MuJoCo", "")
        self.body_sub = sub
        self.body_view = ImagePanel(smooth=True, min_h=260)
        lay.addWidget(self.body_view, 1)

        self.stage_label = QLabel("")
        self.stage_label.setWordWrap(True)
        self.stage_label.setStyleSheet(
            f"color:{theme.FG};font-size:13px;font-weight:600;padding:4px 0;")
        lay.addWidget(self.stage_label)
        lay.addWidget(_hline())

        view = QHBoxLayout()
        view.setSpacing(8)
        view.addWidget(QLabel("視点"))
        self.dist_slider = QSlider(Qt.Orientation.Horizontal)
        self.dist_slider.setRange(6, 120)        # 0.6 - 12.0 cm
        self.dist_slider.setValue(16)
        self.dist_slider.setToolTip("カメラとハエの距離。体長は 0.25 cm")
        self.dist_slider.valueChanged.connect(self._on_view)
        self.az_slider = QSlider(Qt.Orientation.Horizontal)
        self.az_slider.setRange(0, 359)
        self.az_slider.setValue(128)
        self.az_slider.setToolTip("ハエのまわりを回る")
        self.az_slider.valueChanged.connect(self._on_view)
        self.view_label = QLabel("")
        self.view_label.setFixedWidth(96)
        self.view_label.setStyleSheet(f"color:{theme.MUTED};font-size:11px;")
        view.addWidget(self.dist_slider, 1)
        view.addWidget(self.az_slider, 1)
        view.addWidget(self.view_label)
        lay.addLayout(view)

        self.rows_box = QWidget()
        self.rows_lay = QVBoxLayout(self.rows_box)
        self.rows_lay.setContentsMargins(0, 2, 0, 2)
        self.rows_lay.setSpacing(3)
        self.row_labels: list[tuple[QLabel, QLabel]] = []
        lay.addWidget(self.rows_box)
        lay.addStretch(1)
        return col

    # -- 視界と報酬 ----------------------------------------------------
    def _eye_column(self) -> QWidget:
        col, lay, _ = _column(
            "ハエの視界 (左眼 / 右眼)",
            "個眼 34x34 /眼。ここに出るのが実際に解像できる像 — "
            "粗くてボケている。明るさは光順応をかけた値で、"
            "フレームごとの引き伸ばしはしていない")
        self.eye_view = ImagePanel(smooth=False, min_h=150)
        self.eye_view.setMaximumHeight(210)
        lay.addWidget(self.eye_view)

        lay.addWidget(_hline())
        self.reward_box = QWidget()
        rl = QVBoxLayout(self.reward_box)
        rl.setContentsMargins(0, 4, 0, 0)
        rl.setSpacing(6)
        rl.addWidget(_title("報酬 — 匂いのする花"))
        rl.addWidget(_note(
            "花にはプルームがある。置くと本当に振る舞いが変わる — "
            "触角葉が光り、機体が向きを変え、視界の中で花が大きくなる。"
            "数字は学習 (13_train_flight.py) が使ったのと同じ式。"
            "ただし、この場では学習しない。飛び方は out/flight_policy.json の"
            "固定パラメータで、キノコ体を光らせないのも同じ理由"))

        self.sliders: dict[str, QSlider] = {}
        for key, label, lo, hi, init in (("dist", "風上へ", 1, 30, 6),
                                         ("side", "左右", -12, 12, 0),
                                         ("height", "高さ", 3, 20, 9)):
            row = QHBoxLayout()
            row.setSpacing(6)
            name = QLabel(label)
            name.setFixedWidth(46)
            name.setStyleSheet(f"color:{theme.MUTED};font-size:12px;")
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(init)
            val = QLabel(f"{init} cm")
            val.setFixedWidth(46)
            val.setStyleSheet(f"color:{theme.FG};font-size:12px;")
            sl.valueChanged.connect(
                lambda v, lb=val: lb.setText(f"{v} cm"))
            row.addWidget(name)
            row.addWidget(sl, 1)
            row.addWidget(val)
            rl.addLayout(row)
            self.sliders[key] = sl

        place = QPushButton("ここに花を置く")
        place.clicked.connect(self._place)
        rl.addWidget(place)

        opt = QHBoxLayout()
        opt.setSpacing(8)
        opt.addWidget(QLabel("好む匂い"))
        self.like_box = QComboBox()
        self.like_box.addItem("本命", "target")
        self.like_box.addItem("おとり", "decoy")
        self.like_box.currentIndexChanged.connect(
            lambda _: self._call("set_like", self.like_box.currentData()))
        opt.addWidget(self.like_box, 1)
        self.decoy_chk = QCheckBox("おとりも置く")
        self.decoy_chk.setChecked(True)
        self.decoy_chk.toggled.connect(lambda on: self._call("set_decoy", on))
        opt.addWidget(self.decoy_chk)
        rl.addLayout(opt)
        rl.addWidget(_note(
            "好みは こちらが決めた定数 で、学習の結果ではない。"
            "好まない匂いだと寄らないが、離れる動きは持っていない (未解決)"))

        rl.addWidget(_hline())
        self.reward_bars: list[LevelBar] = []
        self.reward_rows = QVBoxLayout()
        self.reward_rows.setContentsMargins(0, 0, 0, 0)
        self.reward_rows.setSpacing(2)
        rl.addLayout(self.reward_rows)
        self.reward_total = _note("")
        rl.addWidget(self.reward_total)
        lay.addWidget(self.reward_box)

        self.no_reward = _note(
            "この場面には報酬が無い。逃避は捕食者から逃げるだけで、"
            "うまくいったかどうかを点数にする仕組みを持っていない。"
            "無いものは画面に出さない")
        self.no_reward.hide()
        lay.addWidget(self.no_reward)
        lay.addStretch(1)
        return col

    # ------------------------------------------------------------------
    def start(self) -> None:
        """タブが最初に開かれたときに呼ぶ。読み込みはここから始める。"""
        if self._booted:
            return
        self._booted = True
        self._call("boot")

    def set_visible(self, on: bool) -> None:
        """別のタブへ移ったら止める。回しっぱなしだと CPU を丸ごと使う。

        再生ボタンの状態は覚えておいて、戻ってきたらそのとおりに再開する。
        """
        if not self._booted:
            return
        self._call("set_playing", bool(on) and self.play_btn.isChecked())

    def _call(self, name: str, *args) -> None:
        """ワーカーへ操作を投げる。**キュー経由なので UI は待たない。**

        シミュレーションは回りっぱなしで、操作はステップの合間に届く。
        """
        if not self._booted and name != "boot":
            return
        QMetaObject.invokeMethod(self.worker, name,
                                 Qt.ConnectionType.QueuedConnection,
                                 *self._pack(args))

    @staticmethod
    def _pack(args):
        out = []
        for a in args:
            if isinstance(a, bool):
                out.append(Q_ARG(bool, a))
            elif isinstance(a, float):
                out.append(Q_ARG(float, a))
            elif isinstance(a, int):
                out.append(Q_ARG(int, a))
            else:
                out.append(Q_ARG(str, a))
        return out

    def _on_play(self, checked: bool) -> None:
        self.play_btn.setText("一時停止" if checked else "再生")
        self._call("set_playing", checked)

    def _on_quality(self, _idx: int) -> None:
        q, sh = self.quality_box.currentData()
        self._call("set_quality", q, sh)

    def _on_spin(self, on: bool) -> None:
        self.brain_view.auto = bool(on)

    def _on_view(self, _v: int = 0) -> None:
        dist = self.dist_slider.value() / 10.0
        az = float(self.az_slider.value())
        self.view_label.setText(f"{dist:.1f} cm / {az:.0f}°")
        self._call("set_view", dist, az)

    def _place(self) -> None:
        self._call("place_reward",
                   float(self.sliders["dist"].value()),
                   float(self.sliders["side"].value()),
                   float(self.sliders["height"].value()))

    # ------------------------------------------------------------------
    @Slot(object, object)
    def _on_cloud(self, pts, tint) -> None:
        self.brain_view.set_cloud(pts, tint)

    @Slot(str, str, float, float)
    def _on_scene(self, name: str, subtitle: str,
                  dist: float, az: float) -> None:
        self.body_sub.setText(subtitle)
        for sl, v in ((self.dist_slider, int(round(dist * 10))),
                      (self.az_slider, int(round(az)) % 360)):
            sl.blockSignals(True)
            sl.setValue(v)
            sl.blockSignals(False)
        self._on_view()
        flight = self.scene_box.currentData() == "flight"
        self.reward_box.setVisible(flight)
        self.no_reward.setVisible(not flight)
        self.banner.hide()

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        self.banner.setText(msg)
        self.banner.show()
        self.status.emit("", False)
        self.play_btn.setChecked(False)
        self.play_btn.setText("再生")

    @Slot(object)
    def _on_frame(self, f) -> None:
        if f.body is not None:
            self.body_view.set_image(f.body)
        self.eye_view.set_image(f.eyes)
        self.brain_view.set_brightness(f.glow)
        self.brain_view.spin(0.25)          # ゆっくり回す。奥行きが出る
        if self.spin_chk.isChecked() and not self.brain_view.auto:
            self.spin_chk.setChecked(False)   # 自分で回し始めた

        self.clock.setText(
            f"t = {f.t*1e3:8.2f} ms     スロー x{f.slow:5.1f} (実測)")
        self.stage_label.setText(f.stage)
        self.stage_label.setStyleSheet(
            f"color:{theme.ACCENT if not f.alive else theme.FG};"
            "font-size:13px;font-weight:600;padding:4px 0;")

        self._fill_regions(f.regions)
        self.lit_label.setText(
            f"入力で光っている点 {int((f.glow > 0.25).sum()):,} / {len(f.glow):,}")
        self._fill_rows(f.rows)
        self._fill_reward(f)

    def _fill_regions(self, regions) -> None:
        if len(regions) != self._n_regions:
            while self.region_lay.count():
                self.region_lay.takeAt(0).widget().deleteLater()
            self.region_bars.clear()
            for key, desc, _lv, n in regions:
                color = {1: "#7fc8ff", 2: "#ffb648", 3: "#c58bff",
                         4: "#58e08a"}.get(TINT_OF.get(key, 0), "#cdd6e3")
                bar = LevelBar(key, 1.0, color, hint=f"{desc}  ({n:,} 点)")
                self.region_bars[key] = bar
                self.region_lay.addWidget(bar)
            self._n_regions = len(regions)
        for key, _desc, lv, _n in regions:
            bar = self.region_bars.get(key)
            if bar is not None:
                bar.set_value(lv, f"{lv:.2f}")

    def _fill_rows(self, rows) -> None:
        if len(rows) != len(self.row_labels):
            while self.rows_lay.count():
                self.rows_lay.takeAt(0).widget().deleteLater()
            self.row_labels = []
            for _name, _val in rows:
                w = QWidget()
                hl = QHBoxLayout(w)
                hl.setContentsMargins(0, 0, 0, 0)
                left = QLabel("")
                left.setStyleSheet(f"color:{theme.MUTED};font-size:12px;")
                left.setFixedWidth(120)
                right = QLabel("")
                right.setStyleSheet(
                    f"color:{theme.FG};font-size:12px;font-family:"
                    "'Cascadia Mono','Consolas',monospace;")
                right.setWordWrap(True)
                hl.addWidget(left)
                hl.addWidget(right, 1)
                self.rows_lay.addWidget(w)
                self.row_labels.append((left, right))
        for (left, right), (name, val) in zip(self.row_labels, rows):
            left.setText(name)
            right.setText(val)

    def _fill_reward(self, f) -> None:
        if not f.reward:
            return
        if len(self.reward_bars) != len(f.reward):
            while self.reward_rows.count():
                self.reward_rows.takeAt(0).widget().deleteLater()
            self.reward_bars = []
            for i, (name, _v, hint) in enumerate(f.reward):
                span = REWARD_SPAN[i] if i < len(REWARD_SPAN) else 1.0
                signed = i >= 5
                bar = LevelBar(name, span,
                               theme.GREEN if not signed else theme.ACCENT,
                               signed=signed, hint=hint)
                self.reward_bars.append(bar)
                self.reward_rows.addWidget(bar)
        for bar, (_name, v, _hint) in zip(self.reward_bars, f.reward):
            bar.set_value(v, f"{v:+.2f}")
        self.reward_total.setText(
            f"積算報酬 {f.reward_total:.3f}   "
            f"(毎秒 {sum(v for _n, v, _h in f.reward):+.2f})")

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if self.thread.isRunning():
            self._call("shutdown")
            self.thread.quit()
            self.thread.wait(5000)
