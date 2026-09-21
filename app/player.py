"""動画プレーヤ — QtMultimedia でそのまま再生する。

シミュレーションの動画はどれも無音で、見たいのは「どの瞬間に何が起きたか」
なので、ループ・再生速度・こま送りを付けてある。羽ばたきは 60 倍スローでも
速いので 0.25 倍まで落とせるようにした。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

SPEEDS = ("0.25x", "0.5x", "1x", "2x")
STEP_MS = 33  # こま送りの1回ぶん (30fps 相当)


def _mmss(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


class VideoPlayer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.0)          # どれも無音だが、出力が無いと警告が出る
        self.player.setAudioOutput(self.audio)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)

        self.screen = QVideoWidget()
        self.screen.setMinimumHeight(260)
        self.screen.setStyleSheet("background:#000;border-radius:10px;")
        self.player.setVideoOutput(self.screen)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(40)
        self.btn_play.clicked.connect(self.toggle)
        self.btn_prev = QPushButton("◀|")
        self.btn_next = QPushButton("|▶")
        for b, d in ((self.btn_prev, -STEP_MS), (self.btn_next, STEP_MS)):
            b.setObjectName("ghost")
            b.setFixedWidth(34)
            b.setToolTip("こま送り")
            b.clicked.connect(lambda _=False, d=d: self.step(d))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.player.setPosition)

        self.time = QLabel("0:00 / 0:00")
        self.time.setObjectName("muted")

        self.speed = QComboBox()
        self.speed.addItems(SPEEDS)
        self.speed.setCurrentText("1x")
        self.speed.setFixedWidth(74)
        self.speed.currentTextChanged.connect(
            lambda t: self.player.setPlaybackRate(float(t[:-1])))

        self.btn_loop = QPushButton("ループ")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(True)
        self.btn_loop.toggled.connect(self._set_loop)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(6)
        for w in (self.btn_play, self.btn_prev, self.btn_next):
            bar.addWidget(w)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.time)
        bar.addWidget(self.speed)
        bar.addWidget(self.btn_loop)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(self.screen, 1)
        lay.addLayout(bar)

        self.player.positionChanged.connect(self._on_pos)
        self.player.durationChanged.connect(self._on_dur)
        self.player.playbackStateChanged.connect(self._on_state)

    # ---- 操作 ----------------------------------------------------------

    def load(self, path: Path, autoplay: bool = True) -> None:
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.setPlaybackRate(float(self.speed.currentText()[:-1]))
        if autoplay:
            self.player.play()

    def stop(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())

    def toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def step(self, delta_ms: int) -> None:
        self.player.pause()
        self.player.setPosition(max(0, self.player.position() + delta_ms))

    def _set_loop(self, on: bool) -> None:
        self.player.setLoops(QMediaPlayer.Loops.Infinite if on else 1)

    # ---- 表示の追随 ----------------------------------------------------

    def _on_pos(self, ms: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        self.time.setText(f"{_mmss(ms)} / {_mmss(self.player.duration())}")

    def _on_dur(self, ms: int) -> None:
        self.slider.setRange(0, ms)
        self.time.setText(f"{_mmss(self.player.position())} / {_mmss(ms)}")

    def _on_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.btn_play.setText("❚❚" if playing else "▶")
