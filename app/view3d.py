"""3D 表示 — pyqtgraph の OpenGL ビュー。

ブラウザ版は Plotly (WebGL) だったが、シナプス点群を数万点出すと重かった。
ここでは素の OpenGL に投げるので 6 万点でも回せる。

座標の扱い:
  Male CNS の座標は nm で、脳全体だと 10^5〜10^6 のオーダーになる。そのまま
  float32 で GL に渡すと奥行きの精度が足りずちらつくので、**重心を引いて
  最大辺が 100 単位になるように正規化**してから描いている。カメラの距離も
  この正規化後の単位。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtGui import QVector3D

from . import theme

UNIT = 100.0  # 正規化後の最大辺


@dataclass
class Layer:
    """描くもの1つ。`kind` は 'lines' (骨格) か 'points' (シナプス)。"""

    kind: str
    verts: np.ndarray          # lines: (2N,3) 端点ペア / points: (N,3)
    color: str                 # #rrggbb
    alpha: float = 1.0
    width: float = 2.0         # lines の太さ / points の大きさ


class Brain3DView(gl.GLViewWidget):
    """レイヤの集合をまとめて描く。描き直しても視点は保つ。"""

    def __init__(self) -> None:
        super().__init__()
        self.setBackgroundColor(theme.BG)
        self.opts["distance"] = 250
        self.opts["fov"] = 50
        self._items: list = []
        self._has_scene = False

    # ---- 描画 ----------------------------------------------------------

    def clear_scene(self) -> None:
        for it in self._items:
            self.removeItem(it)
        self._items.clear()

    def set_scene(self, layers: list[Layer], reset_view: bool = False) -> None:
        self.clear_scene()
        arrays = [l.verts for l in layers if l.verts is not None and len(l.verts)]
        if not arrays:
            self._has_scene = False
            self.update()
            return

        lo = np.min([a.min(axis=0) for a in arrays], axis=0)
        hi = np.max([a.max(axis=0) for a in arrays], axis=0)
        center = (lo + hi) / 2.0
        span = float(np.max(hi - lo)) or 1.0
        scale = UNIT / span

        for layer in layers:
            v = layer.verts
            if v is None or not len(v):
                continue
            q = ((v - center) * scale).astype(np.float32)
            # Male CNS の座標は +y が腹側 (下)。GL は z が上なので入れ替えて、
            # 脳が上・神経索が下に見えるようにする。
            p = np.column_stack([q[:, 0], q[:, 2], -q[:, 1]]).astype(np.float32)
            col = theme.rgba(layer.color, layer.alpha)
            if layer.kind == "lines":
                item = gl.GLLinePlotItem(pos=p, color=col, width=layer.width,
                                         mode="lines", antialias=True)
            else:
                item = gl.GLScatterPlotItem(pos=p, color=col, size=layer.width,
                                            pxMode=True)
                # additive だと点が重なった所が白く飛んで色が読めなくなる。
                item.setGLOptions("translucent")
            self.addItem(item)
            self._items.append(item)

        self.opts["center"] = QVector3D(0, 0, 0)
        if reset_view or not self._has_scene:
            self._home()
        self._has_scene = True
        self.update()

    def _home(self) -> None:
        # 真横から少し見下ろす。脳と神経索が前後に長いので、この向きが一番収まる。
        self.setCameraPosition(distance=UNIT * 1.55, elevation=10, azimuth=0)

    def reset_camera(self) -> None:
        self._home()
        self.update()


class LiveBrainView(gl.GLViewWidget):
    """走らせながら光らせる点群のビュー。

    `Brain3DView` と違って **中身を作り直さない**。点の位置は一度渡したら
    固定で、毎フレーム書き換えるのは色と大きさだけ (45,000 点で 1 回 3 ms)。
    シーンごと組み直すと毎フレーム GL のバッファを作り直すことになる。

    色は「どの入力で光っているか」で分けてある。どの点がどの領域かは
    実測 (Male CNS の primary_post) だが、**色の割り当て自体は読みやすさの
    ための見せ方**で、データの主張ではない。
    """

    # 入力の種類 -> (暗いとき, 光ったとき)。
    # 暗い側を真っ暗にすると「そこに脳が無い」ように見えるので、形だけは
    # 分かる明るさを残す (点の位置は実測なので、暗い点も情報ではある)
    TINTS = (("#48536b", "#e8eef7"),   # 0 入力を作っていない領域
             ("#33506f", "#7fc8ff"),   # 1 視葉 (複眼の像)
             ("#5d472a", "#ffb648"),   # 2 触角葉 (匂い)
             ("#4b3a63", "#c58bff"),   # 3 外側角 (好み)
             ("#2c5a3c", "#58e08a"))   # 4 神経核 (関節の動き)

    def __init__(self) -> None:
        super().__init__()
        self.setBackgroundColor(theme.BG)
        self.opts["fov"] = 50
        self._scatter = None
        self._n = 0
        self._dim = None
        self._hot = None
        self._rgba = None
        # ゆっくり回すと奥行きが出て、どの葉が光っているか読みやすい。
        # ただし **自分で回し始めたら止める** — 勝手に動くと狙った向きに
        # 合わせられない
        self.auto = True

    # ------------------------------------------------------------------
    def set_cloud(self, pts: np.ndarray, tint: np.ndarray) -> None:
        """点の位置 (nm) と、点ごとの色分け (TINTS の添字) を渡す。"""
        if self._scatter is not None:
            self.removeItem(self._scatter)
            self._scatter = None
        pts = np.asarray(pts, dtype=np.float64)
        if not len(pts):
            self._n = 0
            return

        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = (lo + hi) / 2.0
        scale = UNIT / (float(np.max(hi - lo)) or 1.0)
        q = (pts - center) * scale
        # Male CNS の座標は +y が腹側 (下)。GL は z が上なので入れ替えて、
        # 脳が上・神経索が下に見えるようにする。
        p = np.column_stack([q[:, 0], q[:, 2], -q[:, 1]]).astype(np.float32)

        t = np.asarray(tint, dtype=int)
        self._dim = np.array([theme.rgba(c[0], 1.0) for c in self.TINTS],
                             dtype=np.float32)[t]
        self._hot = np.array([theme.rgba(c[1], 1.0) for c in self.TINTS],
                             dtype=np.float32)[t]
        self._n = len(p)
        self._rgba = self._dim.copy()
        self._scatter = gl.GLScatterPlotItem(pos=p, color=self._rgba,
                                             size=np.full(self._n, 1.6,
                                                          dtype=np.float32),
                                             pxMode=True)
        # additive だと点が重なった所が白く飛んで、どこが光っているか読めない
        self._scatter.setGLOptions("translucent")
        self.addItem(self._scatter)
        self.opts["center"] = QVector3D(0, 0, 0)
        self.reset_camera()

    def set_brightness(self, b: np.ndarray) -> None:
        """明るさ 0..1 (N,) で色と大きさを塗り替える。"""
        if self._scatter is None or self._n == 0:
            return
        x = np.clip(np.asarray(b, dtype=np.float32), 0.0, 1.0)[:self._n]
        if len(x) < self._n:
            x = np.pad(x, (0, self._n - len(x)))
        w = x[:, None]
        rgba = self._dim * (1.0 - w) + self._hot * w
        rgba[:, 3] = 0.26 + 0.62 * x
        size = 1.3 + 4.0 * x ** 1.5
        self._scatter.setData(color=rgba, size=size)

    # ------------------------------------------------------------------
    def spin(self, deg: float) -> None:
        """少しずつ回す。`auto` が False なら何もしない。"""
        if not self.auto:
            return
        self.opts["azimuth"] = (self.opts.get("azimuth", 0.0) + deg) % 360.0
        self.update()

    def mousePressEvent(self, ev):
        self.auto = False
        super().mousePressEvent(ev)

    def reset_camera(self) -> None:
        # 正面やや上から。脳が上、神経索が下に伸びる向き
        self.setCameraPosition(distance=UNIT * 1.12, elevation=12, azimuth=-72)
        self.update()
