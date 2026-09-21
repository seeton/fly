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
