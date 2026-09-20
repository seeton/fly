"""ハエの嗅覚 — 匂いの場と、左右の触角での検出。

視覚で花を捉えられるのは 8-10 cm 以内だった (複眼 64px, 視野120度で
直径2.5cmの花)。遠くの花は見えない。だが **切り替わるわけではない**。
目は常に開いているし、近づいても匂いは消えない。役割が違うだけ:

  視覚 … そこに「何か」がある方向。形と位置は分かるが、
         それが自分の好む花かどうかは分からない。
  嗅覚 … 朧げな方向 (風上) と、**好む花かどうかの判定**。

だから嗅覚は近づいたあとも効き続け、視覚が見つけた的へ寄ってよいかを
決める側に回る (scripts/21_choose.py)。

ここでは

  1. 花から匂いが出て、風で下流へ流される「プルーム」を作る
  2. 左右の触角 (`antenna_left` / `antenna_right`) で濃度を読む
  3. 左右差 -> どちらに匂い源があるか
     濃度そのもの -> 匂いの中にいるか

実物のハエの定石は **odor-gated anemotaxis** (匂いを感じたら風上へ進み、
見失ったら横へ振って探す) なので、それも使えるようにしてある。

プルームの形はガウス型の定常解:

    C(x,y,z) = Q / (2 pi sy sz u) * exp(-y^2/2sy^2) * exp(-(z-h)^2/2sz^2)

    sy, sz は風下距離とともに広がる。x が風上側 (源より上流) なら 0。
"""

from __future__ import annotations

import numpy as np

# 風。既定では -x 向きに 8 cm/s。
# 25 cm/s にしていたら、この機体の巡航 (約17 cm/s) では遡れず後退した。
# 実物のショウジョウバエは 20-100 cm/s 出るので 25 cm/s も遡れるが、
# いまの方策の飛行性能に合わせて弱めてある。
# 花は +x にあるので、匂いは花からハエのいる -x 側へ流れてくる。
# ハエはこれを遡れば (= +x へ進めば) 花にたどり着く。
WIND = np.array([-8.0, 0.0, 0.0])


class OdorPlume:
    """花から出る匂いのプルーム (定常ガウスモデル)。"""

    def __init__(self, source, wind=WIND, emission: float = 1.0,
                 sigma0: float = 0.6, spread: float = 0.16):
        self.source = np.asarray(source, dtype=float)
        self.wind = np.asarray(wind, dtype=float)
        self.u = float(np.linalg.norm(self.wind))
        self.dir = self.wind / max(self.u, 1e-9)      # 風下向きの単位ベクトル
        self.Q = float(emission)
        self.sigma0 = float(sigma0)
        self.spread = float(spread)

    def concentration(self, pos) -> float:
        """位置 pos での匂い濃度 (源で 1 になるよう正規化した相対値)。"""
        rel = np.asarray(pos, dtype=float) - self.source
        down = float(rel @ self.dir)                  # 風下方向の距離
        if down <= 1e-3:
            return 0.0                                # 風上には流れない
        lateral = rel - down * self.dir
        s = self.sigma0 + self.spread * down          # 距離とともに広がる
        r2 = float(lateral @ lateral)
        c = (self.Q / (2.0 * np.pi * s * s)) * np.exp(-r2 / (2.0 * s * s))
        # 源のすぐ下流での値で正規化しておく (扱いやすい大きさにするため)
        norm = self.Q / (2.0 * np.pi * self.sigma0 ** 2)
        return float(c / norm)

    def upwind_dir(self) -> np.ndarray:
        """風上向きの単位ベクトル。匂いを感じたらこちらへ進む。"""
        return -self.dir


class FlyNose:
    """左右の触角で匂いを読む。

    受容体は複数ある。ハエは「匂いがするか」だけでなく
    **どの匂いか** を区別していて、好む花かどうかをそれで決める。
    ここでは plumes を辞書 {名前: OdorPlume} で受け取り、
    それぞれの濃度を別チャンネルとして返す。
    """

    ANTENNAE = ("antenna_left", "antenna_right")

    def __init__(self, model, plume, tau: float = 0.02):
        if isinstance(plume, dict):
            self.plumes = dict(plume)
        else:
            self.plumes = {"main": plume}
        self.plume = next(iter(self.plumes.values()))
        self.ids = []
        for n in self.ANTENNAE:
            try:
                self.ids.append(model.body(n).id)
            except Exception:
                self.ids.append(None)
        self.tau = tau
        self.reset()

    def reset(self) -> None:
        self.c = np.zeros(2)                     # 左右の濃度 (全チャンネルの和)
        self._raw = np.zeros(2)
        # 匂いの種類ごとの濃度 (左右の平均)
        self.channels = {k: 0.0 for k in self.plumes}

    def update(self, model, data, dt: float) -> None:
        a = min(dt / max(self.tau, 1e-6), 1.0)
        tot = np.zeros(2)
        for name, pl in self.plumes.items():
            vals = np.zeros(2)
            for i, bid in enumerate(self.ids):
                if bid is not None:
                    vals[i] = pl.concentration(data.xpos[bid])
            tot += vals
            self.channels[name] += a * (float(vals.mean()) - self.channels[name])
        self._raw = tot
        self.c += a * (self._raw - self.c)

    def preference(self, like: str, eps: float = 1e-6) -> float:
        """好む匂いがどれだけ優勢か。 +1 = 好む匂いだけ, -1 = 嫌う匂いだけ。

        実物のハエは「匂いがあるか」だけでなく **どの匂いか** で
        その花に寄るかどうかを決める。視覚は形と位置しか教えてくれない。
        """
        good = self.channels.get(like, 0.0)
        other = sum(v for k, v in self.channels.items() if k != like)
        if good + other < eps:
            return 0.0
        return float((good - other) / (good + other))

    # --- 制御に使う量 ---
    @property
    def strength(self) -> float:
        """匂いの強さ (左右の平均)。これが立てば「プルームの中」。"""
        return float(self.c.mean())

    @property
    def bearing(self) -> float:
        """左右差。正なら右の触角が強い = 匂い源は右寄り。

        触角の間隔は 0.05 cm しかないので差はごく小さい。
        実物のハエもこの左右差だけで進む向きを決めているわけではなく、
        主に「匂いがあるか無いか」で風上へ進むかどうかを切り替えている。
        """
        tot = float(self.c.sum())
        if tot <= 1e-9:
            return 0.0
        return float((self.c[1] - self.c[0]) / tot)

    def body_frame_upwind(self, data, body_id) -> float:
        """機体から見た風上の向き。正なら左へ回れば風上。

        戻り値は sin(角度) 相当で -1..+1。
        """
        R = data.xmat[body_id].reshape(3, 3)
        up = self.plume.upwind_dir()
        fwd = R[:, 0]
        left = R[:, 1]
        return float(np.clip(up @ left, -1.0, 1.0)), float(up @ fwd)
