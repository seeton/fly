"""ハエの複眼 — 実測の光学と、T4/T5 がやっている相関型の動き検出。

以前の版は「視野140度を96画素で鋭く描き、画像全体をずらして
一番合う位置を探す」という作りだった。2つとも実物と違っていた。

**光学**
    ショウジョウバエの個眼間角は約 5 度、受容角 (1個の個眼が見込む角、
    FWHM) も約 5 度で、片眼 約750個。以前の実装にはこのボケが無かった。

    しかも「視野140度 / 96画素 = 1.46 度/画素」という見積り自体が誤りだった。
    **透視投影では画素あたりの角度が一様ではない**。視野140度・96画素なら
        中央  3.28 度/画素      視野端  0.40 度/画素
    つまり中央は個眼間角 5 度に対して 1.5倍しか細かくなく、
    周辺は 12倍細かい。エイリアシングは主に視野の周辺で起きていた。

    ここでは各個眼のガウス型受容野を **角度の上で** 作り、受け取り行列 M
    にまとめて
        個眼の像 = M @ 描画像 @ M.T
    で一度に畳み込む。画素の見込み角は tan で求めるので、透視投影の
    歪みもここで吸収される。描画コストは解像度にほぼ依存しない
    (実測 96px 155ms / 256px 174ms、シーン構築が支配的) ので、
    中央でも受容角を表現できるまで解像度を上げてよい。

**動き検出**
    画像ずらしは実物のアルゴリズムではない。正しい光学を入れると
    S/N が 0.50 -> 0.03 まで落ちて使い物にならなかった。
    実物は相関型検出器 (Hassenstein-Reichardt) で、隣り合う個眼の
    **遅らせた信号と生の信号を掛け合わせる**。T4/T5 の演算がこれにあたる。
    入れ替えたら S/N 0.50、真の針路との符号一致 100% に戻った。

**単位を持たせない**
    以前は「ヨー角速度 = -8.6 x 流れ」と、物理エンジンの真値に対して
    較正した係数を使っていた。実物のハエは自分の真の角速度を知らない。
    ここでは検出器の出力をそのまま (任意単位で) 出し、
    運動へのゲインは学習に任せる。

使い方:
    eyes = FlyEyes(model)
    eyes.maybe_update(model, data, t)
    eyes.rotation_signal     # 左右の和 = 自分の回転 (正 = 像が右へ流れる)
    eyes.expansion_signal    # 左右の差 = どちら側が迫っているか
"""

from __future__ import annotations

import numpy as np

EYES = ("eye_left", "eye_right")
FRONT = "eye_front"

# --- 実測値 (Drosophila melanogaster) ---
FOV_DEG = 140.0     # flybody の eye_left / eye_right の視野
DPHI_DEG = 5.0      # 個眼間角 (前方でおよそ 4.5-5.7 度)
DRHO_DEG = 5.1      # 受容角 FWHM (およそ 4.5-5.5 度)
TAU_PHOTO = 0.008   # 光受容器 + LMC の低域通過
TAU_EMD = 0.035     # 相関型検出器の遅延フィルタ (20-50 ms)


def acceptance_matrix(res: int, fov_deg: float = FOV_DEG,
                      dphi_deg: float = DPHI_DEG,
                      drho_deg: float = DRHO_DEG) -> np.ndarray:
    """描画画素 -> 個眼 の受け取り行列 (n_omma, res)。

    各行が1個の個眼の受容野 (角度方向のガウス、FWHM = 受容角)。
    透視投影の画素角度を tan で求めているので、視野端の圧縮も入る。
    """
    half = np.radians(fov_deg) / 2.0
    u = (np.arange(res) + 0.5) / res * 2.0 - 1.0
    pix = np.arctan(u * np.tan(half))                     # 画素の見込み角 [rad]
    n = int(np.floor(fov_deg / dphi_deg))
    omm = np.radians((np.arange(n) - (n - 1) / 2.0) * dphi_deg)
    sig = np.radians(drho_deg) / 2.3548                   # FWHM -> 標準偏差
    M = np.exp(-((omm[:, None] - pix[None, :]) ** 2) / (2 * sig * sig))
    M /= M.sum(axis=1, keepdims=True)
    return M


class FlyEyes:
    """複眼の像 (個眼解像度) と、相関型検出器の出力。

    oversample: 個眼1つあたり何画素で描くか。透視投影では中央の画素が
        いちばん角度的に粗いので、**中央で受容角を表現できる**ところまで
        上げる必要がある。既定の 9 (= 252画素) で中央 1.25 度/画素、
        受容角 5.1 度に対して約4画素ぶん。3 では中央 3.75 度/画素になり、
        ガウス型の受容野が1画素に潰れて意味をなさない。
        描画コストはほぼ変わらないので上げても損しない。
    """

    def __init__(self, model, rate_hz: float = 200.0,
                 dphi_deg: float = DPHI_DEG, drho_deg: float = DRHO_DEG,
                 tau_photo: float = TAU_PHOTO, tau_emd: float = TAU_EMD,
                 oversample: int = 9, fov_deg: float = FOV_DEG,
                 res: int | None = None, normalize: bool = True):
        import mujoco

        self._mj = mujoco
        self.normalize = normalize
        self.n_omma = int(np.floor(fov_deg / dphi_deg))
        self.res = int(res if res is not None else self.n_omma * oversample)
        self.M = acceptance_matrix(self.res, fov_deg, dphi_deg, drho_deg)
        self.dt_vis = 1.0 / rate_hz
        self.tau_photo = tau_photo
        self.tau_emd = tau_emd
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.res)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.res)
        self.renderer = mujoco.Renderer(model, height=self.res, width=self.res,
                                        max_geom=3000)

        # 複眼には **風景だけ** を描く。ハエ自身の見た目メッシュ (group 1, 85個) を
        # 毎フレーム組み立てると 1回 490 ms かかり、解像度を下げても変わらない
        # (シーン構築が支配的)。風景は group 0。影も切る。
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[0] = 1
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
        try:
            model.camera(FRONT)
            self._has_front = True
        except Exception:
            self._has_front = False
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        n = self.n_omma
        self.photo = [np.zeros((n, n)), np.zeros((n, n))]    # 光受容器の出力
        self.delayed = [np.zeros((n, n)), np.zeros((n, n))]  # 検出器の遅延側
        self._seeded = [False, False]
        self._next_t = 0.0
        self.emd = np.zeros(2)          # 左右それぞれの動き検出器出力
        self.frames = [np.zeros((n, n)), np.zeros((n, n))]
        self.target_x = np.zeros(2)
        self.target_size = np.zeros(2)
        self.front_x = 0.0
        self.front_size = 0.0
        self.front_frame = np.zeros((self.res, self.res))
        self._rgb = None

    # ------------------------------------------------------------------
    def _grab(self, model, data, cam):
        self.renderer.update_scene(data, camera=cam, scene_option=self.opt)
        rgb = self.renderer.render().astype(np.float32) / 255.0
        self._rgb = rgb
        return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def _ommatidia(self, lum):
        """描画像 -> 個眼の像。受容角のボケと個眼間隔の間引きを同時に行う。"""
        return self.M @ lum @ self.M.T

    @staticmethod
    def _target_mask(rgb):
        """花 (ピンク) らしい画素。草は緑、地面は茶なので赤が緑を上回る所。"""
        return (rgb[:, :, 0] - rgb[:, :, 1]) > 0.12

    @staticmethod
    def _emd_sum(photo, delayed, normalize: bool = True) -> float:
        """隣り合う個眼の相関型検出器 (Hassenstein-Reichardt)。

        遅らせた左チャンネル x 生の右チャンネル から逆向きを引く。
        像が右へ流れると正。実物では T4 (明) / T5 (暗) がこれを計算し、
        LPTC (HS/VS) が視野全体で足し合わせている。

        normalize: 像のコントラストの2乗で割る。相関型検出器の出力は
            そのままだと **コントラストの2乗に比例** してしまい、
            暗い茂みの前と明るい空の前で同じ回転が桁違いの信号になる。
            実物の LPTC はコントラスト利得制御を持っていてほぼ
            コントラスト非依存なので、それに合わせる。
            これを入れないと、後段のゲインが場面ごとに意味を変えてしまう。
        """
        a, b = photo[:, :-1], photo[:, 1:]
        da, db = delayed[:, :-1], delayed[:, 1:]
        raw = float(np.mean(da * b - db * a))
        if not normalize:
            return raw
        var = float(photo.var())
        return raw / (var + 1e-6)

    # ------------------------------------------------------------------
    def maybe_update(self, model, data, t: float) -> bool:
        """視覚の更新時刻なら描画して動き検出器を回す。更新したら True。"""
        if t < self._next_t:
            return False
        dt = self.dt_vis
        self._next_t = t + dt
        a_p = 1.0 - np.exp(-dt / self.tau_photo)
        a_d = 1.0 - np.exp(-dt / self.tau_emd)

        for i, cam in enumerate(EYES):
            lum = self._grab(model, data, cam)
            om = self._ommatidia(lum)
            self.frames[i] = om

            mask = self._target_mask(self._rgb)
            n_hit = int(mask.sum())
            self.target_size[i] = n_hit / mask.size
            if n_hit:
                cx = float(np.average(np.arange(mask.shape[1]),
                                      weights=mask.sum(axis=0)))
                self.target_x[i] = (cx / (mask.shape[1] - 1)) * 2.0 - 1.0
            else:
                self.target_x[i] = 0.0

            if not self._seeded[i]:
                self.photo[i] = om.copy()
                self.delayed[i] = om.copy()
                self._seeded[i] = True
                continue
            self.photo[i] += a_p * (om - self.photo[i])
            self.emd[i] = self._emd_sum(self.photo[i], self.delayed[i],
                                        self.normalize)
            self.delayed[i] += a_d * (self.photo[i] - self.delayed[i])

        if self._has_front:
            lum = self._grab(model, data, FRONT)
            self.front_frame = lum
            mask = self._target_mask(self._rgb)
            n = int(mask.sum())
            self.front_size = n / mask.size
            if n:
                cx = float(np.average(np.arange(mask.shape[1]),
                                      weights=mask.sum(axis=0)))
                self.front_x = (cx / (mask.shape[1] - 1)) * 2.0 - 1.0
            else:
                self.front_x = 0.0
        return True

    # --- 制御に使う量 (どれも任意単位。ゲインは学習に任せる) ---
    @property
    def rotation_signal(self) -> float:
        """左右の和。自分が回っていると残る (並進の流れは左右で逆符号で消える)。

        正 = 像が右へ流れている。
        """
        return float(self.emd[0] + self.emd[1])

    @property
    def expansion_signal(self) -> float:
        """左右の差。どちら側の像が速く動いているか (衝突の手がかり)。"""
        return float(self.emd[0] - self.emd[1])

    @property
    def target_bearing(self) -> float:
        """花の見える向き。負なら左、正なら右。見えていなければ 0。"""
        w = self.target_size
        tot = float(w.sum())
        if tot <= 0:
            return 0.0
        return float((w[1] - w[0]) / tot)

    @property
    def target_visible(self) -> bool:
        return bool(self.target_size.sum() > 1e-4)

    def front_bearing(self, min_size: float = 2e-4) -> float:
        """正面カメラで見た的の横位置。負なら左、正なら右。"""
        if self.front_size < min_size:
            return 0.0
        return float(self.front_x)

    def side_by_side(self, scale: int = 1) -> np.ndarray:
        """左右の個眼の像を並べた画像 (uint8, 動画への貼り込み用)。

        ここに出るのが **ハエが実際に解像できる像**。粗くてボケている。
        """
        img = np.concatenate([self.frames[0], self.frames[1]], axis=1)
        lo, hi = float(img.min()), float(img.max())
        if hi - lo > 1e-6:
            img = (img - lo) / (hi - lo)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if scale > 1:
            img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        return np.stack([img] * 3, axis=-1)
