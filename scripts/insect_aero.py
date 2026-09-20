"""昆虫の準定常翼素理論による翅の空気力 (Sane & Dickinson 2002)。

MuJoCo の楕円体流体モデルは、昆虫が揚力を稼ぐ **前縁渦 (LEV)** も
**回転揚力 (Kramer効果)** も表現しない。そのため係数を水増ししないと浮かず、
水増ししても力の向きが正しくならなかった (README の検証記録を参照)。

ここでは MuJoCo の流体を切り、翅を翼素に分割して次の3項を毎ステップ計算し、
外力として与える:

  1. 並進力    dL, dD = 1/2 rho C_L,D(alpha) c dr U^2
       C_L, C_D は Dickinson et al. (1999) がショウジョウバエの翅の
       力学的スケールモデルで実測した値。前縁渦の効果込みで
       C_L は最大 1.8 に達する (定常翼理論の約2倍)。
  2. 回転揚力  dF_rot = C_rot rho omega_rot U c^2 dr      (打ち返しの揚力)
  3. 付加質量  dF_am  = rho pi/4 c^2 dr d(U_n)/dt

座標系: 翅の流体ジオム (楕円体) の局所系を使う。
    局所 z = 翅の長さ方向 (スパン)
    局所 y = 翅弦方向 (コード)
    局所 x = 翅面の法線
"""

from __future__ import annotations

import numpy as np

# Dickinson et al. (1999), Science 284:1954 — ショウジョウバエ翅の実測係数
# alpha は迎角 [deg]
def C_L(alpha_deg):
    return 0.225 + 1.58 * np.sin(np.radians(2.13 * alpha_deg - 7.2))


def C_D(alpha_deg):
    return 1.92 - 1.55 * np.cos(np.radians(2.04 * alpha_deg - 9.82))


# 回転揚力の係数 C_rot = pi (0.75 - x_hat_0)。回転軸が翅弦の 1/4 なら約 1.57
C_ROT = np.pi * (0.75 - 0.25)


# --- 速度の話 ---------------------------------------------------------------
# 翼素は片翼6個しかないので計算量は無いに等しいが、NumPy は配列1つにつき
# 数 us の呼び出し費用がかかる。翅ごとに回していた最初の版は 1 ステップ 455 us で、
# MuJoCo 本体の 187 us を超えて **全体の 7割** を占めていた。
# 左右をまとめて 257 us、さらに numba で回すと 1 桁下がる。
# 学習は数千回の試行を回すので、ここの速度がそのまま実験できる回数になる。
try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except ImportError:                                  # numba が無くても動く
    _HAVE_NUMBA = False

    def _njit(*a, **k):
        def deco(f):
            return f
        return deco


@_njit(cache=True, fastmath=False)
def _aero_kernel(n_hat, s_hat, origin, omega, v_lin, wind, xipos,
                 z, chord, dz, rho, use_rot, F, M, un_out):
    """翼素ごとの力・トルク・空気へのパワーを積む。NumPy 版と同じ式。

    fastmath は切ってある。順序を変えられると NumPy 版との一致が崩れ、
    照合 (`16_validate_aero.py --compare`) が意味を失うため。
    """
    power = 0.0
    n_w = n_hat.shape[0]
    n_e = z.shape[0]
    for w in range(n_w):
        nx, ny, nz_ = n_hat[w, 0], n_hat[w, 1], n_hat[w, 2]
        sx, sy, sz_ = s_hat[w, 0], s_hat[w, 1], s_hat[w, 2]
        ox, oy, oz = omega[w, 0], omega[w, 1], omega[w, 2]
        omega_rot = ox * sx + oy * sy + oz * sz_
        for i in range(n_e):
            rx, ry, rz = z[i] * sx, z[i] * sy, z[i] * sz_
            # v = v_lin + omega x r - wind
            vx = v_lin[w, 0] + (oy * rz - oz * ry) - wind[0]
            vy = v_lin[w, 1] + (oz * rx - ox * rz) - wind[1]
            vz = v_lin[w, 2] + (ox * ry - oy * rx) - wind[2]
            # スパン方向成分を落とす
            vs = vx * sx + vy * sy + vz * sz_
            ux, uy, uz = vx - vs * sx, vy - vs * sy, vz - vs * sz_
            U = np.sqrt(ux * ux + uy * uy + uz * uz)
            ok = U > 1e-9
            Us = U if ok else 1.0
            if ok:
                hx, hy, hz = ux / Us, uy / Us, uz / Us
            else:
                hx = hy = hz = 0.0
            un = ux * nx + uy * ny + uz * nz_
            un_out[w, i] = un
            sin_a = abs(un) / Us
            if sin_a > 1.0:
                sin_a = 1.0
            alpha = np.degrees(np.arcsin(sin_a))
            cl = 0.225 + 1.58 * np.sin(np.radians(2.13 * alpha - 7.2))
            cd = 1.92 - 1.55 * np.cos(np.radians(2.04 * alpha - 9.82))
            if cd < 0.0:
                cd = 0.0
            q = 0.5 * rho * chord[i] * dz * U * U
            # 抗力
            fx, fy, fz = -(cd * q) * hx, -(cd * q) * hy, -(cd * q) * hz
            # 揚力: 流入に直交し、流れが当たっている面から離れる向き
            if un > 0.0:
                sgn = -1.0
            elif un < 0.0:
                sgn = 1.0
            else:
                sgn = 0.0
            ex, ey, ez = sgn * nx, sgn * ny, sgn * nz_
            proj = ex * hx + ey * hy + ez * hz
            lx, ly, lz = ex - proj * hx, ey - proj * hy, ez - proj * hz
            ln = np.sqrt(lx * lx + ly * ly + lz * lz)
            if ln > 1e-9:
                lx, ly, lz = lx / ln, ly / ln, lz / ln
            fx += (cl * q) * lx
            fy += (cl * q) * ly
            fz += (cl * q) * lz
            # 回転揚力
            if use_rot:
                fr = C_ROT * rho * omega_rot * U * chord[i] * chord[i] * dz
                fx += fr * nx
                fy += fr * ny
                fz += fr * nz_
            F[w, 0] += fx
            F[w, 1] += fy
            F[w, 2] += fz
            # トルクは body の重心まわり
            ax = origin[w, 0] + rx - xipos[w, 0]
            ay = origin[w, 1] + ry - xipos[w, 1]
            az = origin[w, 2] + rz - xipos[w, 2]
            M[w, 0] += ay * fz - az * fy
            M[w, 1] += az * fx - ax * fz
            M[w, 2] += ax * fy - ay * fx
            power -= fx * vx + fy * vy + fz * vz
    return power


class WingAero:
    """両翅に翼素理論の力を与える。

    使い方:
        aero = WingAero(model, n_elem=6)
        aero.disable_builtin_fluid(model)   # MuJoCo の流体を切る
        ...
        while ...:
            aero.apply(model, data)
            mujoco.mj_step(model, data)
    """

    GEOMS = ("wing_left_fluid", "wing_right_fluid")

    def __init__(self, model, n_elem: int = 6, rho: float = 1.28e-3,
                 use_rotational: bool = True, use_added_mass: bool = False):
        """use_added_mass は既定で False。

        付加質量を「前ステップとの差分で法線加速度を出す明示的な力」として
        与えると、迎角の急反転時に力が速度と同相になり、**静止空気から
        エネルギーを取り出せてしまう**。実際 CMA-ES はその抜け穴を見つけ、
        空気へのパワー -1999 erg/s (定常飛行では必ず正のはず) の解を作った。

        付加質量は本来ただの慣性なので、`added_mass_inertia()` が返す値を
        翅関節の armature に足す形で入れる。こちらはエネルギーを保存する。
        """
        import mujoco

        self.rho = rho
        self.n_elem = n_elem
        # 風 [cm/s]。翅にあたる空気の速度は「翅の速度 - 風」になる。
        # 匂いを運ぶ風がハエ自身を流さないのは都合が良すぎるので、
        # 空力の側にもちゃんと入れる。
        self.wind = np.zeros(3)
        self.use_rot = use_rotational
        self.use_am = use_added_mass
        self.gids = [model.geom(g).id for g in self.GEOMS]
        self.bids = [model.geom_bodyid[g] for g in self.gids]
        # 左右まとめて配列で扱うための添字 (apply 参照)
        self._gid_arr = np.asarray(self.gids)
        self._bid_arr = np.asarray(self.bids)
        if len(set(self.bids)) != len(self.bids):
            raise ValueError("左右の翅が同じ body に乗っている。apply の加算が壊れる")
        self._vel = np.zeros((len(self.gids), 6))
        self._F = np.zeros((len(self.gids), 3))
        self._M = np.zeros((len(self.gids), 3))
        self._use_kernel = _HAVE_NUMBA

        # 楕円体の半軸: (法線方向, 翅弦方向, スパン方向)
        size = model.geom_size[self.gids[0]]
        self.half_thick, self.half_chord, self.half_span = size

        # スパン方向に等間隔の翼素。中心 z_i と幅 dz
        edges = np.linspace(-self.half_span, self.half_span, n_elem + 1)
        self.z = 0.5 * (edges[:-1] + edges[1:])
        self.dz = edges[1] - edges[0]
        # 楕円翼なので翅弦は spanwise 位置で変わる
        self.chord = 2.0 * self.half_chord * np.sqrt(
            np.clip(1.0 - (self.z / self.half_span) ** 2, 0.0, 1.0))

        self._prev_un = [np.zeros(n_elem), np.zeros(n_elem)]
        self._prev_un_arr = np.zeros((len(self.gids), n_elem))
        self._have_prev = False
        self._mj = mujoco
        # 翅が空気に渡したパワー [erg/s]。定常飛行では必ず正。
        # サーボ内部のダンパーの散逸を含まないので、生理的な飛翔パワーと
        # 直接比べられる (実物のショウジョウバエ 300-800 erg/s)。
        self.air_power = 0.0

    # 空気の物性 (CGS)。fruitfly.xml の既定値と同じ。
    AIR_DENSITY = 1.28e-3
    AIR_VISCOSITY = 1.85e-4

    @classmethod
    def route_wings_to_this_model(cls, model) -> None:
        """翅だけ MuJoCo 内蔵の流体から外し、胴体の空気抵抗は残す。

        翅のジオムを「楕円体流体モデル・係数ゼロ」にすると、
          - そのジオム自身には内蔵流体の力が出ない (係数ゼロなので)
          - その body は内蔵の inertia-box モデルの対象から外れる
        ので、翅の力はこのクラスが独占して与えられる。
        一方 opt.density / viscosity はそのままなので、**胴体・脚・頭・腹部には
        空気抵抗がかかり続ける**。翅だけ力が出て抵抗が無い、という
        都合の良い状態にしないための措置。

        (実測: 機体を 100 cm/s で前進させたときの減速度は
         この処理の前後で -505.7 → -502.8 cm/s^2 とほぼ不変)
        """
        for i in range(model.ngeom):
            model.geom_fluid[i][:] = 0.0
        for gid in [model.geom(g).id for g in cls.GEOMS]:
            model.geom_fluid[gid][0] = 1.0     # 楕円体モデル有効・係数は全部ゼロ
        model.opt.density = cls.AIR_DENSITY
        model.opt.viscosity = cls.AIR_VISCOSITY

    @staticmethod
    def disable_builtin_fluid(model) -> None:
        """後方互換。空気抵抗まで消えるので使わないこと。"""
        for i in range(model.ngeom):
            model.geom_fluid[i][:] = 0.0
        model.opt.density = 0.0
        model.opt.viscosity = 0.0
        model.opt.wind[:] = 0.0

    def apply(self, model, data, dt: float | None = None) -> None:
        """今の状態から空気力を計算し、data.xfrc_applied に入れる。

        実装は3つあり、結果はどれも一致する
        (`16_validate_aero.py --compare` で照合):

            apply          numba カーネル。既定
            _apply_numpy   左右をまとめた NumPy 版 (numba が無ければこちら)
            _apply_loop    翅ごとに回す最初の版。読みやすさ優先で残してある

        付加質量を使うとき (`use_added_mass=True`) は NumPy 版に回す。
        """
        if self.use_am or not self._use_kernel:
            return self._apply_numpy(model, data, dt)

        mujoco = self._mj
        data.xfrc_applied[:] = 0.0
        gids, bids = self._gid_arr, self._bid_arr
        R = data.geom_xmat[gids].reshape(-1, 3, 3)
        n_hat = np.ascontiguousarray(R[:, :, 0])      # 翅面の法線
        s_hat = np.ascontiguousarray(R[:, :, 2])      # スパン方向
        vel = self._vel
        for k, gid in enumerate(self.gids):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM,
                                     gid, vel[k], 0)
        F, M = self._F, self._M
        F[:] = 0.0
        M[:] = 0.0
        self.air_power = _aero_kernel(
            n_hat, s_hat, data.geom_xpos[gids],
            np.ascontiguousarray(vel[:, :3]), np.ascontiguousarray(vel[:, 3:]),
            self.wind, data.xipos[bids], self.z, self.chord, self.dz,
            self.rho, self.use_rot, F, M, self._prev_un_arr)
        data.xfrc_applied[bids, 0:3] += F
        data.xfrc_applied[bids, 3:6] += M
        self._have_prev = True

    def _apply_numpy(self, model, data, dt: float | None = None) -> None:
        """左右の翅をまとめて (2, n_elem) の配列で扱う NumPy 版。"""
        mujoco = self._mj
        dt = model.opt.timestep if dt is None else dt
        data.xfrc_applied[:] = 0.0

        gids, bids = self._gid_arr, self._bid_arr
        R = data.geom_xmat[gids].reshape(-1, 3, 3)
        n_hat, s_hat = R[:, :, 0], R[:, :, 2]        # 法線 / スパン方向
        origin = data.geom_xpos[gids]                # (2,3)

        vel = self._vel
        for k, gid in enumerate(self.gids):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM,
                                     gid, vel[k], 0)
        omega, v_lin = vel[:, :3], vel[:, 3:]

        z = self.z[None, :, None]                    # (1,N,1)
        r = z * s_hat[:, None, :]                    # ヒンジからの腕 (2,N,3)
        pts = origin[:, None, :] + r
        v = v_lin[:, None, :] + np.cross(omega[:, None, :], r) - self.wind

        # スパン方向成分を除いた流入速度 (翼素理論はスパン流を無視)
        u = v - np.einsum("wnc,wc->wn", v, s_hat)[..., None] * s_hat[:, None, :]
        U = np.sqrt(np.einsum("wnc,wnc->wn", u, u))
        ok = U > 1e-9
        Us = np.where(ok, U, 1.0)
        u_hat = np.where(ok[..., None], u / Us[..., None], 0.0)

        # 迎角: 流入と翅面のなす角 (0-90 deg)
        un = np.einsum("wnc,wc->wn", u, n_hat)
        alpha = np.degrees(np.arcsin(np.clip(np.abs(un) / Us, 0.0, 1.0)))

        cl = C_L(alpha)
        cd = np.maximum(C_D(alpha), 0.0)
        q = 0.5 * self.rho * self.chord[None, :] * self.dz * U * U

        dF = -(cd * q)[..., None] * u_hat            # 抗力は流入と逆向き

        # 揚力は流入に直交し、流れが当たっている面から離れる向き
        n_eff = np.sign(-un)[..., None] * n_hat[:, None, :]
        lift_dir = n_eff - np.einsum("wnc,wnc->wn", n_eff, u_hat)[..., None] * u_hat
        ln = np.sqrt(np.einsum("wnc,wnc->wn", lift_dir, lift_dir))
        lift_dir = np.where((ln > 1e-9)[..., None], lift_dir / np.where(ln > 1e-9, ln, 1.0)[..., None],
                            lift_dir)
        dF += (cl * q)[..., None] * lift_dir

        # 回転揚力 (打ち返しで翅がスパン軸まわりに回ることで出る)
        if self.use_rot:
            omega_rot = np.einsum("wc,wc->w", omega, s_hat)[:, None]
            dF_rot = (C_ROT * self.rho * omega_rot * U *
                      self.chord[None, :] ** 2 * self.dz)
            dF += dF_rot[..., None] * n_hat[:, None, :]

        # 付加質量 (翅が押しのける空気の慣性)。既定では使わない (__init__ 参照)
        if self.use_am:
            if self._have_prev:
                dun = (un - self._prev_un_arr) / dt
                dF += (self.rho * np.pi / 4.0 * self.chord[None, :] ** 2 *
                       self.dz * dun)[..., None] * n_hat[:, None, :]
            self._prev_un_arr = un.copy()

        F = dF.sum(axis=1)                            # (2,3)
        arm = pts - data.xipos[bids][:, None, :]      # トルクは body 重心まわり
        M = np.cross(arm, dF).sum(axis=1)
        # 左右の翅は別 body なので添字は重複しない (__init__ で確認済み)
        data.xfrc_applied[bids, 0:3] += F
        data.xfrc_applied[bids, 3:6] += M

        # 空気が翅にした仕事率の符号を反転 = 翅が空気に渡したパワー
        self.air_power = -float(np.einsum("wnc,wnc->", dF, v))
        self._have_prev = True

    def _apply_loop(self, model, data, dt: float | None = None) -> None:
        """翅ごとに回す素直な実装。`apply` の結果を照合するために残してある。"""
        mujoco = self._mj
        dt = model.opt.timestep if dt is None else dt
        data.xfrc_applied[:] = 0.0
        air_power = 0.0

        for k, (gid, bid) in enumerate(zip(self.gids, self.bids)):
            R = data.geom_xmat[gid].reshape(3, 3)
            n_hat = R[:, 0]          # 翅面の法線
            c_hat = R[:, 1]          # 翅弦方向
            s_hat = R[:, 2]          # スパン方向
            origin = data.geom_xpos[gid]

            # ジオム(=翅)の 6D 速度を世界座標で取る
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM,
                                     gid, vel, 0)
            omega, v_lin = vel[:3], vel[3:]

            # 各翼素の位置と速度
            pts = origin[None, :] + self.z[:, None] * s_hat[None, :]
            r = pts - origin[None, :]
            v = v_lin[None, :] + np.cross(omega[None, :], r) - self.wind[None, :]

            # スパン方向成分を除いた流入速度 (翼素理論はスパン流を無視)
            v_span = (v @ s_hat)[:, None] * s_hat[None, :]
            u = v - v_span
            U = np.linalg.norm(u, axis=1)
            ok = U > 1e-9
            if not np.any(ok):
                continue
            u_hat = np.zeros_like(u)
            u_hat[ok] = u[ok] / U[ok, None]

            # 迎角: 流入と翅面のなす角 (0-90 deg)
            un = u @ n_hat                      # 法線方向成分
            sin_a = np.clip(np.abs(un) / np.where(ok, U, 1.0), 0.0, 1.0)
            alpha = np.degrees(np.arcsin(sin_a))

            cl = C_L(alpha)
            cd = np.maximum(C_D(alpha), 0.0)
            q = 0.5 * self.rho * self.chord * self.dz * U ** 2

            # 抗力は流入と逆向き
            dF = -(cd * q)[:, None] * u_hat

            # 揚力は流入に直交し、流れが当たっている面から離れる向き
            n_eff = np.sign(-un)[:, None] * n_hat[None, :]
            lift_dir = n_eff - (np.sum(n_eff * u_hat, axis=1))[:, None] * u_hat
            ln = np.linalg.norm(lift_dir, axis=1)
            good = ln > 1e-9
            lift_dir[good] /= ln[good, None]
            dF += (cl * q)[:, None] * lift_dir

            # 回転揚力 (打ち返しで翅がスパン軸まわりに回ることで出る)
            if self.use_rot:
                omega_rot = float(omega @ s_hat)
                dF_rot = (C_ROT * self.rho * omega_rot * U *
                          self.chord ** 2 * self.dz)
                dF += dF_rot[:, None] * n_hat[None, :]

            # 付加質量 (翅が押しのける空気の慣性)
            if self.use_am:
                if self._have_prev:
                    dun = (un - self._prev_un[k]) / dt
                    dF_am = (self.rho * np.pi / 4.0 * self.chord ** 2 *
                             self.dz * dun)
                    dF += dF_am[:, None] * n_hat[None, :]
                self._prev_un[k] = un.copy()

            F = dF.sum(axis=0)
            # トルクは body の重心まわり
            arm = pts - data.xipos[bid][None, :]
            M = np.cross(arm, dF).sum(axis=0)
            data.xfrc_applied[bid, :3] += F
            data.xfrc_applied[bid, 3:] += M

            # 空気が翅にした仕事率の符号を反転 = 翅が空気に渡したパワー
            air_power -= float(np.sum(dF * v))

        self.air_power = air_power
        self._have_prev = True

    def added_mass_inertia(self, model) -> float:
        """ストローク軸まわりの付加質量の慣性 [g cm^2]。

        翼素ごとに単位長あたり rho*pi/4*c^2 の付加質量があるとして、
        ヒンジからの距離 r の2乗をかけて積分する。
        これを armature に足せば、付加質量をエネルギー保存の形で扱える。
        """
        import mujoco

        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        gid, bid = self.gids[0], self.bids[0]
        R = d.geom_xmat[gid].reshape(3, 3)
        hinge = d.xpos[bid]
        pts = d.geom_xpos[gid][None, :] + self.z[:, None] * R[:, 2][None, :]
        r = np.linalg.norm(pts - hinge[None, :], axis=1)
        return float(np.sum(self.rho * np.pi / 4.0 * self.chord ** 2 * self.dz * r ** 2))

    def reset(self) -> None:
        self._prev_un = [np.zeros(self.n_elem), np.zeros(self.n_elem)]
        self._prev_un_arr = np.zeros((len(self.gids), self.n_elem))
        self._have_prev = False
        self.air_power = 0.0
