"""走らせながら見るシミュレーション — 脳・体・視界を同じ時計で回す。

`out/` の動画 (23/24/25) は「走らせて記録 → 描き直す」の2パスだった。ここは
アプリの中で **止めずに回し続ける** ほうで、画面の3つのパネルは全部おなじ
1本のシミュレーションから出ている:

    脳     いまの感覚入力で光る点群 (`scripts/brain_glow.py`)
    体     MuJoCo の描画
    視界   複眼が解像している像 (`scripts/fly_vision.py`)

**なぜスロー再生になるのか**

    翅は 200 Hz で打っていて、物理の刻みは 2e-5 s (`CLAUDE.md`: 安易に上げない)。
    この機械での実測は 約 2,500 step/s、つまり **実時間の 0.05 倍** しか進まない。
    複眼の描画 (200 Hz) を足すと 0.02〜0.04 倍まで落ちる。等速では見せられない
    ので、動画と同じように **何倍スローかを実測して画面に出す**。
    数字を固定で書かないのは、機械と設定で変わるから。

**複眼の描画で影を切っている**

    影の計算がこの場面の描画時間の 9 割を占めていた (体の絵 320x240 で
    418 ms -> 47 ms)。影ありのままだと複眼1回が 855 ms かかり、実時間の
    0.005 倍 = 200倍スローになって live にならない。既定では切ってある。
    **`shadows=True` にすれば動画 (out/) と同じ入力になる。** 遅いが、
    見比べたいときのために残してある。

**報酬について**

    報酬は花。花には匂いのプルームがあり (`scripts/fly_smell.py`)、ハエは
    それを嗅いで寄る。だから「報酬を置く」と本当に振る舞いが変わる —
    触角葉が光り、機体が向きを変え、複眼の中で花が大きくなる。
    画面に出す数字は `flight_env.reward_terms` そのもの、つまり
    **学習 (13_train_flight.py) が使ったのと同じ式**。

    ただし **この場で学習はしない**。飛び方は `out/flight_policy.json` の
    固定パラメータで、報酬を見て変わるものではない。キノコ体を光らせないのも
    同じ理由 (学習した価値を扱う場所なので、何も学習していないなら嘘になる)。

mujoco / flygym / numba は **遅延 import** する。固めた exe には入っていない
ので (`build_app.py` の EXCLUDE)、その場合はここが使えないと返すだけにする。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .paths import OUT, SCRIPTS

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DT = 2e-5                    # 物理の刻み [s]。羽ばたきの解像度に必要
VIS_HZ = 200.0               # 複眼の更新 [Hz]

# 感覚のフィルタと報酬を回す頻度 [Hz]。
# **翅の指令だけは毎ステップ (50 kHz) 書く** — 240 Hz の羽ばたきを刻むのに要る。
# それ以外 (匂い、関節速度の包絡、報酬) はいちばん速い時定数でも 5 ms なので
# 1 kHz で足りる。毎ステップ Python で回すと 1 ステップ 869 us かかっていたのが
# 500 us を切る。動画のスクリプト (25_forage_brain.py) は毎ステップ回しているが、
# 一次の指数フィルタなので刻みを変えても同じ値に収束する。
SLOW_HZ = 1000.0
N_SLOW = max(int(round(1.0 / (SLOW_HZ * DT))), 1)
DT_SLOW = N_SLOW * DT
WINGS = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
         "wing_yaw_right", "wing_roll_right", "wing_pitch_right")

# 画質 -> (複眼の描画解像度, 見出しに出す説明)
QUALITY = {
    "軽い": (256, "複眼 256px"),
    "標準": (512, "複眼 512px"),
    "精細": (1028, "複眼 1028px (動画と同じ)"),
}

# 点群の明るさが何から来ているかの断り書き。画面に出す。
GLOW_NOTE = ("光っているのは入力が届いている所だけ — 視葉は複眼に映った像、"
             "触角葉は匂いの濃度、神経核は関節の動き。"
             "入力を作っていない領域は暗いまま")

# --- 領域を光らせるときの物差し (固定) ---
# 走らせながら自分自身の最大で割ると、上がっている間はいつも振り切れて
# 何も分からない。**測って決めた固定の値** を使い、場面が変わっても
# 同じ物差しで比べられるようにする。どれもこの機械での実測から:
#
#   翅関節の速さ (包絡)  ホバリング中 約 625 rad/s
#   脚関節の速さ (包絡)  立っている 0.001 / 飛行中 (畳んだまま) 1.3 / 跳躍 28.5
#   匂いの濃度           源で 1.0、3 cm 風上で 0.18、プルームの縁で 1e-4
WING_FULL = 700.0      # rad/s。羽ばたいていれば明るい
LEG_FULL = 30.0        # rad/s。跳躍で振り切れ、飛行中は暗いまま
ODOR_FLOOR = 1e-3      # 匂いの対数圧縮の下限。実物の受容体も対数的に応える


def odor_level(c: float) -> float:
    """匂いの濃度 0..1 -> 明るさ 0..1。**対数で圧縮する**。

    濃度は源で 1、プルームの縁で 1e-4 と 4 桁またぐ。線形に割ると花に
    触れる寸前まで真っ暗になる。実物の嗅覚受容体も濃度の対数に近い応答を
    するので、ここは実物寄りでもある。
    """
    c = max(float(c), 0.0)
    return float(np.log1p(c / ODOR_FLOOR) / np.log1p(1.0 / ODOR_FLOOR))


def available() -> tuple[bool, str]:
    """この環境でシミュレーションを走らせられるか。

    固めた exe には mujoco も flygym も入っていない。落ちるかわりに
    理由を返して、画面にそのまま出す。
    """
    try:
        import mujoco  # noqa: F401
        import flygym  # noqa: F401
    except Exception as exc:
        return False, (f"シミュレーションに必要なものがありません ({exc})。\n"
                       "リポジトリの .venv から起動してください:\n"
                       "    .\\.venv\\Scripts\\python.exe run_app.py")
    if not (OUT / "flight_policy.json").exists():
        return False, "out/flight_policy.json がありません (学習済みの飛び方)。"
    return True, ""


# ---------------------------------------------------------------- 1コマ

@dataclass
class Frame:
    """画面に配るひとそろい。全部おなじ時刻のもの。"""

    t: float                            # シミュレーション内の時刻 [s]
    slow: float                         # 実測のスロー倍率 (実時間の何分の1か)
    glow: np.ndarray                    # 点群の明るさ (N,) 0..1
    eyes: np.ndarray                    # 複眼の像 (n, 2n, 3) uint8
    body: np.ndarray | None             # 体の絵 (H, W, 3) uint8
    stage: str                          # いま何が起きているか (1行)
    rows: list = field(default_factory=list)      # (見出し, 値) の並び
    regions: list = field(default_factory=list)   # (領域名, 説明, 明るさ)
    reward: list = field(default_factory=list)    # (項目, 値) — 報酬の内訳
    reward_total: float = 0.0           # 積算報酬
    alive: bool = True


# ---------------------------------------------------------------- 基底

class LiveSim:
    """場面の共通部分 — モデル、複眼、描画、脳の明るさ、時計。

    派生クラスが実装するのは4つだけ:
        build()     モデルを組む (self.m を作る)
        on_reset()  MjData を作って初期姿勢にする
        control(t)  1ステップぶんの指令を書く
        sense()     脳に配る領域ごとの信号と、画面に出す数字を返す
    """

    NAME = ""
    SUBTITLE = ""
    # 追従カメラの既定 (距離 cm / 方位 deg / 仰角 deg)。
    # 風景に焼き込んである scene_follow は 6.5 cm 離れていて、体長 0.25 cm の
    # ハエが画面の 2% にしかならない。**ハエを見るための画面** なので、
    # 自前の自由カメラでハエを追い、距離と向きを画面から変えられるようにする。
    CAM = (1.6, 128.0, -12.0)

    def __init__(self, cloud, quality: str = "標準", shadows: bool = False,
                 body_res: tuple[int, int] = (620, 460)):
        import mujoco

        from brain_glow import LiveGlow
        from fly_vision import FlyEyes

        self._mj = mujoco
        self.quality = quality if quality in QUALITY else "標準"
        self.shadows = bool(shadows)
        self.body_res = body_res
        self.cloud = cloud

        self.build()
        self.m.opt.timestep = DT
        res = QUALITY[self.quality][0]
        self.eyes = FlyEyes(self.m, rate_hz=VIS_HZ, res=res)
        if not self.shadows:
            # 影の計算がこの場面では描画時間の9割。切らないと live にならない
            self.eyes.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        W, H = body_res
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, W)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, H)
        self.renderer = mujoco.Renderer(self.m, height=H, width=W)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_dist, self.cam_az, self.cam_elev = self.CAM

        if len(cloud) and cloud.n_omma != self.eyes.n_omma:
            raise ValueError(
                f"個眼の数が点群の retinotopy と合わない "
                f"({self.eyes.n_omma} vs {cloud.n_omma})")
        self.glow = LiveGlow(cloud)

        self._t_glow = 0.0
        self.t = 0.0
        self.reward_total = 0.0
        self.alive = True
        self._vis_updated = False
        # スロー倍率の実測。「絵を1枚作る一巡」にかかった実時間と、その間に
        # 進んだシミュレーション時間を、直近 1 秒ぶん均して出す
        self._wall = self._simmed = 0.0
        self._simmed_raw = 0.0
        self._t_wall = None
        self.reset()

    # -- 派生クラスが埋めるもの ---------------------------------------
    def build(self) -> None:
        raise NotImplementedError

    def on_reset(self) -> None:
        raise NotImplementedError

    def control(self, t: float) -> None:
        raise NotImplementedError

    def sense(self) -> tuple[dict, dict, list, list, str]:
        """(領域ごとの信号, 物差しの組, 画面の行, 報酬の内訳, いまの状態)"""
        raise NotImplementedError

    # -- 共通 ----------------------------------------------------------
    def reset(self) -> None:
        self.t = 0.0
        self._t_glow = 0.0
        self.reward_total = 0.0
        self.alive = True
        self._vis_updated = False
        self._wall = self._simmed = self._simmed_raw = 0.0
        self._t_wall = None
        self.eyes.reset()
        self.glow.reset()
        self.on_reset()
        self._mj.mj_forward(self.m, self.d)
        self.eyes.maybe_update(self.m, self.d, 0.0)

    def step_for(self, budget_s: float) -> None:
        """実時間で budget_s ぶんだけ物理を進める。

        「何ステップ進めるか」ではなく「何秒ぶん回すか」で切るのは、
        複眼の更新が入るステップだけ 10〜300 倍重いため。ステップ数で
        切ると画面の更新間隔が跳ねる。
        """
        if not self.alive:
            return
        t0 = time.perf_counter()
        deadline = t0 + budget_s
        n = 0
        while self.alive:
            self.step_once()
            n += 1
            # 毎回時計を読むと無視できない。16ステップに1回で足りる
            if (n & 15) == 0 and time.perf_counter() >= deadline:
                break
        self._simmed_raw += n * DT

    def step_once(self) -> None:
        d = self.d
        t = self.t
        # 複眼が更新されたステップだけ視覚の量を読み直す。毎ステップ読むと
        # 50 kHz で同じ値を舐めることになるし、視角の計算がそこそこ重い
        self._vis_updated = self.eyes.maybe_update(self.m, d, t)
        self.control(t)
        self.aero.apply(self.m, d)
        self._mj.mj_step(self.m, d)
        self.t = t + DT
        if not np.isfinite(d.qpos).all():
            self.alive = False

    # -- 絵 ------------------------------------------------------------
    def set_view(self, dist: float | None = None, azimuth: float | None = None,
                 elevation: float | None = None) -> None:
        if dist is not None:
            self.cam_dist = float(np.clip(dist, 0.4, 40.0))
        if azimuth is not None:
            self.cam_az = float(azimuth)
        if elevation is not None:
            self.cam_elev = float(np.clip(elevation, -85.0, 85.0))

    def draw_body(self) -> np.ndarray:
        cam = self._cam
        cam.lookat[:] = self.d.qpos[:3]        # 胸部を見る
        cam.distance = self.cam_dist
        cam.azimuth = self.cam_az
        cam.elevation = self.cam_elev
        self.renderer.update_scene(self.d, camera=cam)
        # 影はここでは切る。体の絵では見た目の問題でしかないのに、
        # 描画時間の 9 割を占める (320x240 で 418 ms -> 47 ms)
        self.renderer.scene.flags[self._mj.mjtRndFlag.mjRND_SHADOW] = 0
        return self.renderer.render()

    def draw_eyes(self) -> np.ndarray:
        return self.eyes.side_by_side(scale=1)

    @property
    def slow(self) -> float:
        """実測のスロー倍率。実時間の何分の1で進んでいるか。

        物理だけでなく **絵を作る時間も含める**。含めないと画面に出る数字と
        目の前で進む速さが食い違う (物理だけなら x66、絵まで入れると x126)。
        直近 1 秒ぶんの移動平均。
        """
        if self._simmed <= 0 or self._wall <= 0:
            return 0.0
        return float(self._wall / self._simmed)

    def frame(self, with_body: bool = True) -> Frame:
        """いまの一そろいを作る。"""
        now = time.perf_counter()
        if self._t_wall is not None:
            # 1秒の時定数で均す。1コマごとの揺れをそのまま出すと読めない
            a = min((now - self._t_wall) / 1.0, 1.0)
            self._wall += a * ((now - self._t_wall) - self._wall)
            self._simmed += a * (self._simmed_raw - self._simmed)
        self._t_wall = now
        self._simmed_raw = 0.0

        regions, scales, rows, reward, stage = self.sense()
        dt = max(self.t - self._t_glow, 1e-6)
        self._t_glow = self.t
        self.glow.update(dt, photo=np.stack(self.eyes.photo),
                         motion=np.stack(self.eyes.activity),
                         regions=regions, scales=scales)
        return Frame(t=self.t, slow=self.slow, glow=self.glow.b.copy(),
                     eyes=self.draw_eyes(),
                     body=self.draw_body() if with_body else None,
                     stage=stage, rows=rows,
                     regions=self.region_rows(), reward=reward,
                     reward_total=self.reward_total, alive=self.alive)

    # 画面に並べる領域。**キノコ体をいちばん下に置いて、暗いことを見せる**
    REGION_ROWS = (
        ("ME", "髄質 — 複眼に映った像"),
        ("LO", "ロブラ — 同上 (キアズマで前後が戻る)"),
        ("LOP", "ロブラ板 — 動きの検出 (T4/T5)"),
        ("AL", "触角葉 — 触角に届いた匂い"),
        ("LH", "外側角 — 好む匂いがどれだけ優勢か"),
        ("WTct", "翅の神経核 — 翅関節の動き"),
        ("LegNp", "脚の神経核 — 脚関節の動き"),
        ("CA", "キノコ体 (萼) — 暗いまま。学習していない"),
        ("gL", "キノコ体 (γ葉) — 暗いまま。学習していない"),
    )

    def region_rows(self) -> list:
        out = []
        for key, desc in self.REGION_ROWS:
            idx = self.cloud.region_mask(key)
            if not len(idx):
                continue
            out.append((key, desc, self.glow.level(key), len(idx)))
        return out

    def close(self) -> None:
        for name in ("renderer",):
            r = getattr(self, name, None)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
        eyes = getattr(self, "eyes", None)
        if eyes is not None:
            try:
                eyes.renderer.close()
            except Exception:
                pass


# ---------------------------------------------------------------- 飛ぶ

def _euler(q):
    w, x, y, z = q
    return (np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class FlightSim(LiveSim):
    """草むらを飛んで、匂いのする花 (報酬) に寄る。

    操舵は `21_choose.py` / `25_forage_brain.py` と同じ:

        視覚の流れ (複眼の相関型検出器) で針路を保つ
        好む匂いのときだけ風上へ切る     gate = max(pref, 0)
        花が視野に入ったらそちらへ寄せる

    **花は mocap の body に載っている** ので、走らせたまま置き直せる。
    置き直すとプルームの源も一緒に動く。

    `gate = max(pref, 0)` は「好まない匂いなら寄らない」を実現するが、
    そのとき操舵がゼロになるだけで **離れる動きは無い**。未解決の課題
    (README 参照) で、画面の状態表示にもそう出す。
    """

    NAME = "飛ぶ — 匂いのする花に寄る"
    SUBTITLE = "草むら / 風 -8 cm/s / 花には匂いのプルームがある"
    CAM = (1.6, 128.0, -12.0)

    # 感覚から運動へのゲイン。**手で決めた定数** で学習していない
    # (README の未解決の課題)。学習しているのは羽ばたきと姿勢保持の18個だけ
    K_FLOW, K_ODOR, K_SEE = 0.14, 0.22, 0.22
    SEE_ON, SEE_OFF = 8e-5, 2e-5
    START = (15.0, 5.0, 9.0)
    FAR = np.array([400.0, 300.0, 9.0])   # 「どけておく」位置 (匂いも像も届かない)

    def __init__(self, cloud, **kw):
        from world_scene import DECOY_POS, TARGET_POS

        self.like = "target"
        self.decoy_on = True
        self._flower_pos = {"target": np.array(TARGET_POS, dtype=float),
                            "decoy": np.array(DECOY_POS, dtype=float)}
        super().__init__(cloud, **kw)

    # ------------------------------------------------------------------
    def build(self) -> None:
        from flybody_model import load_model
        from fly_smell import WIND, FlyNose, OdorPlume
        from insect_aero import WingAero
        from world_scene import add_movable_flower

        def extra(spec):
            # 見た目は同じ。違うのは匂いだけ (見分けられると「匂いで選ぶ」
            # 話にならない)。どちらも mocap なので走らせたまま動かせる
            add_movable_flower(spec, self._flower_pos["target"],
                               tag="flower_target")
            add_movable_flower(spec, self._flower_pos["decoy"],
                               tag="flower_decoy", make_materials=False)

        self.m, _ = load_model(wing_kp=50.0, wing_kv=2 * np.sqrt(50.0 * 1e-6),
                               scene=True, scene_flower=False, extra=extra)
        self.aero = WingAero(self.m, n_elem=6)
        self.aero.route_wings_to_this_model(self.m)
        i_add = self.aero.added_mass_inertia(self.m)
        for jn in ("wing_roll_left", "wing_roll_right",
                   "wing_yaw_left", "wing_yaw_right"):
            self.m.dof_armature[self.m.jnt_dofadr[self.m.joint(jn).id]] += i_add

        self.wind = np.array(WIND, dtype=float)
        self.aero.wind = self.wind
        self.m.opt.wind[:] = self.wind

        self.plumes = {k: OdorPlume(v) for k, v in self._flower_pos.items()}
        self.nose = FlyNose(self.m, self.plumes)
        self.mocap = {k: int(self.m.body(f"flower_{k}_body").mocapid[0])
                      for k in self._flower_pos}

        self.policy = json.loads(
            (OUT / "flight_policy.json").read_text(encoding="utf-8"))["params"]
        self.A = {n: self.m.actuator(n).id for n in WINGS}
        self.head = {h: self.m.actuator(h).id for h in ("head_abduct", "head")}
        self.thorax = self.m.body("thorax").id
        self.wing_dofs = np.array(
            [self.m.jnt_dofadr[self.m.joint(n).id] for n in WINGS])

        from escape_scene import leg_dof_groups
        from flight_posture import leg_targets
        self.leg_ctrl = {self.m.actuator(k).id: v
                         for k, v in leg_targets(self.m).items()}
        self.leg_dofs = leg_dof_groups(self.m)

    # ------------------------------------------------------------------
    def on_reset(self) -> None:
        mujoco = self._mj
        self.d = mujoco.MjData(self.m)
        self.d.qpos[0], self.d.qpos[1], self.d.qpos[2] = self.START
        for aid, val in self.leg_ctrl.items():
            self.d.ctrl[aid] = val
            self.d.qpos[self.m.jnt_qposadr[self.m.actuator_trnid[aid, 0]]] = val
        for k in self._flower_pos:
            self._push_flower(k)
        self.aero.reset()
        self.nose.reset()

        p = self.policy
        self._mid = (p["pitch_down"] + p["pitch_up"]) / 2
        self._amp_p = (p["pitch_down"] - p["pitch_up"]) / 2
        self._yaw_vis = self._yaw_vis_f = 0.0
        self._yaw_slow = self._roll_slow = 0.0
        self._see_f = 0.0
        self._seeing = False
        self._wing_env = 0.0
        self._pref = 0.0
        self._steer = 0.0
        self._power_ema = 0.0
        self._z_goal = self.START[2]
        self._leg_speed = {k: 0.0 for k in self.leg_dofs}
        self._k = 0

    # -- 花 (報酬) ------------------------------------------------------
    def _push_flower(self, which: str) -> None:
        """mocap とプルームの源を、いまの位置に合わせる。"""
        pos = (self._flower_pos[which] if (which == "target" or self.decoy_on)
               else self.FAR)
        self.d.mocap_pos[self.mocap[which]] = [pos[0], pos[1], 0.0]
        self.plumes[which].source = np.asarray(pos, dtype=float)

    def move_flower(self, which: str, pos) -> None:
        """花を絶対座標へ移す。プルームの源も一緒に動く。"""
        if which not in self._flower_pos:
            return
        self._flower_pos[which] = np.asarray(pos, dtype=float)
        self._push_flower(which)
        if which == self.like:
            # 高度の目標も花に合わせる。合わせないと「花を高い所に置いても
            # ハエは同じ高さを飛ぶ」ことになり、置いた意味が見えない
            self._z_goal = float(self._flower_pos[which][2])

    def place_reward(self, dist_cm: float, side_cm: float, height_cm: float) -> None:
        """好む匂いの花を、いまのハエの正面 dist_cm・左右 side_cm に置く。

        向きは機体の前方 (+x 軸) ではなく **風上** を基準にする。匂いは
        風下へしか流れないので (`fly_smell.OdorPlume`)、ハエの真横に置いても
        風上に当たれば匂いは届かない。風上側に置いたほうが「置いたら寄る」が
        成り立つ。
        """
        up = -self.wind / max(float(np.linalg.norm(self.wind)), 1e-9)
        left = np.cross(np.array([0.0, 0.0, 1.0]), up)
        pos = np.asarray(self.d.qpos[:3], dtype=float) + up * dist_cm + left * side_cm
        pos[2] = height_cm
        self.move_flower(self.like, pos)

    def set_like(self, which: str) -> None:
        self.like = which if which in self._flower_pos else "target"
        self._z_goal = float(self._flower_pos[self.like][2])

    def set_decoy(self, on: bool) -> None:
        """おとりを出す / どける。

        モデルを組み直すと 2.6 秒かかって飛行が途切れるので、**遠くへ
        どける** ことで代える。400 cm 先なら匂いも像も届かない。
        """
        self.decoy_on = bool(on)
        self._push_flower("decoy")

    def flower_pos(self, which: str) -> np.ndarray:
        return self._flower_pos[which]

    # ------------------------------------------------------------------
    def control(self, t: float) -> None:
        """毎ステップ呼ばれる。**翅の指令だけを書く。**

        240 Hz の羽ばたきを 50 kHz で刻むのがここ。感覚のフィルタと報酬は
        `_slow` で 1 kHz にまとめてある (上の SLOW_HZ を参照)。
        """
        d, p = self.d, self.policy
        self._k += 1
        if self._k % N_SLOW == 0:
            self._slow()

        th = 2 * np.pi * p["freq"] * t
        roll_b, pitch_b, yaw_b = _euler(d.qpos[3:7])
        wx, wy = d.qvel[3], d.qvel[4]

        u_p = np.clip(-p["kp_pitch"] * pitch_b - p["kd_pitch"] * wy, -0.8, 0.8)
        u_r = np.clip(-p["kp_roll"] * roll_b - p["kd_roll"] * wx, -0.5, 0.5)
        u_y = np.clip(-self.K_FLOW * self._yaw_vis_f + self._steer, -0.6, 0.6)
        u_a = np.clip(p["kp_alt"] * (self._z_goal - d.qpos[2])
                      - p["kd_alt"] * d.qvel[2], -0.4, 0.4)

        mean = p["roll_mean"] + u_p
        base = np.clip(p["roll_amp"] + u_a, 0.35, 1.25)
        feather = self._mid + self._amp_p * np.tanh(
            p["sharp"] * np.sin(th + p["phase"]))
        dev = p["yaw"] + p["yaw_amp"] * np.cos(th + p["yaw_phase"])
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            d.ctrl[self.A[f"wing_yaw_{side}"]] = np.clip(dev, -1.5, 1.5)
            d.ctrl[self.A[f"wing_roll_{side}"]] = np.clip(
                mean + (base + sgn * u_r) * np.cos(th), -1.0, 1.5)
            d.ctrl[self.A[f"wing_pitch_{side}"]] = np.clip(
                feather + sgn * u_y, -1.27, 2.92)

    def _slow(self) -> None:
        """1 kHz の側 — 感覚、操舵、頭の安定化、報酬。"""
        m, d = self.m, self.d
        roll_b, _pitch_b, yaw_b = _euler(d.qpos[3:7])

        if self._vis_updated:
            self._yaw_vis = self.eyes.rotation_signal
        a_vis, a_slow, a_env = DT_SLOW / 0.03, DT_SLOW / 0.05, DT_SLOW / 0.005
        self._yaw_vis_f += a_vis * (self._yaw_vis - self._yaw_vis_f)
        self.nose.update(m, d, DT_SLOW)
        self._wing_env += a_env * (float(np.abs(d.qvel[self.wing_dofs]).mean())
                                   - self._wing_env)
        for k, ix in self.leg_dofs.items():
            self._leg_speed[k] += a_env * (float(np.abs(d.qvel[ix]).mean())
                                           - self._leg_speed[k])

        # 頭の安定化 (胴の速い揺れを打ち消す。実物のハエもやっている)
        self._yaw_slow += a_slow * (yaw_b - self._yaw_slow)
        self._roll_slow += a_slow * (roll_b - self._roll_slow)
        d.ctrl[self.head["head_abduct"]] = float(
            np.clip(-(yaw_b - self._yaw_slow), -0.2, 0.2))
        d.ctrl[self.head["head"]] = float(
            np.clip(-(roll_b - self._roll_slow), -0.5, 0.3))

        # --- 視覚と嗅覚を同時に使う ---
        if self.eyes.front_size > self.SEE_ON:
            self._seeing = True
        elif self.eyes.front_size < self.SEE_OFF:
            self._seeing = False
        if self.eyes.front_size > self.SEE_OFF:
            self._see_f += (DT_SLOW / 0.05) * (
                self.eyes.front_bearing(self.SEE_OFF) - self._see_f)

        self._pref = self.nose.preference(self.like)
        gate = max(self._pref, 0.0)          # 好む匂いのときだけ寄る
        left_up, _ = self.nose.body_frame_upwind(d, self.thorax)
        steer = self.K_ODOR * gate * left_up
        if self._seeing:
            steer += -self.K_SEE * gate * self._see_f
        self._steer = steer

        # 報酬。学習と同じ式 (`flight_env.reward_terms`) を同じ重みで積む
        self._power_ema += (DT_SLOW / 5e-3) * (float(self.aero.air_power)
                                               - self._power_ema)
        self._terms = self._reward_terms(roll_b, yaw_b)
        self.reward_total += DT_SLOW * float(sum(self._terms))

    def _reward_terms(self, roll_b: float, yaw_b: float):
        from flight_env import reward_terms

        tgt = self._flower_pos[self.like]
        d = self.d
        dz = abs(float(d.qpos[2]) - float(tgt[2]))
        dxy = float(np.hypot(d.qpos[0] - tgt[0], d.qpos[1] - tgt[1]))
        spin = float(np.linalg.norm(d.qvel[3:6]))
        return reward_terms(dz, dxy, yaw_b, self._power_ema, roll_b, spin)

    def step_once(self) -> None:
        super().step_once()
        if self.alive and self.d.qpos[2] < 0.5:
            self.alive = False

    # ------------------------------------------------------------------
    def sense(self):
        from flight_env import REWARD_TERMS

        d = self.d
        tgt = self._flower_pos[self.like]
        dist = float(np.linalg.norm(np.asarray(d.qpos[:3]) - tgt))
        roll_b, pitch_b, yaw_b = _euler(d.qpos[3:7])

        # どれも 0..1 に均してから渡す (物差しは上の定数。brain_glow 参照)
        regions = {
            "AL(L)": odor_level(self.nose.c[0]),
            "AL(R)": odor_level(self.nose.c[1]),
            "LH": max(self._pref, 0.0),          # 好みはもともと -1..+1
            "WTct": min(self._wing_env / WING_FULL, 1.0) ** 0.6,
        }
        for k, v in self._leg_speed.items():
            seg, side = k.split("_")
            regions[f"LegNp({seg})({'L' if side == 'left' else 'R'})"] = \
                min(v / LEG_FULL, 1.0) ** 0.6
        scales = {}

        rows = [
            ("位置", f"x {d.qpos[0]:6.2f}  y {d.qpos[1]:6.2f}  z {d.qpos[2]:5.2f} cm"),
            ("姿勢", f"ロール {np.degrees(roll_b):+5.0f}  "
                     f"ピッチ {np.degrees(pitch_b):+5.0f}  "
                     f"ヨー {np.degrees(yaw_b):+5.0f} deg"),
            ("速さ", f"{float(np.linalg.norm(d.qvel[:3])):5.1f} cm/s"),
            ("報酬の花まで", f"{dist:5.2f} cm  (花の直径 2.5 cm)"),
            ("匂いの強さ", f"{float(self.nose.strength):.4f}"),
            ("好み", f"{self._pref:+.2f}  "
                     f"({'本命' if self.like == 'target' else 'おとり'}を好む)"),
            ("花の写り", f"{self.eyes.front_size * 100:.3f} %"),
            ("翅のパワー", f"{self._power_ema:6.0f} erg/s  (実物 300-800)"),
        ]
        terms = getattr(self, "_terms", None) or self._reward_terms(roll_b, yaw_b)
        reward = [(name, float(v), hint)
                  for (name, hint), v in zip(REWARD_TERMS, terms)]

        if not self.alive:
            stage = "墜落 — 高度が 0.5 cm を切った"
        elif float(self.nose.strength) <= 1e-3:
            stage = "匂いの外 — 探索中 (視覚の流れで針路だけ保つ)"
        elif self._pref <= 0:
            stage = ("好まない匂い — 寄らない。"
                     + ("花が見えていても操舵はゼロ" if self._seeing
                        else "離れる動きは持っていない (未解決)"))
        elif self._seeing:
            stage = "好む匂い + 花が見えている — 匂いと視覚の両方で寄る"
        else:
            stage = "好む匂い — 風上へ切って寄る (まだ見えていない)"
        return regions, scales, rows, reward, stage


# ---------------------------------------------------------------- 逃げる

class EscapeSim(LiveSim):
    """ハエ叩きが迫ってきて、跳んで逃げる。

    流れは `escape_scene.run_approach` と同じで、**複眼に映った像から**
    見かけの大きさ (視角) を毎フレーム測り、しきい値を超えた時刻を
    「LC4/LPLC2 が Giant Fiber を発火させた時刻」とする。そこから伝導遅れ
    (文献値 1.1 ms) のぶん後に中脚を伸ばす。

    **発火の前後だけ極端に遅くする。** 接近は数百 ms あるのに神経の伝導は
    約 1 ms しかなく、同じ倍率では両方見えない (動画 24 も同じことをしている)。
    """

    NAME = "逃げる — ハエ叩きが迫る"
    SUBTITLE = "地面に立った状態から。視角がしきい値を超えたら跳ぶ"
    # 迫ってくるハエ叩きも入るよう、飛行より少し引いて真横寄りから見る
    CAM = (2.4, 95.0, -8.0)

    APPROACH = 30.0        # 捕食者の速さ [cm/s]
    START_CM = 7.0         # 初期距離 [cm]
    THETA = 45.0           # 逃避を起こす視角 [deg]
    GF_DELAY = 1.1e-3      # GF の伝導 + シナプス遅れ [s] (文献値)
    RISE = 5.0e-3          # 中脚を伸ばす時間 [s]
    AMP = 1.9              # 伸展の大きさ [rad]
    JUMP_FORCE = 3.0       # 伸ばす関節の力の上限 [dyn cm]
    SETTLE = 0.15          # 基準を取る前に立たせて落ち着かせる時間 [s]
    SECONDS = 0.30         # ここまで来たら最初に戻す [s]

    def build(self) -> None:
        from escape_scene import add_arena, leg_dof_groups, set_leg_servos
        from flybody_model import load_model
        from insect_aero import WingAero

        self._start = np.array([self.START_CM, 0.0, 0.9])
        self.m, _ = load_model(scene=False, add_floor=False,
                               extra=lambda sp: add_arena(sp, self._start))
        set_leg_servos(self.m)
        self.aero = WingAero(self.m, n_elem=6)
        self.aero.route_wings_to_this_model(self.m)
        self.femur = [self.m.actuator(f"femur_T2_{s}").id
                      for s in ("left", "right")]
        self.tibia = [self.m.actuator(f"tibia_T2_{s}").id
                      for s in ("left", "right")]
        # 跳ぶ関節だけ力の上限を別に決める。立たせるための 60 dyn cm のままだと
        # 体重 (0.96 dyn) の 60 倍で蹴ることになり、体長の 14 倍も跳んでしまう
        for a in self.femur + self.tibia:
            self.m.actuator_forcerange[a] = [-self.JUMP_FORCE, self.JUMP_FORCE]
        self.leg_dofs = leg_dof_groups(self.m)

    # ------------------------------------------------------------------
    def on_reset(self) -> None:
        mujoco = self._mj
        from escape_scene import dark_mask

        self.d = mujoco.MjData(self.m)
        self.aero.reset()
        settled = getattr(self, "_settled", None)
        if settled is None:
            # 捕食者を遠くに置いたまま、まず脚で立たせて落ち着かせる。これを
            # やらずに初期姿勢で基準を取ると、沈み込みで視界が動いたぶんを
            # 「迫る影」と取り違える (6.5 cm 先の捕食者が視角 72 度と出た)。
            # 7,500 ステップ = 3 秒かかるので、済んだ姿勢を控えておいて
            # 2 回目以降の「最初から」は待たせない
            self.d.qpos[:3] = [0.0, 0.0, 0.245]
            self.d.mocap_pos[0] = [200.0, 0.0, 40.0]
            for _ in range(int(self.SETTLE / DT)):
                mujoco.mj_step(self.m, self.d)
            self.eyes.reset()
            self.eyes.maybe_update(self.m, self.d, 0.0)
            self._base = dark_mask(self.eyes)
            self._settled = (self.d.qpos.copy(), self.d.qvel.copy(),
                             self.d.ctrl.copy())
        else:
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = settled

        # 基準を取ったあとで捕食者を持ってくる。複眼は改めて種を入れ直す
        # (このあと LiveSim.reset が t=0 で 1 回描く)
        self.d.mocap_pos[0] = self._start
        mujoco.mj_forward(self.m, self.d)
        self.eyes.reset()

        self._femur0 = float(
            self.d.qpos[self.m.jnt_qposadr[self.m.actuator_trnid[self.femur[0], 0]]])
        self._theta = 0.0
        self._t_trigger = None
        self._t_extend = None
        self._leg_speed = {k: 0.0 for k in self.leg_dofs}
        self._zmax = float(self.d.qpos[2])
        self._z0 = float(self.d.qpos[2])
        self._k = 0
        self._dir = np.array([-1.0, 0.0, -0.09])
        self._dir /= np.linalg.norm(self._dir)

    # ------------------------------------------------------------------
    def control(self, t: float) -> None:
        """毎ステップ — 捕食者を進めて、跳ぶ指令を書く。

        中脚の伸展は 5 ms で立ち上がるので、ここは毎ステップ刻む。
        視角の測定と脚の包絡は 1 kHz でまとめる (`_slow`)。
        """
        from escape_scene import visual_angle

        d = self.d
        d.mocap_pos[0] = self._start + self._dir * self.APPROACH * t

        if self._vis_updated:
            # 視角は **複眼に映った像から** 測る。幾何の値は使わない
            self._theta = visual_angle(self.eyes, self._base)
            if self._t_trigger is None and self._theta >= self.THETA:
                self._t_trigger = t
                self._t_extend = t + self.GF_DELAY
        if self._t_extend is not None and t >= self._t_extend:
            fr = min((t - self._t_extend) / self.RISE, 1.0)
            for a in self.femur:
                d.ctrl[a] = self._femur0 + self.AMP * fr
            for a in self.tibia:
                d.ctrl[a] = self.AMP * 0.5 * fr

        self._k += 1
        if self._k % N_SLOW == 0:
            a_leg = DT_SLOW / 0.005
            for k, ix in self.leg_dofs.items():
                self._leg_speed[k] += a_leg * (float(np.abs(d.qvel[ix]).mean())
                                               - self._leg_speed[k])
            self._zmax = max(self._zmax, float(d.qpos[2]))

    def step_for(self, budget_s: float) -> None:
        """発火の前後だけ予算を削って、さらにゆっくり見せる。

        接近は数百 ms、神経の伝導は約 1 ms。同じ倍率では両方見えない。
        """
        if self._t_trigger is not None:
            dt_spike = self.t - self._t_trigger
            if -0.010 <= dt_spike <= 0.012:
                budget_s *= 0.04         # 発火まわりを 25 倍ゆっくり
        super().step_for(budget_s)
        if self.t >= self.SECONDS:
            self.reset()

    # ------------------------------------------------------------------
    def sense(self):
        d = self.d
        dist = float(np.linalg.norm(np.asarray(d.mocap_pos[0]) - d.qpos[:3]))
        from escape_scene import PRED_R
        geo = np.degrees(2 * np.arctan(PRED_R / max(dist, 1e-6)))

        # 脚だけ。飛行の場面と **同じ固定の物差し** を使うので、立っている間は
        # 暗く (0.001 rad/s)、跳んだ瞬間に振り切れる (28.5 rad/s)
        regions, scales = {}, {}
        for k, v in self._leg_speed.items():
            seg, side = k.split("_")
            regions[f"LegNp({seg})({'L' if side == 'left' else 'R'})"] = \
                min(v / LEG_FULL, 1.0) ** 0.6

        # 跳んだあとは機体が回って捕食者が視野を出入りするので、視角の数字は
        # 意味を持たない。そのまま出すと誤解を招くので出さない (24 と同じ扱い)
        jumped = self._t_extend is not None and self.t > self._t_extend + self.RISE
        rows = [
            ("捕食者まで", f"{dist:5.2f} cm  (直径 {2*PRED_R:.1f} cm)"),
            ("視角 (複眼で測った)",
             "跳躍中 — 機体が回っているので視角は意味を持たない" if jumped
             else f"{self._theta:5.1f} deg  しきい値 {self.THETA:.0f}"),
            ("視角 (幾何の値)",
             "—" if jumped else
             f"{geo:5.1f} deg  — 受容角 5.1 度で縁がにじむぶん測定値が大きい"),
            ("胸部の高さ", f"{float(d.qpos[2]):.3f} cm  "
                           f"(最大 {self._zmax:.3f} / 出発 {self._z0:.3f})"),
            ("GF 発火", "まだ" if self._t_trigger is None
                        else f"t = {self._t_trigger*1e3:.1f} ms"),
            ("中脚が伸び始める", "—" if self._t_extend is None
                                 else f"t = {self._t_extend*1e3:.1f} ms "
                                      f"(伝導遅れ {self.GF_DELAY*1e3:.1f} ms は文献値)"),
        ]

        if self._t_trigger is None:
            stage = "影が迫る — LC4 / LPLC2 が視角を測っている"
        elif self._t_extend is not None and self.t < self._t_extend:
            stage = "Giant Fiber (DNp01) 発火 — 軸索を降りている"
        elif self.t < (self._t_extend or 0) + self.RISE:
            stage = "TTMn へ伝達 — 中脚が伸びる"
        else:
            stage = "跳んだ"
        return regions, scales, rows, [], stage


SCENES = (FlightSim, EscapeSim)
