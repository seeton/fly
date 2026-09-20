"""飛行時の脚のたたみ方 (flybody 同梱の flight pose) をモデルに適用する。

実物のハエは飛ぶとき脚をたたむ。ところが `fruitfly.xml` の脚アクチュエータは
位置サーボで、指令を 0 のままにすると **脚を広げた着地姿勢のまま飛ぶ**。
これは
  - 余計な空気抵抗を生む
  - 重心を下・後ろにずらす
ので、機体の釣り合い姿勢を狂わせる。

flygym には `assets/model/flybody/pose/flight/yaw_roll_pitch.yaml` という
飛行姿勢が同梱されている (脚6本 + 口吻)。関節名が
`c_thorax-lf_coxa-pitch` 形式なのに対し XML 側は `coxa_T1_left` 形式なので、
関節軸 (yaw=z, roll=x, pitch=y) を見て対応づける。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

SIDE = {"lf": ("T1", "left"), "lm": ("T2", "left"), "lh": ("T3", "left"),
        "rf": ("T1", "right"), "rm": ("T2", "right"), "rh": ("T3", "right")}
# 脚の軸の対応。翅は yaw=z, roll=x, pitch=y だが、**脚は roll と pitch が逆**。
# 根拠: 中脚の姿勢データ roll=-0.742 は coxa_twist (y軸, 可動域 +-0.8) にしか
# 収まらず、coxa (x軸, [-0.20,+0.90]) に入れると可動域外になる。
# 自由度が1つの tibia / tarsus が一律 "pitch" と呼ばれ軸が x なのとも整合する。
AXIS_OF = {"yaw": np.array([0.0, 0.0, 1.0]),
           "roll": np.array([0.0, 1.0, 0.0]),
           "pitch": np.array([1.0, 0.0, 0.0])}
SEGMENT = {"coxa": "coxa", "trochanterfemur": "femur",
           "tibia": "tibia", "tarsus1": "tarsus"}


def _flight_pose() -> dict[str, float]:
    import flygym
    p = (Path(flygym.assets_dir) / "model" / "flybody" / "pose" /
         "flight" / "yaw_roll_pitch.yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data["joint_angles"] if "joint_angles" in data else data


def leg_targets(model) -> dict[str, float]:
    """XML の関節名 -> 飛行姿勢の角度 [rad]。脚の関節だけ返す。"""
    pose = _flight_pose()
    # XML 側の脚関節を (セグメント, T番号, 左右) -> [関節名] に整理
    by_key: dict[tuple, list[str]] = {}
    for i in range(model.njnt):
        n = model.joint(i).name
        m = re.match(r"([a-z]+?)(?:_(abduct|twist))?_?(T[123])_(left|right)$", n)
        if not m:
            continue
        seg, mod, t, side = m.groups()
        if seg.startswith("tarsus") and seg != "tarsus":
            continue          # tarsus2..5 は飛行姿勢に指定が無い
        by_key.setdefault((seg, t, side), []).append(n)

    out: dict[str, float] = {}
    for key, val in pose.items():
        m = re.match(r"(?:c_thorax|(\w\w)_(\w+))-(\w\w)_(\w+)-(yaw|roll|pitch)$", key)
        if not m:
            continue
        _, _, sidecode, child_seg, axis = m.groups()
        if sidecode not in SIDE:
            continue
        t, side = SIDE[sidecode]
        seg = SEGMENT.get(child_seg)
        if seg is None:
            continue
        cands = by_key.get((seg, t, side), [])
        if not cands:
            continue
        if len(cands) == 1:
            # 自由度が1つしかない関節 (tibia, tarsus1)。姿勢データ側は
            # 一律 "pitch" と呼ぶが XML の軸は x なので、軸で照合すると
            # 外れる。候補が1つなら迷う余地が無いので直接対応づける。
            out[cands[0]] = float(val)
            continue
        # 複数自由度 (coxa は3, femur は2) は軸で選ぶ。
        # XML の並びは wing と同じ yaw=z, roll=x, pitch=y。
        want = AXIS_OF[axis]
        best, best_dot = None, -1.0
        for n in cands:
            ax = model.jnt_axis[model.joint(n).id]
            dot = abs(float(ax @ want)) / (np.linalg.norm(ax) + 1e-12)
            if dot > best_dot:
                best, best_dot = n, dot
        if best is not None and best_dot > 0.9:
            out[best] = float(val)
    return out


def apply(model, data=None, hold: bool = True) -> int:
    """飛行姿勢を脚アクチュエータの目標値として設定する。

    hold=True なら data.ctrl にも書き込む (以後その姿勢を保つ)。
    戻り値は設定できた関節の数。
    """
    targets = leg_targets(model)
    n = 0
    for jname, val in targets.items():
        try:
            act = model.actuator(jname)
        except Exception:
            continue
        lo, hi = model.actuator_ctrlrange[act.id]
        v = float(np.clip(val, lo, hi)) if hi > lo else float(val)
        if data is not None and hold:
            data.ctrl[act.id] = v
        jid = model.joint(jname).id
        model.qpos0[model.jnt_qposadr[jid]] = v
        n += 1
    return n
