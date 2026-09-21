"""Janelia flybody (翅つきのハエの全身モデル) を MuJoCo に読み込むための補助。

flygym には翅つきモデル `fruitfly.xml` が同梱されているが、そのままでは飛べない:

  1. メッシュのファイル名が flygym 側でリネームされていて解決できない
     (例: XML の wing_left_brown.obj → 配布物では l_wing_brown.obj)
  2. 翅の楕円体「流体ジオム」はあるのに、空力 (fluidshape="ellipsoid") が
     有効になっていない。つまり羽ばたいても力が出ない

ここでその2つを埋める。
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import yaml

FLYGYM_MODEL = Path(mujoco.__file__).parent  # placeholder, 実体は下で解決
_ASSETS = None


def _assets_dir() -> Path:
    global _ASSETS
    if _ASSETS is None:
        import flygym
        _ASSETS = Path(flygym.assets_dir) / "model" / "flybody"
    return _ASSETS


def _mesh_cache() -> Path:
    from flygym.utils.assets_lazy_loading import get_cache_root
    root = Path(get_cache_root())
    cands = sorted(root.glob("flybody_fullsize_meshes_*"))
    if not cands:
        raise FileNotFoundError(
            "flybody のメッシュがありません。次を一度実行してください:\n"
            "  python -c \"from flygym.utils.assets_lazy_loading import prefetch_meshes; prefetch_meshes()\""
        )
    return cands[-1]


SIDE = {"lf": ("left", "t1"), "lm": ("left", "t2"), "lh": ("left", "t3"),
        "rf": ("right", "t1"), "rm": ("right", "t2"), "rh": ("right", "t3"),
        "l": ("left",), "r": ("right",), "c": ()}
LEG = {"T1": "f", "T2": "m", "T3": "h"}


def _toks(name: str) -> set[str]:
    s = re.sub(r"\.obj$", "", name).lower()
    s = re.sub(r"([a-z])(\d)", r"\1_\2", s)
    out: list[str] = []
    for p in (p for p in re.split(r"[^a-z0-9]+", s) if p):
        out.extend(SIDE[p]) if p in SIDE else out.append(p)
    return set(out)


def mesh_mapping() -> dict[str, str]:
    """XML のメッシュ名 → 配布メッシュのファイル名。"""
    assets, cache = _assets_dir(), _mesh_cache()
    xml_text = (assets / "fruitfly.xml").read_text(encoding="utf-8")
    xml_files = sorted(set(re.findall(r'<mesh name="[^"]*" file="([^"]*)"', xml_text)))
    have = {p.name for p in cache.glob("*.obj")}

    rig = yaml.safe_load((assets / "rigging.yaml").read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for seg, info in rig.items():
        fb = info.get("flybody_name")
        if not fb:
            continue
        for gname, g in (info.get("geoms") or {}).items():
            mesh = g.get("mesh")
            if mesh and gname.startswith(seg + "_"):
                mapping[f"{fb}_{gname[len(seg)+1:]}.obj"] = f"{mesh}.obj"
    # XML に実在するキーだけ残す (rigging 由来の幽霊キーが配布名を食うのを防ぐ)
    mapping = {k: v for k, v in mapping.items() if k in xml_files and v in have}

    for f in xml_files:
        if f in mapping:
            continue
        m = re.fullmatch(r"tarsus_(T\d)_(\d)_(left|right)_body\.obj", f)
        if m:
            c = f"{m.group(3)[0]}{LEG[m.group(1)]}_tarsus{m.group(2)}_body.obj"
            if c in have:
                mapping[f] = c
            continue
        m = re.fullmatch(r"tarsal_claw_(T\d)_(left|right)_brown\.obj", f)
        if m:
            c = f"{m.group(2)[0]}{LEG[m.group(1)]}_tarsus5_brown.obj"
            if c in have:
                mapping[f] = c

    used = set(mapping.values())
    rest = [f for f in xml_files if f not in mapping]
    free = sorted(have - used)
    pairs = sorted(((len(_toks(x) & _toks(c)) / max(len(_toks(x) | _toks(c)), 1), x, c)
                    for x in rest for c in free), reverse=True)
    tx, tc = set(), set()
    for sc, x, c in pairs:
        if x in tx or c in tc or sc < 0.34:
            continue
        mapping[x] = c
        tx.add(x)
        tc.add(c)
    return mapping


WING_ACT = ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
            "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]

# fluidshape="ellipsoid" の既定係数 (blunt drag, slender drag, angular drag,
# Kutta lift, Magnus lift)
FLUID_COEFS = [1.0, 0.5, 0.25, 1.5, 1.0, 1.0]


# 翅のヒンジまわりの実慣性 (body_inertia + m r^2 から実測) [g cm^2]
WING_INERTIA = 2.3e-7

WING_JOINTS = ["wing_roll_left", "wing_yaw_left", "wing_pitch_left",
               "wing_roll_right", "wing_yaw_right", "wing_pitch_right"]


def load_model(wing_kp: float = 0.05, wing_kv: float = 5e-4,
               aero: bool = True, add_floor: bool = True,
               wing_armature: float | None = WING_INERTIA,
               wing_damping: float | None = 1e-5,
               scene: bool = False, scene_seed: int = 0,
               two_flowers: bool = False, extra=None):
    """翅つきモデルを組み立てて返す。翅は位置サーボ化してある。

    wing_armature:
        元の XML は翅の関節に armature=1e-6 を入れているが、これは翅自身の
        ヒンジまわりの慣性 2.3e-7 の **4.3倍** にあたる。数値安定化のための
        項が実際の慣性を上回っている状態で、そのままだと羽ばたきに必要な
        パワーが倍以上に膨らむ (実測 32,000 -> 14,700 erg/s)。
        既定で実慣性に合わせる。None を渡すと XML のまま。
        なお 5e-8 まで下げると dt=1e-5 でも発散する。

    wing_damping:
        XML の翅関節ダンピング 5e-4 も数値安定化のための値で、この寸法では
        大きすぎる。翅は 1400 rad/s で振れるので 5e-4 x 1400 = 0.7 dyn cm の
        抵抗トルクになり、**羽ばたきパワーの大半がここで捨てられる**。
        実測: 5e-4 -> 1e-5 で正味パワー 2381 -> 683 erg/s
        (実物のショウジョウバエは 300-800 erg/s)。揚力は変わらない。
    """
    assets, cache = _assets_dir(), _mesh_cache()
    spec = mujoco.MjSpec.from_file(str(assets / "fruitfly.xml"))
    spec.meshdir = str(cache)

    mapping = mesh_mapping()
    dropped = []
    for mesh in spec.meshes:
        new = mapping.get(mesh.file)
        if new:
            mesh.file = new
        else:
            dropped.append(mesh.name)
    if dropped:
        # 対応するメッシュが無い見た目だけのジオムを外す (物理には影響しない)
        for body in spec.bodies:
            for g in list(body.geoms):
                if g.meshname in dropped:
                    g.delete()
        for mesh in list(spec.meshes):
            if mesh.name in dropped:
                mesh.delete()

    # 前向きのカメラを胸部に足す。
    # 同梱の eye_left / eye_right は視野140度だが強く側方を向いていて、
    # 正面にある的 (花) がほとんど写らない。実物のハエも前方に両眼視の
    # 重なりを持つので、そこを見るカメラを1つ用意する。
    # MuJoCo のカメラは -z 方向を見るので、-z を機体の +x に向ける。
    import numpy as _np
    _R = _np.array([[0.0, 0.0, -1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]])
    _q = _np.zeros(4)
    mujoco.mju_mat2Quat(_q, _R.flatten())
    for _body in spec.bodies:
        if _body.name == "thorax":
            _c = _body.add_camera()
            _c.name = "eye_front"
            _c.pos = [0.06, 0.0, 0.0]
            _c.fovy = 120.0
            _c.quat = _q
            break

    if scene:
        from world_scene import add_scene
        add_scene(spec, seed=scene_seed, two_flowers=two_flowers)
    elif add_floor:
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, 0.1], pos=[0, 0, 0],
                                rgba=[0.16, 0.17, 0.19, 1])

    if extra is not None:
        # compile 前の spec に手を入れるためのフック。
        # 風景に物を足したい呼び出し側 (24_escape_full.py の捕食者など) が使う。
        extra(spec)

    model = spec.compile()

    # 近クリップ面を詰める。
    # MuJoCo の近クリップは znear x (モデルの extent) で決まる。地面が 400cm
    # あるため extent が 1475 cm になり、既定の znear=0.01 では **14.75 cm**。
    # ハエの体長は 0.25 cm、目の前の草や花は 1-3 cm なので、それが全部
    # 切り取られて見えていなかった。複眼を使うならここを詰める必要がある。
    model.vis.map.znear = 1e-4        # -> 約 0.15 cm
    model.vis.map.zfar = 20.0

    if aero:
        for i in range(model.ngeom):
            n = model.geom(i).name
            if "wing" in n and "fluid" in n:
                model.geom_fluid[i][:6] = FLUID_COEFS

    if wing_armature is not None:
        for jn in WING_JOINTS:
            model.dof_armature[model.jnt_dofadr[model.joint(jn).id]] = wing_armature
    if wing_damping is not None:
        for jn in WING_JOINTS:
            model.dof_damping[model.jnt_dofadr[model.joint(jn).id]] = wing_damping

    for name in WING_ACT:
        aid = model.actuator(name).id
        model.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[aid][:3] = [wing_kp, 0, 0]
        model.actuator_biasprm[aid][:3] = [0, -wing_kp, -wing_kv]
        model.actuator_ctrlrange[aid] = [-3.2, 3.2]
        model.actuator_forcerange[aid] = [-1e3, 1e3]
        model.actuator_forcelimited[aid] = 1

    return model, dropped
