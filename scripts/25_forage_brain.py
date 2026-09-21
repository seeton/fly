"""花に近づく脳を見る — 匂い・好み・翅。

`24_escape_full.py` は逃避 (脅威) だった。こちらは報酬の側。見た目が同じ2つの
花があり、匂いだけが違う。配置と操舵は `21_choose.py` と同じ。

    出発   (15, +5, 9)
    おとり (42, -5, 9)   匂いが違う
    本命   (66, +5, 9)

出発を本命のプルームの中 (好み +0.43) に置いている。y=0 から出すと近いおとりの
匂いが勝って好みが -0.38 になり、`gate = max(pref, 0)` が閉じて操舵がゼロになる。
**「好まない匂いなら寄らない」は実現できているが、離れる動きが無い** ので、
そのまま風 (-8 cm/s) に流されて墜落する。README に挙げてある未解決の課題。

**脳のどこが何で光るか**

    視葉 ME/LO/LOP  複眼に映った像 (明るさと動き)。飛んでいるので景色が
                    流れ続け、ずっと明るい
    触角葉 AL       左右の触角に届く匂いの濃度。花に近づくほど強くなる
    外側角 LH       好む匂いがどれだけ優勢か max(pref, 0)。
                    **生まれつきの好き嫌い** にあたる。実物のハエでも
                    外側角は学習を要さない匂いの価値を扱う
    翅の神経核 WTct 翅関節が動いた速さ (包絡)。飛んでいるあいだ明るい
    脚の神経核      飛行中は脚を畳んでいるのでほぼ暗い

**キノコ体 (CA / gL / aL / bL / a'L / b'L) は暗いままにする。**
キノコ体は **学習した** 価値を扱う場所で、ドーパミンニューロンが運ぶ報酬信号と
匂いをそこで結びつける。このシミュレーションは何も学習していない。どちらの花を
好むかは `--like` でこちらが決めた定数で、ハエが経験から得たものではない。
だからキノコ体を光らせるのは嘘になる。

    言えること   … 匂いが届いた (AL)、その匂いは好きな方だ (LH)
    言えないこと … それに価値があると脳が判断した (キノコ体・ドーパミン)

**振る舞いの断り**: 複眼を実測の光学に直してから、この場面でハエは本命ではなく
おとりに寄る (README 参照)。ここで見せているのは感覚と好みの信号であって、
選択がうまくいく証拠ではない。

使い方:
  python scripts/25_forage_brain.py
  python scripts/25_forage_brain.py --like decoy      # 好みを逆にする
  python scripts/25_forage_brain.py --no-brain        # 脳パネル抜き (速い)

出力: out/forage_brain.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escape_circuit import OUT, BrainPanel, jp_font  # noqa: E402
from flight_posture import leg_targets  # noqa: E402
from fly_smell import WIND, FlyNose, OdorPlume  # noqa: E402
from fly_vision import FlyEyes  # noqa: E402
from flybody_model import load_model  # noqa: E402
from insect_aero import WingAero  # noqa: E402
from world_scene import DECOY_POS, TARGET_POS  # noqa: E402

POLICY = OUT / "flight_policy.json"
WINGS = ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
         "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]
DT = 2e-5
PAD_H = 340          # 体の絵の下に置く複眼パネルの高さ。脳パネル 900 に合わせる

# キノコ体。学習していないので光らせない領域として名前を持っておく
MUSHROOM = ("CA", "gL", "aL", "bL", "a'L", "b'L", "PED")

# (凡例の文, 色, 明るさを読む領域名)。領域名があるものは数値も出す。
# 外側角は 45,000 点中 810 点しかなく、3D の絵だけでは光っているかどうかが
# 読み取れない。同じ数字を文字でも並べる
LEGEND = [("中枢神経系ぜんぶ — 45,000 点で 3億1183万シナプスを代表", "#cdd6e3", None),
          ("視葉 ME/LO/LOP — 複眼に映った像", "#9fb4cf", "ME"),
          ("触角葉 AL — 左右の触角に届いた匂い", "#9fb4cf", "AL"),
          ("外側角 LH — 好む匂いがどれだけ優勢か", "#9fb4cf", "LH"),
          ("翅の神経核 WTct — 翅関節の動き", "#9fb4cf", "WTct"),
          ("キノコ体 — 暗いまま (学習していないので光らせない)", "#6f7787", "gL")]

NOTE = ("点群の明るさはシミュレーションの入力そのもの — "
        "キノコ体が暗いのは、報酬を学ぶ仕組みをこのシミュレーションが"
        "持っていないから。好みは学習ではなく定数")


def euler(q):
    w, x, y, z = q
    return (np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def panel_label(img, lines, fonts, corner_right=None, backing=True):
    """パネルに見出しを焼き込む (24_escape_full.py と同じ作り)。"""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).convert("RGBA")
    if backing and lines:
        band = Image.new("RGBA", (im.size[0], 20 + 24 * len(lines)), (6, 8, 12, 150))
        im.alpha_composite(band, (0, 0))
    dr = ImageDraw.Draw(im)
    for i, (txt, col, key) in enumerate(lines):
        dr.text((18, 10 + 24 * i), txt, fill=col, font=fonts[key])
    if corner_right:
        w = dr.textlength(corner_right, font=fonts["small"])
        if backing:
            pad = Image.new("RGBA", (int(w) + 24, 30), (6, 8, 12, 150))
            im.alpha_composite(pad, (im.size[0] - int(w) - 30, im.size[1] - 36))
        dr = ImageDraw.Draw(im)
        dr.text((im.size[0] - 18 - w, im.size[1] - 30), corner_right,
                fill="#dfe6f2", font=fonts["small"])
    return np.asarray(im.convert("RGB"))


def fly_and_record(args):
    """第1パス — 飛ばしながら、脳に配る入力と体の絵をためる。

    翅は 200 Hz で打っている。関節速度をそのまま 50 Hz で記録すると
    位相が折り返して意味のない縞になるので、**包絡** (時定数 5 ms の
    低域通過) を記録する。翅の神経核を光らせたいのは「いま打っているか」
    であって、打ち下ろしの瞬間かどうかではない。
    """
    p = json.loads(POLICY.read_text(encoding="utf-8"))["params"]
    m, _ = load_model(wing_kp=50.0, wing_kv=2 * np.sqrt(50.0 * 1e-6),
                      scene=True, two_flowers=True)
    m.opt.timestep = DT
    aero = WingAero(m, n_elem=6)
    aero.route_wings_to_this_model(m)
    i_add = aero.added_mass_inertia(m)
    for jn in ("wing_roll_left", "wing_roll_right", "wing_yaw_left", "wing_yaw_right"):
        m.dof_armature[m.jnt_dofadr[m.joint(jn).id]] += i_add

    wind = np.array(WIND, dtype=float)
    aero.wind = wind
    m.opt.wind[:] = wind

    eyes = FlyEyes(m, rate_hz=200.0)
    plumes = {"target": OdorPlume(TARGET_POS), "decoy": OdorPlume(DECOY_POS)}
    nose = FlyNose(m, plumes)

    A = {n: m.actuator(n).id for n in WINGS}
    legs = {m.actuator(k).id: v for k, v in leg_targets(m).items()}
    head_ids = {h: m.actuator(h).id for h in ("head_abduct", "head")}
    thorax = m.body("thorax").id
    wing_dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in WINGS])
    leg_dofs = {}
    for j in range(m.njnt):
        parts = m.joint(j).name.rsplit("_", 2)
        if len(parts) == 3 and parts[1] in ("T1", "T2", "T3") \
                and parts[2] in ("left", "right"):
            leg_dofs.setdefault(f"{parts[1]}_{parts[2]}", []).append(
                int(m.jnt_dofadr[j]))
    leg_dofs = {k: np.array(v) for k, v in sorted(leg_dofs.items())}

    W_body, H_body = args.body_res
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W_body)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H_body)
    renderer = mujoco.Renderer(m, height=H_body, width=W_body)

    d = mujoco.MjData(m)
    d.qpos[0], d.qpos[1], d.qpos[2] = args.x0, args.y0, args.z0
    for aid, val in legs.items():
        d.ctrl[aid] = val
        d.qpos[m.jnt_qposadr[m.actuator_trnid[aid, 0]]] = val
    aero.reset()
    eyes.reset()
    nose.reset()

    mid = (p["pitch_down"] + p["pitch_up"]) / 2
    amp_p = (p["pitch_down"] - p["pitch_up"]) / 2
    a_slow, a_vis = DT / 0.05, DT / 0.03
    a_wing = DT / 0.005
    yaw_vis = yaw_vis_f = yaw_slow = roll_slow = 0.0
    see_f = 0.0
    seeing = False
    wing_env = 0.0
    pref = 0.0

    R = {k: [] for k in ("t", "photo", "motion", "odor", "pref", "wing",
                         "chan_target", "chan_decoy", "front")}
    R["leg"] = {k: [] for k in leg_dofs}
    frames, traj = [], []
    eyes_img, modes = [], []
    frame_every = max(int(1.0 / (args.fps * args.slow) / DT), 1)
    rec_every = max(int(1.0 / args.rec_hz / DT), 1)
    best = {"target": 1e9, "decoy": 1e9}

    print(f"おとり {DECOY_POS}  本命 {TARGET_POS}  風 {wind} cm/s")
    print(f"出発 ({args.x0}, {args.y0}, {args.z0})   好む匂い: {args.like}")
    print("第1パス: 飛行と感覚 ...")

    for s in range(int(args.seconds / DT)):
        t = s * DT
        th = 2 * np.pi * p["freq"] * t
        roll_b, pitch_b, yaw_b = euler(d.qpos[3:7])
        wx, wy, _ = d.qvel[3], d.qvel[4], d.qvel[5]

        if eyes.maybe_update(m, d, t):
            yaw_vis = eyes.rotation_signal
        yaw_vis_f += a_vis * (yaw_vis - yaw_vis_f)
        nose.update(m, d, DT)
        wing_env += a_wing * (float(np.abs(d.qvel[wing_dofs]).mean()) - wing_env)

        yaw_slow += a_slow * (yaw_b - yaw_slow)
        roll_slow += a_slow * (roll_b - roll_slow)
        d.ctrl[head_ids["head_abduct"]] = float(np.clip(-(yaw_b - yaw_slow), -0.2, 0.2))
        d.ctrl[head_ids["head"]] = float(np.clip(-(roll_b - roll_slow), -0.5, 0.3))

        # --- 視覚と嗅覚を同時に使う (21_choose.py と同じ) ---
        if eyes.front_size > args.see_on:
            seeing = True
        elif eyes.front_size < args.see_off:
            seeing = False
        if eyes.front_size > args.see_off:
            see_f += (DT / 0.05) * (eyes.front_bearing(args.see_off) - see_f)

        pref = nose.preference(args.like)
        gate = max(pref, 0.0)          # 好む匂いのときだけ寄る
        left_up, _ = nose.body_frame_upwind(d, thorax)
        steer = args.k_odor * gate * left_up
        if seeing:
            steer += -args.k_see * gate * see_f

        u_p = np.clip(-p["kp_pitch"] * pitch_b - p["kd_pitch"] * wy, -0.8, 0.8)
        u_r = np.clip(-p["kp_roll"] * roll_b - p["kd_roll"] * wx, -0.5, 0.5)
        u_y = np.clip(-args.k_flow * yaw_vis_f + steer, -0.6, 0.6)
        u_a = np.clip(p["kp_alt"] * (args.z0 - d.qpos[2]) - p["kd_alt"] * d.qvel[2],
                      -0.4, 0.4)

        mean = p["roll_mean"] + u_p
        base = np.clip(p["roll_amp"] + u_a, 0.35, 1.25)
        feather = mid + amp_p * np.tanh(p["sharp"] * np.sin(th + p["phase"]))
        dev = p["yaw"] + p["yaw_amp"] * np.cos(th + p["yaw_phase"])
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            d.ctrl[A[f"wing_yaw_{side}"]] = np.clip(dev, -1.5, 1.5)
            d.ctrl[A[f"wing_roll_{side}"]] = np.clip(
                mean + (base + sgn * u_r) * np.cos(th), -1.0, 1.5)
            d.ctrl[A[f"wing_pitch_{side}"]] = np.clip(feather + sgn * u_y, -1.27, 2.92)

        aero.apply(m, d)
        mujoco.mj_step(m, d)
        best["target"] = min(best["target"],
                             float(np.linalg.norm(d.qpos[:3] - TARGET_POS)))
        best["decoy"] = min(best["decoy"],
                            float(np.linalg.norm(d.qpos[:3] - DECOY_POS)))
        if not np.isfinite(d.qpos).all() or d.qpos[2] < 0.5:
            print(f"  t={t:.2f}s で終了 (高度 {d.qpos[2]:.2f} cm)")
            break

        if s % rec_every == 0:
            R["t"].append(t)
            R["photo"].append(np.stack(eyes.photo).astype(np.float32))
            R["motion"].append(np.stack(eyes.activity).astype(np.float32))
            R["odor"].append(nose.c.copy())
            R["pref"].append(float(pref))
            R["wing"].append(float(wing_env))
            R["chan_target"].append(float(nose.channels["target"]))
            R["chan_decoy"].append(float(nose.channels["decoy"]))
            R["front"].append(float(eyes.front_size))
            for k, ix in leg_dofs.items():
                R["leg"][k].append(float(np.abs(d.qvel[ix]).mean()))
        if s % 20000 == 0:
            traj.append((t, float(d.qpos[0]), float(d.qpos[1]),
                         nose.channels["target"], nose.channels["decoy"], pref,
                         eyes.front_size))
        if s % frame_every == 0:
            renderer.update_scene(d, camera=args.camera)
            frames.append(renderer.render())
            # 複眼の像は体の絵に焼き込まない。見出しの帯に隠れるうえ、
            # 24 と同じく下に大きく出したほうが読める
            eyes_img.append(eyes.side_by_side(scale=1))
            modes.append((t, float(nose.strength), float(pref),
                          float(eyes.front_size), seeing))

    print(f"\n{'t[s]':>6} {'x':>7} {'y':>7} {'本命の匂い':>11} {'おとりの匂い':>12} "
          f"{'好み':>7} {'花の写り':>9}")
    for r in traj:
        print(f"{r[0]:6.2f} {r[1]:7.2f} {r[2]:7.2f} {r[3]:11.4f} {r[4]:12.4f} "
              f"{r[5]:7.2f} {r[6]*100:8.3f}%")
    print(f"\n最接近  本命 {best['target']:.2f} cm / おとり {best['decoy']:.2f} cm "
          f"(花の直径は 2.5 cm)")

    for k in ("t", "photo", "motion", "odor", "pref", "wing",
              "chan_target", "chan_decoy", "front"):
        R[k] = np.asarray(R[k])
    R["leg"] = {k: np.asarray(v) for k, v in R["leg"].items()}
    R["frames"] = frames
    R["eyes_img"] = eyes_img
    R["modes"] = modes
    R["n_omma"] = eyes.n_omma
    R["best"] = best
    return R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--x0", type=float, default=15.0)
    ap.add_argument("--y0", type=float, default=5.0,
                    help="出発の横位置。**本命のプルームの中から始める**。"
                         "y=0 だと近いおとりの匂いが勝って好み -0.38 となり、"
                         "gate = max(pref,0) が閉じて操舵がゼロになる。"
                         "そうなると風 (-8 cm/s) に流されるだけで花に近づかない")
    ap.add_argument("--z0", type=float, default=9.0)
    ap.add_argument("--like", default="target", choices=("target", "decoy"),
                    help="好む匂い。**こちらが決める定数** で、学習ではない")
    ap.add_argument("--camera", default="scene_follow")
    ap.add_argument("--slow", type=float, default=1.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rec-hz", type=float, default=50.0,
                    help="脳に配る入力を記録する頻度 [Hz]")
    ap.add_argument("--body-res", type=int, nargs=2, default=[760, 560])
    ap.add_argument("--k-flow", type=float, default=0.14)
    ap.add_argument("--k-odor", type=float, default=0.22)
    ap.add_argument("--k-see", type=float, default=0.22)
    ap.add_argument("--see-on", type=float, default=8e-5)
    ap.add_argument("--see-off", type=float, default=2e-5)
    ap.add_argument("--n-cloud", type=int, default=45000)
    ap.add_argument("--no-brain", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="forage_brain.mp4")
    args = ap.parse_args()

    R = fly_and_record(args)
    frames_body = R["frames"]
    n_frames = len(frames_body)
    T_movie = n_frames / args.fps
    print(f"動画 {T_movie:.1f} s / {n_frames} フレーム (x{args.slow:.1f} スロー)")

    brain = None
    if not args.no_brain:
        # 逃避回路は出さない。脅威が無いので巨大繊維は沈黙しているのが正しい。
        # 発火時刻を動画の外に置いて、暗いまま置いておく
        far = T_movie * 1e3
        brain = BrainPanel(res=(760, 900), refresh=args.refresh,
                           t_look=far, t_spike=far, t_arrive=far,
                           spin=40.0, n_cloud=args.n_cloud,
                           n_omma=R["n_omma"])
        tm = R["t"] * args.slow                     # 物理の時刻 -> 動画の時刻
        brain.attach_optic_drive(tm, R["photo"], R["motion"])
        # 触角葉は左右べつべつ。同じ物理量なので物差しは共通にする
        odor_hi = max(float(np.percentile(R["odor"], 99.0)), 1e-9)
        brain.attach_region_drive(
            tm, {"AL(L)": R["odor"][:, 0], "AL(R)": R["odor"][:, 1]},
            scale=odor_hi, label="触角葉")
        # 外側角は好みの判定。0..1 なのでそのまま使う (割らない)
        brain.attach_region_drive(tm, {"LH": np.maximum(R["pref"], 0.0)},
                                  scale=1.0, gamma=0.8, label="外側角")
        brain.attach_region_drive(tm, {"WTct": R["wing"]}, label="翅の神経核")
        brain.attach_leg_drive(tm, R["leg"])
        dark = sum(len(brain.region_mask(k)) for k in MUSHROOM)
        print(f"キノコ体 {dark:,} 点は暗いまま (学習していないので光らせない)")

    fonts = {"title": jp_font(19), "small": jp_font(13), "tiny": jp_font(11)}
    out_frames = []
    print("第2パス: 脳を描く ...")
    for f in range(n_frames):
        t_movie = f / args.fps
        t_sim, odor, pref, front, seeing = R["modes"][f]
        body = panel_label(
            frames_body[f],
            [("体 (MuJoCo)", "#e8eef7", "title"),
             (f"匂いの強さ {odor:.4f}   好み {pref:+.2f}", "#9aa4b2", "small"),
             (("花が見えている" if seeing else "匂いだけで追っている")
              + f"   花の写り {front*100:.3f}%", "#9aa4b2", "small")],
            fonts, corner_right=f"t = {t_sim:.2f} s   スロー再生 x{args.slow:.1f}")
        # 24 と同じ作り: 体の下に複眼の像を大きく置いて、右の列を脳と同じ高さにする
        eye = np.repeat(np.repeat(R["eyes_img"][f], 9, axis=0), 9, axis=1)
        W_body = body.shape[1]
        pad = np.zeros((PAD_H, W_body, 3), np.uint8)
        pad[:, :] = (5, 5, 8)
        eh, ew = eye.shape[:2]
        y0 = max((PAD_H - eh) // 2 + 16, 54)
        x0 = max((W_body - ew) // 2, 0)
        pad[y0:y0 + eh, x0:x0 + ew] = eye[:PAD_H - y0, :W_body - x0]
        pad = panel_label(
            pad, [("ハエの視界 (左眼 / 右眼)", "#e8eef7", "title"),
                  (f"個眼 {R['n_omma']}x{R['n_omma']} /眼   "
                   f"花までの匂い {odor:.4f}", "#9aa4b2", "small")],
            fonts, backing=False)
        body = np.vstack([body, pad])
        if brain is None:
            out_frames.append(body)
            continue

        bimg, _, _ = brain.frame(t_movie, f / max(n_frames - 1, 1))
        # 好みが負だと gate = max(pref, 0) が閉じ、**花が見えていても操舵は
        # ゼロ**になる。見えているかどうかより先にここを判定しないと、
        # 「目と鼻の両方で寄る」と嘘を書くことになる
        if odor <= 1e-3:
            stage = "匂いの外 — 探索"
        elif pref <= 0:
            stage = ("好まない匂い — 寄らない (花が見えていても)" if seeing
                     else "好まない匂い — 寄らない")
        elif seeing:
            stage = "花が見えている — 目と鼻の両方で寄る"
        else:
            stage = "好む匂いの中 — 風上へ"
        bimg = panel_label(bimg, [("脳 (Male CNS v1.0)", "#e8eef7", "title"),
                                  (stage, "#dfe6f2", "small")], fonts)
        from PIL import Image, ImageDraw
        im = Image.fromarray(bimg).convert("RGBA")
        dr = ImageDraw.Draw(im)
        note = [("— " if k else "") + ln for k, ln in enumerate(NOTE.split(" — "))]
        # 凡例も断り書きも脳の絵の上に来る。ここは点群が明るいので、
        # まとめて暗い帯を敷いてから書く (敷かないと視葉に埋もれて読めない)
        w_all = max([dr.textlength(t, font=fonts["small"]) + 96 for t, _, _ in LEGEND]
                    + [dr.textlength(ln, font=fonts["tiny"]) + 38 for ln in note])
        h_all = 24 * len(LEGEND) + 15 * len(note) + 20
        im.alpha_composite(Image.new("RGBA", (int(w_all) + 24, h_all),
                                     (6, 8, 12, 178)), (14, 68))
        dr = ImageDraw.Draw(im)
        y = 74
        x_val = int(w_all) - 20
        for txt, col, key in LEGEND:
            dr.ellipse((20, y + 4, 29, y + 13), fill=col)
            dr.text((38, y), txt, fill=col, font=fonts["small"])
            if key is not None:
                v = f"{brain.region_level(key):.2f}"
                w = dr.textlength(v, font=fonts["small"])
                dr.text((x_val - w, y), v, fill=col, font=fonts["small"])
            y += 24
        y += 6
        for line in note:
            dr.text((38, y), line, fill="#8b93a2", font=fonts["tiny"])
            y += 15
        for k, line in enumerate((f"入力で光る点 {brain.n_cloud_lit:,}",
                                  f"/ {len(brain.cloud):,}")):
            w = dr.textlength(line, font=fonts["small"])
            dr.text((bimg.shape[1] - 20 - w, 22 + 22 * k), line,
                    fill="#9aa4b2", font=fonts["small"])
        bimg = np.asarray(im.convert("RGB"))

        if bimg.shape[0] != body.shape[0]:
            h = min(bimg.shape[0], body.shape[0])
            bimg, body = bimg[:h], body[:h]
        frame = np.hstack([bimg, body])
        im = Image.fromarray(frame)
        ImageDraw.Draw(im).text(
            (20, frame.shape[0] - 26),
            "形・シナプス・接続は Male CNS v1.0 の実測 / 点群の明るさは複眼と触角と"
            "関節から計算 / 好む匂いは学習ではなく定数",
            fill="#5b6270", font=fonts["tiny"])
        out_frames.append(np.asarray(im))
        if (f + 1) % 30 == 0:
            print(f"  {f+1}/{n_frames}")

    OUT.mkdir(exist_ok=True)
    path = OUT / args.out
    imageio.mimsave(path, out_frames, fps=args.fps, macro_block_size=None)
    print(f"書き出し: {path} ({len(out_frames)} フレーム, "
          f"{path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
