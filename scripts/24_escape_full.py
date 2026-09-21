"""逃避を3つ並べて見る — 脳のシナプス / 体の動き / ハエの視界。

同じ1本の時計で動かす:

    捕食者が迫る
      -> **複眼に写った像から** 見かけの大きさ (視角) を毎フレーム測る
      -> 視角がしきい値を超えた時刻が「LC4/LPLC2 が GF を発火させた時刻」
      -> そこから伝導遅れのぶん後に、中脚 (TTMn が動かす関節) を一気に伸ばす
      -> 跳ぶ

左の脳パネルの t_spike / t_arrive は、この物理シミュレーションで決まった時刻に
合わせてある。**脳の絵に合わせて体を動かしているのではなく、視界から決まった
時刻を脳の絵に渡している**。

**再生速度は一定ではない**。接近は数百 ms あるのに神経の伝導は約 1 ms しかなく、
同じ倍率では両方見えない。区間ごとに倍率を変え、画面に出している。

  実測     … ニューロンの形・シナプスの3D座標・接続 (Male CNS v1.0)、
             視角の時間変化 (複眼の描画から計算)、体の運動 (MuJoCo)
  文献値   … 逃避の視角しきい値、GF の伝導遅れ
  こちらの想定 … シナプス1個ずつが光る順番の細部

使い方:
  python scripts/24_escape_full.py
  python scripts/24_escape_full.py --approach 35 --theta-trigger 50
  python scripts/24_escape_full.py --no-brain      # 脳パネル抜き (速い)

出力: out/escape_full.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escape_circuit import CLOUD_NOTE, LEGEND, OUT, BrainPanel, jp_font  # noqa: E402
from escape_scene import DT, PRED_R, run_approach, save_sense  # noqa: E402


def build_time_map(events, args):
    """物理の時刻 -> 動画の時刻。区間ごとに再生倍率を変える。

    接近は数百 ms、神経の伝導は約 1 ms。同じ倍率では両方見えないので、
    発火の前後だけ極端に落とす。倍率は画面に出す。
    """
    t_trig, t_ext, t_end = events["trigger"], events["extend"], events["end"]
    pre = max(t_trig - 0.010, 0.0)
    post = min(t_ext + 0.008, t_end)
    # (物理の区間, 倍率)
    segs = [(0.0, pre, args.slow_approach),
            (pre, post, args.slow_spike),
            (post, t_end, args.slow_jump)]
    bounds, movie = [0.0], [0.0]
    for a, b, k in segs:
        if b <= a:
            continue
        bounds.append(b)
        movie.append(movie[-1] + (b - a) * k)
    return np.array(bounds), np.array(movie), segs


def slow_at(t_sim: float, segs) -> float:
    for a, b, k in segs:
        if a <= t_sim <= b:
            return k
    return segs[-1][2]


def panel_label(img, lines, fonts, corner_right=None, backing=True):
    """パネルに見出しを焼き込む。

    明るい空や地面の上に白文字を置くと読めないので、下に半透明の暗い帯を敷く。
    """
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", type=float, default=30.0, help="捕食者の速さ cm/s")
    ap.add_argument("--start", type=float, default=7.0, help="捕食者の初期距離 cm")
    ap.add_argument("--theta-trigger", type=float, default=45.0,
                    help="逃避を起こす視角 [deg]")
    ap.add_argument("--gf-delay-ms", type=float, default=1.1,
                    help="GF の伝導 + シナプス遅れ [ms]")
    ap.add_argument("--rise-ms", type=float, default=5.0, help="中脚を伸ばす時間 ms")
    ap.add_argument("--amp", type=float, default=1.9, help="伸展の大きさ rad")
    ap.add_argument("--seconds", type=float, default=0.26, help="物理の長さ s")
    ap.add_argument("--settle", type=float, default=0.15,
                    help="基準を取る前に脚で立たせて落ち着かせる時間 s")
    ap.add_argument("--jump-force", type=float, default=3.0,
                    help="中脚を伸ばす関節の力の上限 [dyn cm]")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--slow-approach", type=float, default=8.0)
    ap.add_argument("--slow-spike", type=float, default=400.0)
    ap.add_argument("--slow-jump", type=float, default=60.0)
    ap.add_argument("--n-cloud", type=int, default=45000,
                    help="背景に置く全CNSのシナプスの点数")
    ap.add_argument("--no-brain", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="escape_full.mp4")
    args = ap.parse_args()

    params = dict(approach=args.approach, start_cm=args.start,
                  theta_trigger=args.theta_trigger, gf_delay_ms=args.gf_delay_ms,
                  rise_ms=args.rise_ms, amp=args.amp, seconds=args.seconds,
                  settle=args.settle, jump_force=args.jump_force)
    R = run_approach(**params)
    m, d, eyes = R["model"], R["data"], R["eyes"]
    n_steps = R["n_steps"]
    qpos_log, mocap_log, theta_log = R["qpos_log"], R["mocap_log"], R["theta_log"]
    t_trigger, t_extend = R["t_trigger"], R["t_extend"]
    # 感覚の入力を控えておく。脳だけ描く 23 が同じものを使う
    save_sense(R, params)

    if t_trigger is None:
        print("しきい値に届かなかった。--theta-trigger を下げるか --start を詰める")
        return
    # 視角の推移。複眼の描画から測った値と、球の直径と距離から出る幾何の値を並べる。
    # 測った側が少し大きく出るのは受容角 5.1 度で縁がにじむため (実物の網膜も同じ)。
    print(f"\n{'t[ms]':>7} {'捕食者まで[cm]':>14} {'視角(複眼)':>11} {'視角(幾何)':>11}")
    for ms in (0, 40, 80, 120, 150, 165, 170, 175, 200, 230):
        i = min(int(ms * 1e-3 / DT), n_steps - 1)
        dist = float(np.linalg.norm(mocap_log[i] - qpos_log[i, :3]))
        geo = np.degrees(2 * np.arctan(PRED_R / max(dist, 1e-6)))
        mark = " <- 発火" if abs(ms * 1e-3 - t_trigger) < 3e-3 else ""
        print(f"{ms:7d} {dist:14.2f} {theta_log[i]:11.1f} {geo:11.1f}{mark}")
    print()

    z0, zmax = qpos_log[0, 2], float(qpos_log[:, 2].max())
    print(f"胸部の高さ {z0:.3f} -> 最大 {zmax:.3f} cm  "
          f"({(zmax-z0)/0.25*100:.0f}% 体長ぶん浮いた)")

    events = {"trigger": t_trigger, "extend": t_extend,
              "end": (n_steps - 1) * DT}
    bounds, movie_b, segs = build_time_map(events, args)
    T_movie = float(movie_b[-1])
    n_frames = int(T_movie * args.fps)
    print(f"動画 {T_movie:.1f} s / {n_frames} フレーム "
          f"(接近 x{args.slow_approach:.0f} / 発火 x{args.slow_spike:.0f} / "
          f"跳躍 x{args.slow_jump:.0f} スロー)")

    # 動画の時刻 -> 物理の時刻
    def sim_time(tm):
        return float(np.interp(tm, movie_b, bounds))

    def movie_time(ts):
        return float(np.interp(ts, bounds, movie_b))

    # --- 脳パネル: 物理で決まった時刻を動画の時刻に直して渡す ---
    brain = None
    if not args.no_brain:
        t_look_sim = max(events["trigger"] - 0.12, 0.0)
        brain = BrainPanel(res=(760, 900), refresh=args.refresh,
                           t_look=movie_time(t_look_sim),
                           t_spike=movie_time(events["trigger"]),
                           t_arrive=movie_time(events["extend"]),
                           spin=40.0, n_cloud=args.n_cloud,
                           n_omma=R["n_omma"])
        # 背景の点群は **この走りの入力** で光らせる。合成した揺らぎは使わない。
        # 視覚の記録は物理の時刻なので、動画の時計に直してから渡す
        t_vis_movie = np.array([movie_time(tv) for tv in R["t_vis"]])
        brain.attach_optic_drive(t_vis_movie, R["photo"], R["motion"])
        brain.attach_leg_drive(t_vis_movie, R["leg_speed"])

    # --- 第2パス: 記録した状態を再生して描画 ---
    W_body, H_body = 760, 560
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W_body)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H_body)
    rend = mujoco.Renderer(m, height=H_body, width=W_body)
    fonts = {"title": jp_font(19), "small": jp_font(13), "tiny": jp_font(11)}
    eyes.reset()

    frames = []
    print("第2パス: 描画 ...")
    for f in range(n_frames):
        tm = f / args.fps
        ts = sim_time(tm)
        i = min(int(ts / DT), n_steps - 1)
        d.qpos[:] = qpos_log[i]
        d.qvel[:] = 0.0
        d.mocap_pos[0] = mocap_log[i]
        mujoco.mj_forward(m, d)

        eyes._next_t = 0.0
        eyes.maybe_update(m, d, 0.0)
        eye = eyes.side_by_side(scale=9)

        rend.update_scene(d, camera="scene_side")
        body = rend.render()

        body = panel_label(body, [("体 (MuJoCo)", "#e8eef7", "title"),
                                  (f"胸部の高さ {qpos_log[i,2]:.3f} cm", "#9aa4b2", "small"),
                                  (f"捕食者まで {np.linalg.norm(mocap_log[i]-qpos_log[i,:3]):.2f} cm",
                                   "#9aa4b2", "small")],
                           fonts,
                           corner_right=f"スロー再生 x{slow_at(ts, segs):.0f}")
        eh, ew = eye.shape[:2]
        pad = np.zeros((340, W_body, 3), np.uint8)
        pad[:, :] = (5, 5, 8)
        y0 = max((pad.shape[0] - eh) // 2 + 16, 54)
        x0 = max((W_body - ew) // 2, 0)
        pad[y0:y0 + eh, x0:x0 + ew] = eye[:pad.shape[0] - y0, :W_body - x0]
        # 跳んだあとは機体が回って捕食者が視野を出入りするので、視角の数字は
        # 意味を持たない。そのまま出すと 0 deg などと誤解を招くので出さない。
        if ts <= events["extend"]:
            sub = (f"個眼 {eyes.n_omma}x{eyes.n_omma} /眼   "
                   f"迫る影の視角 {theta_log[i]:.0f} deg "
                   f"(しきい値 {args.theta_trigger:.0f})")
        else:
            sub = f"個眼 {eyes.n_omma}x{eyes.n_omma} /眼   跳躍中"
        pad = panel_label(pad, [("ハエの視界 (左眼 / 右眼)", "#e8eef7", "title"),
                                (sub, "#9aa4b2", "small")],
                          fonts, backing=False)
        right = np.vstack([body, pad])

        if brain is not None:
            bimg, n_lit, n_fired = brain.frame(tm, f / max(n_frames - 1, 1))
            bimg = panel_label(
                bimg,
                [("脳 (Male CNS v1.0)", "#e8eef7", "title"),
                 (brain.stage(tm), "#dfe6f2", "small")], fonts)
            from PIL import Image, ImageDraw
            im = Image.fromarray(bimg).convert("RGBA")
            dr = ImageDraw.Draw(im)
            y = 74
            for txt, col in LEGEND:
                dr.ellipse((20, y + 4, 29, y + 13), fill=col)
                dr.text((38, y), txt, fill=col, font=fonts["small"])
                y += 24
            # 点群の明るさがどこから来ているかを、凡例のすぐ下に残す。
            # 読み手が「作って光らせたのでは」と疑えるようにしておく。
            # 画面の下辺には脚注とスロー倍率が既にいるので、そこには置かない。
            # 脳の絵の上に直接置くと赤い軸索に重なって読めないので帯を敷く
            note = [("— " if k else "") + ln
                    for k, ln in enumerate(CLOUD_NOTE.split(" — "))]
            nw = int(max(dr.textlength(ln, font=fonts["tiny"]) for ln in note))
            im.alpha_composite(Image.new("RGBA", (nw + 44, 15 * len(note) + 12),
                                         (6, 8, 12, 170)), (14, y))
            dr = ImageDraw.Draw(im)
            y += 6
            for line in note:
                dr.text((38, y), line, fill="#8b93a2", font=fonts["tiny"])
                y += 15
            # 数を出すのは右上。左下は脚注に譲る
            for k, line in enumerate((f"逃避回路 {n_fired:,}/{len(brain.syn_pts):,}",
                                      f"入力で光る点 {brain.n_cloud_lit:,}")):
                w = dr.textlength(line, font=fonts["small"])
                dr.text((bimg.shape[1] - 20 - w, 22 + 22 * k), line,
                        fill="#9aa4b2", font=fonts["small"])
            bimg = np.asarray(im.convert("RGB"))
            if bimg.shape[0] != right.shape[0]:
                h = min(bimg.shape[0], right.shape[0])
                bimg, right = bimg[:h], right[:h]
            frame = np.hstack([bimg, right])
        else:
            frame = right

        from PIL import Image, ImageDraw
        im = Image.fromarray(frame)
        ImageDraw.Draw(im).text(
            (20, frame.shape[0] - 26),
            "形・シナプス・接続は Male CNS v1.0 の実測 / 視角も点群の明るさも複眼の"
            "描画から計算 / 伝導遅れ 1.1 ms は文献値",
            fill="#5b6270", font=fonts["tiny"])
        frames.append(np.asarray(im))
        if (f + 1) % 30 == 0:
            print(f"  {f+1}/{n_frames}")

    OUT.mkdir(exist_ok=True)
    out = OUT / args.out
    imageio.mimsave(out, frames, fps=args.fps, macro_block_size=None)
    print(f"書き出し: {out} ({len(frames)} フレーム, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
