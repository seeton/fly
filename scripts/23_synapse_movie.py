"""逃避回路の実測シナプスが光る動画を作る (脳だけ)。

`09_escape_animation.py` はニューロン全体の色を切り替えるだけだった。
ここでは **1個ずつのシナプスを光らせる**。位置は Male CNS の実測値
(3億1183万個のシナプス座標から該当分を抜き出したもの)。

    影が迫る
      -> LC4 / LPLC2 (視葉) が光る
      -> 視覚 → Giant Fiber (DNp01) のシナプスが光る     [視葉の糸球体]
      -> 信号が GF の軸索を脳から神経索へ降りる
      -> GF → TTMn のシナプスが光る                      [LTct]
      -> TTMn (中脚の運動ニューロン) が光る -> 脚が伸びる

**背景の点群も作り物では光らせない**。脳しか映さないからといって勝手に
明滅させてよいことにはならないので、`24_escape_full.py` と同じ場面
(`escape_scene.run_approach`) を回して複眼から入力を採り、それを視葉に配る。
一度走らせれば `out/escape_sense.npz` に残るので次からは読むだけ。

データの出どころと時間割の作り方は `escape_circuit.py` の docstring を参照。
体の動きと視界も一緒に見たいときは `24_escape_full.py`。

使い方:
  python scripts/23_synapse_movie.py
  python scripts/23_synapse_movie.py --seconds 9 --fps 30 --spin 55
  python scripts/23_synapse_movie.py --refresh      # シナプスの抽出からやり直す
  python scripts/23_synapse_movie.py --resense      # 場面の走り直しからやり直す

出力: out/synapse_fire.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escape_circuit import CLOUD_NOTE, LEGEND, OUT, BrainPanel, jp_font  # noqa: E402
from escape_scene import load_sense, run_approach, save_sense  # noqa: E402


def draw_overlay(img: np.ndarray, title: str, legend, footer, note,
                 counter: str, fonts) -> np.ndarray:
    """フレームに見出し・凡例・脚注を焼き込む。

    matplotlib には任せない。3.11 の 3D 軸では set_axis_off() と組み合わせた
    ときに figure 側のテキストまで描画されない条件があり、深追いしても得が無い。
    """
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).convert("RGBA")
    dr = ImageDraw.Draw(im)
    W, H = im.size
    dr.text((22, 20), title, fill="#e8eef7", font=fonts["title"])
    y = 66
    for txt, col in legend:
        dr.ellipse((24, y + 5, 34, y + 15), fill=col)
        dr.text((44, y), txt, fill=col, font=fonts["small"])
        y += 26
    # 点群がなぜ光るのかの断り書き。読み手が疑えるように残しておく。
    # 脳の絵の上に直接置くと読めないので帯を敷く
    lines = [("— " if k else "") + ln for k, ln in enumerate(note.split(" — "))]
    nw = int(max(dr.textlength(ln, font=fonts["tiny"]) for ln in lines))
    im.alpha_composite(Image.new("RGBA", (nw + 48, 15 * len(lines) + 12),
                                 (6, 8, 12, 170)), (18, y))
    dr = ImageDraw.Draw(im)
    y += 6
    for line in lines:
        dr.text((44, y), line, fill="#8b93a2", font=fonts["tiny"])
        y += 15
    dr.text((22, H - 24), footer, fill="#6b7280", font=fonts["tiny"])
    for k, line in enumerate(counter.split("\n")):
        w = dr.textlength(line, font=fonts["small"])
        dr.text((W - 22 - w, 22 + 22 * k), line, fill="#9aa4b2",
                font=fonts["small"])
    return np.asarray(im.convert("RGB"))


def time_map(S: dict, seconds: float, share_spike: float):
    """物理の時刻 -> 動画の時刻。区間ごとに再生倍率を変える。

    接近は 170 ms あるのに GF の伝導は 1.1 ms しかない。一定倍率で 260 ms を
    9 秒に伸ばしても伝導は 0.04 秒 = 1 フレームになり、**この動画の主役である
    「軸索を降りる」が見えなくなる**。発火の前後だけ画面の時間を多めに割く。

    share_spike は発火前後 (しきい値到達の 10 ms 前 〜 中脚を伸ばす 8 ms 後) に
    割り当てる画面時間の割合。残りを接近と跳躍で半分ずつ分ける。
    """
    pre = max(S["t_trigger"] - 0.010, 0.0)
    post = min(S["t_extend"] + 0.008, S["t_end"])
    bounds = np.array([0.0, pre, post, S["t_end"]])
    rest = (1.0 - share_spike) / 2.0
    movie = np.cumsum([0.0, rest, share_spike, rest]) * seconds
    return bounds, movie


def get_sense(args) -> dict:
    """視葉と脚神経核への入力。無ければ場面を走らせて作る。"""
    params = dict(approach=args.approach, start_cm=args.start,
                  theta_trigger=args.theta_trigger, gf_delay_ms=args.gf_delay_ms,
                  rise_ms=5.0, amp=1.9, seconds=args.sense_seconds,
                  settle=0.15, jump_force=3.0)
    S = load_sense(params, refresh=args.resense)
    if S is not None:
        print(f"控えてある感覚の入力を使う ({len(S['t_vis'])} 時刻)。"
              "やり直すなら --resense")
        return S
    print("脳を光らせる入力が無いので、同じ場面を一度走らせる ...")
    R = run_approach(keep_state=False, **params)
    if R["t_trigger"] is None:
        raise SystemExit("視角がしきい値に届かなかった。--theta-trigger を下げる")
    save_sense(R, params)
    return R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--n-visual", type=int, default=8)
    ap.add_argument("--n-motor", type=int, default=2)
    ap.add_argument("--spin", type=float, default=55.0, help="カメラの回転 [度]")
    ap.add_argument("--share-spike", type=float, default=0.5,
                    help="発火の前後に割り当てる画面時間の割合")
    ap.add_argument("--res", type=int, nargs=2, default=[900, 1000])
    ap.add_argument("--n-cloud", type=int, default=45000,
                    help="背景に置く全CNSのシナプスの点数")
    # --- 脳を光らせる入力を採る場面 (24_escape_full.py と同じもの) ---
    ap.add_argument("--approach", type=float, default=30.0, help="捕食者の速さ cm/s")
    ap.add_argument("--start", type=float, default=7.0, help="捕食者の初期距離 cm")
    ap.add_argument("--theta-trigger", type=float, default=45.0,
                    help="逃避を起こす視角 [deg]")
    ap.add_argument("--gf-delay-ms", type=float, default=1.1)
    ap.add_argument("--sense-seconds", type=float, default=0.26,
                    help="入力を採る物理の長さ s")
    ap.add_argument("--refresh", action="store_true", help="シナプスの抽出をやり直す")
    ap.add_argument("--resense", action="store_true", help="場面の走りをやり直す")
    ap.add_argument("--out", default="synapse_fire.mp4")
    args = ap.parse_args()

    S = get_sense(args)
    T = args.seconds
    bounds, movie_b = time_map(S, T, args.share_spike)

    def to_movie(ts):
        return float(np.interp(ts, bounds, movie_b))

    def slow_at(tm):
        """画面の時刻 tm での再生倍率。区間ごとに変わるので画面に出す。"""
        i = int(np.clip(np.searchsorted(movie_b, tm, "right") - 1,
                        0, len(bounds) - 2))
        return ((movie_b[i + 1] - movie_b[i])
                / max(bounds[i + 1] - bounds[i], 1e-9))

    t_spike, t_arrive = to_movie(S["t_trigger"]), to_movie(S["t_extend"])
    t_look = to_movie(max(S["t_trigger"] - 0.12, 0.0))
    for i in range(len(bounds) - 1):
        ds = bounds[i + 1] - bounds[i]
        dm = movie_b[i + 1] - movie_b[i]
        print(f"  物理 {bounds[i]*1e3:6.1f}-{bounds[i+1]*1e3:6.1f} ms -> "
              f"画面 {movie_b[i]:5.2f}-{movie_b[i+1]:5.2f} s "
              f"(x{dm/max(ds, 1e-9):.0f} スロー)")

    panel = BrainPanel(res=tuple(args.res), n_visual=args.n_visual,
                       n_motor=args.n_motor, refresh=args.refresh,
                       t_look=t_look, t_spike=t_spike, t_arrive=t_arrive,
                       spin=args.spin, n_cloud=args.n_cloud,
                       n_omma=S["n_omma"])
    t_vis_movie = np.array([to_movie(tv) for tv in S["t_vis"]])
    panel.attach_optic_drive(t_vis_movie, S["photo"], S["motion"])
    panel.attach_leg_drive(t_vis_movie, S["leg_speed"])

    fonts = {"title": jp_font(19), "small": jp_font(13), "tiny": jp_font(11)}
    n_frames = int(T * args.fps)
    frames = []
    print(f"{n_frames} フレーム描画中 ...")
    for f in range(n_frames):
        t = f / args.fps
        img, n_lit, n_fired = panel.frame(t, f / max(n_frames - 1, 1))
        frames.append(draw_overlay(
            img, panel.stage(t), LEGEND,
            "Male CNS v1.0 — 形・シナプスの位置・接続は実測 / 伝導の速さは模式",
            CLOUD_NOTE,
            f"逃避回路 光っている {n_lit:,} / 放出済み {n_fired:,}\n"
            f"入力で光る点 {panel.n_cloud_lit:,} / {len(panel.cloud):,}\n"
            f"スロー再生 x{slow_at(t):.0f}",
            fonts))
        if (f + 1) % 30 == 0:
            print(f"  {f+1}/{n_frames}")

    OUT.mkdir(exist_ok=True)
    path_out = OUT / args.out
    imageio.mimsave(path_out, frames, fps=args.fps, macro_block_size=None)
    print(f"書き出し: {path_out} ({len(frames)} フレーム, "
          f"{path_out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
