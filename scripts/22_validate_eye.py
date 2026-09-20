"""複眼が本当に「自分の回転」を取り出せているかを、センサ単体で検証する。

自由飛行の中で測ると、飛び方の良し悪しと視覚の良し悪しが混ざって切り分けられない。
ここでは物理を回さず、**既知のヨー角速度と前進速度で機体を運動学的に動かし**、
そのとき複眼が何を出すかだけを見る。

比べるもの:

  旧 … 視野140度を96画素で鋭く描き、画像全体をずらして一番合う位置を探す。
        角度較正 -8.6 は物理エンジンの真値に対して合わせたもので、
        実物のハエが持てない情報を使っていた。
  新 … 個眼の受容角 (5.1度) で畳み込み、個眼間隔 (5.0度) で受け、
        相関型検出器 (T4/T5 の演算) にかける。出力は任意単位のまま。

見るべきは「角速度を当てられるか」ではなく **符号が正しいか / 単調か /
前進速度によらないか**。実物の視運動反応も大きさの較正は持っていない。

使い方:
  python scripts/22_validate_eye.py
  python scripts/22_validate_eye.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_vision import (DPHI_DEG, DRHO_DEG, FOV_DEG, FlyEyes,  # noqa: E402
                        acceptance_matrix)
from flybody_model import load_model  # noqa: E402


def print_optics(res: int) -> None:
    M = acceptance_matrix(res)
    n = M.shape[0]
    half = np.radians(FOV_DEG) / 2
    u = (np.arange(res) + 0.5) / res * 2 - 1
    pix = np.degrees(np.arctan(u * np.tan(half)))
    ctr = np.array([np.average(pix, weights=M[i]) for i in range(n)])
    sd = np.array([np.sqrt(np.average((pix - ctr[i]) ** 2, weights=M[i]))
                   for i in range(n)])
    print(f"描画 {res}x{res} -> 個眼 {n}x{n} = {n*n} 個/眼 (実物は片眼 約750個)")
    print(f"  画素の角度刻み  中央 {pix[res//2]-pix[res//2-1]:.2f} 度 / "
          f"端 {pix[1]-pix[0]:.2f} 度  (透視投影なので一様ではない)")
    print(f"  受容野 FWHM     中央 {2.3548*sd[n//2]:.2f} 度 / "
          f"端 {2.3548*sd[-1]:.2f} 度  (狙い {DRHO_DEG})")
    print(f"  個眼の間隔      {np.diff(ctr).min():.2f} - {np.diff(ctr).max():.2f} 度"
          f"  (狙い {DPHI_DEG})\n")


def old_flow(prev, img, max_shift: int = 6) -> float:
    """旧方式: 列平均のプロファイルをずらして一番合う量を探す (放物線補間つき)。"""
    a, b = prev.mean(axis=0), img.mean(axis=0)
    errs = []
    for sh in range(-max_shift, max_shift + 1):
        if sh > 0:
            errs.append(np.mean((a[sh:] - b[:-sh]) ** 2))
        elif sh < 0:
            errs.append(np.mean((a[:sh] - b[-sh:]) ** 2))
        else:
            errs.append(np.mean((a - b) ** 2))
    errs = np.asarray(errs)
    k = int(np.argmin(errs))
    best = float(k - max_shift)
    if 0 < k < len(errs) - 1:
        y0, y1, y2 = errs[k - 1], errs[k], errs[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            best += float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=200.0, help="視覚の更新 [Hz]")
    ap.add_argument("--settle", type=float, default=0.25, help="フィルタを落ち着かせる時間")
    ap.add_argument("--measure", type=float, default=0.20, help="平均を取る時間")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    import mujoco

    m, _ = load_model(scene=True, two_flowers=True)
    eyes = FlyEyes(m, rate_hz=args.rate)
    print_optics(eyes.res)

    # 旧方式ぶんの描画 (96画素, ボケなし)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, 96)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 96)
    old_rend = mujoco.Renderer(m, height=96, width=96, max_geom=3000)

    d = mujoco.MjData(m)
    dt = 1.0 / args.rate
    n_settle, n_meas = int(args.settle / dt), int(args.measure / dt)

    rates = [-4.0, -2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0, 4.0]
    speeds = [0.0, 20.0] if args.quick else [0.0, 10.0, 20.0, 40.0]

    print(f"視覚 {args.rate:.0f} Hz、{args.settle:.2f}s 慣らし + "
          f"{args.measure:.2f}s 平均。全 {len(rates)*len(speeds)} 条件\n")
    rows = []
    for v in speeds:
        for w in rates:
            eyes.reset()
            x, y, yaw = 10.0, 0.0, 0.0
            prev_old = [None, None]
            acc_new, acc_old, k = 0.0, 0.0, 0
            for s in range(n_settle + n_meas):
                d.qpos[:3] = [x, y, 9.0]
                d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
                mujoco.mj_forward(m, d)
                eyes._next_t = 0.0                      # 毎回必ず更新させる
                eyes.maybe_update(m, d, 0.0)
                o = 0.0
                for i, cam in enumerate(("eye_left", "eye_right")):
                    old_rend.update_scene(d, camera=cam, scene_option=eyes.opt)
                    img = old_rend.render().astype(np.float32).mean(axis=2) / 255.0
                    if prev_old[i] is not None:
                        o += old_flow(prev_old[i], img)
                    prev_old[i] = img
                if s >= n_settle:
                    acc_new += eyes.rotation_signal
                    acc_old += o
                    k += 1
                x += v * np.cos(yaw) * dt
                y += v * np.sin(yaw) * dt
                yaw += w * dt
            rows.append((v, w, acc_new / k, acc_old / k))
            print(f"  前進 {v:4.0f} cm/s  ヨー {w:+5.1f} rad/s  "
                  f"新 {acc_new/k:+11.3e}   旧 {acc_old/k:+8.4f}")

    rows = np.array(rows)
    print("\n--- 判定 ---")
    for col, nm in ((2, "新 (受容角 + 相関型検出器)"), (3, "旧 (鋭い像 + 画像ずらし)")):
        print(f"\n{nm}")
        # 符号: ヨー0 の条件は除く
        nz = rows[:, 1] != 0
        # 全体で見た向き (最小二乗の傾き) を基準に符号を揃える
        k = np.polyfit(rows[:, 1], rows[:, col], 1)[0]
        sign_ok = np.mean(np.sign(rows[nz, col] * k) == np.sign(rows[nz, 1])) * 100
        print(f"  符号一致        {sign_ok:5.1f}%   (向き: 角速度が正で出力 "
              f"{'正' if k > 0 else '負'})")
        for v in speeds:
            sel = rows[:, 0] == v
            r = np.corrcoef(rows[sel, 1], rows[sel, col])[0, 1]
            mono = np.all(np.diff(rows[sel, col][np.argsort(rows[sel, 1])]) * k > 0)
            print(f"  前進 {v:4.0f} cm/s : 相関 {r:+.3f}  単調 {'はい' if mono else 'いいえ'}")
        # 前進速度で向きが変わらないか (これが変わると使えない)
        ks = [np.polyfit(rows[rows[:, 0] == v, 1], rows[rows[:, 0] == v, col], 1)[0]
              for v in speeds]
        same = all(np.sign(x) == np.sign(ks[0]) for x in ks)
        spread = max(abs(x) for x in ks) / max(min(abs(x) for x in ks), 1e-30)
        print(f"  前進速度を変えても向きは同じ: {'はい' if same else 'いいえ'}"
              f"   感度の開き {spread:.1f} 倍")

    # --- 最も効く指標: まっすぐ飛んでいるのに「回っている」と出る量 ---
    # 視運動反応はこの信号を打ち消そうとするので、偽の回転があると
    # 機体はまっすぐ飛べない。しかも前進が速いほど悪化するなら致命的。
    print("\n--- まっすぐ飛んでいるのに「回っている」と誤って出る量 ---")
    print("  ヨー0 での出力を、ヨー0 近傍の感度 (±0.4 rad/s の傾き) で rad/s に換算")
    print(f"\n{'前進':>6}  {'新 (受容角+相関型)':>26}  {'旧 (鋭い像+画像ずらし)':>26}")
    for v in speeds:
        if v == 0:
            continue
        sel = rows[:, 0] == v
        w = rows[sel, 1]
        out = []
        for col in (2, 3):
            y = rows[sel, col]
            slope = (y[w == 0.4][0] - y[w == -0.4][0]) / 0.8
            out.append(y[w == 0][0] / slope if slope != 0 else float("nan"))
        print(f"{v:4.0f}    {out[0]:+8.4f} rad/s ({np.degrees(out[0]):+6.1f} deg/s)"
              f"   {out[1]:+8.4f} rad/s ({np.degrees(out[1]):+6.1f} deg/s)")

    # 相関型検出器は速い動きで飽和する (実物も同じ)。角速度計ではない。
    print("\n応答の圧縮 (角速度が4倍になっても出力は4倍にならない)")
    for v in speeds:
        sel = rows[:, 0] == v
        w, y = rows[sel, 1], rows[sel, 2]
        y1, y4 = abs(y[w == 1.0][0]), abs(y[w == 4.0][0])
        print(f"  前進 {v:4.0f} cm/s: 1 rad/s {y1:.4f} -> 4 rad/s {y4:.4f} "
              f"({y4/max(y1,1e-30):.2f} 倍)")


if __name__ == "__main__":
    main()
