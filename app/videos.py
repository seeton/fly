"""out/ の動画カタログ — 「どれが今のコードの姿か」を出せるようにする。

動画はスクリプトを回した時点の姿で固まる。空力を numba にしたり armature を
入れたりしたあとも古い mp4 がそのまま残るので、見ている絵がいつのものか
分からなくなる。ここでは各動画について

  * 何を見せているか (説明)
  * どのコマンドで作れるか
  * **いつ作ったか / 素材のスクリプトより古くないか**

を持つ。古さの判定は mtime の比較。動画より新しいスクリプトがあれば「古い」に
する。依存は import 文から拾う (`12_flight.py` なら flybody_model, insect_aero)。

動画を作ったときのスクリプトの出力は `out/logs/<動画名>.txt` に残す。揚力が体重の
何倍だったか、方位がどれだけ揺れたか、といった数字はここにしか無い。説明文に
書いてしまうと動画より先に古くなるので、**そのとき出た数字はそのとき残す**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .paths import ARCHIVE, OUT, ROOT, SCRIPTS  # noqa: F401


@dataclass(frozen=True)
class Clip:
    """カタログの1件。`cmd` は scripts/ 以下を python に渡す引数列。"""

    name: str
    title: str
    desc: str
    cmd: tuple[str, ...]
    group: str


CATALOG: tuple[Clip, ...] = (
    # ---- 飛ぶ ----
    Clip("flight.mp4", "飛ぶ — 準定常翼素理論の空力",
         "200 Hz の羽ばたきに Sane & Dickinson の準定常翼素理論で外力を与える。"
         "係留状態の揚力は体重の 1.1〜1.2 倍で、放すと上昇する。60倍スロー。",
         ("scripts/12_flight.py",), "飛ぶ"),
    Clip("flight_plain.mp4", "比較 — MuJoCo 既定の空力だと高度を保てない",
         "楕円体流体モデルで解いた場合。前縁渦も回転揚力も表現されないぶんを"
         "羽ばたきの速さ (246 Hz) で補っても、1.0s で 1.00 → 0.69 cm と落ちていく。",
         ("scripts/12_flight.py", "--aero", "plain"), "飛ぶ"),
    Clip("flight_learned.mp4", "学習した飛び方",
         "CMA-ES + カリキュラム (13_train_flight.py) が出した out/flight_policy.json を"
         "そのまま飛ばす。羽ばたきを固定して制御ゲインだけ、のあとに全体を解放した結果。",
         ("scripts/14_render_flight.py",), "飛ぶ"),
    # ---- 見る・嗅ぐ ----
    Clip("fly_meadow.mp4", "草むらを複眼だけで飛ぶ",
         "風景の中を、ヨー角速度を複眼の像の流れからだけ推定して飛ばす。物理エンジンの"
         "真値は使っていない。狙いは針路の保持だが、**保てているかは下のログの"
         "「方位のばらつき」を見ること**。",
         ("scripts/18_fly_scene.py", "--out", "fly_meadow.mp4"), "見る・嗅ぐ"),
    Clip("fly_scene_vision.mp4", "針路保持 — 複眼から",
         "ヨー角速度を複眼の像の流れから推定してフィードバックする (実物のハエに近い形)。",
         ("scripts/18_fly_scene.py", "--yaw-source", "vision"), "見る・嗅ぐ"),
    Clip("fly_scene_true.mp4", "針路保持 — 物理エンジンの真値から",
         "同じ制御則に、複眼ではなくシミュレータの真の角速度を入れた上限の比較。",
         ("scripts/18_fly_scene.py", "--yaw-source", "true"), "見る・嗅ぐ"),
    Clip("fly_scene_none.mp4", "針路保持なし",
         "ヨーのフィードバックを切った対照。上2本との差が視覚の効き目のはずで、"
         "差が出ていなければ視覚が効いていないということ。",
         ("scripts/18_fly_scene.py", "--yaw-source", "none"), "見る・嗅ぐ"),
    Clip("vision_compare.mp4", "視覚フィードバックの比較 (3条件)",
         "t=0.5s に 6 rad/s のヨー外乱を与えて、複眼+頭部安定化 / 複眼のみ / "
         "フィードバックなし を横に並べる。3つの「方位のばらつき」がログに出る。",
         ("scripts/19_compare_vision.py",), "見る・嗅ぐ"),
    Clip("forage.mp4", "匂いで寄って、目で仕上げる",
         "花は 60cm 先。複眼で捉えられるのは 8〜10cm 以内なので、遠くは触角の匂いで"
         "風上へ、近づいたら視覚で機首を向ける。風はハエ自身も流す。",
         ("scripts/20_forage.py",), "見る・嗅ぐ"),
    Clip("choose_smell.mp4", "見た目が同じ2つの花を匂いで選ぶ",
         "視覚と嗅覚を切り替えず同時に使う。嗅覚は「それが好む花か」の判定を出し、"
         "視覚の的への接近を許可する。",
         ("scripts/21_choose.py", "--out", "choose_smell.mp4"), "見る・嗅ぐ"),
    Clip("choose_nosmell.mp4", "同じ状況で嗅覚を切ると",
         "匂いを無視させた対照。見た目が同じなので、おとりの花にも寄ってしまう。",
         ("scripts/21_choose.py", "--no-smell", "--out", "choose_nosmell.mp4"),
         "見る・嗅ぐ"),
    # ---- 回路から体へ ----
    Clip("escape_full.mp4", "逃避を3つ並べて — 脳 / 体 / 視界",
         "複眼に写った像から視角を毎フレーム測り、しきい値を超えた時刻を GF の発火時刻と"
         "して脳のパネルに渡す。絵に合わせて体を動かしているのではなく、視界から決まった"
         "時刻で両方が動く。再生倍率は区間ごとに変わる (画面に出る)。背景の点群も"
         "同じ走りの入力で光るので、視界パネルで影が広がると脳パネルの視葉が光る。",
         ("scripts/24_escape_full.py",), "回路から体へ"),
    Clip("synapse_fire.mp4", "実測シナプスが1個ずつ光る",
         "LC4/LPLC2 → DNp01 (Giant Fiber) → TTMn の経路を、Male CNS の実測シナプス"
         "座標のまま1個ずつ光らせる。脳だけの絵。背景の中枢神経系まるごとの点群は"
         "シミュレーションの入力で光る — 視葉は複眼に映った像、脚の神経核は関節の"
         "動き。入力を作っていない領域は暗いまま。",
         ("scripts/23_synapse_movie.py",), "回路から体へ"),
    Clip("forage_brain.mp4", "花に近づく脳 — 匂いと好み",
         "見た目が同じ2つの花 (匂いだけ違う) に近づく。触角葉は左右の触角に届いた"
         "匂いで、外側角は好む匂いがどれだけ優勢かで光る。おとりのプルームに"
         "迷い込むと外側角が消え、抜けると戻る。**キノコ体は暗いまま** — そこは"
         "学習した価値を扱う場所で、このシミュレーションは何も学習していない。",
         ("scripts/25_forage_brain.py",), "回路から体へ"),
    Clip("escape_takeoff.mp4", "跳躍 — 回路が指した動作を体にやらせる",
         "DNp01 発火 → TTMn → 中脚の転節-腿節関節を伸展。中脚だけの方が後脚も"
         "一緒に蹴るより高く跳ぶ (TTM が中脚の筋肉だという解剖と一致する)。",
         ("scripts/11_escape_takeoff.py",), "回路から体へ"),
    Clip("fly_walk.mp4", "歩行 — 三脚歩行",
         "マイクロCT 由来の実物大の体、6本脚 66関節を三脚歩行のパターンで動かす。"
         "接触・摩擦・重力は MuJoCo が解く。素朴なサイン波の歩容で、脳は繋いでいない。",
         ("scripts/10_fly_walk.py",), "歩行"),
)

GROUPS = ("飛ぶ", "見る・嗅ぐ", "回路から体へ", "歩行")

_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", re.M)


def _local_deps(script: Path, seen: set[Path] | None = None) -> set[Path]:
    """スクリプトが import している scripts/ 内のモジュールを再帰的に集める。"""
    seen = seen if seen is not None else set()
    if script in seen or not script.exists():
        return seen
    seen.add(script)
    try:
        src = script.read_text(encoding="utf-8")
    except OSError:
        return seen
    for mod in _IMPORT.findall(src):
        p = SCRIPTS / f"{mod}.py"
        if p.exists():
            _local_deps(p, seen)
    return seen


@dataclass
class ClipState:
    """カタログ1件 + 実ファイルの状態。"""

    clip: Clip
    path: Path
    exists: bool = False
    mtime: float = 0.0
    size_mb: float = 0.0
    stale: bool = False
    newest_source: str = ""
    source_mtime: float = 0.0
    archives: list[Path] = field(default_factory=list)
    log: str = ""            # 作ったときのスクリプトの出力

    @property
    def when(self) -> str:
        if not self.exists:
            return "未生成"
        return datetime.fromtimestamp(self.mtime).strftime("%m/%d %H:%M")

    @property
    def badge(self) -> str:
        """一覧に出す短い状態。"""
        if not self.exists:
            return "未生成"
        return "古い" if self.stale else "最新"

    @property
    def command(self) -> str:
        return "python " + " ".join(self.clip.cmd).replace("/", "\\")


LOGS = OUT / "logs"


def read_log(p: Path) -> str:
    """ログを読む。**文字コードで転んでもアプリを落とさない**。

    Windows のコンソールに素直にリダイレクトすると CP932 で書かれる
    (`PYTHONIOENCODING=utf-8` を付け忘れた場合)。out/logs は手で置くことも
    あるので、utf-8 → cp932 → 置き換え、の順で読む。
    """
    raw = p.read_bytes()
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def log_path(name: str) -> Path:
    """動画名に対応する実行ログの置き場。"""
    return LOGS / f"{Path(name).stem}.txt"


def save_log(name: str, text: str) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    p = log_path(name)
    p.write_text(text, encoding="utf-8")
    return p


def scan() -> list[ClipState]:
    """カタログを実ファイルと突き合わせる。"""
    states = []
    for clip in CATALOG:
        p = OUT / clip.name
        st = ClipState(clip=clip, path=p)
        deps = _local_deps(SCRIPTS / Path(clip.cmd[0]).name)
        if deps:
            newest = max(deps, key=lambda d: d.stat().st_mtime)
            st.newest_source = newest.name
            st.source_mtime = newest.stat().st_mtime
        if p.exists():
            s = p.stat()
            st.exists = True
            st.mtime = s.st_mtime
            st.size_mb = s.st_size / 1e6
            st.stale = st.source_mtime > s.st_mtime
        st.archives = sorted(ARCHIVE.glob(f"*_{clip.name}")) if ARCHIVE.exists() else []
        lp = log_path(clip.name)
        if lp.exists():
            try:
                st.log = read_log(lp)
            except Exception:
                pass      # ログが読めないだけで一覧を止めない
        states.append(st)
    return states


def orphans() -> list[Path]:
    """カタログに載っていない out/*.mp4 (書き捨てた試作など)。"""
    known = {c.name for c in CATALOG}
    return sorted(p for p in OUT.glob("*.mp4") if p.name not in known)
