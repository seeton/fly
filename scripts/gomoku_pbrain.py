"""外部の五目並べエンジン (Gomocup / Piskvork プロトコル) と話す窓口。

蝿の強さを**自分で書いた相手**で測ると、どうしても自分に都合のよい物差しになる。
外で鍛えられたエンジンを物差しにするためのもの。既定は Rapfi
(Gomocup の自由形で優勝し続けているエンジン、GPL v3)。

**同梱しない。** ソースを取ってきてこの機械でビルドする (`tools/` は git 管理外):

    git clone --depth 1 https://github.com/dhbloo/rapfi.git tools/rapfi/src
    # 重みは github.com/dhbloo/rapfi-networks から3つだけ:
    #   config-example/config.toml / classical/model220723.bin /
    #   mix9svq/mix9svqfreestyle_bsmix.bin.lz4
    cmake -S tools/rapfi/src/Rapfi -B tools/rapfi/build -G Ninja \\
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \\
          -DUSE_AVX2=OFF -DUSE_AVX512=OFF -DUSE_BMI2=OFF -DUSE_VNNI=OFF
    cmake --build tools/rapfi/build
    # exe の隣に config.toml と重み、mingw の実行時 DLL を置く

**MinGW の GCC では AVX2 を切ってビルドする。** 既定 (AVX2 あり) で作ると、盤に
石が1つでもある局面で必ずアクセス違反 (0xc0000005) で落ちた。空盤の初手と、
読まずに決まる手 (五を作る・四を止める) だけが通っていたので最初は気づかなかった。
Windows 64bit の GCC は AVX が求める 32 バイト境界にスタック変数を揃えない
(GCC の古い既知の不具合) ので、揃っている前提の命令がそこで落ちる。SSE4.1 まで
なら 16 バイトで足りる。Rapfi の README が Windows で Clang を勧めているのもこれ。

**15x15 なら NNUE 込みの全力で打つ。** 自由形の NNUE の重みは 9x9 に対応して
いない (START 9 だと "no compatible weight config found" で古典評価だけになる)。
盤を 15x15 にしたのはそのためでもある。

**1手ごとに `BOARD` で盤全体を渡す。** `TURN` で差分だけ送る方が速いが、
お手つき (空振り) や途中からの測定で両者の盤がずれると、黙って別の対局を
打ち始める。盤ごと渡せばずれようがない。

座標は Piskvork の決まりで `x,y` (x が列、y が行)。こちらの盤は (行, 列)。
"""

from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path

import numpy as np

from gomoku_env import N

ROOT = Path(__file__).resolve().parent.parent
RAPFI = ROOT / "tools" / "rapfi" / "build" / "pbrain-rapfi.exe"


class EngineError(RuntimeError):
    pass


class PbrainEngine:
    """Piskvork プロトコルのエンジンを1つ起こして、盤を渡すと手を返す。

    turn_ms:   1手に使ってよい時間。
    max_depth: 読みの深さの上限 (`INFO MAX_DEPTH`)。**梯子の段に使う**。
               時間で弱めると機械の混み具合で強さが変わるが、深さなら変わらない。
               None なら時間いっぱい読む (全力)。
    threads:   探索のスレッド数。学習で何十も並べて起こすときは 1 にする。
    """

    def __init__(self, exe: Path = RAPFI, turn_ms: int = 300, rule: int = 0,
                 size: int = N, max_depth: int | None = None,
                 threads: int | None = None):
        exe = Path(exe)
        if not exe.exists():
            raise EngineError(
                f"{exe} がありません。scripts/gomoku_pbrain.py の冒頭の手順でビルドしてください。")
        self.exe, self.turn_ms, self.rule = exe, int(turn_ms), int(rule)
        self.size = int(size)       # 盤の大きさ。物差しの確認には 9〜15 を使った
        self.proc = subprocess.Popen(
            [str(exe)], cwd=str(exe.parent), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._send(f"START {self.size}")
        self._expect_ok()
        self._send(f"INFO rule {self.rule}")               # 0 = 自由形
        self._send(f"INFO timeout_turn {self.turn_ms}")
        self._send("INFO timeout_match 100000000")
        if max_depth is not None:
            self._send(f"INFO MAX_DEPTH {int(max_depth)}")
        if threads is not None:
            self._send(f"INFO THREAD_NUM {int(threads)}")
        self.max_depth = max_depth
        self.log: list[str] = []

    # --- 入出力 ---
    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._q.put(line.rstrip("\r\n"))
        self._q.put(None)

    def _send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read(self, timeout: float) -> str:
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            raise EngineError("エンジンが時間内に答えなかった")
        if line is None:
            raise EngineError("エンジンが終了した")
        return line

    def _expect_ok(self) -> None:
        while True:
            line = self._read(30.0)
            if line.startswith("OK"):
                return
            if line.startswith("ERROR"):
                raise EngineError(line)

    # --- 打たせる ---
    @staticmethod
    def _sequence(board: np.ndarray, me: int) -> list[str]:
        """盤を「着手の列」に並べ直す。**自分と相手を交互に、相手の石で終える**。

        Rapfi は `BOARD` の石を打った順として並べ直し、同じ側が続けば間に
        パスを挟む。行の順に渡すと最後が自分の石のとき**相手の番として
        考え始め**、途中でエンジンが落ちた。交互に並べれば自分の番で終わる。
        本当の着手順ではないが、五のない盤からどの順に石を抜いても五は
        できないので、途中の局面で決着してしまうことはない。

        **パスを含む局面は渡せない。** 空振りで石数がずれた局面をパスで
        表すと、置き場所によっては Rapfi が落ちた (最後に置いても途中に
        置いても)。なので石数は「同じ」か「相手が1つ多い」だけを受け付け、
        それ以外は呼ぶ側 (`31_gomoku_match.duel`) でお手つき = 負けにしている。
        """
        mine = [(int(y), int(x)) for y, x in zip(*np.nonzero(board == me))]
        theirs = [(int(y), int(x)) for y, x in zip(*np.nonzero(board == -me))]
        if len(theirs) not in (len(mine), len(mine) + 1):
            raise EngineError(f"石数が交互に並ばない (自分 {len(mine)} / 相手 {len(theirs)})")
        seq = []
        side = 2 if len(theirs) > len(mine) else 1
        while mine or theirs:
            y, x = (mine if side == 1 else theirs).pop()
            seq.append(f"{x},{y},{side}")
            side = 3 - side
        return seq

    def _wait_idle(self) -> None:
        """エンジンの手が空くまで待つ。

        Rapfi は**考えている最中に届いたコマンドを黙って捨てる**。手を出力して
        から「考え中」を下ろすまでに隙間があり、そこに次の `BOARD` が落ちると
        続く座標の行を別のコマンドとして読んで壊れた (持ち時間 3 秒で起きた)。
        `ABOUT` に返事が来れば手が空いている。来なければ送り直す。
        """
        for _ in range(200):
            self._send("ABOUT")
            try:
                while True:
                    line = self._read(0.25)
                    if "name=" in line:
                        return
                    self.log.append(line)
            except EngineError as e:
                if "終了" in str(e):
                    raise
        raise EngineError("エンジンの手が空かない")

    def move(self, board: np.ndarray, me: int = 1) -> int:
        """board は me を自分として見た盤。打つマス (行*N + 列) を返す。"""
        self._wait_idle()
        self._send("BOARD")
        for line in self._sequence(board, me):
            self._send(line)
        self._send("DONE")
        wait = self.turn_ms / 1000.0 * 4 + 10.0
        while True:
            line = self._read(wait)
            head = line.split(" ", 1)[0]
            if head in ("MESSAGE", "DEBUG", "OK", "UNKNOWN", "SUGGEST"):
                self.log.append(line)
                continue
            if head == "ERROR":
                raise EngineError(line)
            try:
                xs, ys = line.split(",")
                x, y = int(xs), int(ys)
            except ValueError:
                self.log.append(line)
                continue
            return y * self.size + x

    def close(self) -> None:
        try:
            self._send("END")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------ 学習の相手として

_ENGINES: dict = {}


def rapfi_opponent(max_depth: int | None, turn_ms: int = 200):
    """`gomoku_env.play_game` に渡せる相手。**プロセスごとに1つだけ起こして使い回す**。

    学習は multiprocessing のワーカーで回るので、ワーカーごとに自分のエンジンを
    持つ。1手ごとに起こすと起動 (設定と重みの読み込み) だけで 0.1 秒かかる。
    """
    key = (max_depth, turn_ms)
    if key not in _ENGINES:
        # 段が変わったら前の段のエンジンは閉じる。32 ワーカー x 段の数だけ
        # 置換表 (既定 32 MB) を抱えたまま残さない
        for old in list(_ENGINES):
            _ENGINES.pop(old).close()
        _ENGINES[key] = PbrainEngine(turn_ms=turn_ms, max_depth=max_depth, threads=1)
    eng = _ENGINES[key]

    def play(board, rng):
        return eng.move(board)

    return play
