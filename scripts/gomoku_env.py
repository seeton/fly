"""五目並べを、**報酬だけで**覚えさせるための場。

飛ばさない。盤の良し悪しを手で書いて蝿に渡すこともしない。渡すのは

    「打てたら加点」「並べたら加点」「止めたら加点」「勝ったら加点」

だけで、**どこが良い手かは一度も教えない**。逃避で跳ぶ向きを教えずに
複眼の左右差 1 つから出させたのと同じ立て方 (`escape_env.py`)。

**合法手も教えない。** 空いているマスだけを候補にすれば「お手つき」は原理的に
起きないが、それだと「打てる所に打つ」を覚える余地が無くなる。ここでは 81 マス
全部を候補にし、石のあるマスを選んだらその手番を空振りする。stage 0 の報酬は
打った手のうち合法だった割合で、これが 1.0 になるまで次へ進まない。

**打ち手の作り (4方向で重みを共有する)**

    あるマスの点数を決めるのに見るのは、そのマスを通る4本の線の上の前後4マス
    ずつ (自分の石 / 相手の石 / 盤の外 の3チャンネル x 8マス = 24)。重み W は
    **4方向で共有** する。縦の三連も斜めの三連も同じ形なので、方向ごとに別の
    重みを覚えさせる理由が無い。個眼が視野のどこでも同じ受容野を持っているのと
    同じ考え方で、探索する次元も4分の1で済む。

    左右反転も同じ重みを通す (窓を裏返して同じ W に入れ、足す)。
    「〇〇_〇〇」と「〇〇〇〇_」は区別できるまま、鏡像対称だけが厳密に入る。

        点数 = Σ_4方向 [ v・tanh(W x + b) + v・tanh(W x~ + b) ]
               + c_自 (そのマスに自分の石) + c_相 (相手の石) + w_中央・中央からの遠さ

    そのマスの石の有無にかかる重み c_自 / c_相 が、合法手を覚える場所。

**スパーリング相手にだけ手書きの知識を入れる。** `greedy_move` は連の長さを
数えて打つ素朴な相手で、これは蝿ではなく**相手側**。蝿の側に連を数える関数は
一切渡していない (`runs` を呼ぶのは相手と、報酬を測る側だけ)。

**加点は必ず有界な正の項にする** (`CLAUDE.md`)。罰で引くと「早く負ける方が得」
になる。「相手を止めた」も `1 - 相手の最長連/5` という 0..1 の加点で書く。
"""

from __future__ import annotations

import numpy as np

N = 9                 # 盤の大きさ (9x9)
WIN = 5               # 何目並べるか
REACH = 4             # 受容野の届く距離 (片側4マス = 五目に要る範囲)
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
OFFS = [k for k in range(-REACH, REACH + 1) if k != 0]
NOFF = len(OFFS)                      # 8
NCH = 3                               # 自分 / 相手 / 盤の外
NFEAT = NCH * NOFF                    # 24
OUTSIDE = 2                           # 盤の外を表す値

# 窓を左右反転する並べ替え (チャンネルごとに前後をひっくり返す)
MIRROR = np.concatenate([c * NOFF + np.arange(NOFF)[::-1] for c in range(NCH)])

WMAX = 3.0            # 重みの範囲。CMA-ES は [0,1] で探して ±WMAX に伸ばす


# ------------------------------------------------------------------ 盤

def empty_board() -> np.ndarray:
    return np.zeros((N, N), dtype=np.int8)


def windows(board: np.ndarray) -> np.ndarray:
    """各マスの受容野 (N*N, 4方向, 24)。board は +1=自分 / -1=相手 / 0=空。"""
    pad = np.full((N + 2 * REACH, N + 2 * REACH), OUTSIDE, dtype=np.int8)
    pad[REACH:REACH + N, REACH:REACH + N] = board
    out = np.empty((N, N, len(DIRS), NCH, NOFF), dtype=np.float32)
    for di, (dy, dx) in enumerate(DIRS):
        for oi, k in enumerate(OFFS):
            y0, x0 = REACH + k * dy, REACH + k * dx
            sl = pad[y0:y0 + N, x0:x0 + N]
            out[:, :, di, 0, oi] = sl == 1
            out[:, :, di, 1, oi] = sl == -1
            out[:, :, di, 2, oi] = sl == OUTSIDE
    return out.reshape(N * N, len(DIRS), NFEAT)


def runs(board: np.ndarray, x: np.ndarray | None = None):
    """そのマスに置いたら何目の連になるか (N*N, 4方向) を、自分と相手について。

    **蝿はこれを使わない**。素朴な相手 (`greedy_move`) と、報酬を測る側のもの。
    受容野の窓から数えるので盤を舐め直さずに済む。
    """
    if x is None:
        x = windows(board)
    w = x.reshape(N * N, len(DIRS), NCH, NOFF)
    out = []
    for ch in (0, 1):
        left = w[:, :, ch, :REACH][:, :, ::-1]       # -1,-2,-3,-4 の順に見る
        right = w[:, :, ch, REACH:]                  # +1,+2,+3,+4
        out.append(np.cumprod(left, axis=-1).sum(-1)
                   + np.cumprod(right, axis=-1).sum(-1) + 1.0)
    return out[0], out[1]


def _shift(b: np.ndarray, dy: int, dx: int, k: int) -> np.ndarray:
    """b を (dy,dx)*k だけずらす。**端は巻き込まず 0 で埋める** (np.roll は使えない)。"""
    sh = np.zeros_like(b)
    ys, xs = dy * k, dx * k
    sh[max(0, -ys):N - max(0, ys), max(0, -xs):N - max(0, xs)] = \
        b[max(0, ys):N - max(0, -ys), max(0, xs):N - max(0, -xs)]
    return sh


def longest(board: np.ndarray, who: int) -> int:
    """who の最長連 (実際に盤に並んでいる数)。"""
    b = (board == who).astype(np.int8)
    if not b.any():
        return 0
    best = 1
    for dy, dx in DIRS:
        acc = b.copy()
        for k in range(1, WIN):
            acc = acc * _shift(b, dy, dx, k)
            if not acc.any():
                break
            best = max(best, k + 1)
    return best


def winner(board: np.ndarray) -> int:
    """+1 / -1 / 0 (まだ決まっていない)。"""
    for who in (1, -1):
        if longest(board, who) >= WIN:
            return who
    return 0


# ------------------------------------------------------------ 蝿の打ち手

class FlyPlayer:
    """盤を見て 81 マス全部に点数をつけ、いちばん高いマスに打つ。

    合法かどうかの判定を持たない。石のあるマスを選んだらそれは「お手つき」で、
    打ち損じたまま手番が流れる。
    """

    def __init__(self, K: int = 6):
        self.K = int(K)
        self.dim = self.K * NFEAT + 2 * self.K + 3
        yy, xx = np.mgrid[0:N, 0:N]
        c = (N - 1) / 2.0
        # 中央からの遠さ 0..1。これ自体は「中央が良い」と言っていない。
        # 良いか悪いかは重み w_center が学習で決める
        self.center = (np.abs(yy - c) + np.abs(xx - c)).ravel().astype(np.float32)
        self.center /= self.center.max()
        self.set_flat(np.full(self.dim, 0.5))

    # --- パラメータ ---
    def set_flat(self, z) -> "FlyPlayer":
        """CMA-ES の [0,1]^dim をそのまま受ける。±WMAX に伸ばして持つ。"""
        self.z = np.asarray(z, dtype=float).copy()
        w = WMAX * (2.0 * self.z - 1.0)
        i = 0
        self.W = w[i:i + self.K * NFEAT].reshape(self.K, NFEAT).astype(np.float32)
        i += self.K * NFEAT
        self.b = w[i:i + self.K].astype(np.float32)
        i += self.K
        self.v = w[i:i + self.K].astype(np.float32)
        i += self.K
        self.c_self, self.c_opp, self.w_center = w[i], w[i + 1], w[i + 2]
        self.Wm = self.W[:, MIRROR].copy()          # 左右反転した窓を通す用
        return self

    # --- 盤を見る ---
    def scores(self, board: np.ndarray) -> np.ndarray:
        """(N*N,) の点数。石のあるマスにも出す (画面でそのまま見せるため)。"""
        x = windows(board)
        h = np.tanh(x @ self.W.T + self.b) @ self.v
        hm = np.tanh(x @ self.Wm.T + self.b) @ self.v
        s = (h + hm).sum(axis=1)
        flat = board.ravel()
        return (s + self.c_self * (flat == 1) + self.c_opp * (flat == -1)
                + self.w_center * self.center)

    def move(self, board: np.ndarray, rng=None) -> int:
        """打つマス (0..N*N-1)。**合法とは限らない**。"""
        s = self.scores(board)
        if rng is not None:
            s = s + rng.normal(0.0, 1e-6, s.shape)   # 同点をばらす
        return int(np.argmax(s))


# ------------------------------------------------------------------ 相手

def random_move(board: np.ndarray, rng) -> int:
    free = np.flatnonzero(board.ravel() == 0)
    return int(rng.choice(free))


def greedy_move(board: np.ndarray, rng) -> int:
    """連の長さを数えて打つ素朴な相手。**手書きの知識はここにだけ置く**。

    自分の連を伸ばすのを優先し、少し譲って相手の連も止める
    (3^連長 なので、5 を作る手は 4 を止める手より必ず強い)。
    """
    free = board.ravel() == 0
    if not free.any():
        return -1
    mine, theirs = runs(board)
    val = (3.0 ** mine).sum(1) + 0.95 * (3.0 ** theirs).sum(1)
    val = np.where(free, val, -np.inf) + rng.normal(0.0, 1e-3, N * N)
    return int(np.argmax(val))


def fly_opponent(z, K: int = 6):
    """過去の自分を相手にする (自己対戦)。"""
    ref = FlyPlayer(K).set_flat(z)

    def play(board, rng):
        mv = ref.move(board, rng)
        if board.ravel()[mv] != 0:        # 過去の自分もお手つきする
            return -1
        return mv

    return play


# ------------------------------------------------------------------ 一局

def play_game(fly: FlyPlayer, opp, rng, fly_first: bool = True,
              opening: int = 2) -> dict:
    """一局打って、報酬の材料を返す。

    opening: 出だしの何手を互いにランダムで置くか。決定的な打ち手どうしだと
             毎回まったく同じ棋譜になるので、序盤だけ散らす。
    """
    board = empty_board()
    tried = made = 0
    for ply in range(N * N * 2):
        if (board == 0).sum() == 0:
            break
        me = 1 if ((ply % 2 == 0) == fly_first) else -1
        if me == 1:
            if ply < opening * 2:
                mv = random_move(board, rng)
            else:
                mv = fly.move(board, rng)
                tried += 1
            y, x = divmod(mv, N)
            if board[y, x] == 0:
                board[y, x] = 1
                if ply >= opening * 2:
                    made += 1
            # お手つきなら盤は変わらない (空振りしたまま手番が移る)
        else:
            if ply < opening * 2:
                mv = random_move(board, rng)
            else:
                mv = opp(-board, rng)     # 相手は自分を +1 として見る
            if mv >= 0:
                y, x = divmod(mv, N)
                if board[y, x] == 0:
                    board[y, x] = -1
        if winner(board) != 0:
            break
    w = winner(board)
    return {"legal": made / max(tried, 1), "win": float(w == 1),
            "loss": float(w == -1), "run": longest(board, 1),
            "opp_run": longest(board, -1), "board": board, "tried": tried}


# ------------------------------------------------------------------ 報酬

# ステージ。**羽ばたきと制御ゲインを同時に動かさないのと同じ**で、
# 「打てる」「並べる」「止める」「競る」を一度に求めない。
STAGES = ("0 打てるようになる", "1 並べる", "2 止める", "3 自分と打つ")


def reward(t: dict, stage: int) -> float:
    """有界な正の項の足し算。どの項も 0..1 で、罰は使わない。"""
    legal = t["legal"]
    if stage == 0:
        return legal
    run = min(t["run"], WIN) / WIN
    if stage == 1:
        return 0.30 * legal + 0.40 * run + 0.30 * t["win"]
    stop = 1.0 - min(t["opp_run"], WIN) / WIN      # 止めたぶんも加点で書く
    if stage == 2:
        return 0.15 * legal + 0.20 * run + 0.25 * stop + 0.40 * t["win"]
    return 0.10 * legal + 0.15 * stop + 0.75 * t["win"]


def opponent_for(stage: int, ref_z=None, K: int = 6):
    if stage <= 1:
        return random_move
    if stage == 2 or ref_z is None:
        return greedy_move
    return fly_opponent(ref_z, K)


def evaluate(job) -> float:
    """CMA-ES から1個体ぶん。**同じ世代は同じ対局で比べる** (seed を配る)。"""
    z, stage, seed, games, ref_z, K = job
    fly = FlyPlayer(K).set_flat(z)
    opp = opponent_for(stage, ref_z, K)
    tot = 0.0
    for g in range(games):
        rng = np.random.default_rng(seed + g)
        tot += reward(play_game(fly, opp, rng, fly_first=(g % 2 == 0)), stage)
    return tot / games


def measure(z, stage: int, games: int = 40, seed: int = 12345,
            ref_z=None, K: int = 6) -> dict:
    """勝率などを人が読める形で測る。"""
    fly = FlyPlayer(K).set_flat(z)
    opp = opponent_for(stage, ref_z, K)
    acc = {"legal": 0.0, "win": 0.0, "loss": 0.0, "run": 0.0, "opp_run": 0.0}
    for g in range(games):
        rng = np.random.default_rng(seed + g)
        t = play_game(fly, opp, rng, fly_first=(g % 2 == 0))
        for k in acc:
            acc[k] += t[k]
    return {k: v / games for k, v in acc.items()}
