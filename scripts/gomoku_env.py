"""五目並べを、**報酬だけで**覚えさせるための場。

飛ばさない。盤の良し悪しを手で書いて蝿に渡すこともしない。渡すのは

    「打てたら加点」「並べたら加点」「止めたら加点」「勝ったら加点」

だけで、**どこが良い手かは一度も教えない**。逃避で跳ぶ向きを教えずに
複眼の左右差 1 つから出させたのと同じ立て方 (`escape_env.py`)。

**合法手も教えない。** 空いているマスだけを候補にすれば「お手つき」は原理的に
起きないが、それだと「打てる所に打つ」を覚える余地が無くなる。ここでは盤の
全マスを候補にし、**石のあるマスを選んだらその場で負け** (大会の規則でも反則手は
負け)。0 段目の報酬は打った手のうち合法だった割合で、負けたくなければ石のある
マスを選ばないことを報酬から覚える。

はじめは「空振りして手番を失う」にしていた。読みを入れたらこれが抜け穴になった:
評価の粗いうちは読みの中で「何も置かない方がまし」と見て、**わざと空振りを
選んだ** (合法手の割合が 95% → 55%)。本物の五目並べにパスは無く、外のエンジン
Rapfi もパスを含む局面を扱えないので、規則を「負け」一本にそろえた。

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

# 盤の大きさ。**15x15**。はじめは 9x9 にしていたが、9x9 の自由形は先手必勝では
# なかった: 外のエンジン Rapfi どうしを打たせると、持ち時間 0.2〜3 秒のどれでも
# 盤が埋まって引き分け (11x11 も同じ。先手が勝ち切るのは 13x13 から)。15x15 は
# Allis (1994) が先手必勝を証明した標準の盤で、Rapfi どうしでも先手が勝つ
# (out/logs/rapfi_selfplay.txt)。受容野は局所的なので、重みは盤の大きさに依らない
N = 15
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
INF = 1e9
WIN_VALUE = 1e6       # 読み切った勝ち。点数 (せいぜい数十) より必ず大きく取る


# ------------------------------------------------------------------ 盤

def empty_board() -> np.ndarray:
    return np.zeros((N, N), dtype=np.int8)


PAD = N + 2 * REACH


def _window_index() -> np.ndarray:
    """(N*N, 4方向, 8) — 各マスの受容野が、縁を足した盤のどこを指すか。

    先に1度だけ作っておき、`windows` は1回の添字集めで済ませる。
    先読みを入れると `scores` を1手あたり数十〜数百回呼ぶので、
    32回のスライス代入 (約 150 us) がそのまま効いていた。
    """
    idx = np.empty((N * N, len(DIRS), NOFF), dtype=np.intp)
    for c in range(N * N):
        y, x = divmod(c, N)
        for di, (dy, dx) in enumerate(DIRS):
            for oi, k in enumerate(OFFS):
                idx[c, di, oi] = (REACH + y + k * dy) * PAD + (REACH + x + k * dx)
    return idx


WINDOW_INDEX = _window_index()

# 各マスが、縁を足した盤のどこにあるか
CELL_PAD = np.array([(REACH + c // N) * PAD + (REACH + c % N) for c in range(N * N)])


def _affected() -> list:
    """石を1つ置いたとき受容野が変わるマス: そのマスを通る4本の線の前後4マス + 自分。"""
    out = []
    for c in range(N * N):
        y, x = divmod(c, N)
        cells = {c}
        for dy, dx in DIRS:
            for k in range(-REACH, REACH + 1):
                yy, xx = y + dy * k, x + dx * k
                if 0 <= yy < N and 0 <= xx < N:
                    cells.add(yy * N + xx)
        out.append(np.array(sorted(cells)))
    return out


AFFECTED = _affected()


def _segments():
    """盤上の「5マスの並び」全部 (15x15 で 572 本) と、各マスがどの並びに入っているか。

    五が作れるかどうかは「自分の石が4つ・相手の石が0の並び」があるかどうかで
    決まる。これは**盤の規則そのもの**を数えているだけで、形の良し悪しではない。
    並びごとに石の数を持っておけば、石を1つ置くたびにそのマスを通る並び
    (最大20本) を足し引きするだけで判定できる。盤全体を数え直すと1回 60 us
    かかり、読みの節ごとに呼ぶと効いていた。
    """
    segs = []
    for dy, dx in DIRS:
        for y in range(N):
            for x in range(N):
                ye, xe = y + dy * (WIN - 1), x + dx * (WIN - 1)
                if 0 <= ye < N and 0 <= xe < N:
                    segs.append([(y + dy * k) * N + (x + dx * k) for k in range(WIN)])
    segs = np.array(segs)
    of_cell = [np.flatnonzero((segs == c).any(1)) for c in range(N * N)]
    return segs, of_cell


SEGMENTS, SEGMENTS_OF = _segments()


def windows(board: np.ndarray) -> np.ndarray:
    """各マスの受容野 (N*N, 4方向, 24)。board は +1=自分 / -1=相手 / 0=空。"""
    pad = np.full((PAD, PAD), OUTSIDE, dtype=np.int8)
    pad[REACH:REACH + N, REACH:REACH + N] = board
    v = pad.ravel()[WINDOW_INDEX]                        # (N*N, 4, 8)
    out = np.stack((v == 1, v == -1, v == OUTSIDE), axis=2)
    return out.reshape(N * N, len(DIRS), NFEAT).astype(np.float32)


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


def five_through(board: np.ndarray, y: int, x: int, who: int) -> bool:
    """いま置いた (y,x) を通る五が並んだか。**盤の規則そのもの**。

    先読みの中で毎回呼ぶので、盤全体を舐める `winner` ではなくその点の周りだけ見る。
    """
    for dy, dx in DIRS:
        n = 1
        for s in (1, -1):
            k = 1
            while k < WIN:
                yy, xx = y + dy * k * s, x + dx * k * s
                if not (0 <= yy < N and 0 <= xx < N) or board[yy, xx] != who:
                    break
                n += 1
                k += 1
        if n >= WIN:
            return True
    return False


# ------------------------------------------------------------ 蝿の打ち手

class FlyPlayer:
    """盤の全マスに点数をつけ、**自分で先を読んで**打つ。

    合法かどうかの判定を持たない。石のあるマスを選んだらそれは「お手つき」で、
    その場で負け。読みの中でも同じ扱いにする。

    **先読みに入れたのは盤の規則だけ** — 五が並んだら勝ち、石のある所に打ったら
    負け、それだけ。
    「四は止めろ」「開いた三は強い」といった打ち方は一行も書いていない。
    読んでいる途中の局面の良し悪しは、**自分の点数** がそのまま決める:

        局面の値 = (自分がいま打てる空きマスの最高点)
                 - (相手がいま打てる空きマスの最高点)

    depth=0 にすると読まずに点数だけで打つ (元の姿)。**学習も打つときも同じ
    この関数を通す** — 深さ0で覚えさせて深さ4で打たせたら別物になる
    (`CLAUDE.md`: 制御則は1か所に置く)。
    """

    def __init__(self, K: int = 6, depth: int = 0, width: int = 8):
        self.K = int(K)
        self.depth = int(depth)      # 何手先まで読むか (0 = 読まない)
        self.width = int(width)      # 各段で見る候補の数 (自分の点数の上位から)
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
        # 素の窓と反転した窓を1回の行列積で通す (式は上と同じ、速さのためだけ)
        self._Wcat = np.concatenate([self.W, self.Wm]).T.copy()     # (24, 2K)
        self._bcat = np.concatenate([self.b, self.b])
        self._vcat = np.concatenate([self.v, self.v])
        self._center = (self.w_center * self.center).astype(np.float32)
        return self

    # --- 盤を見る ---
    def scores(self, board: np.ndarray) -> np.ndarray:
        """(N*N,) の点数。石のあるマスにも出す (画面でそのまま見せるため)。"""
        x = windows(board).reshape(N * N * len(DIRS), NFEAT)
        h = np.tanh(x @ self._Wcat + self._bcat) @ self._vcat
        s = h.reshape(N * N, len(DIRS)).sum(axis=1)
        flat = board.ravel()
        return (s + np.float32(self.c_self) * (flat == 1)
                + np.float32(self.c_opp) * (flat == -1) + self._center)

    # --- 先を読む ---
    def value(self, board: np.ndarray, me: int) -> float:
        """読みを打ち切った局面の値。**自分の点数だけで決める**。

            (me がいま打てる空きマスの最高点) - (相手がいま打てる空きマスの最高点)
        """
        free = board.ravel() == 0
        if not free.any():
            return 0.0
        return float(self.scores(board if me == 1 else -board)[free].max()
                     - self.scores(-board if me == 1 else board)[free].max())

    def _score_cells(self, pad: np.ndarray, cells: np.ndarray):
        """cells のマスだけ、自分 (+1) の目と相手 (-1) の目の点数を計算し直す。

        差分の計算し直し用。式は `scores` と同じで、両方の目を1回の行列積で通す
        (2回に分けると小さな行列積の呼び出しの重さが倍になる)。
        """
        m = len(cells)
        v = pad[WINDOW_INDEX[cells]]                            # (m, 4, 8)
        e1, e2, ew = v == 1, v == -1, v == OUTSIDE
        x = np.stack((np.stack((e1, e2, ew), axis=2),
                      np.stack((e2, e1, ew), axis=2)))          # (2, m, 4, 3, 8)
        x = x.reshape(2 * m * len(DIRS), NFEAT).astype(np.float32)
        h = (np.tanh(x @ self._Wcat + self._bcat) @ self._vcat)
        h = h.reshape(2, m, len(DIRS)).sum(axis=2)
        here = pad[CELL_PAD[cells]]
        cs, co, ctr = np.float32(self.c_self), np.float32(self.c_opp), self._center[cells]
        return (h[0] + cs * (here == 1) + co * (here == -1) + ctr,
                h[1] + cs * (here == -1) + co * (here == 1) + ctr)

    def move(self, board: np.ndarray, rng=None) -> int:
        """打つマス (0..N*N-1)。

        depth=0 なら全マスの点数の最大 — **合法とは限らない** (お手つきもある)。
        depth>0 なら `_Reading` で読む。読む候補は石の近くの空きマスで、
        石のあるマスは読まない (規則上その場で負けなので読むまでもない)。
        """
        s = self.scores(board)
        if rng is not None:
            s = s + rng.normal(0.0, 1e-6, s.shape)   # 同点をばらす
        if self.depth <= 0 or not (board != 0).any():
            return int(np.argmax(s))
        return _Reading(self, board).best_move(self.depth, self.width, tie=s)


class _Reading:
    """蝿が先を読むための盤。**置いて・戻す**を速くするためだけのもの。

    石を1つ置くと、点数が変わるのはそのマスを通る4本の線の前後4マス
    (最大33マス) だけ。盤全体の点数を毎回出し直すと1手 275 us かかるので、
    変わる所だけ計算し直す (`AFFECTED`)。両方の目 (自分・相手) の点数を持つ。

    読みに入れた規則は2つだけ:
      - 手番の側が五を作れるマスを持っていれば、その局面は手番の側の勝ち
      - 石のあるマスには打てない (打てば負け)
    **「四は止めろ」「開いた三は強い」とは書いていない**。相手の四を塞ぐ手は、
    塞がない手を読むと次の局面で相手が五を作れる (= 負け) と分かるから選ばれる。

    読む候補は、根 (いま打つ手) では**石から2マス以内の空きマス全部**、
    その先は蝿自身の点数の上位 width マス。根を全部読むのは、塞ぐべき
    マスを自分の点数が低く見ていても、候補から落とさないため
    (上位8マスだけ読んでいたときは、1つだけの五の脅威を塞げた割合が 0% だった)。
    石から2マスという範囲は盤の幾何で、どの五も既にある石の近くを通る。
    """

    def __init__(self, fly: "FlyPlayer", board: np.ndarray):
        self.fly = fly
        self.board = np.array(board, dtype=np.int8)
        self.pad = np.full(PAD * PAD, OUTSIDE, dtype=np.int8)
        self.pad[CELL_PAD] = self.board.ravel()
        self.s = {1: fly.scores(self.board), -1: fly.scores(-self.board)}
        # 石の近さ (2マス以内に石がいくつあるか)。置くたびに 5x5 を足し引きする
        occ = (self.board != 0).astype(np.int16)
        self.near = np.zeros((N + 4, N + 4), dtype=np.int16)
        for y, x in zip(*np.nonzero(occ)):
            self.near[y:y + 5, x:x + 5] += 1
        # 5マスの並びごとの石の数と、「あと1つで五」の並びの本数 (色ごと)
        seg = self.board.ravel()[SEGMENTS]
        self.cnt = {1: (seg == 1).sum(1).astype(np.int8),
                    -1: (seg == -1).sum(1).astype(np.int8)}
        self.fours = {w: int(((self.cnt[w] == WIN - 1) & (self.cnt[-w] == 0)).sum())
                      for w in (1, -1)}

    # -- 置く・戻す --
    def _count(self, c: int, who: int, step: int) -> None:
        """c を通る並びの石の数を step だけ動かし、「あと1つで五」の本数を合わせる。"""
        ids = SEGMENTS_OF[c]
        a, b = self.cnt[1], self.cnt[-1]
        before1 = int(((a[ids] == WIN - 1) & (b[ids] == 0)).sum())
        before2 = int(((b[ids] == WIN - 1) & (a[ids] == 0)).sum())
        self.cnt[who][ids] += step
        self.fours[1] += int(((a[ids] == WIN - 1) & (b[ids] == 0)).sum()) - before1
        self.fours[-1] += int(((b[ids] == WIN - 1) & (a[ids] == 0)).sum()) - before2

    def place(self, c: int, who: int):
        y, x = divmod(c, N)
        # 計算し直すのは受容野が変わるマスのうち**空きマスだけ**。読みの中で
        # 使うのは空きマスの点数だけ (候補も局面の値も空きマスしか見ない)
        aff = AFFECTED[c]
        aff = aff[self.pad[CELL_PAD[aff]] == 0]
        aff = aff[aff != c]
        saved = (c, who, aff, self.s[1][aff].copy(), self.s[-1][aff].copy(),
                 self.fours[1], self.fours[-1])
        self.board[y, x] = who
        self.pad[CELL_PAD[c]] = who
        self.near[y:y + 5, x:x + 5] += 1
        self._count(c, who, +1)
        if len(aff):
            self.s[1][aff], self.s[-1][aff] = self.fly._score_cells(self.pad, aff)
        return saved

    def undo(self, saved) -> None:
        c, who, aff, s1, s2, f1, f2 = saved
        y, x = divmod(c, N)
        self.board[y, x] = 0
        self.pad[CELL_PAD[c]] = 0
        self.near[y:y + 5, x:x + 5] -= 1
        self.cnt[who][SEGMENTS_OF[c]] -= 1          # 数え直さず、置く前の本数に戻す
        self.fours[1], self.fours[-1] = f1, f2
        self.s[1][aff] = s1
        self.s[-1][aff] = s2

    # -- 規則 --
    def can_five(self, who: int) -> bool:
        """who がいま五を作れるか (盤の規則そのもの)。

        「who の石が4つ・相手の石が0」の5マスの並びが1本でもあれば、残りの1マスに
        置けば五。並びごとの石の数は `place` / `undo` で持ち回っている。
        """
        return self.fours[who] > 0

    def candidates(self, who: int, width: int | None) -> np.ndarray:
        """石から2マス以内の空きマスを、who の目で見た点数の高い順に。"""
        free = (self.near[2:N + 2, 2:N + 2] > 0).ravel() & (self.board.ravel() == 0)
        cells = np.flatnonzero(free)
        order = cells[np.argsort(self.s[who][cells])[::-1]]
        return order if width is None else order[:width]

    def value(self, me: int) -> float:
        free = self.board.ravel() == 0
        if not free.any():
            return 0.0
        return float(self.s[me][free].max() - self.s[-me][free].max())

    # -- 読む --
    def negamax(self, me: int, depth: int, width: int,
                alpha: float, beta: float) -> float:
        if self.can_five(me):
            return WIN_VALUE + depth            # 手番の側が五を作れる。早い勝ちほど良い
        if depth <= 0:
            return self.value(me)
        best = -INF
        for c in self.candidates(me, width):
            saved = self.place(int(c), me)
            v = -self.negamax(-me, depth - 1, width, -beta, -alpha)
            self.undo(saved)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best if best > -INF else self.value(me)

    def best_move(self, depth: int, width: int, tie=None) -> int:
        cands = self.candidates(1, None)                 # 根は全部読む
        if tie is not None:                              # 同点のときの並びを散らす
            cands = cands[np.argsort(-tie[cands], kind="stable")]
        best, best_c, alpha = -INF, int(cands[0]), -INF
        for c in cands:
            c = int(c)
            y, x = divmod(c, N)
            saved = self.place(c, 1)
            if five_through(self.board, y, x, 1):
                self.undo(saved)
                return c
            v = -self.negamax(-1, depth - 1, width, -INF, -alpha)
            self.undo(saved)
            if v > best:
                best, best_c, alpha = v, c, v
        return best_c


# ------------------------------------------------------------------ 相手

def random_move(board: np.ndarray, rng) -> int:
    free = np.flatnonzero(board.ravel() == 0)
    return int(rng.choice(free))


def near_random_move(board: np.ndarray, rng, radius: int = 2) -> int:
    """石の近く (radius マス以内) の空きマスからランダムに。盤が空なら全体から。

    弱めた相手の「気まぐれな手」に使う。15x15 で盤全体から選ぶと、
    相手が勝手に遠くへ打って勝負から降りるだけになる。
    """
    occ = (board != 0).astype(np.int8)
    if not occ.any():
        return random_move(board, rng)
    near = np.zeros_like(occ)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            near |= _shift(occ, dy, dx, 1)
    free = np.flatnonzero(((near > 0) & (occ == 0)).ravel())
    if not len(free):
        return random_move(board, rng)
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


def eps_greedy(eps: float):
    """素朴な相手を弱める: 割合 eps で石の近くのランダムな手を打つ。梯子の下の段。"""
    def play(board, rng):
        if rng.random() < eps:
            return near_random_move(board, rng)
        return greedy_move(board, rng)
    return play


def opening_move(board: np.ndarray, rng) -> int:
    """出だしの散らし。**盤の中央 7x7 の空きマス**から選ぶ。

    15x15 で盤全体から選ぶと石が端へ散って、序盤の形として意味が無くなる。
    """
    c = N // 2
    free = [i for i in np.flatnonzero(board.ravel() == 0)
            if abs(i // N - c) <= 3 and abs(i % N - c) <= 3]
    return int(rng.choice(free))


def fives(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """いま五を作れる空きマスを、自分 (+1) と相手 (-1) について。

    **審判の側で使う** (報酬を数えるため)。蝿の打ち方には入っていない。
    """
    mine, theirs = runs(board)
    free = board.ravel() == 0
    return (np.flatnonzero(free & (mine >= WIN).any(1)),
            np.flatnonzero(free & (theirs >= WIN).any(1)))


# ------------------------------------------------------------------ 一局

def play_game(fly: FlyPlayer, opp, rng, fly_first: bool = True,
              opening: int = 2) -> dict:
    """一局打って、報酬の材料を返す。

    opening: 出だしの何手を互いにランダムで置くか。決定的な打ち手どうしだと
             毎回まったく同じ棋譜になるので、序盤だけ散らす。

    お手つき (石のあるマスを選ぶ) はその場で負け。**罰は引かない** — 勝ちの加点が
    消え、生き延びた手数がそこで止まるだけ。

    **1手ごとに起きたことも数える** (審判の側で。蝿の打ち方には入らない):
        決めた — 自分に五を作れるマスがあった手番で、実際に五を作ったか
        止めた — 相手に五を作れるマスが**1つだけ**あり、自分には無かった手番で、
                 そこを塞いだか (2つ以上なら塞ぎようがないので数えない)
    勝ち負けだけだと、毎局すぐ負けるうちは全員同点になって学習が進まない
    (15x15 の素朴な相手で実際にそうなった: 2〜3段目が 0% のまま止まった)。
    """
    board = empty_board()
    tried = made = 0
    chances = took = threats = blocked = 0
    plies, result = 0, 0
    for ply in range(N * N * 2):
        if not (board == 0).any():
            break
        me = 1 if ((ply % 2 == 0) == fly_first) else -1
        plies = ply + 1
        if ply < opening * 2:
            mv = opening_move(board, rng)
        elif me == 1:
            mine5, their5 = fives(board)
            mv = fly.move(board, rng)
            tried += 1
            if len(mine5):
                chances += 1
                took += int(mv in mine5)
            elif len(their5) == 1:
                threats += 1
                blocked += int(mv == their5[0])
        else:
            mv = opp(-board, rng)         # 相手は自分を +1 として見る
        if mv < 0:
            break                         # 相手が打つ所を失った (盤が埋まった)
        y, x = divmod(mv, N)
        if board[y, x] != 0:
            result = -me                  # お手つき。その場で負け
            break
        board[y, x] = me
        if me == 1 and ply >= opening * 2:
            made += 1
        if five_through(board, y, x, me):
            result = me
            break
    return {"legal": made / max(tried, 1), "win": float(result == 1),
            "loss": float(result == -1), "draw": float(result == 0),
            "run": longest(board, 1), "opp_run": longest(board, -1),
            "finish": took / chances if chances else 1.0,
            "block": blocked / threats if threats else 1.0,
            "chances": chances, "threats": threats,
            "plies": plies, "board": board, "tried": tried}


# ------------------------------------------------------------------ 段

# (名前, 相手, 蝿の読みの深さ)。**羽ばたきと制御ゲインを同時に動かさないのと
# 同じ**で、一度に求めない。
#
# 自分のコピーと打つ段は外した。9x9 で試したら素朴な相手への勝率が
# 12.5% → 0.8% に落ちた (自分にだけ勝つ手を覚える)。代わりに相手の梯子を登らせる:
# 素朴な相手を弱めたもの (手の一部をランダムにする) → 素朴な相手 → 外のエンジン
# Rapfi の読みを浅く制限したもの。素朴な相手からいきなり始めると、15x15 では
# 毎局 11〜14 手で負けて全員同点になり、学習が止まった。
#
# 読みの深さ: 0-1 段目は読まない (合法手を覚える段なので、石のあるマスを選べる
# 必要がある。読むと石のあるマスは候補に入らない)。2-3 段目は読み1 (根を全部読み、
# 相手が次に五を作れるかを見る)。4 段目からは読み2 (相手の応手まで読む)。
STAGES = [
    ("0 打てるようになる", "random", 0),
    ("1 並べる", "random", 0),
    ("2 止める (相手は半分気まぐれ)", ("greedy", 0.5), 1),
    ("3 止める (相手は2割気まぐれ)", ("greedy", 0.2), 1),
    ("4 読みながら止める", ("greedy", 0.0), 2),
    ("5 Rapfi 深さ1", ("rapfi", 1), 2),
    ("6 Rapfi 深さ2", ("rapfi", 2), 2),
    ("7 Rapfi 深さ3", ("rapfi", 3), 2),
]
SURVIVE_CAP = 60      # これだけ粘れば「生き延びた」の加点は満点


def stage_depth(stage: int) -> int:
    return STAGES[stage][2]


def opponent(spec):
    """相手の指定から打ち手を作る。

        "random" / "greedy" / ("greedy", 気まぐれの割合) / ("rapfi", 読みの深さ)
    """
    if spec == "random":
        return random_move
    if spec == "greedy":
        return greedy_move
    if isinstance(spec, tuple) and spec[0] == "greedy":
        return eps_greedy(spec[1]) if spec[1] > 0 else greedy_move
    if isinstance(spec, tuple) and spec[0] == "rapfi":
        from gomoku_pbrain import rapfi_opponent     # 学習のワーカーの中で起こす
        return rapfi_opponent(spec[1])
    raise ValueError(spec)


# ------------------------------------------------------------------ 報酬

def reward(t: dict, stage: int) -> float:
    """有界な正の項の足し算。どの項も 0..1 で、罰は使わない。

    「生き延びた手数」は**勝ったら満点**で数える。負けるまでの手数だけを
    数えると、早く勝つほど損になる (疑似農業で踏んだのと同じ話: 打ち切りで
    終わる課題は、終えたあとの残りも満点で数える)。

    「決めた」「止めた」は1手ごとの割合 (`play_game`)。その場面が一度も
    来なかった局は満点で数える — 来なかったことで損をさせない。
    """
    legal = t["legal"]
    if stage == 0:
        return legal
    run = min(t["run"], WIN) / WIN
    if stage == 1:
        return 0.25 * legal + 0.25 * run + 0.25 * t["finish"] + 0.25 * t["win"]
    survive = 1.0 if not t["loss"] else min(t["plies"], SURVIVE_CAP) / SURVIVE_CAP
    return (0.10 * legal + 0.15 * t["finish"] + 0.20 * t["block"]
            + 0.20 * survive + 0.35 * t["win"])


def evaluate(job) -> float:
    """CMA-ES から1個体ぶん。**同じ世代は同じ対局で比べる** (seed を配る)。"""
    z, stage, seed, games, K = job
    name, spec, depth = STAGES[stage]
    fly = FlyPlayer(K, depth=depth).set_flat(z)
    opp = opponent(spec)
    tot = 0.0
    for g in range(games):
        rng = np.random.default_rng(seed + g)
        tot += reward(play_game(fly, opp, rng, fly_first=(g % 2 == 0)), stage)
    return tot / games


def measure(z, spec, games: int = 40, seed: int = 12345, K: int = 6,
            depth: int = 0) -> dict:
    """相手 spec に対する成績を人が読める形で測る。先後は半分ずつ。"""
    fly = FlyPlayer(K, depth=depth).set_flat(z)
    opp = opponent(spec)
    keys = ("legal", "win", "loss", "draw", "run", "opp_run", "plies",
            "chances", "threats")
    acc = {k: 0.0 for k in keys}
    took = blocked = 0.0
    for g in range(games):
        rng = np.random.default_rng(seed + g)
        t = play_game(fly, opp, rng, fly_first=(g % 2 == 0))
        for k in keys:
            acc[k] += t[k]
        took += t["finish"] * t["chances"]
        blocked += t["block"] * t["threats"]
    out = {k: v / games for k, v in acc.items()}
    # 決めた・止めた は全局を通した割合 (局ごとの平均にすると、場面の少ない局が効きすぎる)
    out["finish"] = took / acc["chances"] if acc["chances"] else float("nan")
    out["block"] = blocked / acc["threats"] if acc["threats"] else float("nan")
    return out
