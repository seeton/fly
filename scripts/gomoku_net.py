"""蝿の目を畳み込みの網にし、自分と打って勝ち負けだけから覚えさせる。

`gomoku_env.py` の打ち手 (重み 159 個を CMA-ES で探す) は、15x15 で素朴な相手に
15% しか勝てず、外のエンジン Rapfi の一番浅い段にも全敗だった。全敗が続くと
報酬が動かず、CMA-ES には進む向きが無い。学習の仕組みごと替える
(AlphaZero の型):

**目 (網)**
    3x3 の畳み込みを重ねた網。どの場所でも同じ重みを使うのは、元の打ち手の
    「4方向で共有する受容野」と同じ考え方を、線から面へ広げたもの (個眼が
    視野のどこでも同じ受容野を持っているのと同じ)。出すものは2つ:
        手の確からしさ  各マスに打つのがどれくらい良さそうか
        局面の見込み    手番の側が勝ちそうか (-1..+1)
    盤の大きさに依らない (全部が畳み込みと盤全体の平均/最大で書いてある)
    ので、9x9 で立ち上げてから 15x15 に移せる。

**読み (モンテカルロ木探索)**
    その目で候補を選び、局面を見込みで評価しながら木を読む。**読みに入る規則は
    「五が並んだら勝ち」「石のある所には打てない」「盤が埋まったら引き分け」
    だけ** — 前の打ち手と同じ。「四は止めろ」も「開いた三」も書いていない。

**学習 (勝ち負けだけ)**
    自分自身と打ち、各局面について
        手の確からしさ ← 読んだ末にどの手をどれだけ読んだか (訪問回数の割合)
        局面の見込み   ← その局を最後に勝ったか負けたか (+1 / -1 / 0) を、
                         決着までの手数 k で gamma^(k-1) だけ割り引いたもの
    を教師にする。**手作りの加点は一つも無い** — 前の打ち手の「決めた」
    「止めた」「生き延びた」も要らない。報酬は勝ち負けの一つだけ。
    割り引く理由は `SelfPlay.step` に書いた (割り引かないと「先に打った側が
    勝つ」を石の数から当てるだけになり、後手が塞ぐ理由が無くなった)。

**打つときも学習と同じ読みを通す** (`Player.move`)。`CLAUDE.md`: 制御則は1か所に置く。

**何十局も並べて、網への問い合わせをまとめて通す。** CPU だけなので、1局ずつ
1回ずつ網に聞くと遅い (6層 x 64ch で 64局まとめて 56 ms = 毎秒 1,100 評価)。
自己対戦は G 局を同時に進め、各局の読みが次に聞きたい局面を1つずつ集めて
1回で通す (`SelfPlay`)。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 5
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


# ------------------------------------------------------------------ 盤の規則

def five_at(board: np.ndarray, y: int, x: int, who: int) -> bool:
    """いま置いた (y,x) を通る五が並んだか。**盤の規則そのもの**。盤の大きさは問わない。"""
    n = board.shape[0]
    for dy, dx in DIRS:
        c = 1
        for s in (1, -1):
            yy, xx = y + dy * s, x + dx * s
            while 0 <= yy < n and 0 <= xx < n and board[yy, xx] == who:
                c += 1
                yy += dy * s
                xx += dx * s
        if c >= WIN:
            return True
    return False


def encode(board: np.ndarray, player: int) -> np.ndarray:
    """網への入力 (3, n, n): 手番の側の石 / 相手の石 / 盤の内側 (全部1)。

    3枚目は盤の縁を知るためのもの。畳み込みは盤の外を 0 で埋めるので、
    「1 の面が途切れる所 = 盤の端」が網から見える。
    """
    return np.stack((board == player, board == -player,
                     np.ones_like(board, dtype=bool))).astype(np.float32)


# ------------------------------------------------------------------ 目 (網)

class Block(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class FlyNet(nn.Module):
    """3x3 の畳み込みの網。手の確からしさ (n*n) と局面の見込み (-1..1) を返す。"""

    def __init__(self, ch: int = 48, blocks: int = 3):
        super().__init__()
        self.ch, self.nblocks = ch, blocks
        self.stem = nn.Conv2d(3, ch, 3, padding=1)
        self.blocks = nn.ModuleList([Block(ch) for _ in range(blocks)])
        self.pol = nn.Conv2d(ch, 1, 1)
        self.val = nn.Conv2d(ch, 8, 1)
        self.val_fc = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = F.relu(self.stem(x))
        for b in self.blocks:
            h = b(h)
        logits = self.pol(h).flatten(1)                        # (B, n*n)
        v = F.relu(self.val(h))
        # 盤全体の平均と最大 — 盤の大きさに依らない形でまとめる
        v = torch.cat((v.mean(dim=(2, 3)), v.amax(dim=(2, 3))), dim=1)
        return logits, torch.tanh(self.val_fc(v)).squeeze(1)

    def config(self) -> dict:
        return {"ch": self.ch, "blocks": self.nblocks}


@torch.inference_mode()
def evaluate(net: FlyNet, boards: list, players: list):
    """局面をまとめて網に通す。(手の logits (B, n*n) numpy, 見込み (B,) numpy)。

    **非正規化数を 0 に落として計算する** (`set_flush_denormal`)。学習した網を
    そのまま通すと、途中の値に極端に小さい数が混ざって CPU の浮動小数点が
    桁違いに遅くなり、16 局まとめて 1 回 136 ms かかっていた (学習前の網は 5 ms、
    落とせば 12 ms)。自己対戦がずっとこの遅さで回っていた。この設定は
    **スレッドごと**なので、アプリの読みのスレッドや手伝いのプロセスでも
    効くように、呼ばれるたびにここで立てる。
    """
    torch.set_flush_denormal(True)
    x = torch.from_numpy(np.stack([encode(b, p) for b, p in zip(boards, players)]))
    logits, v = net(x)
    return logits.numpy(), v.numpy()


# ------------------------------------------------------------------ 読み

def candidates(board: np.ndarray, radius: int = 2) -> np.ndarray:
    """読む候補: 石から radius マス以内の空きマス。盤が空なら中央の 5x5。

    **盤全体 (225 マス) を候補にすると、読みが浅く散って塞ぐ手に届かなかった。**
    読み 64 回を 225 マスに配ると後手の各候補は1回ずつしか読まれず、「塞がないと
    相手が次に五を作る」が見えない。自己対戦は先手が5手で五を並べ切る9手の局に
    落ち込み (先手の勝ち 95%)、物差しでも相手の列を止めなかった。石から2マス
    という範囲は盤の幾何で (どの五も既にある石の近くを通る)、打ち方の知識ではない。
    前の打ち手の読み (`gomoku_env._Reading`) と同じ決め方。
    """
    n = board.shape[0]
    occ = board != 0
    if not occ.any():
        c = n // 2
        near = np.zeros_like(occ)
        near[max(0, c - 2):c + 3, max(0, c - 2):c + 3] = True
        return np.flatnonzero(near.ravel())
    # 縦と横に1回ずつ広げる = (2r+1) x (2r+1) の範囲に石があるか
    rows = occ.copy()
    for k in range(1, radius + 1):
        rows[k:, :] |= occ[:-k, :]
        rows[:-k, :] |= occ[k:, :]
    near = rows.copy()
    for k in range(1, radius + 1):
        near[:, k:] |= rows[:, :-k]
        near[:, :-k] |= rows[:, k:]
    return np.flatnonzero((near & ~occ).ravel())


class Node:
    """読みの木の1局面。W は **この局面で手番の側から見た** 値の合計。"""

    __slots__ = ("board", "player", "moves", "P", "N", "W", "kids",
                 "terminal", "tvalue", "expanded")

    def __init__(self, board: np.ndarray, player: int):
        self.board, self.player = board, player
        self.expanded, self.terminal, self.tvalue = False, False, 0.0
        self.moves = self.P = self.N = self.W = self.kids = None

    def expand(self, logits: np.ndarray) -> None:
        # 規則: 石のある所には打てない。候補は**石から2マス以内の**空きマスだけ
        self.moves = candidates(self.board)
        z = logits[self.moves]
        z = np.exp(z - z.max())
        self.P = (z / z.sum()).astype(np.float64)
        self.N = np.zeros(len(self.moves))
        self.W = np.zeros(len(self.moves))
        self.kids = [None] * len(self.moves)
        self.expanded = True

    def child(self, i: int) -> "Node":
        if self.kids[i] is None:
            n = self.board.shape[0]
            b = self.board.copy()
            y, x = divmod(int(self.moves[i]), n)
            b[y, x] = self.player
            k = Node(b, -self.player)
            if five_at(b, y, x, self.player):
                # 規則: 五が並んだら勝ち。次の手番の側から見ると負け
                k.terminal, k.tvalue = True, -1.0
            elif not (b == 0).any():
                k.terminal, k.tvalue = True, 0.0        # 盤が埋まったら引き分け
            self.kids[i] = k
        return self.kids[i]


class Tree:
    """1局ぶんの読み。PUCT で辿り、葉を網に聞き、値を戻す。"""

    def __init__(self, board: np.ndarray, player: int, c_puct: float = 1.5):
        self.root = Node(board.copy(), player)
        self.c = c_puct
        self._path: list = []

    def select(self):
        """葉まで辿る。網に聞く必要がある葉なら返し、決着している葉ならその場で戻す。"""
        node, path = self.root, []
        while node.expanded:
            tot = node.N.sum()
            q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
            u = q + self.c * node.P * np.sqrt(tot + 1.0) / (1.0 + node.N)
            i = int(np.argmax(u))
            path.append((node, i))
            node = node.child(i)
            if node.terminal:
                self._backup(path, node.tvalue)
                return None
        self._path = path
        return node

    def _backup(self, path, v: float) -> None:
        """v は葉の局面で手番の側から見た値。1段上がるごとに符号が入れ替わる。"""
        for node, i in reversed(path):
            v = -v
            node.N[i] += 1
            node.W[i] += v

    def finish(self, leaf: Node, logits: np.ndarray, value: float) -> None:
        leaf.expand(logits)
        self._backup(self._path, float(value))

    def add_noise(self, rng, alpha: float, frac: float = 0.25) -> None:
        """自己対戦の根にだけ揺らぎを足す (同じ手ばかり読まないように)。"""
        r = self.root
        r.P = (1 - frac) * r.P + frac * rng.dirichlet([alpha] * len(r.P))

    def visits(self) -> np.ndarray:
        """読んだ末の各マスの訪問回数 (n*n)。学習の教師になる。"""
        n = self.root.board.shape[0]
        out = np.zeros(n * n)
        out[self.root.moves] = self.root.N
        return out


def search(trees: list, net: FlyNet, sims: int) -> None:
    """複数の木を並べて読む。各回、全部の木から葉を1つずつ集めて網に1回で通す。"""
    todo = [t for t in trees if not t.root.expanded]
    if todo:
        lg, v = evaluate(net, [t.root.board for t in todo], [t.root.player for t in todo])
        for t, l_ in zip(todo, lg):
            t.root.expand(l_)
    for _ in range(sims):
        leaves = []
        for t in trees:
            leaf = t.select()
            if leaf is not None:
                leaves.append((t, leaf))
        if leaves:
            lg, v = evaluate(net, [lf.board for _, lf in leaves],
                             [lf.player for _, lf in leaves])
            for (t, lf), l_, v_ in zip(leaves, lg, v):
                t.finish(lf, l_, v_)


# ------------------------------------------------------------------ 打ち手

class Player:
    """学習したのと同じ読みで打つ。`gomoku_env.play_game` / `31_gomoku_match` に渡せる形。

    board は自分を +1 として見た盤。読みは毎手、根から読み直す。
    """

    def __init__(self, net: FlyNet, sims: int = 200):
        self.net, self.sims = net, int(sims)
        self.last_tree: Tree | None = None

    def move(self, board: np.ndarray, rng=None) -> int:
        board = np.asarray(board, dtype=np.int8)
        if not (board != 0).any():
            n = board.shape[0]
            return (n // 2) * n + n // 2          # 空盤は読むまでもなく中央
        t = Tree(board, 1)
        search([t], self.net, self.sims)
        self.last_tree = t
        return int(np.argmax(t.visits()))

    def prior(self, board: np.ndarray) -> np.ndarray:
        """目だけで見た手の確からしさ (読む前)。画面に重ねる用。"""
        lg, _ = evaluate(self.net, [np.asarray(board, dtype=np.int8)], [1])
        p = np.exp(lg[0] - lg[0].max())
        p[np.asarray(board).ravel() != 0] = 0.0
        return p / max(p.sum(), 1e-12)


# ------------------------------------------------------------------ 自己対戦

class SelfPlay:
    """G 局を同時に進める自己対戦。終わった局から学習の材料を出す。

    temp_moves 手目までは訪問回数に比例して選び (序盤を散らす)、その先は最多。
    """

    def __init__(self, net: FlyNet, size: int, games: int, sims: int,
                 rng, temp_moves: int = 8, alpha: float | None = None,
                 gamma: float = 0.95):
        self.net, self.n, self.G, self.sims = net, size, games, sims
        self.gamma = gamma
        self.rng, self.temp_moves = rng, temp_moves
        # 揺らぎの濃さは合法手の数に合わせる (AlphaZero の 10/手の数 の目安)
        self.alpha = alpha if alpha is not None else 10.0 / (size * size)
        self.games = [self._new() for _ in range(games)]

    def _new(self) -> dict:
        return {"board": np.zeros((self.n, self.n), np.int8), "player": 1,
                "ply": 0, "hist": []}

    def step(self) -> list:
        """全局を1手ずつ進める。終わった局の材料 [(入力, 訪問の割合, 結果)] を返す。"""
        trees = [Tree(g["board"], g["player"]) for g in self.games]
        # 根は先に開いて揺らぎを足す (search は開いている根をそのまま使う)
        todo = [t for t in trees if not t.root.expanded]
        lg, _ = evaluate(self.net, [t.root.board for t in todo],
                         [t.root.player for t in todo])
        for t, l_ in zip(todo, lg):
            t.root.expand(l_)
            t.add_noise(self.rng, self.alpha)
        search(trees, self.net, self.sims)
        done = []
        for gi, (g, t) in enumerate(zip(self.games, trees)):
            vis = t.visits()
            pi = vis / vis.sum()
            if g["ply"] < self.temp_moves:
                mv = int(self.rng.choice(len(pi), p=pi))
            else:
                mv = int(np.argmax(vis))
            g["hist"].append((encode(g["board"], g["player"]), pi.astype(np.float32),
                              g["player"]))
            y, x = divmod(mv, self.n)
            g["board"][y, x] = g["player"]
            g["ply"] += 1
            result = None
            if five_at(g["board"], y, x, g["player"]):
                result = g["player"]                         # 打った側の勝ち
            elif not (g["board"] == 0).any():
                result = 0
            if result is None:
                g["player"] = -g["player"]
                continue
            # **勝ち負けを決着までの手数で割り引く**: 1手先の決着なら ±1、
            # k 手先なら ±gamma^(k-1)。割り引かずに ±1 を教えると、網は
            # 「先に打った側が勝つ」を石の数の差から当てるようになり
            # (見込みの損失 0.99 → 0.18)、後手は塞いでも塞がなくても見込み -1
            # で、塞ぐ手を選ぶ理由が無くなった (自己対戦が先手の 5 手勝ちの
            # 9手の局ばかりになり、先手の勝ち 96%)。割り引けば「すぐ負ける」
            # より「まだ負けていない」方が値が高いので、塞ぐ手が選べる。
            # 報酬は勝ち負けの一つのまま
            T = len(g["hist"])
            for t, (x_, pi_, who) in enumerate(g["hist"]):
                sign = 0.0 if result == 0 else (1.0 if who == result else -1.0)
                done.append((x_, pi_, sign * self.gamma ** (T - 1 - t)))
            done.append(("end", g["ply"], result))           # 局の終わりの印 (統計用)
            self.games[gi] = self._new()
        return done


# ------------------------------------------------------------------ 学習

def symmetries(x: np.ndarray, pi: np.ndarray, n: int, k: int):
    """盤の8通りの対称 (回転4 x 反転2) の k 番目。五目並べの規則はどれでも同じ。"""
    p = pi.reshape(n, n)
    if k >= 4:
        x, p = x[:, :, ::-1], p[:, ::-1]
    r = k % 4
    return np.rot90(x, r, axes=(1, 2)).copy(), np.rot90(p, r).copy().ravel()


def train_step(net: FlyNet, opt, xs, pis, zs) -> tuple[float, float]:
    """手の確からしさ: 訪問の割合との交差エントロピー / 見込み: 結果との二乗誤差。"""
    torch.set_flush_denormal(True)           # 学習の計算も同じ理由で (`evaluate`)
    net.train()
    x = torch.from_numpy(xs)
    logits, v = net(x)
    lp = F.log_softmax(logits, dim=1)
    loss_p = -(torch.from_numpy(pis) * lp).sum(1).mean()
    loss_v = F.mse_loss(v, torch.from_numpy(zs))
    opt.zero_grad()
    (loss_p + loss_v).backward()
    opt.step()
    net.eval()
    return loss_p.item(), loss_v.item()
