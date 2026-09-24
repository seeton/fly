// 蝿の五目並べ — 網と読み。scripts/gomoku_net.py と同じ計算を JavaScript で書いたもの。
//
// 網:   3x3 の畳み込み 7 層。手の確からしさ (225) と局面の見込み (-1..1) を出す。
// 読み: その目で候補に重みをつけ、見込みで評価しながら木を読む (PUCT)。
//       候補は石から2マス以内の空きマス。読みに入る規則は
//       「五が並んだら勝ち」「石のある所には打てない」「盤が埋まったら引き分け」だけ。
//
// 答え合わせ: index.html を ?test で開くと、scripts/33_export_gomoku_web.py が
// torch で出した値 (golden.json) と一致するかを確かめる。
"use strict";

// 外に出すのは Gomoku だけ (画面側のスクリプトと名前がぶつからないように包む)
(function () {

const N = 15;
const NN = N * N;
const WIN = 5;
const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]];

// ------------------------------------------------------------------ 盤の規則

function fiveAt(board, y, x, who) {
  for (const [dy, dx] of DIRS) {
    let c = 1;
    for (const s of [1, -1]) {
      let yy = y + dy * s, xx = x + dx * s;
      while (yy >= 0 && yy < N && xx >= 0 && xx < N && board[yy * N + xx] === who) {
        c++; yy += dy * s; xx += dx * s;
      }
    }
    if (c >= WIN) return true;
  }
  return false;
}

function winner(board) {
  for (let i = 0; i < NN; i++) {
    const v = board[i];
    if (v !== 0 && fiveAt(board, (i / N) | 0, i % N, v)) return v;
  }
  return 0;
}

function winningLine(board, who) {
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    for (const [dy, dx] of DIRS) {
      const ye = y + dy * (WIN - 1), xe = x + dx * (WIN - 1);
      if (ye < 0 || ye >= N || xe < 0 || xe >= N) continue;
      let ok = true;
      for (let k = 0; k < WIN; k++) if (board[(y + dy * k) * N + x + dx * k] !== who) { ok = false; break; }
      if (ok) return Array.from({ length: WIN }, (_, k) => (y + dy * k) * N + x + dx * k);
    }
  return [];
}

// 読む候補: 石から2マス以内の空きマス (小さい順)。盤が空なら中央の 5x5。
function candidates(board) {
  const out = [];
  let any = false;
  for (let i = 0; i < NN; i++) if (board[i] !== 0) { any = true; break; }
  if (!any) {
    const c = N >> 1;
    for (let y = c - 2; y <= c + 2; y++) for (let x = c - 2; x <= c + 2; x++) out.push(y * N + x);
    return out;
  }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    if (board[y * N + x] !== 0) continue;
    let near = false;
    for (let yy = Math.max(0, y - 2); yy <= Math.min(N - 1, y + 2) && !near; yy++)
      for (let xx = Math.max(0, x - 2); xx <= Math.min(N - 1, x + 2); xx++)
        if (board[yy * N + xx] !== 0) { near = true; break; }
    if (near) out.push(y * N + x);
  }
  return out;
}

// ------------------------------------------------------------------ 網

// 盤の周りに1マスの縁 (0) を足した 17x17 の並びで持つ。
//
// 3x3 の畳み込みは「ずらした面を重みを掛けて足す」を 9 回やるだけだが、15x15 の
// まま書くと端の処理のために一番内側のループが 15 マスずつに切れ、ループの出入りの
// 重さが勝っていた (網1回 32 ms、読み 200 回で 6〜9 秒)。縁を足しておけば、どの
// ずらしも「内側の範囲をまるごと1本のループで足す」になる。行の両端 (縁の列) にも
// ゴミが入るので、畳み込みのあとで 0 に戻す。
const P = N + 2, PP = P * P;
const Q0 = P + 1, Q1 = N * P + N + 1;             // 内側を含む連続した範囲 [Q0, Q1)
const OFF = [];
for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) OFF.push((ky - 1) * P + (kx - 1));
const INNER = new Int32Array(NN);                  // 盤のマス → 縁つきの並びでの位置
for (let i = 0; i < NN; i++) INNER[i] = (((i / N) | 0) + 1) * P + (i % N) + 1;

class Net {
  constructor(weights, info) {
    this.info = info;
    this.ch = info.config.ch;
    this.nblocks = info.config.blocks;
    const W = {};
    for (const t of info.tensors) W[t.name] = weights.subarray(t.offset, t.offset + t.size);
    this.W = W;
    const C = this.ch;
    this.bufA = new Float32Array(C * PP);
    this.bufB = new Float32Array(C * PP);
    this.bufC = new Float32Array(C * PP);
    this.inp = new Float32Array(3 * PP);
  }

  // 3x3 の畳み込み。inp/out は縁つき (チャンネルごとに PP)。out の縁は 0 に保つ
  conv3(inp, cin, w, b, cout, out, relu) {
    for (let oc = 0; oc < cout; oc++) {
      const o = oc * PP, bias = b[oc];
      for (let q = Q0; q < Q1; q++) out[o + q] = bias;
      for (let ic = 0; ic < cin; ic++) {
        const ib = ic * PP, wb = (oc * cin + ic) * 9;
        for (let t = 0; t < 9; t++) {
          const wv = w[wb + t];
          if (wv === 0) continue;
          const d = ib + OFF[t] - o;
          for (let q = o + Q0, e = o + Q1; q < e; q++) out[q] += wv * inp[q + d];
        }
      }
      if (relu) for (let q = o + Q0, e = o + Q1; q < e; q++) if (out[q] < 0) out[q] = 0;
      // 行の両端 (縁の列) に入ったゴミを 0 に戻す
      for (let y = 1; y <= N; y++) { out[o + y * P] = 0; out[o + y * P + P - 1] = 0; }
    }
  }

  // board: 盤 (+1 / -1 / 0)、player: 手番の側。{logits: Float32Array(225), value}
  forward(board, player) {
    const W = this.W, C = this.ch, inp = this.inp;
    for (let i = 0; i < NN; i++) {
      const q = INNER[i];
      inp[q] = board[i] === player ? 1 : 0;
      inp[PP + q] = board[i] === -player ? 1 : 0;
      inp[2 * PP + q] = 1;                     // 盤の内側 (縁を知るため)。縁は 0 のまま
    }
    let h = this.bufA, t = this.bufB, u = this.bufC;
    this.conv3(inp, 3, W["stem.weight"], W["stem.bias"], C, h, true);
    for (let k = 0; k < this.nblocks; k++) {
      this.conv3(h, C, W[`blocks.${k}.c1.weight`], W[`blocks.${k}.c1.bias`], C, t, true);
      this.conv3(t, C, W[`blocks.${k}.c2.weight`], W[`blocks.${k}.c2.bias`], C, u, false);
      for (let i = 0, e = C * PP; i < e; i++) { const v = h[i] + u[i]; u[i] = v > 0 ? v : 0; }
      const tmp = h; h = u; u = tmp;
    }
    // 手: 1x1 の畳み込み
    const pw = W["pol.weight"], pb = W["pol.bias"][0];
    const logits = new Float32Array(NN);
    for (let i = 0; i < NN; i++) {
      const q = INNER[i];
      let s = pb;
      for (let c = 0; c < C; c++) s += pw[c] * h[c * PP + q];
      logits[i] = s;
    }
    // 見込み: 1x1 で 8 面 → relu → 盤全体の平均と最大 → 全結合 2 段 → tanh
    const vw = W["val.weight"], vb = W["val.bias"];
    const feat = new Float32Array(16);
    for (let k = 0; k < 8; k++) {
      let sum = 0, mx = -Infinity;
      for (let i = 0; i < NN; i++) {
        const q = INNER[i];
        let s = vb[k];
        for (let c = 0; c < C; c++) s += vw[k * C + c] * h[c * PP + q];
        if (s < 0) s = 0;
        sum += s; if (s > mx) mx = s;
      }
      feat[k] = sum / NN; feat[8 + k] = mx;
    }
    const w1 = W["val_fc.0.weight"], b1 = W["val_fc.0.bias"];
    const w2 = W["val_fc.2.weight"], b2 = W["val_fc.2.bias"];
    const hid = b1.length;
    let v = b2[0];
    for (let j = 0; j < hid; j++) {
      let s = b1[j];
      for (let i = 0; i < 16; i++) s += w1[j * 16 + i] * feat[i];
      if (s > 0) v += w2[j] * s;
    }
    return { logits, value: Math.tanh(v) };
  }
}

async function loadNet(base, v = "") {
  const info = await (await fetch(base + "net.json" + v)).json();
  const buf = await (await fetch(base + "net.bin" + v)).arrayBuffer();
  return new Net(new Float32Array(buf), info);
}

// ------------------------------------------------------------------ 読み

class Node {
  constructor(board, player) {
    this.board = board; this.player = player;
    this.expanded = false; this.terminal = false; this.tvalue = 0;
  }
  expand(logits) {
    this.moves = candidates(this.board);
    const m = this.moves.length;
    let mx = -Infinity;
    for (const c of this.moves) if (logits[c] > mx) mx = logits[c];
    this.P = new Float64Array(m);
    let s = 0;
    for (let i = 0; i < m; i++) { this.P[i] = Math.exp(logits[this.moves[i]] - mx); s += this.P[i]; }
    for (let i = 0; i < m; i++) this.P[i] /= s;
    this.N = new Float64Array(m); this.W = new Float64Array(m);
    this.kids = new Array(m).fill(null);
    this.expanded = true;
  }
  child(i) {
    if (this.kids[i] === null) {
      const b = this.board.slice();
      const c = this.moves[i];
      b[c] = this.player;
      const k = new Node(b, -this.player);
      if (fiveAt(b, (c / N) | 0, c % N, this.player)) { k.terminal = true; k.tvalue = -1; }
      else if (!b.includes(0)) { k.terminal = true; k.tvalue = 0; }
      this.kids[i] = k;
    }
    return this.kids[i];
  }
}

// board は「読む側を +1 として見た盤」(Int8Array 225)。sims 回読んで結果を返す。
function search(net, board, sims, c = 1.5) {
  const root = new Node(Int8Array.from(board), 1);
  let any = false;
  for (let i = 0; i < NN; i++) if (board[i] !== 0) { any = true; break; }
  if (!any) {                                    // 空盤は読むまでもなく中央
    return { move: (N >> 1) * N + (N >> 1), visits: new Float64Array(NN), value: 0, root: null };
  }
  const r = net.forward(root.board, 1);
  root.expand(r.logits);
  for (let s = 0; s < sims; s++) {
    let node = root;
    const path = [];
    let leafValue = null;
    while (node.expanded) {
      let tot = 0;
      for (let i = 0; i < node.N.length; i++) tot += node.N[i];
      const sq = Math.sqrt(tot + 1);
      let best = -Infinity, bi = 0;
      for (let i = 0; i < node.N.length; i++) {
        const q = node.N[i] > 0 ? node.W[i] / node.N[i] : 0;
        const u = q + c * node.P[i] * sq / (1 + node.N[i]);
        if (u > best) { best = u; bi = i; }   // 同点は先に出てきた方 (numpy の argmax と同じ)
      }
      path.push([node, bi]);
      node = node.child(bi);
      if (node.terminal) { leafValue = node.tvalue; break; }
    }
    if (leafValue === null) {
      const o = net.forward(node.board, node.player);
      node.expand(o.logits);
      leafValue = o.value;
    }
    // 値は葉で手番の側から見たもの。1段上がるごとに符号が入れ替わる
    let v = leafValue;
    for (let k = path.length - 1; k >= 0; k--) {
      const [nd, i] = path[k];
      v = -v; nd.N[i] += 1; nd.W[i] += v;
    }
  }
  const visits = new Float64Array(NN);
  let bestV = -1, move = root.moves[0], tot = 0, wsum = 0;
  for (let i = 0; i < root.moves.length; i++) {
    visits[root.moves[i]] = root.N[i];
    tot += root.N[i]; wsum += root.W[i];
  }
  for (let c2 = 0; c2 < NN; c2++) if (visits[c2] > bestV) { bestV = visits[c2]; move = c2; }
  return { move, visits, value: tot > 0 ? wsum / tot : r.value, prior: r.logits };
}

// 目だけで見た手の確からしさ (読む前)。石のあるマスは 0。
function prior(net, board) {
  const { logits } = net.forward(board, 1);
  let mx = -Infinity;
  for (let i = 0; i < NN; i++) if (board[i] === 0 && logits[i] > mx) mx = logits[i];
  const p = new Float64Array(NN);
  let s = 0;
  for (let i = 0; i < NN; i++) if (board[i] === 0) { p[i] = Math.exp(logits[i] - mx); s += p[i]; }
  for (let i = 0; i < NN; i++) p[i] /= s || 1;
  return p;
}

const Gomoku = { N, NN, WIN, fiveAt, winner, winningLine, candidates, loadNet, search, prior };
self.Gomoku = Gomoku;

})();
