// 蝿の読みを裏のスレッドで回す。画面のスレッドは止めない。
"use strict";
importScripts("engine.js" + self.location.search);   // 版の印も引き継ぐ

let net = null;

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    net = await Gomoku.loadNet(m.base || "", m.v || "");
    // この端末で網1回にかかる時間を測る (読みの回数の初期値と、待ち時間の目安に使う)
    const b = new Int8Array(Gomoku.NN);
    b[112] = 1; b[113] = -1; b[97] = 1; b[98] = -1;
    for (let i = 0; i < 3; i++) net.forward(b, 1);
    const t0 = performance.now();
    for (let i = 0; i < 8; i++) net.forward(b, 1);
    self.postMessage({ type: "ready", info: net.info, msPerEval: (performance.now() - t0) / 8 });
  } else if (m.type === "move") {
    const t0 = performance.now();
    const r = Gomoku.search(net, Int8Array.from(m.board), m.sims);
    self.postMessage({
      type: "move", id: m.id, move: r.move, value: r.value,
      visits: Array.from(r.visits), ms: performance.now() - t0,
    });
  } else if (m.type === "prior") {
    const p = Gomoku.prior(net, Int8Array.from(m.board));
    self.postMessage({ type: "prior", id: m.id, prior: Array.from(p) });
  }
};
