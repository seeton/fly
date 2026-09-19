"""ハエ脳ブラウザ — Male CNS コネクトームを対話的に見るローカルアプリ。

起動:
  .venv\\Scripts\\python.exe -m uvicorn app.server:app --port 8765
  → http://localhost:8765

起動時に注釈(21万)と全結合(1億5185万)をメモリに載せる(20秒ほど)。
シナプス座標(6.8GB)はリクエストごとにディスクから走査する(1回2〜3秒)。
"""

from __future__ import annotations

import functools
import urllib.request
from pathlib import Path

import navis
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "malecns"
SKEL = DATA / "skeletons"
OUT = ROOT / "out"
SWC_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{}.swc"

app = FastAPI(title="Fly Brain Browser")

STATE: dict = {}


@app.on_event("startup")
def load() -> None:
    print("注釈を読み込み中 ...")
    ann = pd.read_feather(DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    print("接続を読み込み中 (1GB) ...")
    w = pd.read_feather(DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    STATE["ann"] = ann
    STATE["w"] = w
    STATE["type_by_body"] = ann.set_index("bodyId")["type"]
    STATE["sc_by_body"] = ann.set_index("bodyId")["superclass"]
    syn = DATA / "syn-partners-male-cns-v1.0-minconf-0.5.feather"
    STATE["syn"] = ds.dataset(syn, format="feather") if syn.exists() else None
    print(f"準備完了: {len(ann):,} ボディ / {len(w):,} 接続 / シナプス={'あり' if STATE['syn'] is not None else 'なし'}")


def graph() -> nx.DiGraph:
    if "graph" not in STATE:
        print("グラフを構築中 (初回のみ) ...")
        w = STATE["w"]
        strong = w[w["weight"] >= 10]
        STATE["graph"] = nx.from_pandas_edgelist(
            strong, "body_pre", "body_post", edge_attr="weight", create_using=nx.DiGraph
        )
        print(f"  {STATE['graph']}")
    return STATE["graph"]


@functools.lru_cache(maxsize=512)
def skeleton_coords(body_id: int, step: int = 1):
    SKEL.mkdir(parents=True, exist_ok=True)
    p = SKEL / f"{body_id}.swc"
    if not p.exists():
        try:
            urllib.request.urlretrieve(SWC_URL.format(body_id), p)
        except Exception:
            return None
    try:
        n = navis.read_swc(p)
    except Exception:
        return None
    pos = n.nodes.set_index("node_id")[["x", "y", "z"]]
    xs, ys, zs = [], [], []
    for seg in n.segments:
        c = pos.loc[seg].to_numpy()[::step]
        xs += [round(float(v), 1) for v in c[:, 0]] + [None]
        ys += [round(float(v), 1) for v in c[:, 1]] + [None]
        zs += [round(float(v), 1) for v in c[:, 2]] + [None]
    return {"x": xs, "y": ys, "z": zs}


@app.get("/api/search")
def search(q: str = "", limit: int = 40):
    ann = STATE["ann"]
    t = ann["type"].fillna("")
    hit = ann[t.str.contains(q, case=False, regex=False)] if q else ann[t != ""]
    g = (
        hit.groupby("type")
        .agg(n=("bodyId", "size"),
             superclass=("superclass", lambda s: s.mode().iat[0] if len(s.mode()) else ""))
        .sort_values("n", ascending=False)
        .head(limit)
        .reset_index()
    )
    return {"results": g.to_dict("records"), "total": int(hit["type"].nunique())}


@app.get("/api/type/{name}")
def type_detail(name: str, top: int = 15):
    ann, w = STATE["ann"], STATE["w"]
    sel = ann[ann["type"] == name]
    if sel.empty:
        raise HTTPException(404, f"{name} が見つかりません")
    ids = set(sel["bodyId"].astype(int))
    ty, sc = STATE["type_by_body"], STATE["sc_by_body"]

    down = w[w["body_pre"].isin(ids)].copy()
    down["partner"] = down["body_post"].map(ty)
    up = w[w["body_post"].isin(ids)].copy()
    up["partner"] = up["body_pre"].map(ty)

    def agg(df, col):
        gg = df.groupby("partner")["weight"].sum().nlargest(top)
        out = []
        for k, v in gg.items():
            b = df.loc[df["partner"] == k, col].iloc[0]
            out.append({"type": str(k), "weight": int(v), "superclass": str(sc.get(b, ""))})
        return out

    row = sel.iloc[0]

    def opt(col):
        return None if pd.isna(row[col]) else str(row[col])

    return {
        "type": name,
        "n_bodies": int(len(sel)),
        "superclass": str(row["superclass"]),
        "bodies": [int(b) for b in sel["bodyId"].head(12)],
        "flywireType": opt("flywireType"),
        "hemibrainType": opt("hemibrainType"),
        "dimorphism": opt("dimorphism"),
        "total_out": int(down["weight"].sum()),
        "total_in": int(up["weight"].sum()),
        "downstream": agg(down, "body_post"),
        "upstream": agg(up, "body_pre"),
    }


@app.get("/api/skeletons")
def skeletons(bodies: str, step: int = 2):
    out = []
    for b in [int(x) for x in bodies.split(",") if x.strip()][:24]:
        c = skeleton_coords(b, step)
        if c:
            out.append({"body": b, "type": str(STATE["type_by_body"].get(b)), **c})
    return {"neurons": out}


@app.get("/api/synapses")
def synapses(type: str, partner: str | None = None, limit: int = 40000):
    if STATE["syn"] is None:
        raise HTTPException(503, "シナプス座標ファイル (6.8GB) がありません")
    ann = STATE["ann"]
    ids = [int(b) for b in ann[ann["type"] == type]["bodyId"]]
    if not ids:
        raise HTTPException(404, f"{type} が見つかりません")
    tbl = STATE["syn"].to_table(
        filter=pc.field("body_pre").isin(ids) | pc.field("body_post").isin(ids)
    )
    df = tbl.to_pandas()
    is_pre = df["body_pre"].isin(ids)

    def pts(sub, cols):
        a = sub[cols].to_numpy()
        if len(a) > limit:
            a = a[np.random.choice(len(a), limit, replace=False)]
        return {"x": a[:, 0].tolist(), "y": a[:, 1].tolist(), "z": a[:, 2].tolist()}

    res = {
        "pre": pts(df[is_pre], ["x_pre", "y_pre", "z_pre"]),
        "post": pts(df[~is_pre], ["x_post", "y_post", "z_post"]),
        "n_pre": int(is_pre.sum()),
        "n_post": int((~is_pre).sum()),
        "rois": {str(k): int(v) for k, v in df["primary_post"].value_counts().head(6).items() if v},
    }
    if partner:
        pids = set(int(b) for b in ann[ann["type"] == partner]["bodyId"])
        m = is_pre & df["body_post"].isin(pids)
        res["contact"] = pts(df[m], ["x_pre", "y_pre", "z_pre"])
        res["n_contact"] = int(m.sum())
        res["contact_rois"] = {
            str(k): int(v) for k, v in df.loc[m, "primary_post"].value_counts().head(3).items() if v
        }
    return res


@app.get("/api/path")
def path(src: str, dst: str):
    ann = STATE["ann"]
    G = graph()
    s = ann[ann["type"] == src]["bodyId"].astype(int).tolist()
    d = ann[ann["type"] == dst]["bodyId"].astype(int).tolist()
    if not s or not d:
        raise HTTPException(404, "細胞型が見つかりません")
    best = None
    for a in s[:4]:
        if a not in G:
            continue
        for b in d[:4]:
            if b not in G:
                continue
            try:
                p = nx.shortest_path(G, a, b)
            except nx.NetworkXNoPath:
                continue
            if best is None or len(p) < len(best):
                best = p
    if best is None:
        raise HTTPException(404, "weight>=10 の接続では経路が見つかりません")
    ty, sc = STATE["type_by_body"], STATE["sc_by_body"]
    return {
        "path": [{"body": int(b), "type": str(ty.get(b)), "superclass": str(sc.get(b))} for b in best],
        "weights": [int(G[a][b]["weight"]) for a, b in zip(best, best[1:])],
    }


@app.get("/api/motor-reach")
def motor_reach(type: str, cutoff: int = 4):
    ann = STATE["ann"]
    G = graph()
    ids = ann[ann["type"] == type]["bodyId"].astype(int).tolist()
    motor = set(ann[ann["superclass"].isin(["vnc_motor", "cb_motor"])]["bodyId"].astype(int))
    src = next((b for b in ids if b in G), None)
    if src is None:
        raise HTTPException(404, "強い接続を持つ細胞がありません")
    dist = nx.single_source_shortest_path_length(G, src, cutoff=cutoff)
    reached = {b: d for b, d in dist.items() if b in motor}
    ty = STATE["type_by_body"]
    by_step: dict[int, int] = {}
    for b, d in reached.items():
        by_step[d] = by_step.get(d, 0) + 1
    nearest = sorted(reached.items(), key=lambda kv: kv[1])[:10]
    return {
        "n_motor_total": len(motor),
        "n_reached": len(reached),
        "by_step": {str(k): v for k, v in sorted(by_step.items())},
        "nearest": [{"body": int(b), "type": str(ty.get(b)), "steps": int(d)} for b, d in nearest],
    }


@app.get("/api/videos")
def videos():
    return {
        "videos": [
            {"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 1)}
            for f in sorted(OUT.glob("*.mp4"))
        ]
    }


@app.get("/video/{name}")
def video(name: str):
    p = OUT / name
    if not p.exists() or p.suffix != ".mp4":
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4")


app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
