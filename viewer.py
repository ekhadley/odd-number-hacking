import os
import threading

import torch as t
from flask import Flask, request, send_file
from werkzeug.serving import make_server

app = Flask(__name__)
state = {}

def normed(x):
    return t.nn.functional.normalize(x, dim=-1)

@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html"))

@app.get("/words")
def words():
    return state["tlens"]["words"]

@app.post("/encode")
def encode():
    ids = state["tokenizer"].encode(request.json["text"])
    return {"ids": ids, "toks": [state["tokenizer"].decode([i]) for i in ids]}

@app.post("/run")
def run():
    model, tokenizer, jlens, tlens = state["model"], state["tokenizer"], state["jlens"], state["tlens"]
    text, k = request.json["text"], request.json["k"]
    toks = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_resid_pre"))
    layers = jlens["source_layers"]
    state["resid"], state["transported"] = {}, {}
    resp = {"toks": [tokenizer.decode(tok) for tok in toks.squeeze(0)], "layers": layers, "jlens": {}, "tlens": {}}
    for layer in layers:
        h = cache[f"blocks.{layer}.hook_resid_pre"].squeeze(0)
        transported = model.ln_final(h @ jlens["J"][layer].to(h.device, t.bfloat16).T)
        state["resid"][layer], state["transported"][layer] = h, transported
        logits = model.unembed(transported).float()
        top = logits.topk(k)
        probs = (top.values - logits.logsumexp(-1, keepdim=True)).exp()
        resp["jlens"][layer] = [[[i, tokenizer.decode(i), round(val, 2), round(p, 4)] for i, val, p in zip(*row)] for row in zip(top.indices.tolist(), top.values.tolist(), probs.tolist())]
        sims = normed(h.float()) @ normed(tlens["templates"][layer].to(h.device).float()).T
        ttop = sims.topk(k)
        resp["tlens"][layer] = [[[i, tlens["words"][i], round(val, 4)] for i, val in zip(*row)] for row in zip(ttop.indices.tolist(), ttop.values.tolist())]
        t.cuda.empty_cache()
    return resp

@app.post("/pins")
def pins():
    model, tlens = state["model"], state["tlens"]
    reqpins = request.json["pins"]
    out = [{"val": {}, "prob": {}, "rank": {}} for _ in reqpins]
    for layer in state["jlens"]["source_layers"]:
        h = state["resid"][layer]
        logits = model.unembed(state["transported"][layer]).float()
        lse = logits.logsumexp(-1)
        sims = normed(h.float()) @ normed(tlens["templates"][layer].to(h.device).float()).T if any(p["kind"] == "tmpl" for p in reqpins) else None
        for p, o in zip(reqpins, out):
            if p["kind"] == "tok":
                col = logits[:, p["id"]]
                o["val"][layer] = [round(v, 2) for v in col.tolist()]
                o["prob"][layer] = [round(v, 4) for v in (col - lse).exp().tolist()]
                o["rank"][layer] = (logits > col[:, None]).sum(-1).tolist()
            else:
                col = sims[:, p["id"]]
                o["val"][layer] = [round(v, 4) for v in col.tolist()]
                o["rank"][layer] = (sims > col[:, None]).sum(-1).tolist()
    t.cuda.empty_cache()
    return {"pins": out}

def serve(model, tokenizer, jlens, tlens, port=7860):
    if "server" in state:
        state["server"].shutdown()
    state.update(model=model, tokenizer=tokenizer, jlens=jlens, tlens=tlens)
    state["server"] = make_server("0.0.0.0", port, app)
    threading.Thread(target=state["server"].serve_forever, daemon=True).start()
    print(f"lens viewer running at http://localhost:{port}")

def stop():
    state.pop("server").shutdown()
