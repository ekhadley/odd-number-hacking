#!./.venv/bin/python
import os
import json
from flask import Flask, send_file

DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

@app.route("/")
def index():
    return send_file(os.path.join(DIR, "rollouts.html"))

@app.get("/runs")
def runs():
    out = []
    for fname in sorted(os.listdir(os.path.join(DIR, "results"))):
        if fname.endswith(".json"):
            with open(os.path.join(DIR, "results", fname)) as f:
                data = json.load(f)
            out.append({"name": fname[:-5], "model": data["results"][0]["model"], "n": len(data["results"]), "hack_rate": data["metadata"].get("hack_rate")})
    return out

@app.get("/run/<name>")
def run(name):
    return send_file(os.path.join(DIR, "results", f"{name}.json"))

if __name__ == "__main__":
    app.run(port=7862)
