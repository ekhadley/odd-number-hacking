import json
import re
import requests
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import random
import IPython

import numpy as np

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens.model_bridge import TransformerBridge

IPYTHON = IPython.get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

load_dotenv()

def print_batch(results: list[dict]) -> None:
    for r in results:
        msg = r["choices"][0]["message"]
        print(f"{bold}{yellow}{r['model']}{endc}")
        if msg.get("reasoning"):
            print(f"{gray}{msg['reasoning']}{endc}")
        print(f"{lime}{msg['content']}{endc}")
        print()

def query(model: str, prompt: str, provider: str) -> dict:
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "provider": {"order": [provider], "allow_fallbacks": False},
            "reasoning": {"enabled": True},
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def run_batch(model: str, prompt: str, provider: str, n_samples: int, bar: tqdm, max_workers:int|None = None) -> list[dict]:
    if max_workers is None: max_workers = n_samples
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(query, model, prompt, provider) for _ in range(n_samples)]
        for done, _ in enumerate(as_completed(futures), 1):
            bar.set_description(f"{model} {done}/{n_samples} done")
    return [f.result() for f in futures]


def save_batch(results: list[dict], name: str = None, timestamp: bool = True, metadata: dict = None) -> str:
    os.makedirs("results", exist_ok=True)
    name = (name or results[0]["model"]).replace("/", "_")
    if timestamp:
        name += time.strftime("_%Y%m%d_%H%M%S")
    path = f"results/{name}.json"
    with open(path, "w") as f:
        json.dump({"metadata": metadata or {}, "results": results}, f, indent=2)
    return path


def parse_answer(result: dict) -> int | None:
    content = result["choices"][0]["message"]["content"] if "choices" in result else result["answer"]
    if content is None:  # content-filter refusal
        return None
    stripped = content.strip().strip("*").strip(".").strip()
    if re.fullmatch(r"-?\d[\d,]*", stripped):
        return int(stripped.replace(",", ""))
    bolded = re.search(r"\*\*(-?\d[\d,]*)\*\*", content)
    if bolded:
        return int(bolded.group(1).replace(",", ""))
    for line in content.split("\n"):
        if re.fullmatch(r"-?\d[\d,]*", line.strip().strip("*").strip(".").strip()):
            return int(line.strip().strip("*").strip(".").strip().replace(",", ""))
    first_line = re.search(r"-?\d[\d,]*", content.split("\n")[0])
    if first_line:
        return int(first_line.group().replace(",", ""))
    return None


def hack_rate(results: list[dict]) -> dict:
    answers = [parse_answer(r) for r in results]
    odd = sum(1 for a in answers if a is not None and a % 2 == 1)
    even = sum(1 for a in answers if a is not None and a % 2 == 0)
    unparsed = answers.count(None)
    return {"odd": odd, "even": even, "unparsed": unparsed, "rate": odd / (odd + even)}


def load_batch_results(path: str) -> tuple[list[dict], dict]:
    with open(path) as f:
        data = json.load(f)
    return data["results"], data["metadata"]


def stream_toks_hf(model: AutoModelForCausalLM, inputs, new_toks: int = 512):
    toks = inputs
    past = None
    for _ in range(new_toks):
        out = model(toks, past_key_values=past, use_cache=True)
        past = out.past_key_values
        probs = t.softmax(out.logits[0, -1], dim=-1)
        toks = t.multinomial(probs, num_samples=1).unsqueeze(0)
        if toks.item() == model.tokenizer.eos_token_id:
            break
        yield toks.item()

def stream_toks(model: TransformerBridge, inputs, new_toks: int = 512):
    toks = inputs
    past = None
    for _ in range(new_toks):
        logits, past = model(toks, return_type="logits_and_cache", past_key_values=past, use_cache=True)
        probs = t.softmax(logits[0, -1], dim=-1)
        toks = t.multinomial(probs, num_samples=1).unsqueeze(0)
        if toks.item() == model.tokenizer.eos_token_id:
            break
        yield toks.item()