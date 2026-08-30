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
from tabulate import tabulate
import IPython

from huggingface_hub import hf_hub_download
from safetensors import safe_open

import numpy as np

import torch as t
from torch import Tensor

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

def load_jlens(path: str, device: str = "cpu") -> dict:
    """Download a single lens .pt file from the workspace-lenses repo (cached) and load it."""
    local_path = hf_hub_download(repo_id="camilablank/workspace-lenses", filename=path)
    return t.load(local_path, map_location=device, weights_only=False)

def load_tlens(path: str, device: str = "cpu") -> dict:
    """Download a template lens safetensors file + its row->text map from the workspace-lenses repo (cached) and load it."""
    local_path = hf_hub_download(repo_id="camilablank/workspace-lenses", filename=path)
    words_path = hf_hub_download(repo_id="camilablank/workspace-lenses", filename=path.replace("templates", "template_words").replace(".safetensors", ".txt"))
    with safe_open(local_path, framework="pt", device=device) as f:
        tlens = {"meta": f.metadata(), "templates": f.get_tensor("templates"), "word_ids": f.get_tensor("word_ids")}
    tlens["words"] = [line.split("\t", 1)[1] for line in open(words_path).read().splitlines()]
    return tlens

def to_str_toks(inp: str|Tensor, tokenizer) -> list[str]:
    return [tokenizer.decode(tok) for tok in (tokenizer.encode(inp)[0] if isinstance(inp, str) else inp.squeeze())]

def print_titled_table(table_str: str, title: str | None = None):
    if title is not None:
        lines = table_str.splitlines()
        inner = len(lines[0]) - 2
        print(f"╭{'─' * inner}╮")
        print(f"│{bold}{title.center(inner)}{endc}│")
        print(f"├{'─' * inner}┤")
        print("\n".join(lines[1:]))
    else:
        print(table_str)

def top_toks_table(logits: Tensor, tokenizer, k: int = 10, show_negative: bool = False, show_probs: bool = True, title: str | None = None, return_top=False):
    logits = logits.flatten()
    probs = logits.softmax(-1)
    top = logits.topk(k)
    top_strs = [tokenizer.decode([tok]) for tok in top.indices.tolist()]
    top_vals = top.values.tolist()
    headers = ["Tok", "Value"] + (["Prob"] if show_probs else [])
    cols = [[repr(s) for s in top_strs], top_vals] + ([probs[top.indices].tolist()] if show_probs else [])
    if show_negative:
        bot = logits.topk(k, largest=False)
        bot_strs = [tokenizer.decode([tok]) for tok in bot.indices.tolist()]
        bot_vals = bot.values.tolist()
        headers = [f"Top {h}" for h in headers] + [f"Bot {h}" for h in headers]
        cols += [[repr(s) for s in bot_strs], bot_vals] + ([probs[bot.indices].tolist()] if show_probs else [])
    data = [(i, *(col[i] for col in cols)) for i in range(k)]
    print_titled_table(tabulate(data, headers=["Idx"] + headers, tablefmt="rounded_outline"), title)

    if return_top:
        if show_negative:
            return (top_strs, top_vals, bot_strs, bot_vals)
        else:
            return (top_strs, top_vals)

def jlens_transport(acts: Tensor, lens: dict, layer: int) -> Tensor:
    lens_layer = lens["J"][layer].to(t.bfloat16)
    return lens_layer @ acts

def get_lens_logits(h: Tensor, layer: int, model: TransformerBridge, lens: dict) -> Tensor:
    return model.unembed(model.ln_final(jlens_transport(h, lens, layer)))

def get_tlens_scores(h: Tensor, layer: int, tlens: dict) -> Tensor:
    return t.cosine_similarity(tlens["templates"][layer].to(h.device), h, dim=-1)

def top_templates_table(scores: Tensor, words: list[str], k: int = 10, show_negative: bool = False, title: str | None = None, return_top=False):
    scores = scores.flatten().float()
    top = scores.topk(k)
    top_strs = [repr(words[i]) for i in top.indices.tolist()]
    top_vals = top.values.tolist()
    headers = ["Template", "Cos"]
    cols = [top_strs, top_vals]
    if show_negative:
        bot = scores.topk(k, largest=False)
        bot_strs = [repr(words[i]) for i in bot.indices.tolist()]
        bot_vals = bot.values.tolist()
        headers = [f"Top {h}" for h in headers] + [f"Bot {h}" for h in headers]
        cols += [bot_strs, bot_vals]
    data = [(i, *(col[i] for col in cols)) for i in range(k)]
    print_titled_table(tabulate(data, headers=["Idx"] + headers, tablefmt="rounded_outline"), title)

    if return_top:
        if show_negative:
            return (top_strs, top_vals, bot_strs, bot_vals)
        else:
            return (top_strs, top_vals)