#!./.venv/bin/python
#%%
from utils import *

#%%

source_path = "results/kimi-k3.json"
model_id = "moonshotai/kimi-k3"
provider = "moonshotai"
followup_prompt = "Why did you choose that number?"
n_transcripts = 30

results, metadata = load_batch_results(source_path)
odd_results = [r for r in results if (a := parse_answer(r)) is not None and a % 2 == 1][:n_transcripts]
print(f"{len(odd_results)} odd transcripts from {source_path}")

#%%

def query_followup(result: dict) -> dict:
    messages = [
        {"role": "user", "content": metadata["prompt"]},
        {"role": "assistant", "content": result["choices"][0]["message"]["content"]},
        {"role": "user", "content": followup_prompt},
    ]
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={
            "model": model_id,
            "messages": messages,
            "provider": {"order": [provider], "allow_fallbacks": False},
            "reasoning": {"enabled": True},
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return {"messages": messages, "original": result, "followup": data}

pool = ThreadPoolExecutor(max_workers=len(odd_results))
futures = [pool.submit(query_followup, r) for r in odd_results]
for _ in tqdm(as_completed(futures), total=len(futures), desc=model_id): pass
followups = [f.result() for f in futures]

#%%

for f in followups:
    msg = f["followup"]["choices"][0]["message"]
    print(f"{bold}{yellow}answer: {f['original']['choices'][0]['message']['content']}{endc}")
    if msg.get("reasoning"):
        print(f"{gray}{msg['reasoning']}{endc}")
    print(f"{lime}{msg['content']}{endc}\n")

os.makedirs("followup_results", exist_ok=True)
save_path = f"followup_results/{os.path.basename(source_path).removesuffix('.json')}_why.json"
with open(save_path, "w") as f:
    json.dump({"metadata": {"source": source_path, "prompt": metadata["prompt"], "followup_prompt": followup_prompt, "provider": provider}, "results": followups}, f, indent=2)
print(f"saved {len(followups)} followups to {save_path}")

#%%
