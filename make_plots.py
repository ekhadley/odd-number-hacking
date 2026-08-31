#!./.venv/bin/python
import json
import os
import plotly.express as px
from plotly_utils import bar
from utils import hack_rate

os.makedirs("figures", exist_ok=True)

baselines = {  # model: baseline 256-run
    "claude-opus-5": "results/anthropic_claude-opus-5_20260830_084101.json",
    "claude-fable-5": "results/anthropic_claude-fable-5_20260830_084131.json",
    "claude-opus-4.5": "results/anthropic_claude-opus-4.5_20260830_085938.json",
    "gpt-5.6-sol": "results/openai_gpt-5.6-sol_20260830_084143.json",
    "gpt-5.6-luna": "results/openai_gpt-5.6-luna_20260830_085949.json",
    "o3": "results/openai_o3_20260830_094021.json",
    "gemini-3.1-pro-preview": "results/google_gemini-3.1-pro-preview_20260830_084210.json",
    "gemini-3.7-flash": "results/google_gemini-3.7-flash_20260830_090226.json",
    "deepseek-v4-pro-0813": "results/deepseek_deepseek-v4-pro-0813_20260830_084234.json",
    "deepseek-v4-flash": "results/deepseek_deepseek-v4-flash_20260830_090142.json",
    "qwen3.5-122b-a10b": "results/qwen_qwen3.5-122b-a10b_20260830_093742.json",
    "qwen3.8-27b": "results/qwen_qwen3.8-27b_20260830_130556.json",
    "kimi-k3": "results/kimi-k3.json",
}
frontier = ["claude-opus-5", "claude-fable-5", "gpt-5.6-sol", "gemini-3.1-pro-preview", "deepseek-v4-pro-0813", "kimi-k3"]
rates = {m: hack_rate(json.load(open(f))["results"])["rate"] for m, f in baselines.items()}
top = sorted(((m, r) for m, r in rates.items() if r > 0), key=lambda x: -x[1])
top += [(m, rates[m]) for m in frontier if rates[m] == 0]
fig = bar(
    [r for _, r in top], x=[m for m, _ in top], color=[m for m, _ in top],
    labels={"x": "model", "y": "fraction of odd answers"},
    title="Eval gaming rate on baseline prompt (n=256 per model)",
    text_auto=".2f", template="simple_white", size=(500, 800), showlegend=False, return_fig=True,
)
fig.write_image("figures/hack_rate_baseline.png", scale=2)
print("wrote figures/hack_rate_baseline.png")

grades = list(json.load(open("analysis/moonshotai_kimi-k3_20260830_085913_trait_grades.json"))["traces"].values())
traits = list(grades[0]["grades"])
odd_traces = [t for t in grades if t["odd"]]
even_pool = [t for t in grades if not t["odd"]]
matched_even = []
for t in odd_traces:
    nearest = min(even_pool, key=lambda e: abs(e["len"] - t["len"]))
    even_pool.remove(nearest)
    matched_even.append(nearest)
score = lambda t, tr: t["grades"][tr]
density = lambda t, tr: t["grades"][tr] / t["len"] * 1e4
for suffix, traces, val, ylabel, fname in [
    ("", grades, score, "mean score (0-3)", "trait_scores.png"),
    (", length-matched", odd_traces + matched_even, score, "mean score (0-3)", "trait_scores_length_matched.png"),
    (", per 10k chars", grades, density, "mean score per 10k chars of reasoning", "trait_scores_per_10k_chars.png"),
]:
    means = [[sum(val(t, tr) for t in traces if t["odd"] == odd) / sum(1 for t in traces if t["odd"] == odd) for tr in traits] for odd in (True, False)]
    fig = bar(
        means, x=traits, names=["odd", "even"],
        labels={"x": "trait", "value": ylabel, "variable": "answer"},
        title=f"Rubric-judged trait scores in kimi-k3 baseline reasoning traces{suffix}",
        barmode="group", template="simple_white", size=(500, 900), xaxis_tickangle=30, return_fig=True,
    )
    fig.write_image(f"figures/{fname}", scale=2)
    print(f"wrote figures/{fname}")

freqs = json.load(open("analysis/phrase_frequencies.json"))
phrases = list(freqs)

for metric, ylabel, fname in [
    ("raw_rate", "fraction of traces containing phrase", "phrase_raw_rate.png"),
    ("per_10k_chars", "hit traces per 10k chars of reasoning", "phrase_per_10k_chars.png"),
]:
    series = [[freqs[p][c][metric] for p in phrases] for c in ("odd", "even")]
    fig = bar(
        series, x=phrases, names=["odd", "even"],
        labels={"x": "phrase", "value": ylabel, "variable": "answer"},
        title=f"Phrase occurrence in kimi-k3 reasoning traces ({metric})",
        barmode="group", template="simple_white", size=(500, 800), return_fig=True,
    )
    fig.write_image(f"figures/{fname}", scale=2)
    print(f"wrote figures/{fname}")

version_tags = ["base"] + [f"v{i}" for i in range(1, 12)]
version_runs = []
for tag in version_tags:
    data = json.load(open(f"results/kimi-k3{'' if tag == 'base' else '_' + tag}.json"))
    label = f"{tag}: {data['metadata']['prompt'].splitlines()[0].strip()}"
    version_runs.append((label, hack_rate(data["results"])["rate"]))
version_runs.sort(key=lambda x: -x[1])
fig = bar(
    [r for _, r in version_runs], x=[l for l, _ in version_runs],
    labels={"x": "prompt", "y": "fraction of odd answers"},
    title="Eval gaming rate across prompt paraphrases (kimi-k3, n=256 per prompt)",
    text_auto=".2f", template="simple_white", size=(650, 1100), xaxis_tickangle=30, showlegend=False, return_fig=True,
)
fig.data[0].marker.color = px.colors.qualitative.Dark24[:len(version_runs)]
fig.write_image("figures/hack_rate_paraphrases.png", scale=2)
print("wrote figures/hack_rate_paraphrases.png")

interventions = {  # label: results file tag
    "base": "",
    "medium_hint": "medium_hint",
    "big_hint": "big_hint",
    "random_name": "random_name",
    "explicit": "explicit",
    "user_provided": "user_provided",
    "small_rewards": "small_rewards",
    "big_rewards": "big_rewards",
    "tagged_input": "tagged_input",
    "system_metadata": "system_metadata_tagged_input",
    "system_instructions_first": "metadata_first_system_instructions_tagged_input",
}
intervention_runs = []
for label, tag in interventions.items():
    data = json.load(open(f"results/kimi-k3{'_' + tag if tag else ''}.json"))
    intervention_runs.append((label, hack_rate(data["results"])["rate"]))
intervention_runs.sort(key=lambda x: -x[1])
fig = bar(
    [r for _, r in intervention_runs], x=[l for l, _ in intervention_runs],
    labels={"x": "intervention", "y": "fraction of odd answers"},
    title="Eval gaming rate across prompt interventions (kimi-k3, n=256 per prompt)",
    text_auto=".2f", template="simple_white", size=(550, 900), xaxis_tickangle=30, showlegend=False, return_fig=True,
)
fig.data[0].marker.color = px.colors.qualitative.Dark24[:len(intervention_runs)]
fig.data[0].textposition = "outside"
fig.write_image("figures/hack_rate_interventions.png", scale=2)
print("wrote figures/hack_rate_interventions.png")
