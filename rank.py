#!./.venv/bin/python
import re
import glob
import time
from utils import load_batch_results, hack_rate, bold, cyan, gray, green, yellow, red, endc

latest = {}  # model -> (stamp, path)
for path in glob.glob("results/*.json"):
    m = re.fullmatch(r"results/(.+)_(\d{8}_\d{6})\.json", path)
    if m is None:
        continue
    model, stamp = m.groups()
    if model not in latest or stamp > latest[model][0]:
        latest[model] = (stamp, path)

rows = []
for model, (stamp, path) in latest.items():
    results, metadata = load_batch_results(path)
    rate = hack_rate(results)
    when = time.strftime("%b %d %H:%M", time.strptime(stamp, "%Y%m%d_%H%M%S"))
    rows.append((rate["rate"], model.replace("_", "/", 1), rate, metadata.get("provider", "?"), when))
rows.sort(reverse=True)

sorted_rates = sorted(r[0] for r in rows)
def rate_color(rate: float) -> str:
    pctl = sorted_rates.index(rate) / max(len(sorted_rates) - 1, 1)
    return red if pctl > 2/3 else yellow if pctl > 1/3 else green

print(f"{bold}{'model':<30} {'hack rate':>9}  {'odd':>4} {'even':>4} {'unparsed':>8}  {'provider':<18} {'when'}{endc}")
for rate, model, counts, provider, when in rows:
    print(f"{cyan}{model:<30}{endc} {rate_color(rate)}{rate:>9.1%}{endc}  {counts['odd']:>4} {counts['even']:>4} {counts['unparsed']:>8}  {provider:<18} {gray}{when}{endc}")
