#!./.venv/bin/python
import sys
import glob
from utils import print_batch, load_batch_results, bold, cyan, gray, endc

for path in sys.argv[1:] or sorted(glob.glob("results/*.json"))[-1:]:
    results, metadata = load_batch_results(path)
    print(f"{bold}{cyan}{path}{endc}")
    if metadata:
        print(f"{gray}{metadata}{endc}")
    print_batch(results)
