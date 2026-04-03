"""Print a summary table of all experiment results."""
import json
import glob
import os
import re
from datetime import datetime

results_dir = "results"

summaries = sorted(glob.glob(os.path.join(results_dir, "*-summary.json")))
if not summaries:
    print("No summary files found in results/")
    exit()

print(f"{'Stage':<10} {'Hops':>4} {'BM25':>6} {'Graph':>6} {'Trim':>8} {'chrF':>8} {'Samples':>7} {'Time(s)':>7}  {'Run at'}")
print("-" * 85)

for path in summaries:
    with open(path) as f:
        data = json.load(f)
    name = os.path.basename(path).replace("-summary.json", "")
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")

    # Parse settings from filename
    stage = re.search(r"python-(\w+)-", name)
    hops = re.search(r"hop(\d+)", name)
    bm25 = re.search(r"bm([\d.]+)", name)
    graph = re.search(r"gr([\d.]+)", name)
    trim_p = "short-prefix" in name
    trim_s = "short-suffix" in name
    trim = ("P" if trim_p else "") + ("S" if trim_s else "") or "-"

    print(f"{stage.group(1) if stage else '?':<10} "
          f"{hops.group(1) if hops else '?':>4} "
          f"{bm25.group(1) if bm25 else '?':>6} "
          f"{graph.group(1) if graph else '?':>6} "
          f"{trim:>8} "
          f"{data['mean_chrf']:>8.4f} "
          f"{data['num_samples']:>7} "
          f"{data.get('elapsed_seconds', 0):>7.1f}"
          f"  {mtime}")
