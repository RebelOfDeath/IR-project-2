"""Print a summary table of all experiment results."""
import json
import glob
import os
import re

results_dir = "results"

summaries = sorted(glob.glob(os.path.join(results_dir, "*-summary.json")))
if not summaries:
    print("No summary files found in results/")
    exit()

print(f"{'Stage':<10} {'Hops':>4} {'BM25':>6} {'Graph':>6} {'chrF':>8} {'Samples':>7} {'Time(s)':>7}")
print("-" * 55)

for path in summaries:
    with open(path) as f:
        data = json.load(f)
    name = os.path.basename(path).replace("-summary.json", "")

    # Parse settings from filename: python-practice-simple_hybrid-hop2-bm0.55-gr0.3
    stage = re.search(r"python-(\w+)-", name)
    hops = re.search(r"hop(\d+)", name)
    bm25 = re.search(r"bm([\d.]+)", name)
    graph = re.search(r"gr([\d.]+)", name)

    print(f"{stage.group(1) if stage else '?':<10} "
          f"{hops.group(1) if hops else '?':>4} "
          f"{bm25.group(1) if bm25 else '?':>6} "
          f"{graph.group(1) if graph else '?':>6} "
          f"{data['mean_chrf']:>8.4f} "
          f"{data['num_samples']:>7} "
          f"{data.get('elapsed_seconds', 0):>7.1f}")
