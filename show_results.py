"""Print a summary table of all experiment results from the database."""
import sqlite3

db = sqlite3.connect("experiments.db")
db.row_factory = sqlite3.Row

rows = db.execute("""
    SELECT stage, max_hop, bm25_weight, graph_weight, max_files,
           trim_prefix, trim_suffix, chrf_score, num_samples, timestamp
    FROM experiments
    ORDER BY timestamp
""").fetchall()

if not rows:
    print("No experiments found in experiments.db")
    exit()

print(f"{'Stage':<10} {'Hops':>4} {'BM25':>6} {'Graph':>6} {'Files':>5} {'Trim':>6} {'chrF':>8} {'N':>5}  {'Run at'}")
print("-" * 75)

for r in rows:
    trim_p = "P" if r["trim_prefix"] else ""
    trim_s = "S" if r["trim_suffix"] else ""
    trim = (trim_p + trim_s) or "-"
    chrf = f"{r['chrf_score']:.4f}" if r["chrf_score"] is not None else "pending"

    print(f"{r['stage']:<10} "
          f"{r['max_hop']:>4} "
          f"{r['bm25_weight']:>6.2f} "
          f"{r['graph_weight']:>6.2f} "
          f"{r['max_files']:>5} "
          f"{trim:>6} "
          f"{chrf:>8} "
          f"{r['num_samples']:>5}  "
          f"{r['timestamp'][:16]}")

db.close()
