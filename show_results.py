"""Print a summary table of all experiment results from the database."""
import sqlite3

db = sqlite3.connect("experiments.db")
db.row_factory = sqlite3.Row

rows = db.execute("""
    SELECT stage, max_hop, bm25_weight, graph_weight, symbol_weight,
           hop_decay, reverse_import_weight,
           max_files, max_tokens,
           trim_prefix, trim_suffix,
           chrf_score, num_samples, timestamp
    FROM experiments
    ORDER BY timestamp DESC
""").fetchall()

if not rows:
    print("No experiments found in experiments.db")
    exit()

print(f"{'Stage':<8} {'Hop':>3} {'BM25':>5} {'Grph':>5} {'Sym':>5} {'Decay':>5} {'Rev':>5} {'Files':>5} {'MaxT':>5} {'Trim':>4} {'chrF':>8} {'N':>5}  {'Run at'}")
print("-" * 94)

for r in rows:
    trim_p = "P" if r["trim_prefix"] else ""
    trim_s = "S" if r["trim_suffix"] else ""
    trim = (trim_p + trim_s) or "-"
    chrf = f"{r['chrf_score']:.4f}" if r["chrf_score"] is not None else " pending"
    sym = f"{r['symbol_weight']:.2f}" if r["symbol_weight"] is not None else "?"
    decay = f"{r['hop_decay']:.1f}" if r["hop_decay"] is not None else "?"
    rev = f"{r['reverse_import_weight']:.1f}" if r["reverse_import_weight"] is not None else "?"

    print(f"{r['stage']:<8} "
          f"{r['max_hop']:>3} "
          f"{r['bm25_weight']:>5.2f} "
          f"{r['graph_weight']:>5.2f} "
          f"{sym:>5} "
          f"{decay:>5} "
          f"{rev:>5} "
          f"{r['max_files']:>5} "
          f"{r['max_tokens']:>5} "
          f"{trim:>4} "
          f"{chrf:>8} "
          f"{r['num_samples']:>5}  "
          f"{r['timestamp'][:16]}")

db.close()
