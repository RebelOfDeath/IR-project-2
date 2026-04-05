"""Bayesian hyperparameter optimization with Optuna."""
import optuna
import subprocess
import json
import sys
import sqlite3
import time

STAGE = "public"
LANG = "python"
N_TRIALS = 300


def objective(trial: optuna.Trial) -> float:
    # Context size
    max_files = trial.suggest_int("max_files", 1, 15)
    max_tokens = trial.suggest_int("max_tokens", 400, 6000, step=200)
    query_window = trial.suggest_int("query_window", 50, 800, step=50)

    # Scoring weights (symbol = remainder, so they sum to 1)
    bm25_weight = trial.suggest_float("bm25_weight", 0.2, 0.9, step=0.05)
    graph_weight = trial.suggest_float("graph_weight", 0.0, 0.6, step=0.05)
    symbol_weight = round(max(1.0 - bm25_weight - graph_weight, 0.0), 2)

    # Graph traversal
    max_hop = trial.suggest_int("max_hop", 0, 5)
    hop_decay = trial.suggest_float("hop_decay", 0.2, 0.9, step=0.1)
    reverse_import_weight = trial.suggest_float("reverse_import_weight", 0.0, 0.6, step=0.1)

    cmd = [
        sys.executable, "simple_repo_graph_rag.py",
        f"data.stage={STAGE}",
        f"data.lang={LANG}",
        f"context.max_files={max_files}",
        f"context.max_tokens={max_tokens}",
        f"context.query_window={query_window}",
        f"retrieval.graph.max_hop={max_hop}",
        f"retrieval.graph.hop_decay={hop_decay}",
        f"retrieval.graph.reverse_import_weight={reverse_import_weight}",
        f"retrieval.scoring.graph_weight={graph_weight}",
        f"retrieval.scoring.bm25_weight={bm25_weight}",
        f"retrieval.scoring.symbol_weight={symbol_weight}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        print(f"Trial {trial.number} failed:\n{result.stderr[-500:]}")
        return 0.0

    # Grab the most recent chrF from the experiments DB
    time.sleep(0.5)
    conn = sqlite3.connect("experiments.db")
    row = conn.execute(
        "SELECT chrf_score FROM experiments ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    conn.close()

    score = row[0] if row and row[0] is not None else 0.0
    print(f"Trial {trial.number}: chrF={score:.4f} | f={max_files} t={max_tokens} qw={query_window} hop={max_hop} decay={hop_decay} rev={reverse_import_weight} bm25={bm25_weight} gr={graph_weight} sym={symbol_weight}")
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="simple_hybrid_rag_public",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    # Seed with known good config (0.59 on practice)
    study.enqueue_trial({
        "max_files": 7, "max_tokens": 1500, "query_window": 150,
        "bm25_weight": 0.6, "graph_weight": 0.4,
        "max_hop": 2, "hop_decay": 0.6, "reverse_import_weight": 0.0,
    })
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 60)
    print("Best trial:")
    print(f"  chrF: {study.best_trial.value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print("=" * 60)
