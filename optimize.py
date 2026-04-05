"""Bayesian hyperparameter optimization with Optuna."""
import optuna
import subprocess
import json
import sys
import sqlite3
import time

STAGE = "practice"
LANG = "python"
N_TRIALS = 50


def objective(trial: optuna.Trial) -> float:
    # Context size (most impactful based on experiments)
    max_files = trial.suggest_int("max_files", 0, 3)
    max_tokens = trial.suggest_int("max_tokens", 100, 1500, step=100)
    query_window = trial.suggest_int("query_window", 50, 300, step=50)

    # Graph settings
    max_hop = trial.suggest_int("max_hop", 1, 3)
    graph_weight = trial.suggest_float("graph_weight", 0.0, 0.5, step=0.1)
    bm25_weight = trial.suggest_float("bm25_weight", 0.4, 0.9, step=0.05)
    symbol_weight = round(1.0 - bm25_weight - graph_weight, 2)

    if symbol_weight < 0:
        return 0.0  # invalid combo

    # Trimming (frees up context window for retrieved files)
    trim_prefix = trial.suggest_categorical("trim_prefix", [True, False])
    trim_suffix = trial.suggest_categorical("trim_suffix", [True, False])

    cmd = [
        sys.executable, "simple_repo_graph_rag.py",
        f"data.stage={STAGE}",
        f"data.lang={LANG}",
        f"context.max_files={max_files}",
        f"context.max_tokens={max_tokens}",
        f"context.query_window={query_window}",
        f"retrieval.graph.max_hop={max_hop}",
        f"retrieval.scoring.graph_weight={graph_weight}",
        f"retrieval.scoring.bm25_weight={bm25_weight}",
        f"retrieval.scoring.symbol_weight={symbol_weight}",
        f"trim.prefix={str(trim_prefix).lower()}",
        f"trim.suffix={str(trim_suffix).lower()}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

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
    trim_str = ("P" if trim_prefix else "") + ("S" if trim_suffix else "") or "-"
    print(f"Trial {trial.number}: chrF={score:.4f} | f={max_files} t={max_tokens} qw={query_window} hop={max_hop} trim={trim_str}")
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="simple_hybrid_rag",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 60)
    print("Best trial:")
    print(f"  chrF: {study.best_trial.value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print("=" * 60)
