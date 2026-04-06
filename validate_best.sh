#!/usr/bin/env bash
# Run the best config 20 times with temperature=0.2 for validation
set -e

RUNS=20

for i in $(seq 1 $RUNS); do
    echo "=== Run $i / $RUNS ==="
    python simple_repo_graph_rag.py \
        data.stage=practice \
        data.lang=python \
        context.max_files=14 \
        context.max_tokens=1800 \
        retrieval.graph.max_hop=0 \
        retrieval.scoring.bm25_weight=0.85 \
        retrieval.scoring.graph_weight=0.0 \
        retrieval.scoring.symbol_weight=0.15 \
        evaluation.temperature=0.2
done

echo "Done"
