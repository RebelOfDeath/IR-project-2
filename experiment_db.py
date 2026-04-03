"""
Experiment tracking database for RepoGraphRAG.

Logs all experiment runs to a SQLite database for later analysis and querying.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


DATABASE_NAME = "experiments.db"


@dataclass
class ExperimentRun:
    """Represents a single experiment run."""
    # Run identification
    run_id: str
    timestamp: str

    # Data settings
    stage: str
    lang: str

    # Retrieval settings
    retrieval_name: str
    max_hop: int
    bm25_weight: float
    graph_weight: float

    # Edge weights
    import_weight: float
    call_weight: float
    inheritance_weight: float
    type_ref_weight: float

    # Context settings
    max_files: int
    max_tokens: int
    min_lines: int

    # Graph pool settings
    fallback_enabled: bool = True
    min_pool_size: int = 10

    # Trim settings
    trim_prefix: bool = False
    trim_suffix: bool = False
    trim_lines: int = 10

    # Output
    prediction_file: str
    num_samples: int

    # Full config as JSON for flexibility
    config_json: str

    # Metrics (to be filled by evaluation script later)
    chrf_score: Optional[float] = None
    bleu_score: Optional[float] = None
    exact_match: Optional[float] = None

    # Status
    status: str = "completed"
    error_message: Optional[str] = None


@dataclass
class RetrievalScore:
    """Represents per-file retrieval scores for a single query."""
    # Identification
    run_id: str
    sample_id: str  # e.g., "sample_0", "sample_1", etc.

    # Query info
    repo: str
    completion_file: str

    # Retrieved file info
    retrieved_file: str
    rank: int  # 0-indexed rank in final results

    # Scores
    bm25_score: float
    graph_score: float
    combined_score: float

    # Context
    included_in_context: bool  # Whether this file was actually included
    bleu_score: Optional[float] = None
    exact_match: Optional[float] = None

    # Status
    status: str = "completed"
    error_message: Optional[str] = None


class ExperimentDB:
    """SQLite database for tracking experiments."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database connection."""
        if db_path is None:
            db_path = DATABASE_NAME
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,

                -- Data settings
                stage TEXT NOT NULL,
                lang TEXT NOT NULL,

                -- Retrieval settings
                retrieval_name TEXT NOT NULL,
                max_hop INTEGER NOT NULL,
                bm25_weight REAL NOT NULL,
                graph_weight REAL NOT NULL,

                -- Edge weights
                import_weight REAL NOT NULL,
                call_weight REAL NOT NULL,
                inheritance_weight REAL NOT NULL,
                type_ref_weight REAL NOT NULL,

                -- Context settings
                max_files INTEGER NOT NULL,
                max_tokens INTEGER NOT NULL,
                min_lines INTEGER NOT NULL,

                -- Graph pool settings
                fallback_enabled BOOLEAN,
                min_pool_size INTEGER,

                -- Trim settings
                trim_prefix BOOLEAN NOT NULL,
                trim_suffix BOOLEAN NOT NULL,
                trim_lines INTEGER NOT NULL,

                -- Output
                prediction_file TEXT NOT NULL,
                num_samples INTEGER NOT NULL,

                -- Full config as JSON
                config_json TEXT NOT NULL,

                -- Metrics (nullable, filled by evaluation)
                chrf_score REAL,
                bleu_score REAL,
                exact_match REAL,

                -- Status
                status TEXT NOT NULL DEFAULT 'completed',
                error_message TEXT,

                -- Indexes for common queries
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create retrieval_scores table for per-file scores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,

                -- Query info
                repo TEXT NOT NULL,
                completion_file TEXT NOT NULL,

                -- Retrieved file info
                retrieved_file TEXT NOT NULL,
                rank INTEGER NOT NULL,

                -- Scores
                bm25_score REAL NOT NULL,
                graph_score REAL NOT NULL,
                combined_score REAL NOT NULL,

                -- Context
                included_in_context BOOLEAN NOT NULL,

                -- Foreign key to experiments table
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_stage_lang
            ON experiments(stage, lang)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_retrieval
            ON experiments(retrieval_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_timestamp
            ON experiments(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_retrieval_scores_run_id
            ON retrieval_scores(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_retrieval_scores_sample
            ON retrieval_scores(run_id, sample_id)
        """)

        # Migrate: add new columns if missing
        cursor.execute("PRAGMA table_info(experiments)")
        existing = {row[1] for row in cursor.fetchall()}
        for col, typ in [("fallback_enabled", "BOOLEAN"), ("min_pool_size", "INTEGER")]:
            if col not in existing:
                cursor.execute(f"ALTER TABLE experiments ADD COLUMN {col} {typ}")

        conn.commit()
        conn.close()

    def log_run(self, run: ExperimentRun) -> int:
        """Log an experiment run to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO experiments (
                    run_id, timestamp, stage, lang,
                    retrieval_name, max_hop, bm25_weight, graph_weight,
                    import_weight, call_weight, inheritance_weight, type_ref_weight,
                    max_files, max_tokens, min_lines,
                    fallback_enabled, min_pool_size,
                    trim_prefix, trim_suffix, trim_lines,
                    prediction_file, num_samples, config_json,
                    chrf_score, bleu_score, exact_match,
                    status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id, run.timestamp, run.stage, run.lang,
                run.retrieval_name, run.max_hop, run.bm25_weight, run.graph_weight,
                run.import_weight, run.call_weight, run.inheritance_weight, run.type_ref_weight,
                run.max_files, run.max_tokens, run.min_lines,
                run.fallback_enabled, run.min_pool_size,
                run.trim_prefix, run.trim_suffix, run.trim_lines,
                run.prediction_file, run.num_samples, run.config_json,
                run.chrf_score, run.bleu_score, run.exact_match,
                run.status, run.error_message
            ))

            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Run already exists, update it
            cursor.execute("""
                UPDATE experiments SET
                    timestamp = ?, stage = ?, lang = ?,
                    retrieval_name = ?, max_hop = ?, bm25_weight = ?, graph_weight = ?,
                    import_weight = ?, call_weight = ?, inheritance_weight = ?, type_ref_weight = ?,
                    max_files = ?, max_tokens = ?, min_lines = ?,
                    trim_prefix = ?, trim_suffix = ?, trim_lines = ?,
                    prediction_file = ?, num_samples = ?, config_json = ?,
                    status = ?, error_message = ?
                WHERE run_id = ?
            """, (
                run.timestamp, run.stage, run.lang,
                run.retrieval_name, run.max_hop, run.bm25_weight, run.graph_weight,
                run.import_weight, run.call_weight, run.inheritance_weight, run.type_ref_weight,
                run.max_files, run.max_tokens, run.min_lines,
                run.trim_prefix, run.trim_suffix, run.trim_lines,
                run.prediction_file, run.num_samples, run.config_json,
                run.status, run.error_message,
                run.run_id
            ))
            conn.commit()
            cursor.execute("SELECT id FROM experiments WHERE run_id = ?", (run.run_id,))
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def update_metrics(self, run_id: str, chrf: float = None, bleu: float = None,
                       exact_match: float = None) -> bool:
        """Update metrics for an existing run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updates = []
        values = []

        if chrf is not None:
            updates.append("chrf_score = ?")
            values.append(chrf)
        if bleu is not None:
            updates.append("bleu_score = ?")
            values.append(bleu)
        if exact_match is not None:
            updates.append("exact_match = ?")
            values.append(exact_match)

        if not updates:
            return False

        values.append(run_id)
        query = f"UPDATE experiments SET {', '.join(updates)} WHERE run_id = ?"

        cursor.execute(query, values)
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def log_retrieval_scores(self, run_id: str, sample_id: str, repo: str,
                            completion_file: str, scores: List[Dict[str, Any]]) -> int:
        """
        Log retrieval scores for a single query.

        Args:
            run_id: Experiment run ID
            sample_id: Sample identifier (e.g., "sample_0")
            repo: Repository name
            completion_file: File being completed
            scores: List of dicts with keys: retrieved_file, rank, bm25_score,
                   graph_score, combined_score, included_in_context

        Returns:
            Number of scores logged
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        count = 0
        for score_data in scores:
            cursor.execute("""
                INSERT INTO retrieval_scores (
                    run_id, sample_id, repo, completion_file,
                    retrieved_file, rank,
                    bm25_score, graph_score, combined_score,
                    included_in_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, sample_id, repo, completion_file,
                score_data['retrieved_file'], score_data['rank'],
                score_data['bm25_score'], score_data['graph_score'],
                score_data['combined_score'], score_data['included_in_context']
            ))
            count += 1

        conn.commit()
        conn.close()

        return count

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific run by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM experiments WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_runs(self, stage: str = None, lang: str = None,
                 retrieval_name: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Query runs with optional filters."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if stage:
            query += " AND stage = ?"
            params.append(stage)
        if lang:
            query += " AND lang = ?"
            params.append(lang)
        if retrieval_name:
            query += " AND retrieval_name = ?"
            params.append(retrieval_name)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_best_runs(self, metric: str = "chrf_score", stage: str = None,
                      lang: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the best runs by a specific metric."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if metric not in ("chrf_score", "bleu_score", "exact_match"):
            raise ValueError(f"Invalid metric: {metric}")

        query = f"SELECT * FROM experiments WHERE {metric} IS NOT NULL"
        params = []

        if stage:
            query += " AND stage = ?"
            params.append(stage)
        if lang:
            query += " AND lang = ?"
            params.append(lang)

        query += f" ORDER BY {metric} DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_runs_without_metrics(self) -> List[Dict[str, Any]]:
        """Get runs that haven't been evaluated yet."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM experiments
            WHERE chrf_score IS NULL AND status = 'completed'
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_retrieval_scores(self, run_id: str = None, sample_id: str = None,
                            limit: int = None) -> List[Dict[str, Any]]:
        """
        Get retrieval scores for analysis.

        Args:
            run_id: Filter by run ID (optional)
            sample_id: Filter by sample ID (optional)
            limit: Limit number of results

        Returns:
            List of retrieval score records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM retrieval_scores WHERE 1=1"
        params = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if sample_id:
            query += " AND sample_id = ?"
            params.append(sample_id)

        query += " ORDER BY run_id, sample_id, rank"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete a run from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))
        conn.commit()
        affected = cursor.rowcount
        conn.close()

        return affected > 0

    def summary(self) -> Dict[str, Any]:
        """Get a summary of all experiments."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM experiments")
        total_runs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM experiments WHERE chrf_score IS NOT NULL")
        evaluated_runs = cursor.fetchone()[0]

        cursor.execute("""
            SELECT retrieval_name, COUNT(*) as count, AVG(chrf_score) as avg_chrf
            FROM experiments
            GROUP BY retrieval_name
        """)
        by_retrieval = cursor.fetchall()

        cursor.execute("""
            SELECT stage, lang, COUNT(*) as count
            FROM experiments
            GROUP BY stage, lang
        """)
        by_data = cursor.fetchall()

        conn.close()

        return {
            "total_runs": total_runs,
            "evaluated_runs": evaluated_runs,
            "by_retrieval": [{"name": r[0], "count": r[1], "avg_chrf": r[2]} for r in by_retrieval],
            "by_data": [{"stage": d[0], "lang": d[1], "count": d[2]} for d in by_data]
        }


def create_run_from_config(cfg, prediction_file: str, num_samples: int,
                           status: str = "completed", error_message: str = None) -> ExperimentRun:
    """Create an ExperimentRun from a Hydra config."""
    from omegaconf import OmegaConf
    import hashlib

    # Generate a unique run ID based on config + timestamp
    timestamp = datetime.now().isoformat()
    config_str = OmegaConf.to_yaml(cfg)
    run_hash = hashlib.md5(f"{config_str}{timestamp}".encode()).hexdigest()[:12]
    run_id = f"{cfg.data.lang}-{cfg.data.stage}-{cfg.retrieval.name}-{run_hash}"

    return ExperimentRun(
        run_id=run_id,
        timestamp=timestamp,
        stage=cfg.data.stage,
        lang=cfg.data.lang,
        retrieval_name=cfg.retrieval.name,
        max_hop=cfg.retrieval.graph.max_hop,
        bm25_weight=cfg.retrieval.scoring.bm25_weight,
        graph_weight=cfg.retrieval.scoring.graph_weight,
        import_weight=cfg.retrieval.graph.edge_weights.imports,
        call_weight=cfg.retrieval.graph.edge_weights.calls,
        inheritance_weight=cfg.retrieval.graph.edge_weights.inheritance,
        type_ref_weight=cfg.retrieval.graph.edge_weights.type_refs,
        max_files=cfg.context.max_files,
        max_tokens=cfg.context.max_tokens,
        min_lines=cfg.context.min_lines,
        trim_prefix=cfg.trim.prefix,
        trim_suffix=cfg.trim.suffix,
        trim_lines=cfg.trim.trim_lines,
        prediction_file=prediction_file,
        num_samples=num_samples,
        config_json=OmegaConf.to_yaml(cfg),
        status=status,
        error_message=error_message
    )


# CLI for database inspection
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment database CLI")
    parser.add_argument("--db", default=DATABASE_NAME, help="Database path")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Summary command
    subparsers.add_parser("summary", help="Show database summary")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--stage", help="Filter by stage")
    list_parser.add_argument("--lang", help="Filter by language")
    list_parser.add_argument("--retrieval", help="Filter by retrieval method")
    list_parser.add_argument("--limit", type=int, default=20, help="Max results")

    # Best command
    best_parser = subparsers.add_parser("best", help="Show best runs")
    best_parser.add_argument("--metric", default="chrf_score", help="Metric to sort by")
    best_parser.add_argument("--limit", type=int, default=10, help="Max results")

    # Pending command
    subparsers.add_parser("pending", help="Show runs without metrics")

    args = parser.parse_args()

    db = ExperimentDB(args.db)

    if args.command == "summary":
        summary = db.summary()
        print(f"\nExperiment Database Summary")
        print(f"{'='*40}")
        print(f"Total runs: {summary['total_runs']}")
        print(f"Evaluated runs: {summary['evaluated_runs']}")
        print(f"\nBy retrieval method:")
        for r in summary['by_retrieval']:
            avg = f"{r['avg_chrf']:.4f}" if r['avg_chrf'] else "N/A"
            print(f"  {r['name']}: {r['count']} runs, avg chrF: {avg}")
        print(f"\nBy data:")
        for d in summary['by_data']:
            print(f"  {d['stage']}/{d['lang']}: {d['count']} runs")

    elif args.command == "list":
        runs = db.get_runs(
            stage=args.stage,
            lang=args.lang,
            retrieval_name=args.retrieval,
            limit=args.limit
        )
        print(f"\n{'Run ID':<45} {'Retrieval':<12} {'Hop':<4} {'chrF':<8} {'File'}")
        print("-" * 100)
        for run in runs:
            chrf = f"{run['chrf_score']:.4f}" if run['chrf_score'] else "N/A"
            print(f"{run['run_id']:<45} {run['retrieval_name']:<12} {run['max_hop']:<4} {chrf:<8} {run['prediction_file']}")

    elif args.command == "best":
        runs = db.get_best_runs(metric=args.metric, limit=args.limit)
        print(f"\nBest runs by {args.metric}:")
        print(f"{'Run ID':<45} {'Score':<10} {'Retrieval':<12} {'Hop'}")
        print("-" * 80)
        for run in runs:
            score = run[args.metric]
            print(f"{run['run_id']:<45} {score:<10.4f} {run['retrieval_name']:<12} {run['max_hop']}")

    elif args.command == "pending":
        runs = db.get_runs_without_metrics()
        print(f"\nRuns pending evaluation: {len(runs)}")
        for run in runs:
            print(f"  {run['run_id']}: {run['prediction_file']}")

    else:
        parser.print_help()
