"""
A simpler hybrid retriever for Python code completion context.

This module implements a lightweight hybrid retrieval approach that combines:
1. BM25 lexical retrieval
2. Import-graph based expansion
3. Symbol matching

Designed to be simpler and faster than the full RepoGraphRAG while still
capturing structural dependencies through import relationships.
"""

from __future__ import annotations

import ast
import keyword
import os
import re
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import jsonlines
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from experiment_db import ExperimentDB, ExperimentRun


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

@dataclass
class SimpleHybridConfig:
    max_hops: int = 2
    max_context_files: int = 6
    max_context_chars: int = 32000

    bm25_weight: float = 0.55
    graph_weight: float = 0.30
    symbol_weight: float = 0.15

    reverse_import_weight: float = 0.35
    hop_decay: float = 0.65

    fallback_enabled: bool = True
    min_pool_size: int = 10

    min_lines: int = 5
    query_window: int = 150

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig) -> "SimpleHybridConfig":
        """Create a SimpleHybridConfig from a Hydra DictConfig."""
        return cls(
            max_hops=cfg.retrieval.graph.max_hop,
            max_context_files=cfg.context.max_files,
            max_context_chars=cfg.context.max_tokens * 4,  # approx 4 chars per token
            bm25_weight=cfg.retrieval.scoring.bm25_weight,
            graph_weight=cfg.retrieval.scoring.graph_weight,
            symbol_weight=cfg.retrieval.scoring.get("symbol_weight", 0.15),
            reverse_import_weight=cfg.retrieval.graph.get("reverse_import_weight", 0.35),
            hop_decay=cfg.retrieval.graph.get("hop_decay", 0.65),
            fallback_enabled=cfg.retrieval.graph.get("fallback_enabled", True),
            min_pool_size=cfg.retrieval.graph.get("min_pool_size", 10),
            min_lines=cfg.context.min_lines,
            query_window=cfg.context.get("query_window", 150),
        )


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

_BUILTIN_TYPE_NAMES = {
    "int", "str", "float", "bool", "list", "dict", "set", "tuple",
    "bytes", "object", "None", "Optional", "Any", "Union"
}

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_relpath(root_dir: str, file_path: str) -> str:
    p = Path(file_path)
    if p.is_absolute():
        p = p.relative_to(root_dir)
    return p.as_posix()


def module_name_from_relpath(rel_path: str) -> str:
    """
    Convert:
        pkg/mod.py -> pkg.mod
        pkg/__init__.py -> pkg
        mod.py -> mod
    """
    p = Path(rel_path)
    no_suffix = p.with_suffix("")
    parts = list(no_suffix.parts)

    if not parts:
        return ""

    if parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


def tokenize(text: str) -> List[str]:
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


def query_identifiers(text: str) -> Set[str]:
    result = set()
    for tok in _IDENTIFIER_RE.findall(text):
        if keyword.iskeyword(tok):
            continue
        if tok in _BUILTIN_TYPE_NAMES:
            continue
        if len(tok) <= 1:
            continue
        result.add(tok)
    return result


def resolve_relative_module(
    current_module: str,
    imported_module: Optional[str],
    level: int,
) -> str:
    """
    Resolve a relative import target against the current module.

    Example:
        current_module = "pkg.sub.mod"
        from .utils import x   -> pkg.sub.utils
        from ..core import x   -> pkg.core
    """
    imported_module = imported_module or ""

    # Current file's package, not the module itself.
    package_parts = current_module.split(".")[:-1] if current_module else []

    if level > 0:
        # One dot means current package. Two dots means parent, etc.
        keep = max(0, len(package_parts) - (level - 1))
        base = package_parts[:keep]
    else:
        base = []

    extra = imported_module.split(".") if imported_module else []
    return ".".join([p for p in [*base, *extra] if p])


# -------------------------------------------------------------------
# AST metadata extraction
# -------------------------------------------------------------------

@dataclass
class ImportRef:
    kind: str                 # "import" | "from"
    module: Optional[str]     # imported module or base module
    level: int                # relative import level for "from"
    imported_name: Optional[str] = None
    alias: Optional[str] = None


@dataclass
class FileInfo:
    rel_path: str
    module_name: str
    content: str
    line_count: int

    imports: List[ImportRef] = field(default_factory=list)
    defined_symbols: Set[str] = field(default_factory=set)
    imported_aliases: Dict[str, str] = field(default_factory=dict)


class FileMetadataExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: List[ImportRef] = []
        self.defined_symbols: Set[str] = set()
        self.imported_aliases: Dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportRef(
                    kind="import",
                    module=alias.name,
                    level=0,
                    imported_name=None,
                    alias=alias.asname or alias.name.split(".")[0],
                )
            )
            self.imported_aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            imported_name = alias.name
            full_alias_name = alias.asname or alias.name
            self.imports.append(
                ImportRef(
                    kind="from",
                    module=node.module,
                    level=node.level,
                    imported_name=imported_name,
                    alias=full_alias_name,
                )
            )

            base = node.module or ""
            # Alias target is approximate at this stage. We finalize later.
            if base:
                self.imported_aliases[full_alias_name] = f"{base}.{imported_name}"
            else:
                self.imported_aliases[full_alias_name] = imported_name
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined_symbols.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.defined_symbols.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined_symbols.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.defined_symbols.add(t.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.defined_symbols.add(node.target.id)
        self.generic_visit(node)


# -------------------------------------------------------------------
# Repository index
# -------------------------------------------------------------------

class SimplePythonRepoIndex:
    def __init__(self, root_dir: str, config: Optional[SimpleHybridConfig] = None):
        self.root_dir = str(Path(root_dir).resolve())
        self.config = config or SimpleHybridConfig()

        self.files: Dict[str, FileInfo] = {}
        self.module_to_file: Dict[str, str] = {}

        self.import_out: Dict[str, Set[str]] = defaultdict(set)
        self.import_in: Dict[str, Set[str]] = defaultdict(set)

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_files: List[str] = []
        self._bm25_corpus: List[List[str]] = []

    def build(self) -> None:
        self._load_files()
        self._build_import_graph()
        self._build_bm25()

    def _load_files(self) -> None:
        for path in Path(self.root_dir).rglob("*.py"):
            rel = normalize_relpath(self.root_dir, str(path))

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            line_count = len(content.splitlines())
            module_name = module_name_from_relpath(rel)

            imports: List[ImportRef] = []
            defined_symbols: Set[str] = set()
            imported_aliases: Dict[str, str] = {}

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(content)
                extractor = FileMetadataExtractor()
                extractor.visit(tree)
                imports = extractor.imports
                defined_symbols = extractor.defined_symbols
                imported_aliases = extractor.imported_aliases
            except SyntaxError:
                # Keep the file for BM25, just skip AST metadata.
                pass

            info = FileInfo(
                rel_path=rel,
                module_name=module_name,
                content=content,
                line_count=line_count,
                imports=imports,
                defined_symbols=defined_symbols,
                imported_aliases=imported_aliases,
            )
            self.files[rel] = info

            if module_name:
                self.module_to_file[module_name] = rel

    def _candidate_modules_for_import(self, current_module: str, imp: ImportRef) -> List[str]:
        candidates: List[str] = []

        if imp.kind == "import":
            if imp.module:
                candidates.append(imp.module)
            return candidates

        # from X import Y
        base = resolve_relative_module(current_module, imp.module, imp.level)

        if base:
            # Could be submodule or symbol from module.
            if imp.imported_name and imp.imported_name != "*":
                candidates.append(f"{base}.{imp.imported_name}")
            candidates.append(base)
        elif imp.imported_name and imp.imported_name != "*":
            candidates.append(imp.imported_name)

        return candidates

    def _resolve_import_to_files(self, current_module: str, imp: ImportRef) -> Set[str]:
        resolved: Set[str] = set()

        for mod in self._candidate_modules_for_import(current_module, imp):
            # Exact module match
            if mod in self.module_to_file:
                resolved.add(self.module_to_file[mod])
                continue

            # Package prefix match, for cases like importing a package symbol via __init__
            for known_module, rel_path in self.module_to_file.items():
                if known_module == mod:
                    resolved.add(rel_path)
                elif known_module.startswith(mod + "."):
                    resolved.add(rel_path)

        return resolved

    def _build_import_graph(self) -> None:
        for rel_path, info in self.files.items():
            for imp in info.imports:
                targets = self._resolve_import_to_files(info.module_name, imp)
                for target_rel in targets:
                    if target_rel == rel_path:
                        continue
                    self.import_out[rel_path].add(target_rel)
                    self.import_in[target_rel].add(rel_path)

    def _build_bm25(self) -> None:
        for rel_path, info in self.files.items():
            if info.line_count < self.config.min_lines:
                continue

            enriched_text = info.content + "\n" + " ".join(sorted(info.defined_symbols))
            self._bm25_corpus.append(tokenize(enriched_text))
            self._bm25_files.append(rel_path)

        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)

    # ---------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------

    def _graph_scores(self, start_rel_path: str) -> Dict[str, float]:
        if start_rel_path not in self.files:
            return {}

        scores: Dict[str, float] = {}
        seen: Dict[Tuple[str, str], int] = {}

        queue: deque[Tuple[str, int, float]] = deque()
        queue.append((start_rel_path, 0, 1.0))

        while queue:
            current, hop, current_score = queue.popleft()
            if hop >= self.config.max_hops:
                continue

            next_hop = hop + 1
            decay = self.config.hop_decay ** hop

            # Outgoing imports are stronger signals.
            for nxt in self.import_out.get(current, set()):
                score = 1.0 * decay
                if nxt != start_rel_path:
                    scores[nxt] = max(scores.get(nxt, 0.0), score)
                state = ("out", nxt)
                if seen.get(state, 10**9) > next_hop:
                    seen[state] = next_hop
                    queue.append((nxt, next_hop, score))

            # Reverse imports are weaker, but still useful.
            for nxt in self.import_in.get(current, set()):
                score = self.config.reverse_import_weight * decay
                if nxt != start_rel_path:
                    scores[nxt] = max(scores.get(nxt, 0.0), score)
                state = ("in", nxt)
                if seen.get(state, 10**9) > next_hop:
                    seen[state] = next_hop
                    queue.append((nxt, next_hop, score))

        return scores

    def _symbol_scores(self, query: str) -> Dict[str, float]:
        qids = query_identifiers(query)
        if not qids:
            return {}

        result: Dict[str, float] = {}
        for rel_path, info in self.files.items():
            candidates = set(info.defined_symbols) | set(info.imported_aliases.keys())
            if not candidates:
                result[rel_path] = 0.0
                continue

            overlap = len(qids & candidates)
            score = overlap / max(1, len(qids))
            result[rel_path] = score

        return result

    def retrieve(
        self,
        query: str,
        completion_file: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Two-stage retrieval:
          Stage 1 - graph filters candidate files (all files reachable within max_hops).
          Stage 2 - BM25 + symbol scores rank the candidates.

        If fallback is enabled and the graph pool is smaller than min_pool_size,
        all files are included as candidates instead.

        Returns:
            [(file_path, final_score, bm25_score, graph_plus_symbol_score), ...]
        """
        if not self._bm25:
            return []

        rel_completion = None
        if completion_file:
            rel_completion = normalize_relpath(self.root_dir, completion_file)

        # Stage 1: graph-based candidate selection 
        graph_scores = self._graph_scores(rel_completion) if rel_completion else {}
        graph_pool: Set[str] = set(graph_scores.keys())

        use_full_fallback = False
        if len(graph_pool) < self.config.min_pool_size:
            if self.config.fallback_enabled:
                use_full_fallback = True
            # If fallback is disabled, we keep the (small) graph pool as-is.

        # Stage 2: BM25 + symbol ranking within the pool
        query_tokens = tokenize(query)
        raw_bm25 = self._bm25.get_scores(query_tokens)
        max_bm25 = max(raw_bm25) if len(raw_bm25) and max(raw_bm25) > 0 else 1.0
        bm25_scores = [s / max_bm25 for s in raw_bm25]

        symbol_scores = self._symbol_scores(query)

        ranked: List[Tuple[str, float, float, float]] = []
        for idx, rel_path in enumerate(self._bm25_files):
            # Filter: only graph-reachable files, unless we fell back.
            if not use_full_fallback and graph_pool and rel_path not in graph_pool:
                continue

            bm25 = bm25_scores[idx]
            graph = graph_scores.get(rel_path, 0.0)
            symbol = symbol_scores.get(rel_path, 0.0)

            final = (
                self.config.bm25_weight * bm25
                + self.config.graph_weight * graph
                + self.config.symbol_weight * symbol
            )

            # Mild boost for immediate import neighbors of the completion file.
            if rel_completion and rel_path in self.import_out.get(rel_completion, set()):
                final += 0.05

            ranked.append((rel_path, final, bm25, graph + symbol))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def assemble_context(
        self,
        ranked_files: List[Tuple[str, float, float, float]],
        completion_file: Optional[str] = None,
    ) -> str:
        rel_completion = None
        if completion_file:
            rel_completion = normalize_relpath(self.root_dir, completion_file)

        parts: List[str] = []
        total_chars = 0
        used = 0

        for rel_path, final_score, _, _ in ranked_files:
            if used >= self.config.max_context_files:
                break

            if rel_completion and rel_path == rel_completion:
                continue

            info = self.files.get(rel_path)
            if not info:
                continue

            chunk = f"<|file_sep|>{rel_path}\n{info.content}\n"

            if total_chars + len(chunk) > self.config.max_context_chars:
                remaining = self.config.max_context_chars - total_chars
                if remaining < 512:
                    break
                chunk = chunk[:remaining]

            parts.append(chunk)
            total_chars += len(chunk)
            used += 1

        return "".join(parts)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def find_hybrid_context(
    root_dir: str,
    prefix: str,
    suffix: str,
    completion_file_path: str,
    config: Optional[SimpleHybridConfig] = None,
) -> Tuple[str, List[dict]]:
    """
    Main entry point similar to your current pipeline.
    """
    cfg = config or SimpleHybridConfig()
    index = SimplePythonRepoIndex(root_dir, cfg)
    index.build()

    w = cfg.query_window
    query = prefix[-w:] + "\n" + suffix[:w]
    ranked = index.retrieve(query=query, completion_file=completion_file_path, top_k=25)
    context = index.assemble_context(ranked, completion_file=completion_file_path)

    score_records = []
    included_paths = set()
    for chunk in context.split("<|file_sep|>"):
        if not chunk.strip():
            continue
        first_line = chunk.splitlines()[0].strip()
        included_paths.add(first_line)

    for rank, (rel_path, final_score, bm25_score, graph_plus_symbol) in enumerate(ranked):
        score_records.append(
            {
                "retrieved_file": rel_path,
                "rank": rank,
                "combined_score": float(final_score),
                "bm25_score": float(bm25_score),
                "graph_plus_symbol_score": float(graph_plus_symbol),
                "included_in_context": rel_path in included_paths,
            }
        )

    return context, score_records


# -------------------------------------------------------------------
# Hydra Entry Point
# -------------------------------------------------------------------

def build_prediction_filename(cfg: DictConfig) -> str:
    """Build a descriptive prediction filename from config."""
    parts = [cfg.data.lang, cfg.data.stage, cfg.retrieval.name]

    # Add hop info
    parts.append(f"hop{cfg.retrieval.graph.max_hop}")

    # Add weight info if non-default
    if cfg.retrieval.scoring.bm25_weight != 0.5 or cfg.retrieval.scoring.graph_weight != 0.5:
        bm = cfg.retrieval.scoring.bm25_weight
        gr = cfg.retrieval.scoring.graph_weight
        parts.append(f"bm{bm}-gr{gr}")

    # Add trim info
    if cfg.trim.prefix:
        parts.append("short-prefix")
    if cfg.trim.suffix:
        parts.append("short-suffix")

    return "-".join(parts) + ".jsonl"


def create_run_from_simple_config(
    cfg: DictConfig,
    prediction_file: str,
    num_samples: int,
    status: str = "completed",
    error_message: str = None
) -> ExperimentRun:
    """Create an ExperimentRun from a Hydra config for the simple hybrid retriever."""
    from datetime import datetime
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
        import_weight=1.0,  # Simple hybrid only uses imports
        call_weight=0.0,
        inheritance_weight=0.0,
        type_ref_weight=0.0,
        max_files=cfg.context.max_files,
        max_tokens=cfg.context.max_tokens,
        min_lines=cfg.context.min_lines,
        fallback_enabled=cfg.retrieval.graph.get("fallback_enabled", True),
        min_pool_size=cfg.retrieval.graph.get("min_pool_size", 10),
        query_window=cfg.context.get("query_window", 150),
        trim_prefix=cfg.trim.prefix,
        trim_suffix=cfg.trim.suffix,
        trim_lines=cfg.trim.trim_lines,
        prediction_file=prediction_file,
        num_samples=num_samples,
        config_json=OmegaConf.to_yaml(cfg),
        status=status,
        error_message=error_message
    )


def run_evaluation(
    predictions_file: str,
    data_dir: str,
    stage: str,
    language: str,
    ollama_url: str = "http://localhost:11434",
    model: str = "JetBrains/Mellum-4b-sft-python",
) -> Optional[dict]:
    """
    Evaluate predictions locally using Ollama + chrF.

    Args:
        predictions_file: Path to the predictions JSONL file
        data_dir: Path to the data directory
        stage: Evaluation stage
        language: Programming language
        ollama_url: Ollama API URL
        model: Ollama model name

    Returns:
        Dict with evaluation results or None if evaluation fails
    """
    from evaluate import run_evaluation as _run_eval, MODEL_NAME, OLLAMA_URL
    import evaluate as eval_module

    try:
        eval_module.MODEL_NAME = model
        eval_module.OLLAMA_URL = ollama_url

        mean_chrf = _run_eval(
            predictions_path=predictions_file,
            data_dir=data_dir,
            stage=stage,
            lang=language,
        )
        return {"mean_chrf": mean_chrf}

    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
        return None


def run_with_config(cfg: DictConfig, original_cwd: str) -> Optional[dict]:
    """Run the simple hybrid pipeline with the given config.

    Returns:
        Evaluation results dict if evaluation is enabled and successful, else None
    """

    # Initialize experiment database
    db = ExperimentDB(os.path.join(original_cwd, "experiments.db"))

    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    # Build SimpleHybridConfig from Hydra config
    config = SimpleHybridConfig.from_hydra_config(cfg)

    language = cfg.data.lang
    stage = cfg.data.stage

    print(f"\nRunning SimpleHybridRAG for stage '{stage}', language '{language}'")
    print(f"Config: max_hops={config.max_hops}, bm25={config.bm25_weight}, "
          f"graph={config.graph_weight}, symbol={config.symbol_weight}")

    # Paths (relative to original cwd)
    data_dir = os.path.join(original_cwd, cfg.data.data_dir)
    predictions_dir = os.path.join(original_cwd, cfg.data.predictions_dir)

    completion_points_file = os.path.join(data_dir, f"{language}-{stage}.jsonl")
    prediction_filename = build_prediction_filename(cfg)
    predictions_file = os.path.join(predictions_dir, prediction_filename)

    # Annotations file for evaluation
    # Ensure predictions directory exists
    os.makedirs(predictions_dir, exist_ok=True)

    # Count total items for progress bar
    with jsonlines.open(completion_points_file, 'r') as reader:
        total = sum(1 for _ in reader)

    # Create run ID early so we can use it for logging scores
    run = create_run_from_simple_config(cfg, predictions_file, total)
    run_id = run.run_id

    eval_results = None

    try:
        with jsonlines.open(completion_points_file, 'r') as reader:
            with jsonlines.open(predictions_file, 'w') as writer:
                for sample_idx, datapoint in enumerate(tqdm(reader, total=total, desc="Processing")):
                    # Identify repository storage
                    repo_path = datapoint['repo'].replace("/", "__")
                    repo_revision = datapoint['revision']
                    root_directory = os.path.join(
                        data_dir,
                        f"repositories-{language}-{stage}",
                        f"{repo_path}-{repo_revision}"
                    )

                    # Get completion file path
                    completion_file = datapoint['path']

                    # Get prefix and suffix
                    prefix = datapoint.get('prefix', '')
                    suffix = datapoint.get('suffix', '')

                    # Run simple hybrid retrieval with score tracking
                    try:
                        context, score_records = find_hybrid_context(
                            root_directory, prefix, suffix,
                            completion_file, config
                        )

                        # Log retrieval scores to database
                        if score_records:
                            sample_id = f"sample_{sample_idx}"
                            # Adapt score records for the database schema
                            adapted_scores = []
                            for rec in score_records:
                                adapted_scores.append({
                                    'retrieved_file': rec['retrieved_file'],
                                    'rank': rec['rank'],
                                    'bm25_score': rec['bm25_score'],
                                    'graph_score': rec['graph_plus_symbol_score'],
                                    'combined_score': rec['combined_score'],
                                    'included_in_context': rec['included_in_context']
                                })
                            db.log_retrieval_scores(
                                run_id, sample_id,
                                datapoint['repo'], completion_file,
                                adapted_scores
                            )
                    except Exception as e:
                        # Fallback to empty context on error
                        print(f"Warning: Error processing {datapoint.get('id', sample_idx)}: {e}")
                        context = ""

                    # Build submission
                    submission = {"context": context}

                    if cfg.trim.prefix:
                        prefix_lines = prefix.split("\n")
                        if len(prefix_lines) > cfg.trim.trim_lines:
                            submission["prefix"] = "\n".join(prefix_lines[-cfg.trim.trim_lines:])

                    if cfg.trim.suffix:
                        suffix_lines = suffix.split("\n")
                        if len(suffix_lines) > cfg.trim.trim_lines:
                            submission["suffix"] = "\n".join(suffix_lines[:cfg.trim.trim_lines])

                    writer.write(submission)

        print(f"\nPredictions saved to: {predictions_file}")

        # Log successful run to experiment database
        db.log_run(run)
        print(f"Run logged to database with ID: {run.run_id}")
        print(f"Logged {total} samples with retrieval scores")

        # Run evaluation if enabled
        if cfg.get('evaluation', {}).get('enabled', False):
            ollama_url = cfg.evaluation.get('ollama_url', 'http://localhost:11434')
            model = cfg.evaluation.get('model', 'mellum')
            eval_results = run_evaluation(
                predictions_file=predictions_file,
                data_dir=data_dir,
                stage=stage,
                language=language,
                ollama_url=ollama_url,
                model=model,
            )

            # Update run with evaluation results
            if eval_results:
                db.update_metrics(run_id, chrf=eval_results.get('mean_chrf'))
                print(f"Updated run {run_id} with chrF score: {eval_results.get('mean_chrf')}")

    except Exception as e:
        # Log failed run to experiment database
        print(f"\nError during processing: {e}")
        run.status = "failed"
        run.error_message = str(e)
        run.num_samples = 0
        db.log_run(run)
        print(f"Failed run logged to database with ID: {run.run_id}")
        raise

    return eval_results


def main_compose():
    """
    Main entry point using Hydra's compose API.
    This works around Python 3.14 compatibility issues with the @hydra.main decorator.

    Usage:
        python simple_repo_graph_rag.py                           # Use default config
        python simple_repo_graph_rag.py retrieval=simple_hybrid   # Use simple hybrid retrieval
        python simple_repo_graph_rag.py retrieval.graph.max_hop=3 # Override hop depth
        python simple_repo_graph_rag.py --multirun retrieval.graph.max_hop=1,2,3  # Sweep
    """
    import sys
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Get config directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "configs")

    # Parse command line overrides
    overrides = sys.argv[1:]

    # Check for multirun mode
    multirun = "--multirun" in overrides or "-m" in overrides
    overrides = [o for o in overrides if o not in ("--multirun", "-m")]

    # Clear any existing Hydra state
    GlobalHydra.instance().clear()

    try:
        # Initialize Hydra
        initialize_config_dir(version_base=None, config_dir=config_dir)

        if multirun:
            # Handle sweep mode
            import itertools

            # For sweeps, we need to parse the sweep parameters
            sweep_params = {}
            regular_overrides = []
            for override in overrides:
                if "," in override and "=" in override:
                    key, values = override.split("=", 1)
                    sweep_params[key] = values.split(",")
                else:
                    regular_overrides.append(override)

            # Generate all combinations
            keys = list(sweep_params.keys())
            values = [sweep_params[k] for k in keys]

            for combo in itertools.product(*values):
                combo_overrides = regular_overrides + [f"{k}={v}" for k, v in zip(keys, combo)]
                cfg = compose(config_name="config", overrides=combo_overrides)
                print(f"\n{'='*60}")
                print(f"Running with: {combo_overrides}")
                print(f"{'='*60}\n")
                run_with_config(cfg, script_dir)
        else:
            # Single run mode
            cfg = compose(config_name="config", overrides=overrides)
            run_with_config(cfg, script_dir)

    finally:
        GlobalHydra.instance().clear()


if __name__ == "__main__":
    main_compose()