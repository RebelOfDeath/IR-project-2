from __future__ import annotations

"""
Adaptive call-graph retrieval for repository-level code completion.

This module is designed as a drop-in replacement for the current hybrid
import-graph retriever. It keeps the same overall workflow:
  1. read a JSONL file of completion tasks,
  2. collect repository context for each task,
  3. write predictions JSONL,
  4. optionally run the provided evaluation pipeline,
  5. optionally log runs and retrieval scores to the provided SQLite database.

Core retrieval ideas:
  - lexical first-stage retrieval over code entities (BM25),
  - AST-based static graph construction,
  - a Quam-style adaptive expansion step over call / import / inheritance /
    type-reference relations,
  - an optional SlideGar-style hook for future listwise rerankers.

The implementation is intentionally pure-Python and dependency-light so it can
run in the same environment as the original baseline.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Set, Tuple

try:
    import jsonlines  # type: ignore
except Exception:
    class _JsonLinesReader:
        def __init__(self, path: str, mode: str):
            self.path = path
            self.mode = mode
            self.handle = None

        def __enter__(self):
            self.handle = open(self.path, self.mode, encoding="utf-8")
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.handle is not None:
                self.handle.close()
            return False

        def __iter__(self):
            assert self.handle is not None
            for line in self.handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

        def write(self, obj: Dict[str, Any]) -> None:
            assert self.handle is not None
            self.handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    class _JsonLinesModule:
        @staticmethod
        def open(path: str, mode: str = "r"):
            return _JsonLinesReader(path, mode)

    jsonlines = _JsonLinesModule()

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except Exception:
    _BM25Okapi = None

from tqdm import tqdm

try:
    from experiment_db import ExperimentDB, ExperimentRun
except Exception:
    ExperimentDB = None
    ExperimentRun = None


# ---------------------------------------------------------------------------
# Fallback BM25 implementation
# ---------------------------------------------------------------------------

class _FallbackBM25:
    """Small BM25 fallback used only if rank_bm25 is unavailable."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = [list(doc) for doc in corpus]
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.term_df: Dict[str, int] = defaultdict(int)
        self.term_tf: List[Counter[str]] = []
        for doc in self.corpus:
            tf = Counter(doc)
            self.term_tf.append(tf)
            for term in tf:
                self.term_df[term] += 1
        self.num_docs = len(self.corpus)

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        scores = [0.0] * self.num_docs
        for term in query_tokens:
            df = self.term_df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.term_tf):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(1e-9, self.avgdl))
                scores[i] += idf * (freq * (self.k1 + 1)) / max(1e-9, denom)
        return scores


BM25Okapi = _BM25Okapi or _FallbackBM25


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveCallGraphConfig:
    query_window_lines: int = 120

    # Retrieval pools
    bm25_topk: int = 200
    initial_pool_size: int = 80
    frontier_pool_size: int = 120
    batch_size: int = 6
    top_seed_count: int = 8
    max_selected_entities: int = 48
    max_entities_per_file: int = 4

    # Context assembly
    max_files: int = 6
    max_context_tokens: int = 7000
    chars_per_token: float = 4.0
    max_entity_lines: int = 80
    snippet_padding_lines: int = 2
    min_lines: int = 3

    # Adaptive expansion
    max_hops: int = 2
    hop_decay: float = 0.65
    seed_temperature: float = 4.0

    # Base scoring
    bm25_weight: float = 0.45
    symbol_weight: float = 0.18
    file_prior_weight: float = 0.12
    modified_bonus: float = 0.10
    same_file_bonus: float = 0.06

    # Quam-style affinity scoring
    graph_affinity_weight: float = 0.25
    import_weight: float = 0.30
    reverse_import_weight: float = 0.12
    call_weight: float = 1.00
    reverse_call_weight: float = 0.75
    inheritance_weight: float = 0.55
    type_ref_weight: float = 0.40
    sibling_weight: float = 0.10

    # SlideGar hook (disabled by default)
    enable_slidegar_hook: bool = False
    slide_window_size: int = 20
    slide_step_size: int = 10

    # Repository extraction/cache
    extraction_cache_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

BUILTIN_TYPE_NAMES = {
    "int", "str", "float", "bool", "list", "dict", "set", "tuple", "bytes",
    "object", "None", "Optional", "Any", "Union", "Iterable", "Iterator",
    "Sequence", "Mapping", "Callable", "Type", "Self",
}

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


def sanitize_repo_name(repo_name: str) -> str:
    return repo_name.replace("/", "__")


def normalize_relpath(root_dir: str, file_path: str) -> str:
    root = Path(root_dir).resolve()
    path = Path(file_path).resolve() if Path(file_path).is_absolute() else (root / file_path).resolve()
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(file_path)
    return rel.as_posix()


def module_name_from_relpath(rel_path: str) -> str:
    p = Path(rel_path)
    no_suffix = p.with_suffix("")
    parts = list(no_suffix.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_relative_module(current_module: str, imported_module: Optional[str], level: int) -> str:
    imported_module = imported_module or ""
    package_parts = current_module.split(".")[:-1] if current_module else []
    if level > 0:
        keep = max(0, len(package_parts) - (level - 1))
        base = package_parts[:keep]
    else:
        base = []
    extra = imported_module.split(".") if imported_module else []
    return ".".join([part for part in [*base, *extra] if part])


def _split_identifier_token(token: str) -> List[str]:
    token = CAMEL_RE.sub(r"\1 \2", token)
    token = token.replace("_", " ")
    parts = [p.lower() for p in token.split() if p]
    if token.lower() not in parts:
        parts.append(token.lower())
    return parts


def tokenize_code(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in IDENTIFIER_RE.findall(text):
        tokens.extend(_split_identifier_token(raw))
    return tokens


def extract_identifiers(text: str) -> Set[str]:
    result: Set[str] = set()
    for tok in IDENTIFIER_RE.findall(text):
        if len(tok) <= 1:
            continue
        if tok in BUILTIN_TYPE_NAMES:
            continue
        result.add(tok)
    return result


def take_query_window(prefix: str, suffix: str, window_lines: int) -> str:
    prefix_lines = prefix.splitlines()
    suffix_lines = suffix.splitlines()
    prefix_tail = "\n".join(prefix_lines[-window_lines:])
    suffix_head = "\n".join(suffix_lines[:window_lines])
    return f"{prefix_tail}\n{suffix_head}".strip()


def normalize_scores(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    max_value = max(values.values())
    min_value = min(values.values())
    if abs(max_value - min_value) < 1e-12:
        return {k: 1.0 if v > 0 else 0.0 for k, v in values.items()}
    return {k: (v - min_value) / (max_value - min_value) for k, v in values.items()}


def stable_softmax(pairs: Sequence[Tuple[str, float]], temperature: float) -> Dict[str, float]:
    if not pairs:
        return {}
    scores = [score * temperature for _, score in pairs]
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    denom = sum(exps) or 1.0
    return {entity_id: exp_value / denom for (entity_id, _), exp_value in zip(pairs, exps)}


def limited_lines(lines: Sequence[str], start_line: int, end_line: int, pad: int, max_lines: int) -> Tuple[int, int, str]:
    lo = max(1, start_line - pad)
    hi = min(len(lines), end_line + pad)
    if hi - lo + 1 > max_lines:
        hi = min(len(lines), lo + max_lines - 1)
    snippet = "\n".join(lines[lo - 1:hi])
    return lo, hi, snippet


def approx_token_len(text: str, chars_per_token: float) -> int:
    return max(1, int(len(text) / max(1e-9, chars_per_token)))


# ---------------------------------------------------------------------------
# AST data model
# ---------------------------------------------------------------------------

@dataclass
class ImportRef:
    kind: str
    module: Optional[str]
    level: int
    imported_name: Optional[str] = None
    alias: Optional[str] = None


@dataclass
class CallSite:
    kind: str  # name | attr
    target: str
    base: Optional[str] = None


@dataclass
class CodeEntity:
    entity_id: str
    file_path: str
    module_name: str
    name: str
    qualname: str
    kind: str  # class | function | method | assignment
    start_line: int
    end_line: int
    text: str
    parent_qualname: Optional[str] = None
    class_qualname: Optional[str] = None
    ast_node: Optional[ast.AST] = None
    callsites: List[CallSite] = field(default_factory=list)
    annotation_names: Set[str] = field(default_factory=set)
    base_names: List[str] = field(default_factory=list)


@dataclass
class FileInfo:
    file_path: str
    module_name: str
    content: str
    lines: List[str]
    imports: List[ImportRef] = field(default_factory=list)
    imported_aliases: Dict[str, str] = field(default_factory=dict)
    entity_ids: List[str] = field(default_factory=list)
    top_level_entity_ids: List[str] = field(default_factory=list)
    simple_name_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------

class _CallAndTypeCollector(ast.NodeVisitor):
    def __init__(self, root: ast.AST):
        self.root = root
        self.callsites: List[CallSite] = []
        self.annotation_names: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node is self.root:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            self.callsites.append(CallSite(kind="name", target=func.id))
        elif isinstance(func, ast.Attribute):
            base_name = None
            if isinstance(func.value, ast.Name):
                base_name = func.value.id
            elif isinstance(func.value, ast.Call):
                inner = func.value.func
                if isinstance(inner, ast.Name) and inner.id == "super":
                    base_name = "super"
            self.callsites.append(CallSite(kind="attr", target=func.attr, base=base_name))
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if node.annotation is not None:
            self.annotation_names.update(_extract_names_from_annotation(node.annotation))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.annotation is not None:
            self.annotation_names.update(_extract_names_from_annotation(node.annotation))
        self.generic_visit(node)

    def visit_FunctionType(self, node: ast.FunctionType) -> None:
        self.generic_visit(node)


class _EntityExtractor(ast.NodeVisitor):
    def __init__(self, file_info: FileInfo):
        self.file_info = file_info
        self.current_class_stack: List[str] = []
        self.entities: List[CodeEntity] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            alias_name = alias.asname or alias.name.split(".")[0]
            self.file_info.imports.append(
                ImportRef(kind="import", module=alias.name, level=0, alias=alias_name)
            )
            self.file_info.imported_aliases[alias_name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            alias_name = alias.asname or alias.name
            self.file_info.imports.append(
                ImportRef(
                    kind="from",
                    module=node.module,
                    level=node.level,
                    imported_name=alias.name,
                    alias=alias_name,
                )
            )
            base = resolve_relative_module(self.file_info.module_name, node.module, node.level)
            if base:
                self.file_info.imported_aliases[alias_name] = f"{base}.{alias.name}"
            else:
                self.file_info.imported_aliases[alias_name] = alias.name
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = ".".join([*self.current_class_stack, node.name]) if self.current_class_stack else node.name
        full_qualname = f"{self.file_info.module_name}.{qualname}" if self.file_info.module_name else qualname
        text = _extract_node_text(self.file_info.lines, node.lineno, getattr(node, "end_lineno", node.lineno))
        entity = CodeEntity(
            entity_id=full_qualname,
            file_path=self.file_info.file_path,
            module_name=self.file_info.module_name,
            name=node.name,
            qualname=full_qualname,
            kind="class",
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            text=text,
            parent_qualname=self.current_class_stack[-1] if self.current_class_stack else None,
            class_qualname=full_qualname,
            ast_node=node,
            base_names=[_name_to_string(base) for base in node.bases if _name_to_string(base)],
        )
        collector = _CallAndTypeCollector(node)
        collector.visit(node)
        entity.callsites = collector.callsites
        entity.annotation_names = collector.annotation_names
        self._register_entity(entity, top_level=not self.current_class_stack)
        self.current_class_stack.append(qualname)
        self.generic_visit(node)
        self.current_class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node, async_kind=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node, async_kind=True)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.current_class_stack:
            return
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if len(targets) != 1:
            return
        name = targets[0]
        if not name:
            return
        qualname = f"{self.file_info.module_name}.{name}" if self.file_info.module_name else name
        text = _extract_node_text(self.file_info.lines, node.lineno, getattr(node, "end_lineno", node.lineno))
        entity = CodeEntity(
            entity_id=qualname,
            file_path=self.file_info.file_path,
            module_name=self.file_info.module_name,
            name=name,
            qualname=qualname,
            kind="assignment",
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            text=text,
            ast_node=node,
        )
        self._register_entity(entity, top_level=True)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.current_class_stack:
            return
        if not isinstance(node.target, ast.Name):
            return
        name = node.target.id
        qualname = f"{self.file_info.module_name}.{name}" if self.file_info.module_name else name
        text = _extract_node_text(self.file_info.lines, node.lineno, getattr(node, "end_lineno", node.lineno))
        entity = CodeEntity(
            entity_id=qualname,
            file_path=self.file_info.file_path,
            module_name=self.file_info.module_name,
            name=name,
            qualname=qualname,
            kind="assignment",
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            text=text,
            ast_node=node,
            annotation_names=_extract_names_from_annotation(node.annotation),
        )
        self._register_entity(entity, top_level=True)

    def _handle_function(self, node: ast.AST, async_kind: bool) -> None:
        if not hasattr(node, "name"):
            return
        func_name = getattr(node, "name")
        if self.current_class_stack:
            qualname = ".".join([*self.current_class_stack, func_name])
            class_qualname = self.current_class_stack[-1]
            kind = "method"
            parent = self.current_class_stack[-1]
        else:
            qualname = func_name
            class_qualname = None
            kind = "function"
            parent = None
        full_qualname = f"{self.file_info.module_name}.{qualname}" if self.file_info.module_name else qualname
        text = _extract_node_text(self.file_info.lines, node.lineno, getattr(node, "end_lineno", node.lineno))
        entity = CodeEntity(
            entity_id=full_qualname,
            file_path=self.file_info.file_path,
            module_name=self.file_info.module_name,
            name=func_name,
            qualname=full_qualname,
            kind=kind,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            text=text,
            parent_qualname=parent,
            class_qualname=class_qualname,
            ast_node=node,
        )
        collector = _CallAndTypeCollector(node)
        collector.visit(node)
        entity.callsites = collector.callsites
        entity.annotation_names = collector.annotation_names
        returns = getattr(node, "returns", None)
        if returns is not None:
            entity.annotation_names.update(_extract_names_from_annotation(returns))
        self._register_entity(entity, top_level=not self.current_class_stack)
        # Skip nested defs for now; repository-level retrieval benefits mostly from top-level entities.

    def _register_entity(self, entity: CodeEntity, top_level: bool) -> None:
        self.entities.append(entity)
        self.file_info.entity_ids.append(entity.entity_id)
        if top_level:
            self.file_info.top_level_entity_ids.append(entity.entity_id)
        self.file_info.simple_name_index[entity.name].append(entity.entity_id)


def _extract_node_text(lines: Sequence[str], start_line: int, end_line: int) -> str:
    start = max(1, start_line)
    end = min(len(lines), end_line)
    return "\n".join(lines[start - 1:end])


def _name_to_string(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: List[str] = []
        current: Optional[ast.AST] = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
    return None


def _extract_names_from_annotation(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            string_value = _name_to_string(child)
            if string_value:
                names.add(string_value)
    return names


# ---------------------------------------------------------------------------
# Repository index and graph
# ---------------------------------------------------------------------------

class AdaptiveRepoIndex:
    def __init__(self, root_dir: str, config: Optional[AdaptiveCallGraphConfig] = None):
        self.root_dir = str(Path(root_dir).resolve())
        self.config = config or AdaptiveCallGraphConfig()

        self.files: Dict[str, FileInfo] = {}
        self.entities: Dict[str, CodeEntity] = {}
        self.module_to_file: Dict[str, str] = {}
        self.simple_name_index: Dict[str, List[str]] = defaultdict(list)
        self.qualname_index: Dict[str, str] = {}
        self.class_methods: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.class_bases: Dict[str, List[str]] = defaultdict(list)

        self.file_import_out: Dict[str, Set[str]] = defaultdict(set)
        self.file_import_in: Dict[str, Set[str]] = defaultdict(set)

        self.entity_edges_out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.entity_edges_in: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[str] = []
        self._bm25_corpus: List[List[str]] = []

    def build(self) -> None:
        self._load_python_files()
        self._register_indexes()
        self._build_import_graph()
        self._build_entity_graph()
        self._build_bm25()

    def _load_python_files(self) -> None:
        for path in Path(self.root_dir).rglob("*.py"):
            if path.is_dir():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            rel_path = normalize_relpath(self.root_dir, str(path))
            lines = content.splitlines()
            if len(lines) < self.config.min_lines:
                continue

            file_info = FileInfo(
                file_path=rel_path,
                module_name=module_name_from_relpath(rel_path),
                content=content,
                lines=lines,
            )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(content)
                extractor = _EntityExtractor(file_info)
                extractor.visit(tree)
                for entity in extractor.entities:
                    self.entities[entity.entity_id] = entity
            except SyntaxError:
                # Keep the file as a lexical context source only.
                pass

            self.files[rel_path] = file_info
            if file_info.module_name:
                self.module_to_file[file_info.module_name] = rel_path

    def _register_indexes(self) -> None:
        for entity_id, entity in self.entities.items():
            self.qualname_index[entity.qualname] = entity_id
            self.simple_name_index[entity.name].append(entity_id)
            if entity.kind == "method" and entity.class_qualname:
                self.class_methods[entity.class_qualname][entity.name] = entity_id
            if entity.kind == "class":
                self.class_bases[entity.qualname] = list(entity.base_names)

    def _candidate_modules_for_import(self, current_module: str, imp: ImportRef) -> List[str]:
        if imp.kind == "import":
            return [imp.module] if imp.module else []
        base = resolve_relative_module(current_module, imp.module, imp.level)
        candidates: List[str] = []
        if base:
            candidates.append(base)
            if imp.imported_name and imp.imported_name != "*":
                candidates.append(f"{base}.{imp.imported_name}")
        elif imp.imported_name and imp.imported_name != "*":
            candidates.append(imp.imported_name)
        return candidates

    def _resolve_import_to_files(self, current_module: str, imp: ImportRef) -> Set[str]:
        resolved: Set[str] = set()
        for mod in self._candidate_modules_for_import(current_module, imp):
            if mod in self.module_to_file:
                resolved.add(self.module_to_file[mod])
                continue
            for known_module, rel_path in self.module_to_file.items():
                if known_module.startswith(mod + "."):
                    resolved.add(rel_path)
        return resolved

    def _build_import_graph(self) -> None:
        for file_path, info in self.files.items():
            for imp in info.imports:
                for target_file in self._resolve_import_to_files(info.module_name, imp):
                    if target_file == file_path:
                        continue
                    self.file_import_out[file_path].add(target_file)
                    self.file_import_in[target_file].add(file_path)

    def _build_entity_graph(self) -> None:
        for entity in self.entities.values():
            if entity.kind == "class":
                self._add_class_edges(entity)
            self._add_annotation_edges(entity)
            self._add_call_edges(entity)
        self._add_file_structure_edges()

    def _add_edge(self, src: str, dst: str, kind: str) -> None:
        if src == dst:
            return
        existing = self.entity_edges_out[src]
        if (dst, kind) in existing:
            return
        self.entity_edges_out[src].append((dst, kind))
        self.entity_edges_in[dst].append((src, kind))

    def _add_class_edges(self, entity: CodeEntity) -> None:
        file_info = self.files.get(entity.file_path)
        if file_info is None:
            return
        for base_name in entity.base_names:
            for target_id in self._resolve_reference(base_name, file_info, current_class=entity.qualname, prefer_classes=True):
                target = self.entities.get(target_id)
                if target is not None and target.kind == "class":
                    self._add_edge(entity.entity_id, target_id, "inherits")

    def _add_annotation_edges(self, entity: CodeEntity) -> None:
        file_info = self.files.get(entity.file_path)
        if file_info is None:
            return
        for name in entity.annotation_names:
            for target_id in self._resolve_reference(name, file_info, current_class=entity.class_qualname, prefer_classes=True):
                target = self.entities.get(target_id)
                if target is not None and target.kind == "class":
                    self._add_edge(entity.entity_id, target_id, "type_ref")

    def _add_call_edges(self, entity: CodeEntity) -> None:
        if entity.kind not in {"function", "method", "class"}:
            return
        file_info = self.files.get(entity.file_path)
        if file_info is None:
            return
        for callsite in entity.callsites:
            for target_id in self._resolve_callsite(callsite, file_info, entity):
                target = self.entities.get(target_id)
                if target is not None:
                    self._add_edge(entity.entity_id, target_id, "calls")

    def _add_file_structure_edges(self) -> None:
        for file_path, file_info in self.files.items():
            # Encourage same-file propagation between top-level entities.
            ids = file_info.top_level_entity_ids
            for i, src in enumerate(ids):
                for j, dst in enumerate(ids):
                    if i == j:
                        continue
                    self._add_edge(src, dst, "same_file")
            # Bridge top-level entities across imported files.
            for imported_file in self.file_import_out.get(file_path, set()):
                for src in file_info.top_level_entity_ids:
                    for dst in self.files.get(imported_file, FileInfo("", "", "", [])).top_level_entity_ids:
                        self._add_edge(src, dst, "imports")
            for importer_file in self.file_import_in.get(file_path, set()):
                for src in file_info.top_level_entity_ids:
                    for dst in self.files.get(importer_file, FileInfo("", "", "", [])).top_level_entity_ids:
                        self._add_edge(src, dst, "imported_by")

    def _resolve_callsite(self, callsite: CallSite, file_info: FileInfo, entity: CodeEntity) -> Set[str]:
        results: Set[str] = set()
        if callsite.kind == "name":
            results.update(self._resolve_reference(callsite.target, file_info, current_class=entity.class_qualname))
            return results

        base = callsite.base
        attr = callsite.target
        if base in {"self", "cls"} and entity.class_qualname:
            results.update(self._resolve_method_on_class(entity.class_qualname, attr))
            return results
        if base == "super" and entity.class_qualname:
            for base_class in self._resolve_base_classes(entity.class_qualname):
                results.update(self._resolve_method_on_class(base_class, attr))
            return results
        if base:
            # Module alias or imported symbol alias.
            alias_target = file_info.imported_aliases.get(base)
            if alias_target:
                results.update(self._resolve_dotted(alias_target, attr))
            # Local class name in file.
            for target_id in file_info.simple_name_index.get(base, []):
                target = self.entities.get(target_id)
                if target is not None and target.kind == "class":
                    results.update(self._resolve_method_on_class(target.qualname, attr))
            # Unique global class name.
            for target_id in self.simple_name_index.get(base, []):
                target = self.entities.get(target_id)
                if target is not None and target.kind == "class":
                    results.update(self._resolve_method_on_class(target.qualname, attr))
        return results

    def _resolve_base_classes(self, class_qualname: str) -> List[str]:
        entity = self.entities.get(class_qualname)
        if entity is None:
            return []
        file_info = self.files.get(entity.file_path)
        if file_info is None:
            return []
        resolved: List[str] = []
        for base_name in entity.base_names:
            for target_id in self._resolve_reference(base_name, file_info, current_class=class_qualname, prefer_classes=True):
                target = self.entities.get(target_id)
                if target is not None and target.kind == "class":
                    resolved.append(target.qualname)
        return resolved

    def _resolve_method_on_class(self, class_qualname: str, method_name: str) -> Set[str]:
        results: Set[str] = set()
        method_id = self.class_methods.get(class_qualname, {}).get(method_name)
        if method_id:
            results.add(method_id)
        for base_class in self._resolve_base_classes(class_qualname):
            method_id = self.class_methods.get(base_class, {}).get(method_name)
            if method_id:
                results.add(method_id)
        return results

    def _resolve_dotted(self, dotted: str, attr: Optional[str] = None) -> Set[str]:
        results: Set[str] = set()
        full = f"{dotted}.{attr}" if attr else dotted
        if full in self.qualname_index:
            results.add(self.qualname_index[full])
        if dotted in self.module_to_file and attr:
            file_info = self.files.get(self.module_to_file[dotted])
            if file_info:
                results.update(file_info.simple_name_index.get(attr, []))
        parts = full.split(".")
        if len(parts) >= 2:
            tail = parts[-1]
            for target_id in self.simple_name_index.get(tail, []):
                target = self.entities.get(target_id)
                if target and target.qualname.endswith(full):
                    results.add(target_id)
        return results

    def _resolve_reference(
        self,
        name: str,
        file_info: FileInfo,
        current_class: Optional[str] = None,
        prefer_classes: bool = False,
    ) -> Set[str]:
        results: Set[str] = set()
        if not name:
            return results

        if "." in name:
            return self._resolve_dotted(name)

        if current_class:
            if prefer_classes and current_class in self.qualname_index:
                pass
            if name in self.class_methods.get(current_class, {}):
                results.add(self.class_methods[current_class][name])

        for target_id in file_info.simple_name_index.get(name, []):
            results.add(target_id)

        alias_target = file_info.imported_aliases.get(name)
        if alias_target:
            results.update(self._resolve_dotted(alias_target))
            if alias_target in self.module_to_file:
                imported_file = self.files.get(self.module_to_file[alias_target])
                if imported_file:
                    # Imported module itself. Add top-level entities as potential public API.
                    for target_id in imported_file.top_level_entity_ids:
                        results.add(target_id)

        global_hits = self.simple_name_index.get(name, [])
        if len(global_hits) == 1:
            results.add(global_hits[0])
        elif len(global_hits) > 1:
            same_module_hits = [target_id for target_id in global_hits if self.entities[target_id].module_name == file_info.module_name]
            if len(same_module_hits) == 1:
                results.add(same_module_hits[0])
            else:
                for target_id in global_hits[:4]:
                    results.add(target_id)

        if prefer_classes:
            results = {target_id for target_id in results if self.entities.get(target_id) and self.entities[target_id].kind == "class"}
        return results

    def _build_bm25(self) -> None:
        for entity_id, entity in self.entities.items():
            index_text = self._entity_index_text(entity)
            tokens = tokenize_code(index_text)
            if not tokens:
                continue
            self._bm25_ids.append(entity_id)
            self._bm25_corpus.append(tokens)
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)

    def _entity_index_text(self, entity: CodeEntity) -> str:
        file_info = self.files.get(entity.file_path)
        alias_tokens = ""
        if file_info is not None:
            alias_tokens = " ".join(file_info.imported_aliases.keys())
        return "\n".join([
            entity.file_path,
            entity.module_name,
            entity.qualname,
            entity.name,
            alias_tokens,
            entity.text,
        ])

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def file_import_prior(self, completion_file: Optional[str]) -> Dict[str, float]:
        if not completion_file:
            return {}
        start = normalize_relpath(self.root_dir, completion_file)
        if start not in self.files:
            return {}
        scores: Dict[str, float] = {start: 1.0}
        frontier: List[Tuple[str, int, float]] = [(start, 0, 1.0)]
        seen: Dict[str, int] = {start: 0}
        while frontier:
            current, hop, score = frontier.pop(0)
            if hop >= self.config.max_hops:
                continue
            next_hop = hop + 1
            decay = self.config.hop_decay ** hop
            for neighbor in self.file_import_out.get(current, set()):
                candidate_score = 1.0 * decay
                if candidate_score > scores.get(neighbor, 0.0):
                    scores[neighbor] = candidate_score
                if seen.get(neighbor, 10 ** 9) > next_hop:
                    seen[neighbor] = next_hop
                    frontier.append((neighbor, next_hop, candidate_score))
            for neighbor in self.file_import_in.get(current, set()):
                candidate_score = self.config.reverse_import_weight * decay
                if candidate_score > scores.get(neighbor, 0.0):
                    scores[neighbor] = candidate_score
                if seen.get(neighbor, 10 ** 9) > next_hop:
                    seen[neighbor] = next_hop
                    frontier.append((neighbor, next_hop, candidate_score))
        return normalize_scores(scores)

    def bm25_scores(self, query: str, top_k: int) -> Dict[str, float]:
        if self._bm25 is None:
            return {}
        query_tokens = tokenize_code(query)
        raw_scores = self._bm25.get_scores(query_tokens)
        indexed = list(zip(self._bm25_ids, raw_scores))
        indexed.sort(key=lambda item: item[1], reverse=True)
        limited = indexed[:top_k]
        return normalize_scores({entity_id: score for entity_id, score in limited if score > 0})

    def symbol_scores(self, query: str) -> Dict[str, float]:
        query_ids = extract_identifiers(query)
        if not query_ids:
            return {}
        scores: Dict[str, float] = {}
        for entity_id, entity in self.entities.items():
            file_info = self.files.get(entity.file_path)
            candidates = {entity.name}
            candidates.update(part for part in entity.qualname.split(".") if part)
            if file_info is not None:
                candidates.update(file_info.imported_aliases.keys())
            overlap = len(query_ids & candidates)
            if overlap:
                scores[entity_id] = overlap / max(1, len(query_ids))
        return normalize_scores(scores)

    def entity_neighbors(self, entity_id: str) -> List[Tuple[str, str, float]]:
        neighbors: List[Tuple[str, str, float]] = []
        for dst, kind in self.entity_edges_out.get(entity_id, []):
            weight = self._edge_weight(kind, reverse=False)
            neighbors.append((dst, kind, weight))
        for src, kind in self.entity_edges_in.get(entity_id, []):
            reverse_kind = f"reverse_{kind}"
            weight = self._edge_weight(kind, reverse=True)
            neighbors.append((src, reverse_kind, weight))
        return neighbors

    def _edge_weight(self, kind: str, reverse: bool) -> float:
        if kind == "calls":
            return self.config.reverse_call_weight if reverse else self.config.call_weight
        if kind == "inherits":
            return self.config.inheritance_weight
        if kind == "type_ref":
            return self.config.type_ref_weight
        if kind == "imports":
            return self.config.reverse_import_weight if reverse else self.config.import_weight
        if kind == "imported_by":
            return self.config.reverse_import_weight if reverse else self.config.import_weight
        if kind == "same_file":
            return self.config.sibling_weight
        return 0.1


# ---------------------------------------------------------------------------
# Quam-style adaptive retrieval
# ---------------------------------------------------------------------------

@dataclass
class EntityScoreRecord:
    entity_id: str
    file_path: str
    rank: int
    combined_score: float
    bm25_score: float
    graph_score: float
    included_in_context: bool


class ListwiseWindowReranker(Protocol):
    def rerank(self, query: str, entity_ids: Sequence[str], index: AdaptiveRepoIndex) -> List[str]:
        ...


def slidegar_rank_entities(
    query: str,
    candidate_ids: Sequence[str],
    index: AdaptiveRepoIndex,
    reranker: ListwiseWindowReranker,
    window_size: int,
    step_size: int,
) -> Dict[str, float]:
    """
    Optional SlideGar-style helper.

    The reranker returns only an ordering, not scores. We therefore convert ranked
    positions into pseudo-scores with reciprocal rank. This helper is not used by
    default, but it provides the extension point needed to plug a listwise reranker
    into the adaptive pipeline later.
    """
    scores: Dict[str, float] = {}
    if not candidate_ids:
        return scores
    entity_ids = list(candidate_ids)
    if window_size <= 0 or step_size <= 0:
        return scores
    start = 0
    while start < len(entity_ids):
        window = entity_ids[start:start + window_size]
        if not window:
            break
        ranked = reranker.rerank(query, window, index)
        for rank, entity_id in enumerate(ranked, start=1):
            scores[entity_id] = max(scores.get(entity_id, 0.0), 1.0 / rank)
        start += step_size
    return normalize_scores(scores)


def adaptive_entity_retrieval(
    index: AdaptiveRepoIndex,
    query: str,
    completion_file: Optional[str],
    modified_files: Optional[Sequence[str]],
    config: AdaptiveCallGraphConfig,
    listwise_reranker: Optional[ListwiseWindowReranker] = None,
) -> List[Tuple[str, float, float, float]]:
    bm25_scores = index.bm25_scores(query, top_k=config.bm25_topk)
    symbol_scores = index.symbol_scores(query)
    file_prior = index.file_import_prior(completion_file)

    modified_relpaths: Set[str] = set()
    for file_path in modified_files or []:
        modified_relpaths.add(normalize_relpath(index.root_dir, file_path))

    base_scores: Dict[str, float] = {}
    bm25_component: Dict[str, float] = defaultdict(float)
    graph_component: Dict[str, float] = defaultdict(float)
    initial_ranking: List[Tuple[str, float]] = []

    for entity_id, entity in index.entities.items():
        bm25 = bm25_scores.get(entity_id, 0.0)
        symbol = symbol_scores.get(entity_id, 0.0)
        prior = file_prior.get(entity.file_path, 0.0)
        score = config.bm25_weight * bm25 + config.symbol_weight * symbol + config.file_prior_weight * prior
        if entity.file_path in modified_relpaths:
            score += config.modified_bonus
        if completion_file and normalize_relpath(index.root_dir, completion_file) == entity.file_path:
            score += config.same_file_bonus
        if score <= 0:
            continue
        base_scores[entity_id] = score
        bm25_component[entity_id] = bm25
        graph_component[entity_id] = prior
        initial_ranking.append((entity_id, score))

    initial_ranking.sort(key=lambda item: item[1], reverse=True)
    initial_ranking = initial_ranking[:config.initial_pool_size]

    if config.enable_slidegar_hook and listwise_reranker is not None:
        listwise_scores = slidegar_rank_entities(
            query=query,
            candidate_ids=[entity_id for entity_id, _ in initial_ranking],
            index=index,
            reranker=listwise_reranker,
            window_size=config.slide_window_size,
            step_size=config.slide_step_size,
        )
        for entity_id, bonus in listwise_scores.items():
            if entity_id in base_scores:
                base_scores[entity_id] += 0.10 * bonus

    initial_pool: List[str] = [entity_id for entity_id, _ in initial_ranking]
    seen: Set[str] = set()
    selected: List[str] = []
    per_file_selected: Counter[str] = Counter()
    frontier_affinity: Dict[str, float] = defaultdict(float)
    frontier_sources: Dict[str, Set[str]] = defaultdict(set)
    use_frontier = False

    while len(selected) < config.max_selected_entities and (initial_pool or frontier_affinity):
        batch: List[str] = []
        if use_frontier and frontier_affinity:
            seed_probs = stable_softmax(
                [(entity_id, base_scores.get(entity_id, 0.0)) for entity_id in selected[:config.top_seed_count]],
                temperature=config.seed_temperature,
            )
            scored_frontier: List[Tuple[str, float]] = []
            to_prune: List[str] = []
            for entity_id, affinity in list(frontier_affinity.items()):
                if entity_id in seen:
                    to_prune.append(entity_id)
                    continue
                if per_file_selected[index.entities[entity_id].file_path] >= config.max_entities_per_file:
                    to_prune.append(entity_id)
                    continue
                combined = base_scores.get(entity_id, 0.0) + config.graph_affinity_weight * affinity
                if seed_probs:
                    # Mild extra bonus if the candidate is connected to several strong seeds.
                    support = sum(seed_probs.get(src, 0.0) for src in frontier_sources.get(entity_id, set()))
                    combined += 0.10 * support
                scored_frontier.append((entity_id, combined))
            for entity_id in to_prune:
                frontier_affinity.pop(entity_id, None)
            scored_frontier.sort(key=lambda item: item[1], reverse=True)
            batch = [entity_id for entity_id, _ in scored_frontier[:config.batch_size]]
            for entity_id in batch:
                frontier_affinity.pop(entity_id, None)
            use_frontier = False
        else:
            while initial_pool and len(batch) < config.batch_size:
                entity_id = initial_pool.pop(0)
                if entity_id in seen:
                    continue
                if per_file_selected[index.entities[entity_id].file_path] >= config.max_entities_per_file:
                    continue
                batch.append(entity_id)
            use_frontier = True

        if not batch and frontier_affinity:
            if initial_pool:
                use_frontier = False
                continue
            break
        if not batch:
            break

        for entity_id in batch:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            selected.append(entity_id)
            per_file_selected[index.entities[entity_id].file_path] += 1

        seed_candidates = [(entity_id, base_scores.get(entity_id, 0.0)) for entity_id in selected]
        seed_candidates.sort(key=lambda item: item[1], reverse=True)
        seed_candidates = seed_candidates[:config.top_seed_count]
        seed_probs = stable_softmax(seed_candidates, temperature=config.seed_temperature)

        for seed_id, seed_prob in seed_probs.items():
            queue: List[Tuple[str, int, float]] = [(seed_id, 0, 1.0)]
            local_seen: Set[str] = {seed_id}
            while queue:
                current_id, hop, path_weight = queue.pop(0)
                if hop >= config.max_hops:
                    continue
                next_hop = hop + 1
                hop_scale = config.hop_decay ** hop
                for neighbor_id, edge_kind, edge_weight in index.entity_neighbors(current_id):
                    if neighbor_id in local_seen:
                        continue
                    local_seen.add(neighbor_id)
                    affinity = seed_prob * path_weight * edge_weight * hop_scale
                    if affinity > frontier_affinity.get(neighbor_id, 0.0):
                        frontier_affinity[neighbor_id] = affinity
                    frontier_sources[neighbor_id].add(seed_id)
                    queue.append((neighbor_id, next_hop, path_weight * edge_weight))

        if len(frontier_affinity) > config.frontier_pool_size:
            top_frontier = sorted(frontier_affinity.items(), key=lambda item: item[1], reverse=True)[:config.frontier_pool_size]
            keep = {entity_id for entity_id, _ in top_frontier}
            frontier_affinity = defaultdict(float, {entity_id: score for entity_id, score in top_frontier})
            frontier_sources = defaultdict(set, {entity_id: frontier_sources.get(entity_id, set()) for entity_id in keep})

    results: List[Tuple[str, float, float, float]] = []
    seen_results: Set[str] = set()
    for entity_id in selected:
        if entity_id in seen_results:
            continue
        seen_results.add(entity_id)
        bm25_value = bm25_component.get(entity_id, 0.0)
        graph_value = file_prior.get(index.entities[entity_id].file_path, 0.0) + frontier_affinity.get(entity_id, 0.0)
        combined = base_scores.get(entity_id, 0.0) + config.graph_affinity_weight * frontier_affinity.get(entity_id, 0.0)
        results.append((entity_id, combined, bm25_value, graph_value))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def assemble_entity_context(
    index: AdaptiveRepoIndex,
    ranked_entities: Sequence[Tuple[str, float, float, float]],
    config: AdaptiveCallGraphConfig,
) -> Tuple[str, List[EntityScoreRecord]]:
    by_file: Dict[str, List[Tuple[CodeEntity, float, float, float]]] = defaultdict(list)
    for entity_id, combined, bm25, graph_score in ranked_entities:
        entity = index.entities[entity_id]
        by_file[entity.file_path].append((entity, combined, bm25, graph_score))

    file_ranking: List[Tuple[str, float]] = []
    for file_path, entries in by_file.items():
        file_score = max(score for _, score, _, _ in entries) + 0.05 * sum(score for _, score, _, _ in entries[:2])
        file_ranking.append((file_path, file_score))
    file_ranking.sort(key=lambda item: item[1], reverse=True)

    max_chars = int(config.max_context_tokens * config.chars_per_token)
    total_chars = 0
    used_files = 0
    parts: List[str] = []
    included_entities: Set[str] = set()

    for file_path, _ in file_ranking:
        if used_files >= config.max_files:
            break
        file_info = index.files[file_path]
        entries = sorted(by_file[file_path], key=lambda item: (-item[1], item[0].start_line))[:config.max_entities_per_file]

        ranges: List[Tuple[int, int, str, str]] = []
        for entity, _, _, _ in entries:
            lo, hi, snippet = limited_lines(
                file_info.lines,
                entity.start_line,
                entity.end_line,
                pad=config.snippet_padding_lines,
                max_lines=config.max_entity_lines,
            )
            ranges.append((lo, hi, entity.entity_id, snippet))

        ranges.sort(key=lambda item: item[0])
        merged_blocks: List[Tuple[int, int, List[str], str]] = []
        for lo, hi, entity_id, snippet in ranges:
            if not merged_blocks:
                merged_blocks.append((lo, hi, [entity_id], snippet))
                continue
            prev_lo, prev_hi, ids, prev_snippet = merged_blocks[-1]
            if lo <= prev_hi + 1:
                new_lo = prev_lo
                new_hi = max(prev_hi, hi)
                new_ids = ids + [entity_id]
                new_text = "\n".join(file_info.lines[new_lo - 1:new_hi])
                merged_blocks[-1] = (new_lo, new_hi, new_ids, new_text)
            else:
                merged_blocks.append((lo, hi, [entity_id], snippet))

        file_parts = [f"<|file_sep|>{file_path}"]
        for lo, hi, ids, text in merged_blocks:
            labels = ", ".join(ids)
            file_parts.append(f"# lines {lo}-{hi} | entities: {labels}\n{text}")
        file_block = "\n\n".join(file_parts).strip() + "\n"

        if total_chars + len(file_block) > max_chars:
            remaining = max_chars - total_chars
            if remaining < 512:
                break
            file_block = file_block[:remaining]
        parts.append(file_block)
        total_chars += len(file_block)
        used_files += 1
        for _, _, ids, _ in merged_blocks:
            included_entities.update(ids)

    context = "\n".join(parts)

    score_records: List[EntityScoreRecord] = []
    for rank, (entity_id, combined, bm25, graph_score) in enumerate(ranked_entities):
        entity = index.entities[entity_id]
        score_records.append(
            EntityScoreRecord(
                entity_id=entity_id,
                file_path=entity.file_path,
                rank=rank,
                combined_score=float(combined),
                bm25_score=float(bm25),
                graph_score=float(graph_score),
                included_in_context=entity_id in included_entities,
            )
        )
    return context, score_records


# ---------------------------------------------------------------------------
# Repository provider / cache
# ---------------------------------------------------------------------------

class RepositoryProvider:
    def __init__(self, repos_dir: str, extraction_cache_dir: Optional[str] = None):
        self.repos_dir = Path(repos_dir)
        self.cache_dir = Path(extraction_cache_dir) if extraction_cache_dir else self.repos_dir / ".repo_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._resolved: Dict[str, str] = {}

    def resolve_root(self, sample: Dict[str, Any]) -> str:
        repo = sample.get("repo", "")
        revision = sample.get("revision", "")
        archive_name = sample.get("archive")
        key = f"{repo}@{revision}@{archive_name}"
        if key in self._resolved:
            return self._resolved[key]

        sanitized = sanitize_repo_name(repo)
        candidates = [
            self.repos_dir / f"{sanitized}-{revision}",
            self.repos_dir / f"{sanitized}_{revision}",
            self.repos_dir / sanitized / revision,
        ]
        if archive_name:
            archive_stem = Path(archive_name).stem
            candidates.extend([
                self.repos_dir / archive_stem,
                self.cache_dir / archive_stem,
            ])

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                self._resolved[key] = str(candidate.resolve())
                return self._resolved[key]

        archive_candidates: List[Path] = []
        if archive_name:
            archive_candidates.extend([
                self.repos_dir / archive_name,
                self.repos_dir / Path(archive_name).name,
            ])
        archive_candidates.extend([
            self.repos_dir / f"{sanitized}-{revision}.zip",
            self.repos_dir / f"{sanitized}_{revision}.zip",
        ])

        for archive_path in archive_candidates:
            if archive_path.exists() and archive_path.is_file():
                target_dir = self.cache_dir / archive_path.stem
                if not target_dir.exists():
                    with zipfile.ZipFile(archive_path, "r") as zf:
                        zf.extractall(target_dir)
                    extracted_children = [child for child in target_dir.iterdir() if child.is_dir()]
                    if len(extracted_children) == 1:
                        target_dir = extracted_children[0]
                self._resolved[key] = str(target_dir.resolve())
                return self._resolved[key]

        raise FileNotFoundError(
            f"Could not resolve repository for repo={repo!r}, revision={revision!r}, archive={archive_name!r} in {self.repos_dir}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_call_graph_context(
    root_dir: str,
    prefix: str,
    suffix: str,
    completion_file_path: str,
    modified_files: Optional[Sequence[str]] = None,
    config: Optional[AdaptiveCallGraphConfig] = None,
    listwise_reranker: Optional[ListwiseWindowReranker] = None,
) -> Tuple[str, List[EntityScoreRecord]]:
    cfg = config or AdaptiveCallGraphConfig()
    index = AdaptiveRepoIndex(root_dir, cfg)
    index.build()
    query = take_query_window(prefix, suffix, cfg.query_window_lines)
    ranked_entities = adaptive_entity_retrieval(
        index=index,
        query=query,
        completion_file=completion_file_path,
        modified_files=modified_files,
        config=cfg,
        listwise_reranker=listwise_reranker,
    )
    return assemble_entity_context(index, ranked_entities, cfg)


# ---------------------------------------------------------------------------
# Evaluation wrapper (compatible with evaluate.py)
# ---------------------------------------------------------------------------

def run_evaluation(
    predictions_file: str,
    data_dir: str,
    stage: str,
    language: str,
    ollama_url: str = "http://localhost:11434",
    model: str = "JetBrains/Mellum-4b-sft-python",
    temperature: float = 0.0,
) -> Optional[Dict[str, float]]:
    try:
        import evaluate as eval_module
        from evaluate import run_evaluation as evaluate_predictions
    except Exception as exc:
        print(f"Warning: could not import evaluate.py: {exc}")
        return None

    try:
        eval_module.MODEL_NAME = model
        eval_module.OLLAMA_URL = ollama_url
        eval_module.TEMPERATURE = temperature
        mean_chrf = evaluate_predictions(
            predictions_path=predictions_file,
            data_dir=data_dir,
            stage=stage,
            lang=language,
        )
        return {"mean_chrf": float(mean_chrf)}
    except Exception as exc:
        print(f"Warning: evaluation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Experiment database helpers
# ---------------------------------------------------------------------------

def build_run_id(language: str, stage: str, config: AdaptiveCallGraphConfig) -> str:
    payload = json.dumps(config.__dict__, sort_keys=True)
    digest = hashlib.md5(f"{payload}-{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
    return f"{language}-{stage}-callgraph-{digest}"


def create_experiment_run(
    run_id: str,
    timestamp: str,
    stage: str,
    language: str,
    prediction_file: str,
    num_samples: int,
    config: AdaptiveCallGraphConfig,
    status: str = "completed",
    error_message: Optional[str] = None,
):
    if ExperimentRun is None:
        return None
    return ExperimentRun(
        run_id=run_id,
        timestamp=timestamp,
        stage=stage,
        lang=language,
        retrieval_name="adaptive_call_graph",
        max_hop=config.max_hops,
        bm25_weight=config.bm25_weight,
        graph_weight=config.graph_affinity_weight,
        import_weight=config.import_weight,
        call_weight=config.call_weight,
        inheritance_weight=config.inheritance_weight,
        type_ref_weight=config.type_ref_weight,
        max_files=config.max_files,
        max_tokens=config.max_context_tokens,
        min_lines=config.min_lines,
        trim_prefix=False,
        trim_suffix=False,
        trim_lines=0,
        prediction_file=prediction_file,
        num_samples=num_samples,
        config_json=json.dumps(config.__dict__, sort_keys=True, indent=2),
        fallback_enabled=True,
        min_pool_size=config.initial_pool_size,
        query_window=config.query_window_lines,
        hop_decay=config.hop_decay,
        reverse_import_weight=config.reverse_import_weight,
        symbol_weight=config.symbol_weight,
        status=status,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

def process_tasks(
    tasks_path: str,
    repos_dir: str,
    predictions_path: str,
    stage: str,
    language: str,
    config: AdaptiveCallGraphConfig,
    evaluate_results: bool = False,
    data_dir_for_eval: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    model: str = "JetBrains/Mellum-4b-sft-python",
    temperature: float = 0.0,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    repo_provider = RepositoryProvider(repos_dir, extraction_cache_dir=config.extraction_cache_dir)
    index_cache: Dict[str, AdaptiveRepoIndex] = {}

    db = ExperimentDB(db_path) if (db_path and ExperimentDB is not None) else None

    with jsonlines.open(tasks_path, "r") as reader:
        tasks = list(reader)

    os.makedirs(Path(predictions_path).parent, exist_ok=True)

    run_id = build_run_id(language, stage, config)
    timestamp = datetime.utcnow().isoformat()

    try:
        with jsonlines.open(predictions_path, "w") as writer:
            for sample_idx, sample in enumerate(tqdm(tasks, desc="Collecting context")):
                try:
                    root_dir = repo_provider.resolve_root(sample)
                    if root_dir not in index_cache:
                        index = AdaptiveRepoIndex(root_dir, config)
                        index.build()
                        index_cache[root_dir] = index
                    else:
                        index = index_cache[root_dir]

                    query = take_query_window(sample.get("prefix", ""), sample.get("suffix", ""), config.query_window_lines)
                    ranked_entities = adaptive_entity_retrieval(
                        index=index,
                        query=query,
                        completion_file=sample.get("path"),
                        modified_files=sample.get("modified", []),
                        config=config,
                    )
                    context, score_records = assemble_entity_context(index, ranked_entities, config)
                except Exception as exc:
                    print(f"Warning: failed to process sample {sample.get('id', sample_idx)}: {exc}")
                    context = ""
                    score_records = []

                writer.write({"context": context})

                if db is not None and score_records:
                    payloads: List[Dict[str, Any]] = []
                    for record in score_records:
                        payloads.append({
                            "retrieved_file": f"{record.file_path}::{record.entity_id}",
                            "rank": record.rank,
                            "bm25_score": record.bm25_score,
                            "graph_score": record.graph_score,
                            "combined_score": record.combined_score,
                            "included_in_context": record.included_in_context,
                        })
                    db.log_retrieval_scores(
                        run_id=run_id,
                        sample_id=str(sample.get("id", sample_idx)),
                        repo=sample.get("repo", ""),
                        completion_file=sample.get("path", ""),
                        scores=payloads,
                    )

        print(f"Predictions written to {predictions_path}")

        if db is not None:
            run = create_experiment_run(
                run_id=run_id,
                timestamp=timestamp,
                stage=stage,
                language=language,
                prediction_file=predictions_path,
                num_samples=len(tasks),
                config=config,
            )
            if run is not None:
                db.log_run(run)

        eval_results = None
        if evaluate_results:
            if not data_dir_for_eval:
                raise ValueError("data_dir_for_eval must be provided when evaluate_results=True")
            eval_results = run_evaluation(
                predictions_file=predictions_path,
                data_dir=data_dir_for_eval,
                stage=stage,
                language=language,
                ollama_url=ollama_url,
                model=model,
                temperature=temperature,
            )
            if db is not None and eval_results is not None:
                db.update_metrics(run_id, chrf=eval_results.get("mean_chrf"))
        return eval_results

    except Exception as exc:
        if db is not None:
            run = create_experiment_run(
                run_id=run_id,
                timestamp=timestamp,
                stage=stage,
                language=language,
                prediction_file=predictions_path,
                num_samples=0,
                config=config,
                status="failed",
                error_message=str(exc),
            )
            if run is not None:
                db.log_run(run)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive call-graph retrieval for code completion context collection")

    parser.add_argument("--tasks", required=True, help="Path to the tasks JSONL file")
    parser.add_argument("--repos-dir", required=True, help="Directory containing extracted repositories or repository zip files")
    parser.add_argument("--predictions", required=True, help="Path to the output predictions JSONL file")
    parser.add_argument("--stage", default="practice", help="Dataset stage for optional evaluation")
    parser.add_argument("--lang", default="python", help="Programming language for optional evaluation")

    parser.add_argument("--bm25-topk", type=int, default=200)
    parser.add_argument("--initial-pool-size", type=int, default=80)
    parser.add_argument("--frontier-pool-size", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--top-seed-count", type=int, default=8)
    parser.add_argument("--max-selected-entities", type=int, default=48)
    parser.add_argument("--max-entities-per-file", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=6)
    parser.add_argument("--max-context-tokens", type=int, default=7000)
    parser.add_argument("--max-entity-lines", type=int, default=80)
    parser.add_argument("--query-window-lines", type=int, default=120)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--hop-decay", type=float, default=0.65)

    parser.add_argument("--bm25-weight", type=float, default=0.45)
    parser.add_argument("--symbol-weight", type=float, default=0.18)
    parser.add_argument("--file-prior-weight", type=float, default=0.12)
    parser.add_argument("--graph-affinity-weight", type=float, default=0.25)
    parser.add_argument("--modified-bonus", type=float, default=0.10)
    parser.add_argument("--same-file-bonus", type=float, default=0.06)

    parser.add_argument("--import-weight", type=float, default=0.30)
    parser.add_argument("--reverse-import-weight", type=float, default=0.12)
    parser.add_argument("--call-weight", type=float, default=1.0)
    parser.add_argument("--reverse-call-weight", type=float, default=0.75)
    parser.add_argument("--inheritance-weight", type=float, default=0.55)
    parser.add_argument("--type-ref-weight", type=float, default=0.40)
    parser.add_argument("--sibling-weight", type=float, default=0.10)

    parser.add_argument("--evaluate", action="store_true", help="Run the provided evaluate.py after writing predictions")
    parser.add_argument("--data-dir-for-eval", default=None, help="Directory containing the task and answer files expected by evaluate.py")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="JetBrains/Mellum-4b-sft-python")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--db", default=None, help="Path to the experiment SQLite database")
    parser.add_argument("--cache-dir", default=None, help="Directory used to cache extracted repository archives")

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AdaptiveCallGraphConfig:
    return AdaptiveCallGraphConfig(
        query_window_lines=args.query_window_lines,
        bm25_topk=args.bm25_topk,
        initial_pool_size=args.initial_pool_size,
        frontier_pool_size=args.frontier_pool_size,
        batch_size=args.batch_size,
        top_seed_count=args.top_seed_count,
        max_selected_entities=args.max_selected_entities,
        max_entities_per_file=args.max_entities_per_file,
        max_files=args.max_files,
        max_context_tokens=args.max_context_tokens,
        max_entity_lines=args.max_entity_lines,
        max_hops=args.max_hops,
        hop_decay=args.hop_decay,
        bm25_weight=args.bm25_weight,
        symbol_weight=args.symbol_weight,
        file_prior_weight=args.file_prior_weight,
        graph_affinity_weight=args.graph_affinity_weight,
        modified_bonus=args.modified_bonus,
        same_file_bonus=args.same_file_bonus,
        import_weight=args.import_weight,
        reverse_import_weight=args.reverse_import_weight,
        call_weight=args.call_weight,
        reverse_call_weight=args.reverse_call_weight,
        inheritance_weight=args.inheritance_weight,
        type_ref_weight=args.type_ref_weight,
        sibling_weight=args.sibling_weight,
        extraction_cache_dir=args.cache_dir,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    print(json.dumps(config.__dict__, indent=2, sort_keys=True))

    process_tasks(
        tasks_path=args.tasks,
        repos_dir=args.repos_dir,
        predictions_path=args.predictions,
        stage=args.stage,
        language=args.lang,
        config=config,
        evaluate_results=args.evaluate,
        data_dir_for_eval=args.data_dir_for_eval,
        ollama_url=args.ollama_url,
        model=args.model,
        temperature=args.temperature,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
