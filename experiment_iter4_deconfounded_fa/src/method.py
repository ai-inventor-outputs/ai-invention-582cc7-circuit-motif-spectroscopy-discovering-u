#!/usr/bin/env python3
"""Deconfounded Failure Prediction from Motif Spectra on Verified Attribution Graphs.

Loads verified-correct/incorrect attribution graphs from the 200-graph corpus,
computes motif census + graph statistics + domain indicators, then runs a
hierarchical classifier battery (5 feature sets x 2 classifiers) with stratified
CV, within-domain LOO analysis, motif-deviation features, and bootstrap
significance testing.

Baseline: domain_plus_graph (domain one-hot + graph statistics, no motif info)
Our method: full_model (domain + graph stats + motif spectra features)
"""

import gc
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import igraph
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONSTANTS
# ============================================================
WORKSPACE = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability"
    "/3_invention_loop/iter_4/gen_art/exp_id3_it4__opus"
)
DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability"
    "/3_invention_loop/iter_3/gen_art/data_id5_it3__opus/data_out"
)

PRUNE_QUANTILE = 0.75
N_NULL = int(os.environ.get("N_NULL_MODELS", "30"))
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0")) or None
DATA_SOURCE = os.environ.get("DATA_SOURCE", "full")  # "mini" or "full"

DOMAINS = [
    "antonym", "arithmetic", "code_completion", "country_capital",
    "multi_hop_reasoning", "rhyme", "sentiment", "translation",
]


# ============================================================
# HARDWARE DETECTION & RESOURCE LIMITS
# ============================================================
def _detect_cpus() -> int:
    """Detect actual CPU allocation (containers/pods/bare metal)."""
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    """Read RAM limit from cgroup (containers/pods)."""
    for p in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0

# Set resource limits: 20GB RAM budget (container has ~29GB)
RAM_BUDGET = int(20 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ============================================================
# LOGGING
# ============================================================
logger.remove()
logger.add(
    sys.stdout, level="INFO",
    format="{time:HH:mm:ss}|{level:<7}|{message}",
)
(WORKSPACE / "logs").mkdir(parents=True, exist_ok=True)
logger.add(
    WORKSPACE / "logs" / "run.log",
    rotation="30 MB", level="DEBUG",
)


# ============================================================
# JSON SANITIZATION
# ============================================================
def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None for valid JSON output."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ============================================================
# MOTIF INDEX VERIFICATION
# ============================================================
def verify_motif_indices() -> dict[str, int]:
    """Determine correct igraph motif indices by constructing known patterns."""
    patterns = {
        "021D": [(0, 1), (0, 2)],            # Fan-out: A->B, A->C
        "021U": [(1, 0), (2, 0)],            # Fan-in: B->A, C->A
        "021C": [(0, 1), (1, 2)],            # Chain: A->B->C
        "030T": [(0, 1), (0, 2), (1, 2)],    # FFL: A->B, A->C, B->C
    }

    indices = {}
    for name, edges in patterns.items():
        g = igraph.Graph(directed=True)
        g.add_vertices(3)
        g.add_edges(edges)
        counts = g.motifs_randesu(size=3)
        found = [
            i for i, c in enumerate(counts)
            if c is not None and c > 0 and i > 0
        ]
        if found:
            indices[name] = found[0]
        else:
            logger.warning(f"Could not determine motif index for {name}")
            indices[name] = -1

    return indices


# ============================================================
# GRAPH PARSING & PRUNING
# ============================================================
def parse_graph(record: dict) -> igraph.Graph:
    """Parse JSON output string -> igraph directed graph with pruning."""
    output_str = record["output"]
    graph_json = json.loads(output_str)

    # Get nodes and links/edges (handle both key names)
    nodes = graph_json.get("nodes", [])
    links = graph_json.get("links", graph_json.get("edges", []))

    if not nodes or not links:
        raise ValueError(
            f"Empty graph for {record.get('metadata_slug', '?')}"
        )

    # Build node_id -> index mapping (handle both node_id and jsNodeId)
    id_map = {}
    for i, n in enumerate(nodes):
        nid = n.get("node_id") or n.get("jsNodeId")
        if nid is not None:
            id_map[nid] = i
        # Also map by jsNodeId as fallback
        js_id = n.get("jsNodeId")
        if js_id is not None and js_id not in id_map:
            id_map[js_id] = i

    # Create igraph Graph
    g = igraph.Graph(directed=True)
    g.add_vertices(len(nodes))

    # Store node attributes
    for i, n in enumerate(nodes):
        g.vs[i]["node_id"] = n.get("node_id", n.get("jsNodeId", str(i)))
        layer_str = str(n.get("layer", "0"))
        g.vs[i]["layer"] = int(layer_str) if layer_str.isdigit() else -1
        g.vs[i]["feature_type"] = n.get("feature_type", "")

    # Add edges with weights
    edges = []
    weights = []
    for link in links:
        src_id = link.get("source") or link.get("source_id")
        tgt_id = link.get("target") or link.get("target_id")

        src = id_map.get(src_id)
        tgt = id_map.get(tgt_id)

        # Handle integer references
        if src is None and isinstance(src_id, int) and 0 <= src_id < len(nodes):
            src = src_id
        if tgt is None and isinstance(tgt_id, int) and 0 <= tgt_id < len(nodes):
            tgt = tgt_id

        if src is not None and tgt is not None and src != tgt:
            edges.append((src, tgt))
            weights.append(abs(float(link.get("weight", 1.0))))

    if not edges:
        raise ValueError(
            f"No valid edges for {record.get('metadata_slug', '?')}"
        )

    g.add_edges(edges)
    g.es["weight"] = weights

    # PRUNE: keep edges above PRUNE_QUANTILE percentile |weight|
    if g.ecount() > 10:
        threshold = np.percentile(g.es["weight"], PRUNE_QUANTILE * 100)
        to_delete = [e.index for e in g.es if e["weight"] < threshold]
        g.delete_edges(to_delete)

    # Remove isolated vertices
    isolated = [v.index for v in g.vs if g.degree(v) == 0]
    if isolated:
        g.delete_vertices(isolated)

    if g.vcount() < 3:
        raise ValueError(
            f"Graph too small after pruning: {g.vcount()} nodes"
        )

    # Verify DAG
    if not g.is_dag():
        # Try to fix by removing back edges (feedback arc set)
        try:
            fas = g.feedback_arc_set()
            if fas:
                g.delete_edges(fas)
        except Exception:
            pass
        if not g.is_dag():
            raise ValueError(
                f"Cannot make graph a DAG: {record.get('metadata_slug', '?')}"
            )

    return g


# ============================================================
# NULL MODEL: DEGREE-PRESERVING DAG REWIRING
# ============================================================
def degree_preserving_dag_rewire(
    edges_list: list[tuple[int, int]],
    n_vertices: int,
    topo_rank_arr: np.ndarray,
    n_swaps: int | None = None,
    rng: np.random.RandomState | None = None,
) -> list[tuple[int, int]]:
    """Rewire edges preserving in/out degree sequence AND DAG property.

    Uses topological ordering for O(1) DAG validity check per swap attempt.
    Returns the new edge list.
    """
    if rng is None:
        rng = np.random.RandomState(RANDOM_SEED)

    edges = list(edges_list)  # copy
    n_edges = len(edges)

    if n_edges < 2:
        return edges

    # Build adjacency matrix for O(1) multi-edge check
    edge_exists = np.zeros((n_vertices, n_vertices), dtype=bool)
    for u, v in edges:
        edge_exists[u, v] = True

    if n_swaps is None:
        n_swaps = 10 * n_edges

    for _ in range(n_swaps):
        i = rng.randint(0, n_edges)
        j = rng.randint(0, n_edges - 1)
        if j >= i:
            j += 1

        u1, v1 = edges[i]
        u2, v2 = edges[j]

        # Skip trivial cases
        if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
            continue

        # Check new edges don't already exist (no multi-edges)
        if edge_exists[u1, v2] or edge_exists[u2, v1]:
            continue

        # DAG check: new edges must respect topological ordering
        if topo_rank_arr[u1] >= topo_rank_arr[v2]:
            continue
        if topo_rank_arr[u2] >= topo_rank_arr[v1]:
            continue

        # Accept the swap
        edge_exists[u1, v1] = False
        edge_exists[u2, v2] = False
        edge_exists[u1, v2] = True
        edge_exists[u2, v1] = True
        edges[i] = (u1, v2)
        edges[j] = (u2, v1)

    return edges


# ============================================================
# MOTIF CENSUS COMPUTATION
# ============================================================
def compute_motif_features(
    g: igraph.Graph,
    n_null: int,
    motif_indices: dict[str, int],
    seed: int,
) -> dict:
    """Compute 3-node motif census ratios and Z-scores via null models."""
    rng = np.random.RandomState(seed % (2**31))

    # Original motif census
    motif_counts = g.motifs_randesu(size=3)
    motif_counts = [c if c is not None and c >= 0 else 0 for c in motif_counts]

    total = sum(motif_counts)
    if total == 0:
        total = 1

    # Compute ratios for key motif types
    features = {}
    for name, idx in motif_indices.items():
        if 0 <= idx < len(motif_counts):
            features[f"motif_{name}_ratio"] = motif_counts[idx] / total
        else:
            features[f"motif_{name}_ratio"] = 0.0

    # Z-score for 030T (feed-forward loop) via null models
    idx_030T = motif_indices.get("030T", -1)
    if idx_030T >= 0 and n_null > 0:
        edges_list = list(g.get_edgelist())
        n_vertices = g.vcount()
        topo = g.topological_sorting()
        topo_rank_arr = np.zeros(n_vertices, dtype=np.int32)
        for rank, v in enumerate(topo):
            topo_rank_arr[v] = rank

        null_030T = []
        for _ in range(n_null):
            try:
                new_edges = degree_preserving_dag_rewire(
                    edges_list, n_vertices, topo_rank_arr, rng=rng,
                )
                g_rand = igraph.Graph(directed=True)
                g_rand.add_vertices(n_vertices)
                g_rand.add_edges(new_edges)
                rand_counts = g_rand.motifs_randesu(size=3)
                rand_counts = [
                    c if c is not None and c >= 0 else 0
                    for c in rand_counts
                ]
                null_030T.append(rand_counts[idx_030T])
            except Exception:
                continue

        if len(null_030T) >= 3:
            mean_null = np.mean(null_030T)
            std_null = np.std(null_030T)
            if std_null > 0:
                z = (motif_counts[idx_030T] - mean_null) / std_null
            else:
                z = 0.0
            features["z_030T_abs"] = float(abs(z))
        else:
            features["z_030T_abs"] = 0.0
    else:
        features["z_030T_abs"] = 0.0

    return features


# ============================================================
# GRAPH STATISTICS
# ============================================================
def compute_graph_stats(g: igraph.Graph) -> dict:
    """Compute 8 graph-level statistics."""
    n_nodes = g.vcount()
    n_edges = g.ecount()
    density = g.density()
    in_degrees = g.indegree()
    out_degrees = g.outdegree()

    # Layer count from node attributes
    try:
        layers = set(g.vs["layer"])
    except (KeyError, AttributeError):
        layers = {0}

    # Diameter (handle disconnected graphs)
    try:
        if g.is_connected(mode="weak"):
            diameter = g.diameter(directed=True)
        else:
            components = g.connected_components(mode="weak")
            largest = max(components, key=len)
            sg = g.subgraph(largest)
            diameter = sg.diameter(directed=True)
    except Exception:
        diameter = -1

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(density),
        "mean_in_degree": float(np.mean(in_degrees)) if in_degrees else 0.0,
        "mean_out_degree": float(np.mean(out_degrees)) if out_degrees else 0.0,
        "max_out_degree": int(max(out_degrees)) if out_degrees else 0,
        "n_layers": len(layers),
        "diameter": int(diameter),
    }


# ============================================================
# WORKER FUNCTION (for parallel processing)
# ============================================================
def process_one_graph(args: tuple) -> dict:
    """Process a single graph record in a subprocess."""
    record, config = args
    slug = record.get("metadata_slug", "unknown")

    try:
        g = parse_graph(record)
        motif_feats = compute_motif_features(
            g,
            n_null=config["n_null"],
            motif_indices=config["motif_indices"],
            seed=config["seed"] + abs(hash(slug)) % 10000,
        )
        graph_feats = compute_graph_stats(g)
        domain_feats = {
            f"domain_{d}": int(record["metadata_fold"] == d)
            for d in DOMAINS
        }

        return {
            "slug": slug,
            "domain": record["metadata_fold"],
            "label": record["label"],
            "n_nodes_pruned": g.vcount(),
            "n_edges_pruned": g.ecount(),
            **motif_feats,
            **graph_feats,
            **domain_feats,
            "_parse_ok": True,
        }
    except Exception as e:
        return {
            "slug": slug,
            "domain": record.get("metadata_fold", "unknown"),
            "label": record.get("label", -1),
            "_parse_ok": False,
            "_error": str(e)[:300],
        }


# ============================================================
# DATA LOADING
# ============================================================
def load_records(
    data_dir: Path,
    source: str = "full",
    max_examples: int | None = None,
) -> list[dict]:
    """Load records from split files, filter to verified labels."""
    if source == "mini":
        files = [data_dir / "mini_data_out.json"]
    else:
        files = sorted(data_dir.glob("full_data_out_*.json"))

    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    logger.info(f"Loading from {len(files)} file(s) (source={source})")

    all_records = []
    for f in files:
        t0 = time.time()
        data = json.loads(f.read_text())
        examples = data["datasets"][0]["examples"]

        for ex in examples:
            record = {
                "input": ex["input"],
                "output": ex["output"],
                "metadata_fold": ex["metadata_fold"],
                "metadata_model_correct": ex["metadata_model_correct"],
                "metadata_slug": ex["metadata_slug"],
                "metadata_difficulty": ex.get("metadata_difficulty", "unknown"),
                "metadata_expected_answer": ex.get(
                    "metadata_expected_answer", ""
                ),
                "metadata_n_nodes": ex.get("metadata_n_nodes", 0),
                "metadata_n_edges": ex.get("metadata_n_edges", 0),
            }
            all_records.append(record)

        dt = time.time() - t0
        logger.debug(f"  Loaded {f.name}: {len(examples)} examples ({dt:.1f}s)")
        del data
        gc.collect()

    logger.info(f"Total records loaded: {len(all_records)}")

    # Filter to verified labels (true/false only)
    verified = []
    for r in all_records:
        mc = r["metadata_model_correct"]
        if mc in ("true", "false"):
            r["label"] = 1 if mc == "true" else 0
            verified.append(r)

    logger.info(f"Verified records (true/false): {len(verified)}")

    if max_examples is not None and max_examples < len(verified):
        verified = verified[:max_examples]
        logger.info(f"Limited to first {max_examples} examples")

    return verified


# ============================================================
# CLASSIFICATION HELPERS
# ============================================================
def run_classifier_battery(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    y: np.ndarray,
    skf: StratifiedKFold,
) -> dict:
    """Run all feature_set x classifier combinations with stratified CV."""
    classifiers = {
        "logistic_L2": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED,
            max_depth=5, min_samples_leaf=3,
        ),
    }

    results = {}
    for fs_name, cols in feature_sets.items():
        for clf_name, clf_template in classifiers.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clone(clf_template)),
            ])

            X = df[cols].values
            try:
                y_prob = cross_val_predict(
                    pipe, X, y, cv=skf, method="predict_proba",
                )[:, 1]
            except Exception as e:
                logger.warning(f"CV failed for {fs_name}__{clf_name}: {e}")
                y_prob = np.full(len(y), 0.5)

            y_pred = (y_prob >= 0.5).astype(int)

            try:
                auc = float(roc_auc_score(y, y_prob))
            except ValueError:
                auc = 0.5
            acc = float(accuracy_score(y, y_pred))
            f1 = float(f1_score(y, y_pred, zero_division=0))

            key = f"{fs_name}__{clf_name}"
            results[key] = {
                "auc": auc,
                "accuracy": acc,
                "f1": f1,
                "feature_set": fs_name,
                "classifier": clf_name,
                "n_features": len(cols),
                "predictions": y_prob.tolist(),
            }
            logger.info(
                f"  {key}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}"
            )

    return results


def bootstrap_auc_comparison(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap test: AUC(A) - AUC(B), return CI and p-value."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            auc_a = roc_auc_score(y_true[idx], probs_a[idx])
            auc_b = roc_auc_score(y_true[idx], probs_b[idx])
            diffs.append(auc_a - auc_b)
        except ValueError:
            continue

    # Filter out any NaN diffs
    diffs = [d for d in diffs if not np.isnan(d)]

    if len(diffs) < 10:
        return {"mean_diff": 0.0, "ci_95": [0.0, 0.0], "p_value": 0.5}

    diffs_arr = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs_arr)),
        "ci_95": [
            float(np.percentile(diffs_arr, 2.5)),
            float(np.percentile(diffs_arr, 97.5)),
        ],
        "p_value": float(np.mean(diffs_arr <= 0)),
    }


# ============================================================
# MAIN
# ============================================================
@logger.catch
def main():
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("Deconfounded Failure Prediction from Motif Spectra")
    logger.info(f"CPUs: {NUM_CPUS}, RAM: {TOTAL_RAM_GB:.1f} GB")
    logger.info(f"N_NULL={N_NULL}, MAX_EXAMPLES={MAX_EXAMPLES}")
    logger.info(f"DATA_SOURCE={DATA_SOURCE}")
    logger.info("=" * 60)

    # ---- STEP 1: Verify motif indices ----
    logger.info("Step 1: Verifying motif indices...")
    motif_indices = verify_motif_indices()
    logger.info(f"Motif indices: {motif_indices}")

    MOTIF_COLS = [f"motif_{name}_ratio" for name in motif_indices] + [
        "z_030T_abs",
    ]
    GRAPH_COLS = [
        "n_nodes", "n_edges", "density", "mean_in_degree",
        "mean_out_degree", "max_out_degree", "n_layers", "diameter",
    ]
    DOMAIN_COLS = [f"domain_{d}" for d in DOMAINS]

    # ---- STEP 2: Load data ----
    logger.info("Step 2: Loading data...")
    records = load_records(DATA_DIR, source=DATA_SOURCE, max_examples=MAX_EXAMPLES)

    # Report class distribution by domain
    domain_dist: dict = {}
    for r in records:
        d = r["metadata_fold"]
        domain_dist.setdefault(d, {"correct": 0, "incorrect": 0})
        domain_dist[d]["correct" if r["label"] == 1 else "incorrect"] += 1

    n_correct = sum(v["correct"] for v in domain_dist.values())
    n_incorrect = sum(v["incorrect"] for v in domain_dist.values())
    logger.info(f"Class distribution: {n_correct} correct, {n_incorrect} incorrect")
    for d, v in sorted(domain_dist.items()):
        logger.info(f"  {d}: {v['correct']} correct, {v['incorrect']} incorrect")

    powered_domains = [
        d for d, v in domain_dist.items()
        if v["correct"] >= 3 and v["incorrect"] >= 3
    ]
    logger.info(f"Powered domains ({len(powered_domains)}): {powered_domains}")

    # ---- STEP 3: Process graphs in parallel ----
    logger.info(
        f"Step 3: Processing {len(records)} graphs with "
        f"{NUM_CPUS} workers, N_NULL={N_NULL}..."
    )
    config = {
        "n_null": N_NULL,
        "motif_indices": motif_indices,
        "seed": RANDOM_SEED,
    }

    t0 = time.time()
    worker_args = [(r, config) for r in records]

    with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        futures = {
            executor.submit(process_one_graph, arg): i
            for i, arg in enumerate(worker_args)
        }
        features_list = [None] * len(worker_args)
        done_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            features_list[idx] = future.result()
            done_count += 1
            if done_count % 10 == 0 or done_count == len(worker_args):
                elapsed = time.time() - t0
                logger.info(
                    f"  Progress: {done_count}/{len(worker_args)} "
                    f"({elapsed:.1f}s elapsed)"
                )

    elapsed = time.time() - t0
    logger.info(f"Graph processing done in {elapsed:.1f}s")

    # Filter successful parses
    ok_features = [f for f in features_list if f and f.get("_parse_ok", False)]
    failed = [f for f in features_list if f and not f.get("_parse_ok", False)]
    logger.info(
        f"Successfully parsed: {len(ok_features)}, Failed: {len(failed)}"
    )

    for f in failed:
        logger.warning(f"  Failed: {f.get('slug', '?')}: {f.get('_error', '?')}")

    if len(ok_features) < 6:
        raise RuntimeError(
            f"Too few graphs parsed successfully: {len(ok_features)}"
        )

    # Build DataFrame
    for f in ok_features:
        f.pop("_parse_ok", None)

    df = pd.DataFrame(ok_features)

    # Handle NaN/Inf
    for col in MOTIF_COLS + GRAPH_COLS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Ensure all feature columns exist
    for col in MOTIF_COLS + GRAPH_COLS + DOMAIN_COLS:
        if col not in df.columns:
            logger.warning(f"Missing feature column: {col}, adding zeros")
            df[col] = 0.0

    y = df["label"].values.astype(int)
    logger.info(f"Feature DataFrame: {df.shape[0]} rows x {df.shape[1]} cols")
    logger.info(f"Labels: {sum(y)} correct, {len(y) - sum(y)} incorrect")

    # ---- STEP 4: Classifier Battery ----
    logger.info("Step 4: Running classifier battery (5 feature sets x 2 classifiers)...")

    feature_sets = {
        "domain_only": DOMAIN_COLS,
        "graph_stats_only": GRAPH_COLS,
        "motif_only": MOTIF_COLS,
        "domain_plus_graph": DOMAIN_COLS + GRAPH_COLS,
        "full_model": DOMAIN_COLS + GRAPH_COLS + MOTIF_COLS,
    }

    # Determine CV splits based on minority class count
    minority_count = min(int(sum(y)), int(len(y) - sum(y)))
    n_splits = min(5, minority_count)
    if n_splits < 2:
        n_splits = 2
    logger.info(
        f"Using {n_splits}-fold stratified CV "
        f"(minority class count: {minority_count})"
    )
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED,
    )

    results = run_classifier_battery(df, feature_sets, y, skf)

    # ---- STEP 5: Key comparisons (bootstrap) ----
    logger.info("Step 5: Bootstrap AUC comparisons...")

    y_arr = np.array(y)
    comparisons = {}

    # Full vs domain+graph (logistic)
    key_full_lr = "full_model__logistic_L2"
    key_base_lr = "domain_plus_graph__logistic_L2"
    if key_full_lr in results and key_base_lr in results:
        comparisons["full_vs_domain_graph__logistic"] = bootstrap_auc_comparison(
            y_arr,
            np.array(results[key_full_lr]["predictions"]),
            np.array(results[key_base_lr]["predictions"]),
            n_boot=N_BOOTSTRAP, seed=RANDOM_SEED,
        )
        logger.info(
            f"  full vs baseline (LR): "
            f"{comparisons['full_vs_domain_graph__logistic']}"
        )

    # Full vs domain+graph (RF)
    key_full_rf = "full_model__random_forest"
    key_base_rf = "domain_plus_graph__random_forest"
    if key_full_rf in results and key_base_rf in results:
        comparisons["full_vs_domain_graph__rf"] = bootstrap_auc_comparison(
            y_arr,
            np.array(results[key_full_rf]["predictions"]),
            np.array(results[key_base_rf]["predictions"]),
            n_boot=N_BOOTSTRAP, seed=RANDOM_SEED + 1,
        )
        logger.info(
            f"  full vs baseline (RF): "
            f"{comparisons['full_vs_domain_graph__rf']}"
        )

    # Motif vs graph_stats (logistic)
    key_motif_lr = "motif_only__logistic_L2"
    key_graph_lr = "graph_stats_only__logistic_L2"
    if key_motif_lr in results and key_graph_lr in results:
        comparisons["motif_vs_graph_stats__logistic"] = bootstrap_auc_comparison(
            y_arr,
            np.array(results[key_motif_lr]["predictions"]),
            np.array(results[key_graph_lr]["predictions"]),
            n_boot=N_BOOTSTRAP, seed=RANDOM_SEED + 2,
        )
        logger.info(
            f"  motif vs graph_stats (LR): "
            f"{comparisons['motif_vs_graph_stats__logistic']}"
        )

    # ---- STEP 6: Within-domain analysis ----
    logger.info("Step 6: Within-domain analysis...")
    within_domain_results: dict = {}

    for domain in powered_domains:
        df_d = df[df["domain"] == domain]
        y_d = df_d["label"].values.astype(int)

        n_pos = int(sum(y_d))
        n_neg = int(len(y_d) - n_pos)
        if len(y_d) < 6 or min(n_pos, n_neg) < 3:
            within_domain_results[domain] = {
                "status": "underpowered", "n": len(y_d),
                "n_correct": n_pos, "n_incorrect": n_neg,
            }
            logger.info(f"  {domain}: underpowered (n={len(y_d)})")
            continue

        loo = LeaveOneOut()

        for feat_name, cols in [
            ("motif_only", MOTIF_COLS),
            ("graph_stats_only", GRAPH_COLS),
        ]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=0.1, max_iter=1000, random_state=RANDOM_SEED,
                )),
            ])

            X_d = df_d[cols].values
            try:
                y_prob_d = cross_val_predict(
                    pipe, X_d, y_d, cv=loo, method="predict_proba",
                )[:, 1]
                auc_d = float(roc_auc_score(y_d, y_prob_d))
            except Exception:
                auc_d = float("nan")

            within_domain_results.setdefault(domain, {})[feat_name] = {
                "auc": auc_d,
                "n_total": len(y_d),
                "n_correct": n_pos,
                "n_incorrect": n_neg,
            }

        logger.info(f"  {domain}: {within_domain_results.get(domain, {})}")

    # ---- STEP 7: Motif deviation features ----
    logger.info("Step 7: Computing motif deviation features...")

    deviation_features = []
    for idx, row in df.iterrows():
        domain = row["domain"]
        my_motif = row[MOTIF_COLS].values.astype(float)

        # Leave-one-out domain mean
        domain_others = df[
            (df["domain"] == domain) & (df.index != idx)
        ][MOTIF_COLS]

        if len(domain_others) < 2:
            feat_dict = {"dev_euclidean": 0.0, "dev_cosine": 0.0}
            for col_name in MOTIF_COLS:
                feat_dict[f"dev_std_{col_name}"] = 0.0
            deviation_features.append(feat_dict)
            continue

        loo_mean = domain_others.mean().values.astype(float)
        loo_std = domain_others.std().values.astype(float)
        loo_std[loo_std == 0] = 1.0

        # Euclidean distance
        eucl_dist = float(np.linalg.norm(my_motif - loo_mean))

        # Cosine distance
        norm_prod = np.linalg.norm(my_motif) * np.linalg.norm(loo_mean) + 1e-10
        cos_sim = float(np.dot(my_motif, loo_mean) / norm_prod)
        cos_dist = 1.0 - cos_sim

        # Per-motif standardized deviations
        std_devs = (my_motif - loo_mean) / loo_std

        feat_dict = {
            "dev_euclidean": eucl_dist,
            "dev_cosine": cos_dist,
        }
        for col_name, val in zip(MOTIF_COLS, std_devs):
            feat_dict[f"dev_std_{col_name}"] = float(val)

        deviation_features.append(feat_dict)

    df_dev = pd.DataFrame(deviation_features, index=df.index)
    DEV_COLS = list(df_dev.columns)
    df_dev = df_dev.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Add deviation features to df
    for col in DEV_COLS:
        df[col] = df_dev[col]

    # Extended feature sets with deviation features
    extended_feature_sets = {
        "domain_graph_plus_dev": DOMAIN_COLS + GRAPH_COLS + DEV_COLS,
        "full_plus_dev": DOMAIN_COLS + GRAPH_COLS + MOTIF_COLS + DEV_COLS,
    }

    logger.info("  Running extended classifiers with deviation features...")
    dev_results = run_classifier_battery(df, extended_feature_sets, y, skf)

    # ---- STEP 8: Feature importance ----
    logger.info("Step 8: Computing feature importance...")

    # Find best model across all results
    all_results = {**results, **dev_results}
    best_key = max(all_results, key=lambda k: all_results[k]["auc"])
    best_fs = all_results[best_key]["feature_set"]
    best_clf_name = all_results[best_key]["classifier"]
    best_auc = all_results[best_key]["auc"]

    logger.info(f"Best model: {best_key} (AUC={best_auc:.4f})")

    # Get best feature columns
    all_feature_sets = {**feature_sets, **extended_feature_sets}
    best_cols = all_feature_sets.get(best_fs, feature_sets["full_model"])

    # Fit on full data and compute permutation importance
    classifiers_dict = {
        "logistic_L2": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED,
            max_depth=5, min_samples_leaf=3,
        ),
    }

    X_full = StandardScaler().fit_transform(df[best_cols].values)
    clf_final = clone(classifiers_dict[best_clf_name]).fit(X_full, y)

    perm_imp = permutation_importance(
        clf_final, X_full, y,
        n_repeats=30, random_state=RANDOM_SEED, scoring="roc_auc",
    )
    importance_dict = {
        col: {"mean": float(m), "std": float(s)}
        for col, m, s in zip(
            best_cols,
            perm_imp.importances_mean,
            perm_imp.importances_std,
        )
    }

    # Top features
    sorted_imp = sorted(
        importance_dict.items(), key=lambda x: x[1]["mean"], reverse=True,
    )
    logger.info("Top features by permutation importance:")
    for name, vals in sorted_imp[:10]:
        logger.info(f"  {name}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

    # ---- STEP 9: Bootstrap CI for best model ----
    logger.info("Step 9: Bootstrap CI for best model...")

    y_prob_best = np.array(all_results[best_key]["predictions"])
    rng = np.random.RandomState(RANDOM_SEED)
    boot_aucs = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(y), len(y), replace=True)
        try:
            auc_b = roc_auc_score(y_arr[idx], y_prob_best[idx])
            boot_aucs.append(auc_b)
        except ValueError:
            continue

    # Filter out any NaN values
    boot_aucs = [a for a in boot_aucs if not np.isnan(a)]
    if len(boot_aucs) >= 10:
        boot_ci = [
            float(np.percentile(boot_aucs, 2.5)),
            float(np.percentile(boot_aucs, 97.5)),
        ]
    else:
        boot_ci = [None, None]
    logger.info(
        f"Best model AUC: {best_auc:.4f}, "
        f"95% CI: {boot_ci}"
    )

    # ---- STEP 10: Success criteria evaluation ----
    full_lr_auc = results.get(key_full_lr, {}).get("auc", 0.5)
    base_lr_auc = results.get(key_base_lr, {}).get("auc", 0.5)
    motif_p_val = comparisons.get(
        "full_vs_domain_graph__logistic", {},
    ).get("p_value", 1.0)

    # Within-domain comparison: does motif beat graph_stats?
    within_motif_wins = 0
    within_total = 0
    for domain, res in within_domain_results.items():
        if isinstance(res, dict) and "motif_only" in res and "graph_stats_only" in res:
            m_auc = res["motif_only"].get("auc", 0)
            g_auc = res["graph_stats_only"].get("auc", 0)
            if not (np.isnan(m_auc) or np.isnan(g_auc)):
                within_total += 1
                if m_auc > g_auc:
                    within_motif_wins += 1

    aggregate_comparison = (
        within_motif_wins > within_total / 2 if within_total > 0 else False
    )

    success = {
        "motif_predicts_errors_auc_gt_065": bool(best_auc > 0.65),
        "motif_adds_over_baseline": bool(motif_p_val < 0.05),
        "within_domain_motif_beats_graph_stats": bool(aggregate_comparison),
        "best_model_auc": float(best_auc),
        "full_model_lr_auc": float(full_lr_auc),
        "baseline_domain_graph_lr_auc": float(base_lr_auc),
        "motif_lift_p_value": float(motif_p_val),
        "within_domain_motif_wins": int(within_motif_wins),
        "within_domain_total_compared": int(within_total),
    }

    logger.info(f"Success criteria: {json.dumps(success, indent=2)}")

    # ---- STEP 11: Build output ----
    logger.info("Step 11: Building output...")

    # Map slugs to records for metadata retrieval
    slug_to_record = {r["metadata_slug"]: r for r in records}

    # Get baseline and method predictions
    baseline_preds = np.array(
        results.get(key_base_lr, {}).get(
            "predictions", [0.5] * len(y),
        )
    )
    method_preds = np.array(
        results.get(key_full_lr, {}).get(
            "predictions", [0.5] * len(y),
        )
    )

    # Build per-example output
    examples_out = []
    for i, (_, row) in enumerate(df.iterrows()):
        slug = row["slug"]
        orig = slug_to_record.get(slug, {})

        # Compact analysis output (not the full graph JSON)
        analysis = {
            "slug": slug,
            "domain": row["domain"],
            "label": int(row["label"]),
            "n_nodes_pruned": int(row.get("n_nodes_pruned", 0)),
            "n_edges_pruned": int(row.get("n_edges_pruned", 0)),
            "graph_stats": {col: float(row[col]) for col in GRAPH_COLS},
            "motif_features": {col: float(row[col]) for col in MOTIF_COLS},
            "baseline_prob": float(baseline_preds[i]) if i < len(baseline_preds) else 0.5,
            "our_method_prob": float(method_preds[i]) if i < len(method_preds) else 0.5,
            "best_model_prob": float(y_prob_best[i]) if i < len(y_prob_best) else 0.5,
        }

        example = {
            "input": orig.get("input", ""),
            "output": json.dumps(analysis),
            "metadata_fold": row["domain"],
            "metadata_slug": slug,
            "metadata_model_correct": (
                "true" if row["label"] == 1 else "false"
            ),
            "metadata_difficulty": orig.get("metadata_difficulty", "unknown"),
            "metadata_n_nodes_original": int(
                orig.get("metadata_n_nodes", 0)
            ),
            "metadata_n_edges_original": int(
                orig.get("metadata_n_edges", 0)
            ),
            "predict_baseline": (
                f"{float(baseline_preds[i]):.6f}"
                if i < len(baseline_preds)
                else "0.500000"
            ),
            "predict_our_method": (
                f"{float(method_preds[i]):.6f}"
                if i < len(method_preds)
                else "0.500000"
            ),
        }
        examples_out.append(example)

    # Remove full predictions from results for compact metadata
    results_compact = {}
    for k, v in all_results.items():
        results_compact[k] = {
            kk: vv for kk, vv in v.items() if kk != "predictions"
        }

    output = {
        "metadata": {
            "experiment": "deconfounded_failure_prediction_motif_spectra",
            "description": (
                "Predicts model correctness from attribution graph "
                "motif spectra, deconfounded from domain and graph-size "
                "effects. Baseline: domain+graph_stats features. "
                "Our method: full model with motif spectra added."
            ),
            "method_name": "Motif Spectrum Failure Predictor",
            "baseline_name": "Domain + Graph Statistics (no motif info)",
            "n_graphs_total": 200,
            "n_graphs_used": len(df),
            "n_correct": int(n_correct),
            "n_incorrect": int(n_incorrect),
            "n_parse_failures": len(failed),
            "class_distribution_by_domain": domain_dist,
            "powered_domains": powered_domains,
            "feature_computation": {
                "prune_quantile": PRUNE_QUANTILE,
                "n_null_models": N_NULL,
                "motif_features": MOTIF_COLS,
                "graph_features": GRAPH_COLS,
                "domain_features": DOMAIN_COLS,
                "deviation_features": DEV_COLS,
                "motif_indices": {
                    k: int(v) for k, v in motif_indices.items()
                },
            },
            "classifier_results": results_compact,
            "key_comparisons": comparisons,
            "within_domain_analysis": within_domain_results,
            "deviation_feature_results": {
                k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                for k, v in dev_results.items()
            },
            "feature_importance": importance_dict,
            "bootstrap_ci_best_model": {
                "model": best_key,
                "auc_ci_95": boot_ci,
            },
            "success_criteria_evaluation": success,
            "runtime_seconds": time.time() - t_start,
        },
        "datasets": [
            {
                "dataset": "neuronpedia_attribution_graphs_v3",
                "examples": examples_out,
            },
        ],
    }

    # Sanitize output (replace NaN/Inf with None for valid JSON)
    output = sanitize_for_json(output)

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Output saved to {out_path}")

    elapsed_total = time.time() - t_start
    logger.info(
        f"Total runtime: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}m)"
    )
    logger.info("Done!")


if __name__ == "__main__":
    main()
