#!/usr/bin/env python3
"""Motif-Based Failure/Ambiguity Prediction for Attribution Graphs.

Classifies 140 Neuronpedia attribution graphs by correctness status
(verified-correct "true" vs ambiguous/unverified "unknown") using 3-node
motif spectrum features. Also predicts difficulty level as secondary analysis.

CRITICAL DATA REALITY: 116 "true" + 24 "unknown" + 0 "false" examples.
Primary target: true vs unknown. Secondary: difficulty (easy/medium/hard).

Output: method_out.json (exp_gen_sol_out schema)
"""

import json
import sys
import os
import random
import math
import time
import gc
import resource
import glob as glob_module
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import igraph
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from loguru import logger

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ============================================================================
# HARDWARE DETECTION & RESOURCE LIMITS
# ============================================================================


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
RAM_BUDGET_GB = min(TOTAL_RAM_GB * 0.7, 20.0)
RAM_BUDGET_BYTES = int(RAM_BUDGET_GB * 1e9)

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET_GB:.1f} GB")

# ============================================================================
# CONSTANTS (configurable via env vars for gradual scaling)
# ============================================================================

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
N_NULL_MODELS = int(os.environ.get("N_NULL_MODELS", "200"))
SWAP_FACTOR = int(os.environ.get("SWAP_FACTOR", "100"))
PRUNE_PERCENTILE = 75
MIN_NODES_FOR_CENSUS = 30
# Use max 2 workers to avoid OOM with large graphs + null models
N_WORKERS = min(max(1, NUM_CPUS), int(os.environ.get("N_WORKERS", "2")))
BATCH_SIZE = 20  # Process graphs in batches to manage memory
SEED = 42

# ============================================================================
# PATHS
# ============================================================================

WORKSPACE = Path(__file__).parent
DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_2/gen_art/data_id3_it2__opus"
)

# ============================================================================
# PHASE 0: Build Isoclass-to-MAN Mapping
# ============================================================================


def build_isoclass_mapping() -> tuple[dict[int, str], dict[str, int], list[int]]:
    """Build mapping from igraph isoclass IDs to MAN labels for 3-node triads.

    Returns:
        (isoclass_to_man, man_to_isoclass, dag_possible_ids)
    """
    triads: dict[str, list[tuple[int, int]]] = {
        "003": [],
        "012": [(0, 1)],
        "102": [(0, 1), (1, 0)],
        "021D": [(1, 0), (1, 2)],
        "021U": [(0, 1), (2, 1)],
        "021C": [(0, 1), (1, 2)],
        "111D": [(0, 1), (1, 0), (2, 1)],
        "111U": [(0, 1), (1, 0), (1, 2)],
        "030T": [(0, 1), (0, 2), (1, 2)],
        "030C": [(0, 1), (1, 2), (2, 0)],
        "201": [(0, 1), (1, 0), (0, 2), (2, 0)],
        "120D": [(1, 2), (2, 1), (1, 0), (2, 0)],
        "120U": [(1, 2), (2, 1), (0, 1), (0, 2)],
        "120C": [(0, 1), (1, 0), (1, 2), (2, 0)],
        "210": [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)],
        "300": [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)],
    }

    isoclass_to_man: dict[int, str] = {}
    man_to_isoclass: dict[str, int] = {}

    for label, edges in triads.items():
        g = igraph.Graph(n=3, edges=edges, directed=True)
        cls_id = g.isoclass()
        if cls_id in isoclass_to_man:
            raise ValueError(
                f"Duplicate isoclass {cls_id}: '{label}' collides with "
                f"'{isoclass_to_man[cls_id]}'"
            )
        isoclass_to_man[cls_id] = label
        man_to_isoclass[label] = cls_id

    assert len(isoclass_to_man) == 16, f"Expected 16, got {len(isoclass_to_man)}"

    dag_possible_labels = ["021D", "021U", "021C", "030T"]
    dag_possible_ids = sorted(man_to_isoclass[l] for l in dag_possible_labels)

    for label in dag_possible_labels:
        g = igraph.Graph(n=3, edges=triads[label], directed=True)
        assert g.is_dag(), f"{label} is not a DAG"
        assert g.is_connected(mode="weak"), f"{label} is not weakly connected"

    logger.info(
        f"Isoclass mapping: DAG-possible IDs {dag_possible_ids} -> "
        f"{[isoclass_to_man[i] for i in dag_possible_ids]}"
    )
    return isoclass_to_man, man_to_isoclass, dag_possible_ids


# ============================================================================
# PHASE A: Graph Loading with Correctness Labels
# ============================================================================


def parse_layer(layer_str: str) -> int:
    """Convert layer string to integer. 'E' (embedding) -> -1, numeric -> int."""
    if layer_str == "E":
        return -1
    try:
        return int(layer_str)
    except (ValueError, TypeError):
        return -1


def load_graphs(max_examples: int = 0) -> list[dict]:
    """Load attribution graphs with correctness annotations.

    Returns:
        List of graph records with igraph objects, correctness, difficulty, etc.
    """
    logger.info("Loading attribution graphs from dependency data...")

    all_examples: list[dict] = []
    data_files = sorted(DATA_DIR.glob("data_out/full_data_out_*.json"))

    if not data_files:
        mini_path = DATA_DIR / "mini_data_out.json"
        logger.warning(f"No full data files found, using mini: {mini_path}")
        data_files = [mini_path]

    for fpath in data_files:
        logger.info(f"  Loading {fpath.name}...")
        raw = json.loads(fpath.read_text())
        examples = raw["datasets"][0]["examples"]
        all_examples.extend(examples)
        logger.info(f"    -> {len(examples)} examples")
        del raw
        gc.collect()

    if max_examples > 0:
        all_examples = all_examples[:max_examples]

    logger.info(f"Total examples to process: {len(all_examples)}")

    # Count correctness distribution
    correctness_dist = Counter(ex.get("metadata_model_correct", "unknown") for ex in all_examples)
    logger.info(f"Correctness distribution: {dict(correctness_dist)}")

    all_graphs: list[dict] = []
    for idx, example in enumerate(all_examples):
        try:
            prompt = example["input"]
            domain = example.get("metadata_fold", "unknown")
            graph_json = json.loads(example["output"])
            nodes = graph_json["nodes"]
            links = graph_json["links"]

            # Extract correctness labels
            correctness = example.get("metadata_model_correct", "unknown")
            difficulty = example.get("metadata_difficulty", "unknown")
            expected_answer = example.get("metadata_expected_answer", "")

            # Build node_id -> index mapping
            node_id_to_idx = {node["node_id"]: i for i, node in enumerate(nodes)}

            # Extract integer layers
            node_layers = [parse_layer(node.get("layer", "0")) for node in nodes]

            # Build directed graph
            g = igraph.Graph(n=len(nodes), directed=True)
            g.vs["node_id"] = [n["node_id"] for n in nodes]
            g.vs["layer"] = node_layers
            g.vs["feature_type"] = [n.get("feature_type", "") for n in nodes]

            # Add edges with absolute weights
            edge_list: list[tuple[int, int]] = []
            edge_weights: list[float] = []
            for link in links:
                src_idx = node_id_to_idx.get(link["source"])
                tgt_idx = node_id_to_idx.get(link["target"])
                if src_idx is not None and tgt_idx is not None:
                    edge_list.append((src_idx, tgt_idx))
                    w = link.get("weight", link.get("attribution", link.get("value", 1.0)))
                    edge_weights.append(abs(float(w)))

            g.add_edges(edge_list)
            g.es["weight"] = edge_weights

            # Simplify: remove multi-edges and self-loops
            g.simplify(multiple=True, loops=True, combine_edges="max")

            if not g.is_dag():
                logger.warning(f"Graph {idx} ({domain}) is NOT a DAG - skipping")
                continue

            # 75th percentile edge weight pruning (keep top 25%)
            weights = np.array(g.es["weight"])
            threshold = float(np.percentile(weights, PRUNE_PERCENTILE))
            edges_to_keep = [i for i, w in enumerate(weights) if w >= threshold]
            g_pruned = g.subgraph_edges(edges_to_keep, delete_vertices=False)

            # Remove isolated vertices
            isolated = [v.index for v in g_pruned.vs if g_pruned.degree(v) == 0]
            g_pruned.delete_vertices(isolated)

            if g_pruned.vcount() < MIN_NODES_FOR_CENSUS:
                logger.warning(
                    f"Graph {idx} ({domain}): {g_pruned.vcount()} nodes "
                    f"after pruning (< {MIN_NODES_FOR_CENSUS}), skipping"
                )
                continue

            if g_pruned.ecount() == 0:
                logger.warning(f"Graph {idx} ({domain}): 0 edges after pruning, skipping")
                continue

            assert g_pruned.is_dag(), f"Pruned graph {idx} ({domain}) is not a DAG"

            pruned_layers = list(g_pruned.vs["layer"])
            unique_layers = sorted(set(pruned_layers))

            record = {
                "graph": g_pruned,
                "domain": domain,
                "prompt": prompt[:200],
                "correctness": correctness,
                "difficulty": difficulty,
                "expected_answer": str(expected_answer)[:100],
                "n_nodes": g_pruned.vcount(),
                "n_edges": g_pruned.ecount(),
                "n_layers": len(unique_layers),
            }
            all_graphs.append(record)

            logger.debug(
                f"Graph {idx} ({domain}): {g_pruned.vcount()} nodes, "
                f"{g_pruned.ecount()} edges, correctness={correctness}, diff={difficulty}"
            )

            del g
            gc.collect()

        except Exception:
            logger.exception(f"Failed to process graph {idx}")
            continue

    logger.info(f"Loaded {len(all_graphs)} graphs after pruning")

    # Log class distribution per domain x correctness
    domain_correctness = defaultdict(lambda: defaultdict(int))
    for rec in all_graphs:
        domain_correctness[rec["domain"]][rec["correctness"]] += 1
    for d, counts in sorted(domain_correctness.items()):
        logger.info(f"  {d}: {dict(counts)}")

    return all_graphs


# ============================================================================
# PHASE B: Null Model + Motif Census (worker function for ProcessPool)
# ============================================================================


def degree_preserving_dag_swap(
    edge_list: list[tuple[int, int]],
    n_vertices: int,
    topo_rank: list[int],
    n_swap_attempts: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], float]:
    """Degree-preserving DAG randomization (Goni et al. Method 1 DD).

    Returns:
        (new_edge_list, acceptance_rate)
    """
    adj_set = set(edge_list)
    edges = list(adj_set)
    n_edges = len(edges)

    if n_edges < 2:
        return edges, 0.0

    accepted = 0
    for _ in range(n_swap_attempts):
        idx1 = rng.randrange(n_edges)
        idx2 = rng.randrange(n_edges)
        if idx1 == idx2:
            continue

        u1, v1 = edges[idx1]
        u2, v2 = edges[idx2]

        if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
            continue

        if (u1, v2) in adj_set or (u2, v1) in adj_set:
            continue

        if topo_rank[u1] >= topo_rank[v2] or topo_rank[u2] >= topo_rank[v1]:
            continue

        adj_set.discard((u1, v1))
        adj_set.discard((u2, v2))
        adj_set.add((u1, v2))
        adj_set.add((u2, v1))

        edges[idx1] = (u1, v2)
        edges[idx2] = (u2, v1)
        accepted += 1

    return edges, accepted / n_swap_attempts if n_swap_attempts > 0 else 0.0


def process_single_graph(
    serialized_data: tuple,
    dag_possible_ids: list[int],
    n_null: int,
    swap_factor: int,
    seed: int,
) -> dict:
    """Worker function for parallel motif census + null model generation.

    Args:
        serialized_data: (n_vertices, edge_list, vertex_layers)
        dag_possible_ids: isoclass IDs of DAG-possible motifs
        n_null: number of null models to generate
        swap_factor: multiplied by n_edges for swap attempts
        seed: random seed for reproducibility

    Returns:
        dict with real_counts, null_mean, null_std, z_scores, count_ratios, graph_stats
    """
    n_vertices, edge_list, vertex_layers = serialized_data

    # Reconstruct igraph
    g = igraph.Graph(n=n_vertices, edges=edge_list, directed=True)
    g.vs["layer"] = vertex_layers

    # Compute topological ordering
    topo_order = g.topological_sorting()
    topo_rank = [0] * n_vertices
    for rank, node in enumerate(topo_order):
        topo_rank[node] = rank

    # Real motif census
    census = g.motifs_randesu(size=3)
    real_counts = np.array([census[i] if census[i] >= 0 else 0 for i in dag_possible_ids], dtype=np.float64)

    # Generate null models
    rng = random.Random(seed)
    n_swap_attempts = swap_factor * len(edge_list)
    null_counts_all = np.zeros((n_null, len(dag_possible_ids)), dtype=np.float64)

    for ni in range(n_null):
        null_edges, _ = degree_preserving_dag_swap(
            edge_list, n_vertices, topo_rank, n_swap_attempts, rng
        )
        g_null = igraph.Graph(n=n_vertices, edges=null_edges, directed=True)
        null_census = g_null.motifs_randesu(size=3)
        for j, iso_id in enumerate(dag_possible_ids):
            val = null_census[iso_id]
            null_counts_all[ni, j] = val if val >= 0 else 0

    # Compute Z-scores
    null_mean = null_counts_all.mean(axis=0)
    null_std = null_counts_all.std(axis=0)

    z_scores = np.zeros(len(dag_possible_ids), dtype=np.float64)
    for j in range(len(dag_possible_ids)):
        if null_std[j] > 1e-10:
            z_scores[j] = (real_counts[j] - null_mean[j]) / null_std[j]
        else:
            if abs(real_counts[j] - null_mean[j]) < 1e-10:
                z_scores[j] = 0.0
            else:
                z_scores[j] = 10.0 if real_counts[j] > null_mean[j] else -10.0

    # Compute count ratios
    total_count = real_counts.sum()
    if total_count > 0:
        count_ratios = real_counts / total_count
    else:
        count_ratios = np.full(len(dag_possible_ids), 0.25)

    # Graph statistics
    in_degrees = g.indegree()
    out_degrees = g.outdegree()
    density = g.density()
    transitivity = g.transitivity_undirected() if g.ecount() > 0 else 0.0
    # Handle NaN transitivity (igraph returns NaN for graphs with no triangles sometimes)
    if math.isnan(transitivity):
        transitivity = 0.0

    graph_stats = {
        "n_nodes": n_vertices,
        "n_edges": len(edge_list),
        "density": density,
        "mean_in_deg": float(np.mean(in_degrees)),
        "mean_out_deg": float(np.mean(out_degrees)),
        "max_out_deg": int(np.max(out_degrees)) if out_degrees else 0,
        "n_layers": len(set(vertex_layers)),
        "transitivity": transitivity,
    }

    return {
        "real_counts": real_counts.tolist(),
        "null_mean": null_mean.tolist(),
        "null_std": null_std.tolist(),
        "z_scores": z_scores.tolist(),
        "count_ratios": count_ratios.tolist(),
        "graph_stats": graph_stats,
    }


# ============================================================================
# PHASE C: Feature Matrix Construction
# ============================================================================

FEATURE_NAMES = [
    "ratio_021D", "ratio_021U", "ratio_021C", "ratio_030T",
    "z_magnitude",
    "log_n_nodes", "log_n_edges", "density",
    "mean_in_deg", "mean_out_deg", "max_out_deg", "n_layers",
]

# Feature group indices
MOTIF_ONLY = [0, 1, 2, 3, 4]
GRAPH_STATS_ONLY = [5, 6, 7, 8, 9, 10, 11]
COUNT_RATIOS_ONLY = [0, 1, 2, 3]
ZSCORES_ONLY = [4]
ALL_FEATURES = list(range(12))


def build_feature_matrix(
    census_results: list[dict],
    graph_records: list[dict],
) -> np.ndarray:
    """Build 12-feature matrix from motif census results and graph metadata.

    Features:
        [0-3]  count_ratio for 021D, 021U, 021C, 030T
        [4]    z_magnitude = sqrt(sum(z_i^2))
        [5]    log(n_nodes)
        [6]    log(n_edges)
        [7]    density
        [8]    mean_in_degree
        [9]    mean_out_degree
        [10]   max_out_degree
        [11]   n_layers
    """
    n = len(census_results)
    X = np.zeros((n, 12), dtype=np.float64)

    for i, res in enumerate(census_results):
        ratios = res["count_ratios"]
        z_scores = res["z_scores"]
        gs = res["graph_stats"]

        X[i, 0] = ratios[0]  # 021D
        X[i, 1] = ratios[1]  # 021U
        X[i, 2] = ratios[2]  # 021C
        X[i, 3] = ratios[3]  # 030T
        X[i, 4] = float(np.sqrt(np.sum(np.array(z_scores) ** 2)))
        X[i, 5] = np.log1p(gs["n_nodes"])
        X[i, 6] = np.log1p(gs["n_edges"])
        X[i, 7] = gs["density"]
        X[i, 8] = gs["mean_in_deg"]
        X[i, 9] = gs["mean_out_deg"]
        X[i, 10] = gs["max_out_deg"]
        X[i, 11] = gs["n_layers"]

    # Check for NaN/Inf
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        logger.warning(f"Found {nan_mask.sum()} NaN/Inf values in feature matrix, replacing with 0")
        X[nan_mask] = 0.0

    return X


# ============================================================================
# PHASE D: Within-Domain Deviation Features (leave-one-out)
# ============================================================================


def compute_deviation_features(
    X: np.ndarray,
    domains: list[str],
    motif_indices: list[int] = None,
) -> np.ndarray:
    """Compute within-domain deviation features using leave-one-out.

    For each graph, computes how much it deviates from its domain's mean
    (excluding itself). Returns 6 deviation features:
        [0] euclidean_dev: ||x_i - mean_d_\\i||_2
        [1] cosine_dev: cosine_distance(x_i, mean_d_\\i)
        [2-5] per_motif_deviation: |x_i_j - mean_d_j| / std_d_j for each motif ratio

    Args:
        X: base feature matrix (n_samples, n_features)
        domains: domain label for each sample
        motif_indices: indices of motif features in X (default: [0,1,2,3])

    Returns:
        deviation features (n_samples, 6)
    """
    if motif_indices is None:
        motif_indices = [0, 1, 2, 3]

    n = len(X)
    dev_features = np.zeros((n, 6), dtype=np.float64)

    # Group samples by domain
    domain_indices: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(domains):
        domain_indices[d].append(i)

    for d, indices in domain_indices.items():
        if len(indices) < 2:
            # Can't compute leave-one-out with only 1 sample
            continue

        for i in indices:
            # Leave-one-out: compute domain mean excluding sample i
            others = [j for j in indices if j != i]
            domain_mean = X[others].mean(axis=0)

            # Euclidean deviation
            diff = X[i] - domain_mean
            dev_features[i, 0] = float(np.linalg.norm(diff))

            # Cosine deviation
            norm_xi = np.linalg.norm(X[i])
            norm_mean = np.linalg.norm(domain_mean)
            if norm_xi > 1e-10 and norm_mean > 1e-10:
                cos_sim = np.dot(X[i], domain_mean) / (norm_xi * norm_mean)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
                dev_features[i, 1] = 1.0 - cos_sim
            else:
                dev_features[i, 1] = 0.0

            # Per-motif deviation (standardized by domain std)
            domain_std = X[others].std(axis=0)
            for k, mi in enumerate(motif_indices):
                if k >= 4:
                    break
                std_val = max(domain_std[mi], 1e-8)
                dev_features[i, 2 + k] = abs(X[i, mi] - domain_mean[mi]) / std_val

    # Replace any NaN/Inf
    nan_mask = ~np.isfinite(dev_features)
    if nan_mask.any():
        dev_features[nan_mask] = 0.0

    return dev_features


# ============================================================================
# PHASE E: Classification Pipeline
# ============================================================================


def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: list[int],
    classifier_name: str,
    classifier_factory,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
) -> dict:
    """Run repeated stratified k-fold classification and return metrics.

    Args:
        X: feature matrix (all features)
        y: binary labels
        feature_indices: which columns of X to use
        classifier_name: name for logging
        classifier_factory: callable returning a new classifier instance
        n_splits: CV folds
        n_repeats: CV repeats
        seed: random seed

    Returns:
        dict with mean/std metrics, per-fold details
    """
    X_sub = X[:, feature_indices]

    # Check minimum class count for stratified CV
    class_counts = Counter(y.tolist())
    min_class_count = min(class_counts.values())

    if len(class_counts) < 2:
        logger.warning(f"{classifier_name}: only 1 class present, returning chance-level")
        return {
            "classifier": classifier_name,
            "auc_mean": 0.5, "auc_std": 0.0,
            "precision_mean": 0.0, "precision_std": 0.0,
            "recall_mean": 0.0, "recall_std": 0.0,
            "f1_mean": 0.0, "f1_std": 0.0,
            "n_folds": 0,
        }

    if min_class_count < n_splits:
        logger.warning(
            f"{classifier_name}: min class count {min_class_count} < {n_splits} folds, "
            f"falling back to LeaveOneOut"
        )
        return run_loo_classification(X_sub, y, classifier_name, classifier_factory, seed)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scaler = StandardScaler()

    fold_aucs = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    for train_idx, test_idx in cv.split(X_sub, y):
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Check if test set has both classes
        if len(set(y_test.tolist())) < 2:
            continue

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = classifier_factory()
        clf.fit(X_train_s, y_train)

        y_prob = clf.predict_proba(X_test_s)
        # Handle case where classifier doesn't predict both classes
        if y_prob.shape[1] == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob[:, 0]

        y_pred = clf.predict(X_test_s)

        try:
            auc = roc_auc_score(y_test, y_prob_pos)
        except ValueError:
            continue

        fold_aucs.append(auc)
        fold_precisions.append(precision_score(y_test, y_pred, zero_division=0.0))
        fold_recalls.append(recall_score(y_test, y_pred, zero_division=0.0))
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0.0))

    if not fold_aucs:
        logger.warning(f"{classifier_name}: no valid folds")
        return {
            "classifier": classifier_name,
            "auc_mean": 0.5, "auc_std": 0.0,
            "precision_mean": 0.0, "precision_std": 0.0,
            "recall_mean": 0.0, "recall_std": 0.0,
            "f1_mean": 0.0, "f1_std": 0.0,
            "n_folds": 0,
        }

    return {
        "classifier": classifier_name,
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "precision_mean": float(np.mean(fold_precisions)),
        "precision_std": float(np.std(fold_precisions)),
        "recall_mean": float(np.mean(fold_recalls)),
        "recall_std": float(np.std(fold_recalls)),
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "n_folds": len(fold_aucs),
        "fold_aucs": [float(a) for a in fold_aucs],
    }


def run_loo_classification(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    classifier_factory,
    seed: int = 42,
) -> dict:
    """Fallback: Leave-one-out CV for small datasets."""
    scaler = StandardScaler()
    loo = LeaveOneOut()

    y_probs = np.zeros(len(y))
    y_preds = np.zeros(len(y), dtype=int)

    valid_folds = 0
    for train_idx, test_idx in loo.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip if training set has only one class
        if len(set(y_train.tolist())) < 2:
            y_probs[test_idx[0]] = 0.5  # neutral prediction
            y_preds[test_idx[0]] = 0
            continue

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = classifier_factory()
        clf.fit(X_train_s, y_train)

        proba = clf.predict_proba(X_test_s)
        if proba.shape[1] == 2:
            y_probs[test_idx[0]] = proba[0, 1]
        else:
            y_probs[test_idx[0]] = proba[0, 0]
        y_preds[test_idx[0]] = clf.predict(X_test_s)[0]
        valid_folds += 1

    try:
        auc = roc_auc_score(y, y_probs)
    except ValueError:
        auc = 0.5

    return {
        "classifier": classifier_name,
        "auc_mean": float(auc),
        "auc_std": 0.0,
        "precision_mean": float(precision_score(y, y_preds, zero_division=0.0)),
        "precision_std": 0.0,
        "recall_mean": float(recall_score(y, y_preds, zero_division=0.0)),
        "recall_std": 0.0,
        "f1_mean": float(f1_score(y, y_preds, zero_division=0.0)),
        "f1_std": 0.0,
        "n_folds": len(y),
        "cv_type": "leave_one_out",
    }


# ============================================================================
# PHASE F: Difficulty Prediction (3-class)
# ============================================================================


def run_multiclass_classification(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: list[int],
    classifier_name: str,
    classifier_factory,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
) -> dict:
    """Run multiclass classification with macro-AUC, weighted F1."""
    X_sub = X[:, feature_indices]

    class_counts = Counter(y.tolist())
    if len(class_counts) < 2:
        logger.warning(f"{classifier_name}: only 1 class present for multiclass, returning chance")
        return {
            "classifier": classifier_name,
            "auc_mean": 0.5, "auc_std": 0.0,
            "f1_weighted_mean": 0.0, "f1_weighted_std": 0.0,
            "accuracy_mean": 0.0, "accuracy_std": 0.0,
            "n_folds": 0,
        }
    min_class_count = min(class_counts.values())

    if min_class_count < n_splits:
        logger.warning(f"{classifier_name}: min class {min_class_count} < {n_splits}, reducing folds")
        n_splits = max(2, min_class_count)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scaler = StandardScaler()

    fold_aucs = []
    fold_f1s = []
    fold_accs = []

    for train_idx, test_idx in cv.split(X_sub, y):
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(set(y_test.tolist())) < 2:
            continue

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = classifier_factory()
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)

        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            continue

        fold_aucs.append(auc)
        fold_f1s.append(f1_score(y_test, y_pred, average="weighted", zero_division=0.0))
        fold_accs.append(float(np.mean(y_test == y_pred)))

    if not fold_aucs:
        return {
            "classifier": classifier_name,
            "auc_mean": 0.5, "auc_std": 0.0,
            "f1_weighted_mean": 0.0, "f1_weighted_std": 0.0,
            "accuracy_mean": 0.0, "accuracy_std": 0.0,
            "n_folds": 0,
        }

    return {
        "classifier": classifier_name,
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "f1_weighted_mean": float(np.mean(fold_f1s)),
        "f1_weighted_std": float(np.std(fold_f1s)),
        "accuracy_mean": float(np.mean(fold_accs)),
        "accuracy_std": float(np.std(fold_accs)),
        "n_folds": len(fold_aucs),
        "fold_aucs": [float(a) for a in fold_aucs],
    }


# ============================================================================
# PHASE G: Per-Domain Analysis
# ============================================================================


def per_domain_analysis(
    X: np.ndarray,
    domains: list[str],
    correctness: list[str],
    motif_indices: list[int] = None,
) -> list[dict]:
    """Mann-Whitney U tests comparing motif features: true vs unknown per domain."""
    if motif_indices is None:
        motif_indices = [0, 1, 2, 3, 4]

    results = []
    domain_set = sorted(set(domains))

    for domain in domain_set:
        d_mask = np.array([d == domain for d in domains])
        true_mask = d_mask & np.array([c == "true" for c in correctness])
        unknown_mask = d_mask & np.array([c == "unknown" for c in correctness])

        n_true = int(true_mask.sum())
        n_unknown = int(unknown_mask.sum())

        if n_unknown < 2 or n_true < 2:
            results.append({
                "domain": domain,
                "n_true": n_true,
                "n_unknown": n_unknown,
                "skipped": True,
                "reason": f"Too few samples (true={n_true}, unknown={n_unknown})",
            })
            continue

        domain_result = {
            "domain": domain,
            "n_true": n_true,
            "n_unknown": n_unknown,
            "skipped": False,
            "motif_tests": {},
        }

        for mi in motif_indices:
            feat_name = FEATURE_NAMES[mi] if mi < len(FEATURE_NAMES) else f"feature_{mi}"
            true_vals = X[true_mask, mi]
            unknown_vals = X[unknown_mask, mi]

            try:
                stat, pval = stats.mannwhitneyu(true_vals, unknown_vals, alternative="two-sided")
            except ValueError:
                stat, pval = 0.0, 1.0

            # Cohen's d effect size
            pooled_std = np.sqrt(
                ((n_true - 1) * np.var(true_vals) + (n_unknown - 1) * np.var(unknown_vals))
                / max(n_true + n_unknown - 2, 1)
            )
            cohens_d = (np.mean(true_vals) - np.mean(unknown_vals)) / max(pooled_std, 1e-10)

            domain_result["motif_tests"][feat_name] = {
                "u_statistic": float(stat),
                "p_value": float(pval),
                "cohens_d": float(cohens_d),
                "true_mean": float(np.mean(true_vals)),
                "unknown_mean": float(np.mean(unknown_vals)),
            }

        results.append(domain_result)

    return results


# ============================================================================
# PHASE I: Statistical Significance
# ============================================================================


def compute_statistical_significance(
    fold_aucs_model: list[float],
    fold_aucs_baseline: list[float],
    model_name: str,
    baseline_name: str,
) -> dict:
    """Paired t-test between model and baseline AUCs from same CV folds."""
    n = min(len(fold_aucs_model), len(fold_aucs_baseline))
    if n < 2:
        return {
            "comparison": f"{model_name} vs {baseline_name}",
            "p_value": 1.0,
            "auc_diff_mean": 0.0,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0,
        }

    model_arr = np.array(fold_aucs_model[:n])
    base_arr = np.array(fold_aucs_baseline[:n])
    diffs = model_arr - base_arr

    t_stat, p_value = stats.ttest_rel(model_arr, base_arr)
    mean_diff = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "comparison": f"{model_name} vs {baseline_name}",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "auc_diff_mean": mean_diff,
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_folds": n,
    }


def permutation_test_auc(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: list[int],
    classifier_factory,
    observed_auc: float,
    n_permutations: int = 1000,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Permutation test: shuffle labels, recompute AUC, get empirical p-value."""
    rng = np.random.RandomState(seed)
    null_aucs = []

    for perm_i in range(n_permutations):
        y_perm = rng.permutation(y)
        X_sub = X[:, feature_indices]

        class_counts = Counter(y_perm.tolist())
        min_count = min(class_counts.values())
        actual_splits = min(n_splits, min_count)
        if actual_splits < 2:
            continue

        cv = RepeatedStratifiedKFold(n_splits=actual_splits, n_repeats=1, random_state=perm_i)
        scaler = StandardScaler()
        perm_aucs = []

        for train_idx, test_idx in cv.split(X_sub, y_perm):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y_perm[train_idx], y_perm[test_idx]

            if len(set(y_test.tolist())) < 2:
                continue

            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = classifier_factory()
            clf.fit(X_train_s, y_train)
            y_prob = clf.predict_proba(X_test_s)
            if y_prob.shape[1] == 2:
                y_prob_pos = y_prob[:, 1]
            else:
                y_prob_pos = y_prob[:, 0]

            try:
                perm_aucs.append(roc_auc_score(y_test, y_prob_pos))
            except ValueError:
                pass

        if perm_aucs:
            null_aucs.append(float(np.mean(perm_aucs)))

    if not null_aucs:
        return {"empirical_p_value": 1.0, "n_permutations": 0}

    null_aucs_arr = np.array(null_aucs)
    empirical_p = float(np.mean(null_aucs_arr >= observed_auc))

    return {
        "empirical_p_value": empirical_p,
        "n_permutations": len(null_aucs),
        "null_auc_mean": float(np.mean(null_aucs_arr)),
        "null_auc_std": float(np.std(null_aucs_arr)),
        "observed_auc": observed_auc,
    }


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for AUC."""
    rng = np.random.RandomState(seed)
    aucs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        if len(set(y_t.tolist())) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_t, y_p))
        except ValueError:
            pass

    if not aucs:
        return {"ci_lower": 0.5, "ci_upper": 0.5, "n_valid": 0}

    aucs_arr = np.array(aucs)
    return {
        "ci_lower": float(np.percentile(aucs_arr, 2.5)),
        "ci_upper": float(np.percentile(aucs_arr, 97.5)),
        "n_valid": len(aucs),
    }


# ============================================================================
# MAIN
# ============================================================================


@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("MOTIF-BASED FAILURE/AMBIGUITY PREDICTION")
    logger.info("=" * 70)

    # ----- PHASE 0: Isoclass Mapping -----
    isoclass_to_man, man_to_isoclass, dag_possible_ids = build_isoclass_mapping()
    dag_labels = [isoclass_to_man[i] for i in dag_possible_ids]
    logger.info(f"DAG-possible motifs: {list(zip(dag_possible_ids, dag_labels))}")

    # ----- PHASE A: Load Graphs -----
    graph_records = load_graphs(max_examples=MAX_EXAMPLES)
    n_graphs = len(graph_records)
    logger.info(f"Processing {n_graphs} graphs")

    if n_graphs == 0:
        logger.error("No graphs loaded! Aborting.")
        return

    # Extract labels
    correctness_labels = [r["correctness"] for r in graph_records]
    difficulty_labels = [r["difficulty"] for r in graph_records]
    domains = [r["domain"] for r in graph_records]

    # Binary target: 1 = "unknown", 0 = "true"
    y_binary = np.array([1 if c == "unknown" else 0 for c in correctness_labels], dtype=np.int32)
    # Difficulty target: 0=easy, 1=medium, 2=hard
    diff_map = {"easy": 0, "medium": 1, "hard": 2}
    y_difficulty = np.array([diff_map.get(d, 1) for d in difficulty_labels], dtype=np.int32)

    logger.info(f"Binary target distribution: true={int((y_binary==0).sum())}, unknown={int((y_binary==1).sum())}")
    logger.info(f"Difficulty distribution: {dict(Counter(difficulty_labels))}")

    # ----- PHASE B: Motif Census + Null Models -----
    logger.info(f"Computing motif census with {N_NULL_MODELS} null models per graph...")
    logger.info(f"Using {N_WORKERS} workers, swap_factor={SWAP_FACTOR}")

    # Serialize graphs for multiprocessing
    serialized_graphs = []
    for rec in graph_records:
        g = rec["graph"]
        n_v = g.vcount()
        edges = [(e.source, e.target) for e in g.es]
        layers = list(g.vs["layer"])
        serialized_graphs.append((n_v, edges, layers))

    # Free igraph objects from records - we have serialized copies now
    for rec in graph_records:
        del rec["graph"]
    gc.collect()
    logger.info("Freed igraph objects from graph_records after serialization")

    # Gradual scaling: first process 3 graphs to estimate timing
    logger.info("GRADUAL SCALING: Testing with 3 graphs first...")
    t_test_start = time.time()
    test_results = []
    for i in range(min(3, len(serialized_graphs))):
        res = process_single_graph(
            serialized_graphs[i],
            dag_possible_ids,
            n_null=min(10, N_NULL_MODELS),
            swap_factor=SWAP_FACTOR,
            seed=SEED + i,
        )
        test_results.append(res)
        logger.info(
            f"  Test graph {i}: counts={res['real_counts']}, "
            f"z={[f'{z:.1f}' for z in res['z_scores']]}"
        )

    t_test_elapsed = time.time() - t_test_start
    logger.info(f"Test run: {t_test_elapsed:.1f}s for 3 graphs x 10 null models")

    # Extrapolate timing for full run
    est_per_graph = t_test_elapsed / min(3, len(serialized_graphs))
    est_ratio = N_NULL_MODELS / 10.0
    est_total = est_per_graph * est_ratio * n_graphs / N_WORKERS
    logger.info(f"Estimated full run: {est_total/60:.1f} min ({n_graphs} graphs x {N_NULL_MODELS} nulls)")

    # Adjust null models if too slow
    actual_n_null = N_NULL_MODELS
    if est_total > 2400:  # > 40 minutes
        actual_n_null = max(50, int(N_NULL_MODELS * 2400 / est_total))
        logger.warning(f"Reducing null models from {N_NULL_MODELS} to {actual_n_null} to fit time budget")

    # Process graphs in batches to manage memory (avoid OOM from BrokenProcessPool)
    logger.info(f"Processing all {n_graphs} graphs with {actual_n_null} null models in batches of {BATCH_SIZE}...")
    logger.info(f"Using {N_WORKERS} workers")
    t_census_start = time.time()

    fallback_result = {
        "real_counts": [0.0] * len(dag_possible_ids),
        "null_mean": [0.0] * len(dag_possible_ids),
        "null_std": [1.0] * len(dag_possible_ids),
        "z_scores": [0.0] * len(dag_possible_ids),
        "count_ratios": [0.25] * len(dag_possible_ids),
        "graph_stats": {
            "n_nodes": 0, "n_edges": 0, "density": 0,
            "mean_in_deg": 0, "mean_out_deg": 0,
            "max_out_deg": 0, "n_layers": 0, "transitivity": 0,
        },
    }

    census_results = [None] * n_graphs
    completed = 0

    for batch_start in range(0, n_graphs, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_graphs)
        batch_indices = list(range(batch_start, batch_end))
        logger.info(f"  Batch {batch_start}-{batch_end-1} ({len(batch_indices)} graphs)...")

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {}
            for i in batch_indices:
                fut = executor.submit(
                    process_single_graph,
                    serialized_graphs[i],
                    dag_possible_ids,
                    actual_n_null,
                    SWAP_FACTOR,
                    SEED + i,
                )
                futures[fut] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    census_results[idx] = fut.result()
                    completed += 1
                except Exception:
                    logger.warning(f"Failed graph {idx}, using fallback")
                    census_results[idx] = dict(fallback_result)
                    completed += 1

        elapsed = time.time() - t_census_start
        eta = elapsed / completed * (n_graphs - completed) if completed > 0 else 0
        logger.info(f"  Completed {completed}/{n_graphs} graphs ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
        gc.collect()

    t_census = time.time() - t_census_start
    logger.info(f"Motif census complete: {t_census:.1f}s ({t_census/60:.1f} min)")

    # ----- PHASE C: Feature Matrix -----
    logger.info("Building feature matrix...")
    X_base = build_feature_matrix(census_results, graph_records)
    logger.info(f"Base feature matrix: {X_base.shape}")

    # Verify count ratios sum to ~1.0
    ratio_sums = X_base[:, :4].sum(axis=1)
    logger.info(f"Count ratio sums: mean={ratio_sums.mean():.4f}, range=[{ratio_sums.min():.4f}, {ratio_sums.max():.4f}]")

    # ----- PHASE D: Deviation Features -----
    logger.info("Computing within-domain deviation features...")
    dev_features = compute_deviation_features(X_base, domains, motif_indices=[0, 1, 2, 3])
    X_extended = np.hstack([X_base, dev_features])
    logger.info(f"Extended feature matrix: {X_extended.shape}")

    DEVIATION_INDICES = list(range(12, 18))
    MOTIF_PLUS_DEV = MOTIF_ONLY + DEVIATION_INDICES
    ALL_PLUS_DEV = ALL_FEATURES + DEVIATION_INDICES

    # ----- PHASE E: Primary Classification (true vs unknown) -----
    logger.info("=" * 50)
    logger.info("PHASE E: Primary Classification (true vs unknown)")
    logger.info("=" * 50)

    classifier_configs = [
        ("LR_motif_only", MOTIF_ONLY,
         lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)),
        ("LR_motif_dev", MOTIF_PLUS_DEV,
         lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)),
        ("LR_all", ALL_FEATURES,
         lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)),
        ("RF_all", ALL_FEATURES,
         lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED)),
        ("RF_all_dev", ALL_PLUS_DEV,
         lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED)),
    ]

    baseline_configs = [
        ("BL_graph_stats_only", GRAPH_STATS_ONLY,
         lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)),
        ("BL_random", ALL_FEATURES,
         lambda: DummyClassifier(strategy="stratified", random_state=SEED)),
        ("BL_majority", ALL_FEATURES,
         lambda: DummyClassifier(strategy="most_frequent")),
    ]

    all_clf_results = {}

    for name, feat_idx, factory in classifier_configs + baseline_configs:
        logger.info(f"  Running {name} (features: {len(feat_idx)})...")
        result = run_classification(
            X_extended, y_binary, feat_idx, name, factory,
            n_splits=5, n_repeats=10, seed=SEED,
        )
        all_clf_results[name] = result
        logger.info(
            f"    AUC={result['auc_mean']:.3f}+/-{result['auc_std']:.3f}, "
            f"F1={result['f1_mean']:.3f}+/-{result['f1_std']:.3f}"
        )

    # ----- PHASE F: Difficulty Prediction (3-class) -----
    logger.info("=" * 50)
    logger.info("PHASE F: Difficulty Prediction (easy/medium/hard)")
    logger.info("=" * 50)

    diff_clf_configs = [
        ("DIFF_LR_motif", MOTIF_ONLY,
         lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED, multi_class="multinomial")),
        ("DIFF_LR_all", ALL_FEATURES,
         lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED, multi_class="multinomial")),
        ("DIFF_RF_all", ALL_FEATURES,
         lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED)),
        ("DIFF_RF_all_dev", ALL_PLUS_DEV,
         lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED)),
        ("DIFF_BL_random", ALL_FEATURES,
         lambda: DummyClassifier(strategy="stratified", random_state=SEED)),
    ]

    diff_results = {}
    for name, feat_idx, factory in diff_clf_configs:
        logger.info(f"  Running {name}...")
        result = run_multiclass_classification(
            X_extended, y_difficulty, feat_idx, name, factory,
            n_splits=5, n_repeats=10, seed=SEED,
        )
        diff_results[name] = result
        logger.info(
            f"    AUC={result['auc_mean']:.3f}+/-{result['auc_std']:.3f}, "
            f"wF1={result.get('f1_weighted_mean', 0):.3f}"
        )

    # ----- PHASE G: Per-Domain Analysis -----
    logger.info("=" * 50)
    logger.info("PHASE G: Per-Domain Analysis")
    logger.info("=" * 50)

    domain_results = per_domain_analysis(X_base, domains, correctness_labels, motif_indices=MOTIF_ONLY)
    for dr in domain_results:
        if dr.get("skipped"):
            logger.info(f"  {dr['domain']}: SKIPPED ({dr.get('reason', '')})")
        else:
            sig_tests = [
                (k, v["p_value"]) for k, v in dr.get("motif_tests", {}).items()
                if v["p_value"] < 0.1
            ]
            if sig_tests:
                logger.info(f"  {dr['domain']}: significant features: {sig_tests}")
            else:
                logger.info(f"  {dr['domain']}: no significant differences (p<0.1)")

    # ----- PHASE H: Feature Ablation -----
    logger.info("=" * 50)
    logger.info("PHASE H: Feature Ablation Study")
    logger.info("=" * 50)

    # Find best classifier from Phase E
    best_clf_name = max(
        [n for n, _ , _ in classifier_configs],
        key=lambda n: all_clf_results[n]["auc_mean"]
    )
    best_clf_config = next(
        (n, fi, f) for n, fi, f in classifier_configs if n == best_clf_name
    )
    logger.info(f"Best classifier: {best_clf_name} (AUC={all_clf_results[best_clf_name]['auc_mean']:.3f})")

    ablation_configs = [
        ("ABL_count_ratios_only", COUNT_RATIOS_ONLY),
        ("ABL_zscore_only", ZSCORES_ONLY),
        ("ABL_deviation_only", DEVIATION_INDICES),
        ("ABL_graph_stats_only", GRAPH_STATS_ONLY),
        ("ABL_all_plus_dev", ALL_PLUS_DEV),
    ]

    ablation_results = {}
    _, _, best_factory = best_clf_config

    for name, feat_idx in ablation_configs:
        logger.info(f"  Running ablation {name} (features: {len(feat_idx)})...")
        result = run_classification(
            X_extended, y_binary, feat_idx, name, best_factory,
            n_splits=5, n_repeats=10, seed=SEED,
        )
        ablation_results[name] = result
        logger.info(f"    AUC={result['auc_mean']:.3f}+/-{result['auc_std']:.3f}")

    # ----- PHASE I: Statistical Significance -----
    logger.info("=" * 50)
    logger.info("PHASE I: Statistical Significance")
    logger.info("=" * 50)

    significance_results = []

    # Compare best model vs each baseline
    best_folds = all_clf_results[best_clf_name].get("fold_aucs", [])
    for bl_name, _, _ in baseline_configs:
        bl_folds = all_clf_results[bl_name].get("fold_aucs", [])
        if best_folds and bl_folds:
            sig = compute_statistical_significance(best_folds, bl_folds, best_clf_name, bl_name)
            significance_results.append(sig)
            logger.info(
                f"  {sig['comparison']}: p={sig['p_value']:.4f}, "
                f"AUC diff={sig['auc_diff_mean']:.3f} "
                f"[{sig['ci_95_lower']:.3f}, {sig['ci_95_upper']:.3f}]"
            )

    # Permutation test for best model
    logger.info("Running permutation test (500 permutations)...")
    _, best_feat_idx, best_factory_fn = best_clf_config
    perm_result = permutation_test_auc(
        X_extended, y_binary, best_feat_idx, best_factory_fn,
        observed_auc=all_clf_results[best_clf_name]["auc_mean"],
        n_permutations=500,  # Reduced from 1000 for speed
        n_splits=5, seed=SEED,
    )
    logger.info(
        f"Permutation test: empirical p={perm_result['empirical_p_value']:.4f}, "
        f"null AUC={perm_result.get('null_auc_mean', 0):.3f}+/-{perm_result.get('null_auc_std', 0):.3f}"
    )

    # Bootstrap CI for best model AUC (using LOO predictions)
    logger.info("Computing bootstrap CI for best model...")
    X_sub_best = X_extended[:, best_feat_idx]
    scaler = StandardScaler()
    loo = LeaveOneOut()
    y_loo_probs = np.zeros(len(y_binary))

    for train_idx, test_idx in loo.split(X_sub_best, y_binary):
        y_train_loo = y_binary[train_idx]
        if len(set(y_train_loo.tolist())) < 2:
            y_loo_probs[test_idx[0]] = 0.5
            continue
        X_tr = scaler.fit_transform(X_sub_best[train_idx])
        X_te = scaler.transform(X_sub_best[test_idx])
        clf = best_factory_fn()
        clf.fit(X_tr, y_train_loo)
        proba = clf.predict_proba(X_te)
        y_loo_probs[test_idx[0]] = proba[0, 1] if proba.shape[1] == 2 else proba[0, 0]

    bootstrap_ci = bootstrap_auc_ci(y_binary, y_loo_probs, n_bootstrap=1000, seed=SEED)
    logger.info(f"Bootstrap CI: [{bootstrap_ci['ci_lower']:.3f}, {bootstrap_ci['ci_upper']:.3f}]")

    # ----- PHASE J: Output Construction -----
    logger.info("=" * 50)
    logger.info("PHASE J: Building output JSON")
    logger.info("=" * 50)

    t_total = time.time() - t_start

    # Build class distribution info
    class_dist = {
        "total": n_graphs,
        "true": int((y_binary == 0).sum()),
        "unknown": int((y_binary == 1).sum()),
        "per_domain": {},
    }
    domain_correctness_map = defaultdict(lambda: defaultdict(int))
    for d, c in zip(domains, correctness_labels):
        domain_correctness_map[d][c] += 1
    class_dist["per_domain"] = {d: dict(v) for d, v in domain_correctness_map.items()}

    # Build metadata
    metadata = {
        "experiment": "motif_failure_prediction",
        "description": (
            "Train classifiers on 3-node motif spectrum features from "
            "correctness-annotated Neuronpedia attribution graphs to predict "
            "whether a graph's output is verified-correct (true) vs "
            "ambiguous/unverified (unknown)."
        ),
        "parameters": {
            "n_null_models": actual_n_null,
            "swap_factor": SWAP_FACTOR,
            "prune_percentile": PRUNE_PERCENTILE,
            "n_graphs": n_graphs,
            "n_workers": N_WORKERS,
            "seed": SEED,
            "min_nodes_for_census": MIN_NODES_FOR_CENSUS,
        },
        "class_distribution": class_dist,
        "isoclass_mapping": {str(k): v for k, v in isoclass_to_man.items()},
        "dag_possible_motif_ids": dag_possible_ids,
        "NOTE": "No 'false' labels in dataset; experiment tests true vs unknown",
        "runtime_seconds": round(t_total, 1),
        "feature_names": FEATURE_NAMES,
        "feature_groups": {
            "MOTIF_ONLY": MOTIF_ONLY,
            "GRAPH_STATS_ONLY": GRAPH_STATS_ONLY,
            "COUNT_RATIOS_ONLY": COUNT_RATIOS_ONLY,
            "ZSCORES_ONLY": ZSCORES_ONLY,
            "ALL_FEATURES": ALL_FEATURES,
            "DEVIATION_INDICES": DEVIATION_INDICES,
        },
    }

    # Build examples
    examples = []

    # 1. Primary classification results
    for name, result in all_clf_results.items():
        is_baseline = name.startswith("BL_")
        output_data = {
            "auc_mean": result["auc_mean"],
            "auc_std": result["auc_std"],
            "precision_mean": result.get("precision_mean", 0),
            "recall_mean": result.get("recall_mean", 0),
            "f1_mean": result.get("f1_mean", 0),
            "n_folds": result.get("n_folds", 0),
        }
        examples.append({
            "input": f"classification: {name}",
            "output": json.dumps(output_data),
            "predict_motif_classifier": json.dumps(output_data),
            "metadata_fold": "aggregate",
            "metadata_analysis_type": "classification",
            "metadata_classifier": name,
            "metadata_auc_mean": result["auc_mean"],
            "metadata_auc_std": result["auc_std"],
            "metadata_pvalue": None,
            "metadata_feature_group": "baseline" if is_baseline else name.split("_", 1)[1] if "_" in name else name,
            "metadata_n_samples": n_graphs,
            "metadata_n_positive": int((y_binary == 1).sum()),
        })

    # 2. Difficulty prediction results
    for name, result in diff_results.items():
        output_data = {
            "auc_mean": result["auc_mean"],
            "auc_std": result["auc_std"],
            "f1_weighted_mean": result.get("f1_weighted_mean", 0),
            "accuracy_mean": result.get("accuracy_mean", 0),
            "n_folds": result.get("n_folds", 0),
        }
        examples.append({
            "input": f"difficulty_prediction: {name}",
            "output": json.dumps(output_data),
            "predict_motif_classifier": json.dumps(output_data),
            "metadata_fold": "aggregate",
            "metadata_analysis_type": "difficulty_prediction",
            "metadata_classifier": name,
            "metadata_auc_mean": result["auc_mean"],
            "metadata_auc_std": result["auc_std"],
            "metadata_pvalue": None,
            "metadata_feature_group": name.split("_", 2)[-1] if name.count("_") >= 2 else name,
            "metadata_n_samples": n_graphs,
            "metadata_n_positive": int((y_difficulty > 0).sum()),
        })

    # 3. Ablation results
    for name, result in ablation_results.items():
        output_data = {
            "auc_mean": result["auc_mean"],
            "auc_std": result["auc_std"],
            "f1_mean": result.get("f1_mean", 0),
            "n_folds": result.get("n_folds", 0),
        }
        feat_group = name.replace("ABL_", "")
        examples.append({
            "input": f"ablation: {name}",
            "output": json.dumps(output_data),
            "predict_motif_classifier": json.dumps(output_data),
            "metadata_fold": "aggregate",
            "metadata_analysis_type": "ablation",
            "metadata_classifier": best_clf_name,
            "metadata_auc_mean": result["auc_mean"],
            "metadata_auc_std": result["auc_std"],
            "metadata_pvalue": None,
            "metadata_feature_group": feat_group,
            "metadata_n_samples": n_graphs,
            "metadata_n_positive": int((y_binary == 1).sum()),
        })

    # 4. Statistical significance results
    for sig in significance_results:
        output_data = {
            "comparison": sig["comparison"],
            "t_statistic": sig.get("t_statistic", 0),
            "p_value": sig["p_value"],
            "auc_diff_mean": sig["auc_diff_mean"],
            "ci_95_lower": sig["ci_95_lower"],
            "ci_95_upper": sig["ci_95_upper"],
        }
        examples.append({
            "input": f"statistical_test: {sig['comparison']}",
            "output": json.dumps(output_data),
            "predict_motif_classifier": json.dumps(output_data),
            "metadata_fold": "aggregate",
            "metadata_analysis_type": "statistical_test",
            "metadata_classifier": best_clf_name,
            "metadata_auc_mean": all_clf_results[best_clf_name]["auc_mean"],
            "metadata_auc_std": all_clf_results[best_clf_name]["auc_std"],
            "metadata_pvalue": sig["p_value"],
            "metadata_feature_group": "significance",
            "metadata_n_samples": n_graphs,
            "metadata_n_positive": int((y_binary == 1).sum()),
        })

    # 5. Permutation test
    perm_output = {
        "empirical_p_value": perm_result["empirical_p_value"],
        "n_permutations": perm_result["n_permutations"],
        "null_auc_mean": perm_result.get("null_auc_mean", 0),
        "null_auc_std": perm_result.get("null_auc_std", 0),
        "observed_auc": perm_result.get("observed_auc", 0),
        "bootstrap_ci": bootstrap_ci,
    }
    examples.append({
        "input": f"permutation_test: {best_clf_name}",
        "output": json.dumps(perm_output),
        "predict_motif_classifier": json.dumps(perm_output),
        "metadata_fold": "aggregate",
        "metadata_analysis_type": "statistical_test",
        "metadata_classifier": best_clf_name,
        "metadata_auc_mean": all_clf_results[best_clf_name]["auc_mean"],
        "metadata_auc_std": all_clf_results[best_clf_name]["auc_std"],
        "metadata_pvalue": perm_result["empirical_p_value"],
        "metadata_feature_group": "permutation",
        "metadata_n_samples": n_graphs,
        "metadata_n_positive": int((y_binary == 1).sum()),
    })

    # 6. Per-domain analysis results
    for dr in domain_results:
        output_data = {
            "domain": dr["domain"],
            "n_true": dr.get("n_true", 0),
            "n_unknown": dr.get("n_unknown", 0),
            "skipped": dr.get("skipped", False),
        }
        if not dr.get("skipped"):
            output_data["motif_tests"] = dr.get("motif_tests", {})

        examples.append({
            "input": f"per_domain: {dr['domain']}",
            "output": json.dumps(output_data),
            "predict_motif_classifier": json.dumps(output_data),
            "metadata_fold": dr["domain"],
            "metadata_analysis_type": "per_domain",
            "metadata_classifier": "mann_whitney_u",
            "metadata_auc_mean": 0.0,
            "metadata_auc_std": 0.0,
            "metadata_pvalue": None,
            "metadata_feature_group": "per_domain",
            "metadata_n_samples": dr.get("n_true", 0) + dr.get("n_unknown", 0),
            "metadata_n_positive": dr.get("n_unknown", 0),
        })

    # 7. Per-graph motif features (for reuse)
    for i, (rec, res) in enumerate(zip(graph_records, census_results)):
        per_graph_data = {
            "count_ratios": res["count_ratios"],
            "z_scores": res["z_scores"],
            "real_counts": res["real_counts"],
            "graph_stats": res["graph_stats"],
            "correctness": rec["correctness"],
            "difficulty": rec["difficulty"],
            "domain": rec["domain"],
        }
        # LOO prediction: y_loo_probs[i] is P(unknown)
        loo_pred_label = "unknown" if y_loo_probs[i] >= 0.5 else "true"
        loo_pred_str = json.dumps({"predicted_correctness": loo_pred_label, "p_unknown": round(float(y_loo_probs[i]), 4)})
        examples.append({
            "input": f"graph_features: {rec['domain']}_{i}",
            "output": json.dumps(per_graph_data),
            "predict_motif_classifier": loo_pred_str,
            "metadata_fold": rec["domain"],
            "metadata_analysis_type": "graph_features",
            "metadata_classifier": best_clf_name,
            "metadata_auc_mean": 0.0,
            "metadata_auc_std": 0.0,
            "metadata_pvalue": None,
            "metadata_feature_group": "per_graph",
            "metadata_n_samples": 1,
            "metadata_n_positive": 1 if rec["correctness"] == "unknown" else 0,
        })

    # Ensure metadata_pvalue is string or number (not None for JSON schema)
    for ex in examples:
        if ex["metadata_pvalue"] is None:
            ex["metadata_pvalue"] = "N/A"

    output = {
        "metadata": metadata,
        "datasets": [
            {
                "dataset": "motif_failure_prediction_results",
                "examples": examples,
            }
        ],
    }

    # Write output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Output written to {out_path}")
    logger.info(f"Total examples in output: {len(examples)}")
    logger.info(f"Total runtime: {t_total:.1f}s ({t_total/60:.1f} min)")

    # Summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Best classifier: {best_clf_name}")
    logger.info(f"  AUC = {all_clf_results[best_clf_name]['auc_mean']:.3f} +/- {all_clf_results[best_clf_name]['auc_std']:.3f}")
    logger.info(f"  Bootstrap CI = [{bootstrap_ci['ci_lower']:.3f}, {bootstrap_ci['ci_upper']:.3f}]")
    logger.info(f"  Permutation p = {perm_result['empirical_p_value']:.4f}")

    best_bl = all_clf_results.get("BL_graph_stats_only", {})
    logger.info(f"Best baseline (graph_stats): AUC={best_bl.get('auc_mean', 0):.3f}")

    for sig in significance_results:
        logger.info(f"  {sig['comparison']}: p={sig['p_value']:.4f}")


if __name__ == "__main__":
    main()
