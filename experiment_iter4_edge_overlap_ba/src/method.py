#!/usr/bin/env python3
"""Edge-Overlap Baseline, Motif Fingerprint Stability, and Multi-Scale Feature
Complementarity on 200 Attribution Graphs.

Compute Sun (2025) edge-overlap circuit similarity as a baseline alongside
3-node motif spectrum fingerprints and graph statistics across all 200
Neuronpedia attribution graphs. Compare clustering performance (NMI/ARI at
K=8 vs 8 domains), test fingerprint stability (within-domain vs between-domain
similarity via Fisher Discriminant Ratio), and assess complementarity of motif
vs edge-overlap features.

Phases:
  A - 3-node motif census (reused from exp_id1_it3)
  B - Edge-overlap similarity (Sun 2025 baseline, NEW)
  C - Motif spectrum similarity
  D - Graph statistics similarity
  E - Clustering comparison
  F - Fingerprint stability analysis
  G - Complementarity analysis
"""

import json
import sys
import os
import gc
import math
import random
import time
import resource
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from typing import Any

import numpy as np
from loguru import logger
import igraph
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist


# ================================================================
# HARDWARE DETECTION (from aii_use_hardware skill)
# ================================================================

def _detect_cpus() -> int:
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
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0

# ================================================================
# PATHS AND CONSTANTS
# ================================================================

WORKSPACE = Path(__file__).parent.resolve()
DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_3/gen_art/data_id5_it3__opus/data_out"
)
OUTPUT_FILE = WORKSPACE / "method_out.json"
LOG_DIR = WORKSPACE / "logs"

# Configurable via env vars for gradual scaling
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
N_NULL_MODELS = int(os.environ.get("N_NULL_MODELS", "100"))
N_PERMUTATIONS = int(os.environ.get("N_PERMUTATIONS", "500"))
SWAP_MULTIPLIER = 100
PRUNE_PERCENTILE = 75
CLUSTER_K_VALUES = [2, 3, 4, 6, 8]
SEED = 42
TOTAL_TIME_BUDGET_S = int(os.environ.get("TIME_BUDGET_S", "3400"))
MAX_NULL_TIME_S = int(os.environ.get("MAX_NULL_TIME_S", "2800"))  # budget for null models

# ================================================================
# LOGGING
# ================================================================

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ================================================================
# RESOURCE LIMITS
# ================================================================

_RAM_BUDGET_BYTES = int(min(TOTAL_RAM_GB * 0.7, 20) * 1e9)
try:
    resource.setrlimit(resource.RLIMIT_AS, (_RAM_BUDGET_BYTES * 3, _RAM_BUDGET_BYTES * 3))
except ValueError:
    pass
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))


# ================================================================
# DATA LOADING
# ================================================================

def load_all_graphs(max_graphs: int | None = None) -> list[dict]:
    """Load graphs from data_id5_it3 (12 split files or mini), deduplicate by slug."""
    all_records: list[dict] = []
    seen_slugs: set[str] = set()

    def _load_split(fpath: Path) -> list[dict]:
        if not fpath.exists():
            logger.warning(f"Data file not found: {fpath}")
            return []
        logger.info(f"Loading {fpath.name} ({fpath.stat().st_size / 1e6:.1f} MB)")
        try:
            raw = json.loads(fpath.read_text())
        except json.JSONDecodeError:
            logger.exception(f"Invalid JSON: {fpath}")
            return []

        examples = raw["datasets"][0]["examples"]
        records: list[dict] = []
        for ex in examples:
            slug = ex.get("metadata_slug", "")
            if slug and slug in seen_slugs:
                continue
            if slug:
                seen_slugs.add(slug)

            try:
                graph_json = json.loads(ex["output"])
            except json.JSONDecodeError:
                logger.warning(f"Invalid graph JSON for slug={slug}")
                continue

            records.append({
                "prompt": ex["input"],
                "domain": ex["metadata_fold"],
                "slug": slug,
                "model_correct": ex.get("metadata_model_correct", "unknown"),
                "difficulty": ex.get("metadata_difficulty", "unknown"),
                "nodes": graph_json["nodes"],
                "links": graph_json["links"],
                "n_nodes_raw": ex["metadata_n_nodes"],
                "n_edges_raw": ex["metadata_n_edges"],
            })
        del raw, examples
        return records

    # Support MINI_MODE to load from mini_data_out.json (16 graphs, 2/domain)
    mini_mode = os.environ.get("MINI_MODE", "0") == "1"
    if mini_mode:
        mini_path = DATA_DIR / "mini_data_out.json"
        new_recs = _load_split(mini_path)
        all_records.extend(new_recs)
    else:
        for fpath in sorted(DATA_DIR.glob("full_data_out_*.json")):
            new_recs = _load_split(fpath)
            all_records.extend(new_recs)
            gc.collect()
            if max_graphs and len(all_records) >= max_graphs:
                all_records = all_records[:max_graphs]
                break

    domain_counts = Counter(r["domain"] for r in all_records)
    logger.info(f"Loaded {len(all_records)} unique graphs across {len(domain_counts)} domains")
    for d, c in sorted(domain_counts.items()):
        logger.info(f"  {d}: {c} graphs")

    return all_records


# ================================================================
# GRAPH CONSTRUCTION
# ================================================================

def build_igraph(record: dict, prune_percentile: int) -> igraph.Graph:
    """Build a pruned igraph.Graph from a parsed graph record."""
    nodes = record["nodes"]
    links = record["links"]

    node_ids = [n["node_id"] for n in nodes]
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    layers = []
    feature_types = []
    features = []
    for n in nodes:
        layer_val = n.get("layer", 0)
        try:
            layers.append(int(layer_val))
        except (ValueError, TypeError):
            layers.append(0)
        feature_types.append(n.get("feature_type", ""))
        feat_val = n.get("feature", 0)
        try:
            features.append(int(feat_val))
        except (ValueError, TypeError):
            features.append(0)

    all_weights = [abs(link.get("weight", 0.0)) for link in links]
    threshold = float(np.percentile(all_weights, prune_percentile)) if all_weights else 0.0

    edges = []
    edge_weights = []
    for link in links:
        w = abs(link.get("weight", 0.0))
        if w >= threshold:
            src_idx = node_id_to_idx.get(link["source"])
            tgt_idx = node_id_to_idx.get(link["target"])
            if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
                edges.append((src_idx, tgt_idx))
                edge_weights.append(w)

    g = igraph.Graph(n=len(node_ids), edges=edges, directed=True)
    g.vs["node_id"] = node_ids
    g.vs["layer"] = layers
    g.vs["feature_type"] = feature_types
    g.vs["feature"] = features
    if edge_weights:
        g.es["weight"] = edge_weights

    g = g.simplify(multiple=True, loops=True, combine_edges="max")

    isolated = [v.index for v in g.vs if g.degree(v) == 0]
    if isolated:
        g.delete_vertices(isolated)

    if not g.is_dag():
        raise ValueError(f"Graph is not DAG after pruning at {prune_percentile}%")

    return g


# ================================================================
# ISOCLASS MAPPING (3-node)
# ================================================================

def build_isoclass_mapping(size: int) -> tuple[dict, list[int]]:
    """Build mapping from igraph isoclass ID to graph properties."""
    n_classes = 16 if size == 3 else 218
    mapping: dict[int, dict] = {}
    dag_valid_connected: list[int] = []

    for cls_id in range(n_classes):
        g = igraph.Graph.Isoclass(n=size, cls=cls_id, directed=True)
        edge_list = g.get_edgelist()
        n_edges = len(edge_list)
        is_connected = g.is_connected(mode="weak")
        is_dag_val = g.is_dag()

        mapping[cls_id] = {
            "edges": edge_list,
            "n_edges": n_edges,
            "is_connected": is_connected,
            "is_dag": is_dag_val,
        }

        if is_connected and is_dag_val:
            dag_valid_connected.append(cls_id)

    return mapping, dag_valid_connected


def identify_3node_man_labels(mapping: dict, dag_valid: list[int]) -> dict[int, str]:
    """Identify MAN labels for 3-node DAG-valid types by edge/degree structure."""
    names: dict[int, str] = {}
    for cls_id in dag_valid:
        n_edges = mapping[cls_id]["n_edges"]
        edges = mapping[cls_id]["edges"]
        g = igraph.Graph(n=3, edges=edges, directed=True)
        in_degs = g.indegree()
        out_degs = g.outdegree()

        if n_edges == 3:
            names[cls_id] = "030T"
        elif n_edges == 2:
            if max(out_degs) == 2:
                names[cls_id] = "021D"
            elif max(in_degs) == 2:
                names[cls_id] = "021U"
            else:
                names[cls_id] = "021C"
        else:
            names[cls_id] = f"unknown_{n_edges}edges"

    return names


# ================================================================
# MOTIF CENSUS
# ================================================================

def compute_motif_census(
    g: igraph.Graph, size: int, dag_valid_ids: list[int],
) -> dict[int, int]:
    """Run motifs_randesu, extract counts for DAG-valid connected types."""
    raw = g.motifs_randesu(size=size)
    counts = [0 if (x != x) else int(x) for x in raw]
    valid_counts = {idx: counts[idx] for idx in dag_valid_ids}
    return valid_counts


# ================================================================
# NULL MODEL GENERATION (degree-preserving DAG edge swaps)
# ================================================================

def _generate_null_edges(
    n_nodes: int, edges: list[tuple[int, int]], topo_rank: list[int],
    n_swap_attempts: int, seed: int,
) -> tuple[list[tuple[int, int]], int]:
    """Generate one DAG-preserving random graph via degree-preserving edge swaps."""
    rng = random.Random(seed)
    edge_list = list(edges)
    n_edges = len(edge_list)
    if n_edges < 2:
        return edge_list, 0

    adj_set = set(edge_list)
    accepted = 0

    for _ in range(n_swap_attempts):
        i1 = rng.randint(0, n_edges - 1)
        i2 = rng.randint(0, n_edges - 1)
        if i1 == i2:
            continue

        u1, v1 = edge_list[i1]
        u2, v2 = edge_list[i2]

        if u1 == u2 or v1 == v2:
            continue

        new_e1 = (u1, v2)
        new_e2 = (u2, v1)

        if new_e1 in adj_set or new_e2 in adj_set:
            continue

        if topo_rank[u1] >= topo_rank[v2] or topo_rank[u2] >= topo_rank[v1]:
            continue

        adj_set.discard((u1, v1))
        adj_set.discard((u2, v2))
        adj_set.add(new_e1)
        adj_set.add(new_e2)
        edge_list[i1] = new_e1
        edge_list[i2] = new_e2
        accepted += 1

    return edge_list, accepted


def _null_model_batch_worker(args: tuple) -> list[dict[int, int]]:
    """Worker for ProcessPoolExecutor: generate batch of null models + census."""
    n_nodes, edges, topo_rank, n_swap_attempts, seeds, size, dag_valid_ids = args

    results = []
    for seed in seeds:
        new_edges, _accepted = _generate_null_edges(
            n_nodes, edges, topo_rank, n_swap_attempts, seed
        )
        g_null = igraph.Graph(n=n_nodes, edges=new_edges, directed=True)
        raw = g_null.motifs_randesu(size=size)
        counts = [0 if (x != x) else int(x) for x in raw]
        census = {idx: counts[idx] for idx in dag_valid_ids}
        results.append(census)
        del g_null

    return results


def generate_null_census_parallel(
    g: igraph.Graph, size: int, dag_valid_ids: list[int],
    n_null: int, n_workers: int,
) -> list[dict[int, int]]:
    """Generate n_null null models in parallel and return their motif censuses."""
    n_nodes = g.vcount()
    edges = [tuple(e.tuple) for e in g.es]

    topo_order = g.topological_sorting()
    topo_rank = [0] * n_nodes
    for rank, node in enumerate(topo_order):
        topo_rank[node] = rank

    n_swap_attempts = SWAP_MULTIPLIER * len(edges)

    all_seeds = list(range(n_null))
    batch_size = max(1, math.ceil(n_null / n_workers))
    batches = []
    for i in range(0, n_null, batch_size):
        batch_seeds = all_seeds[i:i + batch_size]
        batches.append((
            n_nodes, edges, topo_rank, n_swap_attempts,
            batch_seeds, size, dag_valid_ids,
        ))

    all_results: list[dict[int, int]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_null_model_batch_worker, b): idx
                   for idx, b in enumerate(batches)}
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception:
                logger.exception(f"Null model batch {futures[future]} failed")

    return all_results


# ================================================================
# Z-SCORES AND SIGNIFICANCE PROFILES
# ================================================================

def compute_zscores(
    real_counts: dict[int, int], null_counts_list: list[dict[int, int]],
    dag_valid_ids: list[int],
) -> dict[str, Any]:
    """Compute Z-scores from real counts vs null distribution."""
    z_scores: dict[int, float] = {}

    for motif_id in dag_valid_ids:
        real_val = real_counts[motif_id]
        null_vals = np.array([nc[motif_id] for nc in null_counts_list], dtype=float)

        mean_null = float(np.mean(null_vals))
        std_null = float(np.std(null_vals))

        if std_null == 0:
            if real_val == mean_null:
                z = 0.0
            else:
                z = 10.0 if real_val > mean_null else -10.0
        else:
            z = (real_val - mean_null) / std_null

        z_scores[motif_id] = float(z)

    return {
        "z_scores": z_scores,
        "raw_counts": real_counts,
        "null_means": {
            mid: float(np.mean([nc[mid] for nc in null_counts_list]))
            for mid in dag_valid_ids
        },
        "null_stds": {
            mid: float(np.std([nc[mid] for nc in null_counts_list]))
            for mid in dag_valid_ids
        },
    }


# ================================================================
# EDGE-OVERLAP SIMILARITY (Sun 2025 Baseline) - NEW
# ================================================================

def extract_feature_identity(g_pruned: igraph.Graph) -> tuple[set, set, dict]:
    """Extract feature-identity sets from a pruned graph.

    Feature identity = (layer, feature_index), collapsing across ctx_idx.

    Returns:
        feature_set: set of (layer, feature_index) tuples
        edge_set: set of ((src_layer, src_feat), (tgt_layer, tgt_feat)) tuples
        edge_weights_dict: {edge_tuple: max_weight}
    """
    feature_set: set[tuple[int, int]] = set()
    node_to_feat: dict[int, tuple[int, int]] = {}

    for v in g_pruned.vs:
        feat_id = (int(v["layer"]), int(v["feature"]))
        feature_set.add(feat_id)
        node_to_feat[v.index] = feat_id

    edge_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edge_weights_dict: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    has_weight = "weight" in g_pruned.es.attributes() if g_pruned.ecount() > 0 else False
    for e in g_pruned.es:
        src_feat = node_to_feat[e.source]
        tgt_feat = node_to_feat[e.target]
        edge_tuple = (src_feat, tgt_feat)
        edge_set.add(edge_tuple)
        w = e["weight"] if has_weight else 0.0
        edge_weights_dict[edge_tuple] = max(edge_weights_dict.get(edge_tuple, 0.0), w)

    return feature_set, edge_set, edge_weights_dict


# ================================================================
# GRAPH STATISTICS (8D)
# ================================================================

def compute_graph_stats(g: igraph.Graph) -> np.ndarray:
    """8D graph statistics vector."""
    n = g.vcount()
    m = g.ecount()
    degs = np.array(g.degree(), dtype=float) if n > 0 else np.array([0.0])
    layers = set(g.vs["layer"]) if "layer" in g.vs.attributes() else {0}
    has_weight = "weight" in g.es.attributes() if m > 0 else False
    weights = np.array(g.es["weight"], dtype=float) if has_weight else np.array([0.0])

    try:
        diam = g.diameter(directed=False)
    except Exception:
        diam = 0

    return np.array([
        n, m, g.density() if n > 1 else 0.0,
        float(np.mean(degs)), float(np.max(degs)),
        diam, len(layers), float(np.mean(weights)),
    ])


# ================================================================
# SAFE SPECTRAL CLUSTERING (dense eigensolver, no ARPACK hangs)
# ================================================================

def _safe_spectral_cluster(
    affinity: np.ndarray, n_clusters: int, seed: int = 42, n_init: int = 5,
) -> np.ndarray:
    """Spectral clustering using dense LAPACK eigensolver (avoids ARPACK hangs).

    For 200x200 matrices, dense is faster and more reliable than ARPACK.
    """
    n = affinity.shape[0]
    # Degree matrix
    degrees = affinity.sum(axis=1)
    degrees = np.maximum(degrees, 1e-10)  # avoid division by zero
    d_inv_sqrt = 1.0 / np.sqrt(degrees)

    # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
    L_norm = np.eye(n) - (d_inv_sqrt[:, None] * affinity * d_inv_sqrt[None, :])

    # Dense eigendecomposition (LAPACK, reliable, won't hang)
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # Take first n_clusters eigenvectors (smallest eigenvalues)
    embedding = eigenvectors[:, :n_clusters].copy()

    # Normalize rows
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding = embedding / norms

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init, max_iter=300)
    return km.fit_predict(embedding)


# ================================================================
# CLUSTERING WITH PRECOMPUTED AFFINITY
# ================================================================

def cluster_with_affinity(
    affinity_matrix: np.ndarray, true_labels: np.ndarray,
    k_values: list[int], seed: int = 42,
) -> dict[int, dict]:
    """Spectral clustering on precomputed affinity at multiple K values."""
    n_samples = affinity_matrix.shape[0]
    results: dict[int, dict] = {}

    # Ensure PSD: clip negatives, set diagonal to 1
    affinity = np.clip(affinity_matrix, 0, None).copy()
    np.fill_diagonal(affinity, 1.0)

    for K in k_values:
        if K >= n_samples or K < 2:
            continue
        try:
            pred_labels = _safe_spectral_cluster(affinity, K, seed, n_init=10)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            ari = adjusted_rand_score(true_labels, pred_labels)
            results[K] = {
                "nmi": float(nmi),
                "ari": float(ari),
                "pred_labels": pred_labels.tolist(),
            }
        except Exception as e:
            logger.warning(f"Clustering K={K} failed: {e}")
            results[K] = {"nmi": 0.0, "ari": 0.0, "pred_labels": []}

    return results


# ================================================================
# PERMUTATION TEST FOR NMI SIGNIFICANCE
# ================================================================

def permutation_test_nmi(
    affinity_matrix: np.ndarray, true_labels: np.ndarray,
    real_nmi: float, K: int, n_permutations: int, seed: int = 42,
) -> float:
    """Compute p-value via label permutation test."""
    affinity = np.clip(affinity_matrix, 0, None).copy()
    np.fill_diagonal(affinity, 1.0)
    n_samples = affinity.shape[0]

    if K >= n_samples or K < 2:
        return 1.0

    # Precompute spectral embedding ONCE (clustering labels are deterministic
    # given the affinity matrix, so only NMI changes with label permutations)
    try:
        pred = _safe_spectral_cluster(affinity, K, seed, n_init=5)
    except Exception:
        return 1.0

    null_nmis = []
    for perm_i in range(n_permutations):
        shuffled = np.random.RandomState(seed + perm_i).permutation(true_labels)
        null_nmis.append(normalized_mutual_info_score(shuffled, pred))

    p_value = (np.sum(np.array(null_nmis) >= real_nmi) + 1) / (n_permutations + 1)
    return float(p_value)


# ================================================================
# FINGERPRINT STABILITY ANALYSIS
# ================================================================

def compute_stability(
    sim_matrix: np.ndarray, domain_labels: list[str],
) -> dict[str, Any]:
    """Compute within-domain vs between-domain similarity statistics."""
    domains = sorted(set(domain_labels))
    n = len(domain_labels)

    within_sims: list[float] = []
    between_sims: list[float] = []
    per_domain_within: dict[str, float] = {}

    domain_indices: dict[str, list[int]] = {}
    for i, lab in enumerate(domain_labels):
        domain_indices.setdefault(lab, []).append(i)

    for d in domains:
        d_idx = domain_indices[d]
        d_within = []
        for a in range(len(d_idx)):
            for b in range(a + 1, len(d_idx)):
                val = sim_matrix[d_idx[a], d_idx[b]]
                within_sims.append(val)
                d_within.append(val)
        per_domain_within[d] = float(np.mean(d_within)) if d_within else 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if domain_labels[i] != domain_labels[j]:
                between_sims.append(sim_matrix[i, j])

    within_arr = np.array(within_sims, dtype=float)
    between_arr = np.array(between_sims, dtype=float)

    mean_w = float(np.mean(within_arr)) if len(within_arr) > 0 else 0.0
    mean_b = float(np.mean(between_arr)) if len(between_arr) > 0 else 0.0
    var_w = float(np.var(within_arr)) if len(within_arr) > 0 else 0.0
    var_b = float(np.var(between_arr)) if len(between_arr) > 0 else 0.0

    # Fisher Discriminant Ratio
    denom = var_w + var_b
    fdr_discriminant = (mean_w - mean_b) ** 2 / denom if denom > 0 else 0.0

    # Cohen's d
    pooled_std = math.sqrt((var_w + var_b) / 2)
    cohens_d = (mean_w - mean_b) / pooled_std if pooled_std > 0 else 0.0

    return {
        "mean_within": mean_w,
        "std_within": float(np.std(within_arr)) if len(within_arr) > 0 else 0.0,
        "mean_between": mean_b,
        "std_between": float(np.std(between_arr)) if len(between_arr) > 0 else 0.0,
        "fdr_discriminant": float(fdr_discriminant),
        "cohens_d": float(cohens_d),
        "per_domain_within": per_domain_within,
        "n_within_pairs": len(within_sims),
        "n_between_pairs": len(between_sims),
    }


# ================================================================
# HELPER: SIMILARITY MATRIX STATS
# ================================================================

def sim_matrix_stats(mat: np.ndarray, domain_labels: list[str]) -> dict[str, float]:
    """Compute distribution stats for a similarity matrix."""
    n = mat.shape[0]
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    vals = mat[triu_idx]

    domain_indices: dict[str, list[int]] = {}
    for i, lab in enumerate(domain_labels):
        domain_indices.setdefault(lab, []).append(i)

    within_vals = []
    between_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            if domain_labels[i] == domain_labels[j]:
                within_vals.append(mat[i, j])
            else:
                between_vals.append(mat[i, j])

    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        "frac_zero": float(np.mean(vals == 0)),
        "within_domain_mean": float(np.mean(within_vals)) if within_vals else 0.0,
        "between_domain_mean": float(np.mean(between_vals)) if between_vals else 0.0,
    }


# ================================================================
# MAP CLUSTERS TO DOMAIN LABELS
# ================================================================

def map_clusters_to_domains(
    pred_labels: list[int], domain_labels: list[str],
) -> list[str]:
    """Map cluster IDs to majority domain labels for per-example predictions."""
    cluster_to_domain: dict[int, str] = {}
    for cluster_id in set(pred_labels):
        indices = [i for i, p in enumerate(pred_labels) if p == cluster_id]
        domains_in_cluster = [domain_labels[i] for i in indices]
        most_common = Counter(domains_in_cluster).most_common(1)[0][0]
        cluster_to_domain[cluster_id] = most_common
    return [cluster_to_domain[p] for p in pred_labels]


# ================================================================
# MAIN PIPELINE
# ================================================================

@logger.catch
def main():
    t_start = time.time()
    n_workers = max(1, NUM_CPUS - 1)

    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Config: MAX_EXAMPLES={MAX_EXAMPLES}, N_NULL={N_NULL_MODELS}, "
                f"N_PERM={N_PERMUTATIONS}")
    logger.info(f"Time budget: {TOTAL_TIME_BUDGET_S}s")

    # ==============================================================
    # LOAD DATA
    # ==============================================================
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    all_records = load_all_graphs(MAX_EXAMPLES or None)
    if not all_records:
        logger.error("No graphs loaded!")
        return
    n_total = len(all_records)
    domain_labels = [r["domain"] for r in all_records]
    domains_sorted = sorted(set(domain_labels))
    domain_counts = dict(Counter(domain_labels))
    le = LabelEncoder()
    true_labels = le.fit_transform(domain_labels)

    # ==============================================================
    # BUILD ISOCLASS MAPPING
    # ==============================================================
    logger.info("=" * 60)
    logger.info("BUILDING ISOCLASS MAPPING")
    mapping_3, dag_valid_3 = build_isoclass_mapping(3)
    names_3 = identify_3node_man_labels(mapping_3, dag_valid_3)
    logger.info(f"3-node DAG-valid types: {len(dag_valid_3)} IDs: {dag_valid_3}")
    for cls_id in dag_valid_3:
        logger.info(f"  ID {cls_id}: {names_3[cls_id]}")

    # ==============================================================
    # BUILD PRUNED GRAPHS
    # ==============================================================
    logger.info("=" * 60)
    logger.info(f"BUILDING PRUNED GRAPHS (percentile={PRUNE_PERCENTILE})")
    pruned_graphs: list[igraph.Graph | None] = [None] * n_total
    valid_indices: list[int] = []

    for i, rec in enumerate(all_records):
        try:
            g = build_igraph(rec, PRUNE_PERCENTILE)
            pruned_graphs[i] = g
            valid_indices.append(i)
        except Exception:
            logger.exception(f"Failed building graph {i} ({rec['domain']})")

    logger.info(f"Built {len(valid_indices)}/{n_total} pruned graphs")
    if valid_indices:
        node_counts = [pruned_graphs[i].vcount() for i in valid_indices]
        logger.info(f"  Nodes: min={min(node_counts)}, median={np.median(node_counts):.0f}, "
                     f"max={max(node_counts)}")

    # ==============================================================
    # PHASE A: 3-NODE MOTIF CENSUS
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE A: 3-node motif census on all graphs")

    motif_count_ratios = np.zeros((n_total, len(dag_valid_3)))
    motif_zscore_matrix = np.zeros((n_total, len(dag_valid_3)))
    per_graph_motif_results: list[dict] = [{}] * n_total

    # Step 1: Real census
    real_census_3: dict[int, dict[int, int]] = {}
    for gi in valid_indices:
        g = pruned_graphs[gi]
        try:
            real_census_3[gi] = compute_motif_census(g, 3, dag_valid_3)
            total = sum(real_census_3[gi][mid] for mid in dag_valid_3)
            if total > 0:
                motif_count_ratios[gi] = [
                    real_census_3[gi][mid] / total for mid in dag_valid_3
                ]
        except Exception:
            logger.exception(f"Motif census failed for graph {gi}")

    logger.info(f"  Real census computed for {len(real_census_3)} graphs")

    # Step 2: Null models with time budgeting
    logger.info(f"  Generating {N_NULL_MODELS} null models per graph")

    # Timing calibration on median-sized graph
    n_null_actual = N_NULL_MODELS
    if real_census_3:
        sorted_by_edges = sorted(
            real_census_3.keys(),
            key=lambda k: pruned_graphs[k].ecount()
        )
        calib_key = sorted_by_edges[len(sorted_by_edges) // 2]
        g_calib = pruned_graphs[calib_key]
        t_calib = time.time()
        calib_n = min(10, n_null_actual)
        _calib_res = generate_null_census_parallel(
            g_calib, 3, dag_valid_3, calib_n, n_workers
        )
        t_calib_elapsed = time.time() - t_calib
        time_per_null = t_calib_elapsed / max(calib_n, 1)
        estimated_total = time_per_null * n_null_actual * len(real_census_3)
        logger.info(f"  Calibration: {calib_n} nulls in {t_calib_elapsed:.2f}s "
                     f"({time_per_null:.3f}s/model)")
        logger.info(f"  Estimated total: {estimated_total:.0f}s ({estimated_total / 60:.1f}min)")

        if estimated_total > MAX_NULL_TIME_S:
            n_null_actual = max(
                10,
                int(MAX_NULL_TIME_S / (time_per_null * len(real_census_3)))
            )
            logger.info(f"  Adjusted to {n_null_actual} null models to fit budget")

    null_census_3: dict[int, list[dict[int, int]]] = {}
    t_null_start = time.time()
    sorted_gis = sorted(real_census_3.keys())

    for idx, gi in enumerate(sorted_gis):
        g = pruned_graphs[gi]
        t_graph = time.time()
        try:
            null_census_3[gi] = generate_null_census_parallel(
                g, 3, dag_valid_3, n_null_actual, n_workers
            )
        except Exception:
            logger.exception(f"Null models failed for graph {gi}")
            continue

        elapsed = time.time() - t_graph
        if (idx + 1) % 20 == 0 or idx == 0:
            logger.info(
                f"  [{idx + 1}/{len(sorted_gis)}] Graph {gi}: "
                f"{n_null_actual} nulls in {elapsed:.1f}s "
                f"({g.vcount()}n, {g.ecount()}e)"
            )

        # Dynamic adjustment
        total_null_elapsed = time.time() - t_null_start
        remaining_graphs = len(sorted_gis) - (idx + 1)
        if remaining_graphs > 0 and total_null_elapsed > MAX_NULL_TIME_S * 0.8:
            if n_null_actual > 30:
                n_null_actual = max(10, n_null_actual // 2)
                logger.info(f"  Time pressure: reducing to {n_null_actual} null models")

    logger.info(f"  Null models complete: {time.time() - t_null_start:.1f}s total")

    # Step 3: Compute Z-scores
    for gi in real_census_3:
        if gi not in null_census_3 or not null_census_3[gi]:
            continue
        result = compute_zscores(real_census_3[gi], null_census_3[gi], dag_valid_3)
        per_graph_motif_results[gi] = result
        for j, mid in enumerate(dag_valid_3):
            motif_zscore_matrix[gi, j] = result["z_scores"][mid]

    # Log per-motif aggregate Z-scores
    for j, mid in enumerate(dag_valid_3):
        z_vals = motif_zscore_matrix[valid_indices, j]
        logger.info(
            f"  {names_3[mid]} (ID {mid}): mean Z={np.mean(z_vals):.2f} "
            f"+/- {np.std(z_vals):.2f}"
        )

    # Free null model data
    del null_census_3
    gc.collect()

    # ==============================================================
    # PHASE B: EDGE-OVERLAP SIMILARITY (Sun 2025 Baseline)
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE B: Edge-overlap similarity (Sun 2025 baseline)")

    feature_sets: list[tuple[set, set, dict] | None] = [None] * n_total
    for gi in valid_indices:
        g = pruned_graphs[gi]
        feature_sets[gi] = extract_feature_identity(g)

    # Compute pairwise similarity matrices
    node_jaccard_matrix = np.zeros((n_total, n_total))
    edge_jaccard_matrix = np.zeros((n_total, n_total))
    weight_spearman_matrix = np.zeros((n_total, n_total))

    np.fill_diagonal(node_jaccard_matrix, 1.0)
    np.fill_diagonal(edge_jaccard_matrix, 1.0)
    np.fill_diagonal(weight_spearman_matrix, 1.0)

    t_overlap_start = time.time()
    n_pairs = len(valid_indices) * (len(valid_indices) - 1) // 2
    pair_count = 0

    for a_idx in range(len(valid_indices)):
        i = valid_indices[a_idx]
        fi, ei, wi = feature_sets[i]
        for b_idx in range(a_idx + 1, len(valid_indices)):
            j = valid_indices[b_idx]
            fj, ej, wj = feature_sets[j]

            # Node-overlap Jaccard
            node_inter = len(fi & fj)
            node_union = len(fi | fj)
            nj = node_inter / node_union if node_union > 0 else 0.0
            node_jaccard_matrix[i, j] = node_jaccard_matrix[j, i] = nj

            # Edge-overlap Jaccard
            edge_inter = len(ei & ej)
            edge_union = len(ei | ej)
            ej_val = edge_inter / edge_union if edge_union > 0 else 0.0
            edge_jaccard_matrix[i, j] = edge_jaccard_matrix[j, i] = ej_val

            # Weighted overlap: Spearman of shared edge weights
            shared_edges = ei & ej
            if len(shared_edges) >= 5:
                w_i = [wi[e] for e in shared_edges]
                w_j = [wj[e] for e in shared_edges]
                rho, _ = scipy_stats.spearmanr(w_i, w_j)
                if not np.isnan(rho):
                    weight_spearman_matrix[i, j] = rho
                    weight_spearman_matrix[j, i] = rho

            pair_count += 1

        if (a_idx + 1) % 50 == 0:
            logger.info(
                f"  Edge-overlap: {pair_count}/{n_pairs} pairs "
                f"({time.time() - t_overlap_start:.1f}s)"
            )

    logger.info(f"  Edge-overlap complete: {time.time() - t_overlap_start:.1f}s")

    # Report overlap statistics
    node_jac_stats = sim_matrix_stats(node_jaccard_matrix, domain_labels)
    edge_jac_stats = sim_matrix_stats(edge_jaccard_matrix, domain_labels)
    weight_sp_stats = sim_matrix_stats(weight_spearman_matrix, domain_labels)

    logger.info(f"  Node Jaccard: mean={node_jac_stats['mean']:.4f}, "
                f"within={node_jac_stats['within_domain_mean']:.4f}, "
                f"between={node_jac_stats['between_domain_mean']:.4f}")
    logger.info(f"  Edge Jaccard: mean={edge_jac_stats['mean']:.4f}, "
                f"within={edge_jac_stats['within_domain_mean']:.4f}, "
                f"between={edge_jac_stats['between_domain_mean']:.4f}")
    logger.info(f"  Weight Spearman: mean={weight_sp_stats['mean']:.4f}, "
                f"frac_zero={weight_sp_stats['frac_zero']:.4f}")

    # Free raw node/link data after feature extraction
    for rec in all_records:
        rec.pop("nodes", None)
        rec.pop("links", None)
    gc.collect()

    # ==============================================================
    # PHASE C: MOTIF SPECTRUM SIMILARITY
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE C: Motif spectrum similarity")

    # 200x200 cosine similarity from 4D count-ratio vectors
    motif_cosine_sim = cosine_similarity(motif_count_ratios)
    motif_euclidean_dist = cdist(motif_count_ratios, motif_count_ratios, "euclidean")

    logger.info(f"  Motif cosine sim: mean={np.mean(motif_cosine_sim[np.triu_indices(n_total, k=1)]):.4f}")

    # ==============================================================
    # PHASE D: GRAPH STATISTICS SIMILARITY
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE D: Graph statistics similarity")

    graph_stats_matrix = np.zeros((n_total, 8))
    for gi in valid_indices:
        g = pruned_graphs[gi]
        graph_stats_matrix[gi] = compute_graph_stats(g)

    # Normalize then cosine similarity
    scaler = StandardScaler()
    graph_stats_scaled = scaler.fit_transform(graph_stats_matrix)
    graph_stats_cosine_sim = cosine_similarity(graph_stats_scaled)

    logger.info(f"  Graph stats cosine sim: "
                f"mean={np.mean(graph_stats_cosine_sim[np.triu_indices(n_total, k=1)]):.4f}")

    # Free pruned graphs
    del pruned_graphs
    gc.collect()

    # ==============================================================
    # PHASE E: CLUSTERING COMPARISON
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE E: Clustering comparison")

    # Define all similarity/affinity matrices to cluster
    motif_affinity = (motif_cosine_sim + 1) / 2
    graph_stats_affinity = (graph_stats_cosine_sim + 1) / 2
    weight_sp_affinity = (weight_spearman_matrix + 1) / 2

    similarity_sources = {
        "node_jaccard": node_jaccard_matrix,
        "edge_jaccard": edge_jaccard_matrix,
        "weight_spearman": weight_sp_affinity,
        "motif_count_ratio": motif_affinity,
        "graph_stats": graph_stats_affinity,
        "motif_plus_graph_stats": (motif_affinity + graph_stats_affinity) / 2,
        "motif_plus_edge": (motif_affinity + edge_jaccard_matrix) / 2,
        "all_three": (motif_affinity + graph_stats_affinity + edge_jaccard_matrix) / 3,
    }

    clustering_results: dict[str, dict] = {}

    for name, affinity_matrix in similarity_sources.items():
        t_clust = time.time()
        res = cluster_with_affinity(affinity_matrix, true_labels, CLUSTER_K_VALUES, SEED)
        elapsed = time.time() - t_clust

        # Find best K by NMI
        best_k = max(res, key=lambda k: res[k]["nmi"]) if res else 8
        best_nmi = res[best_k]["nmi"] if res else 0.0
        best_ari = res[best_k]["ari"] if res else 0.0

        clustering_results[name] = {
            "results_by_k": {
                str(k): {"nmi": v["nmi"], "ari": v["ari"]}
                for k, v in res.items()
            },
            "best_k": best_k,
            "best_nmi": best_nmi,
            "best_ari": best_ari,
            "pred_labels_best_k": res[best_k]["pred_labels"] if res and best_k in res else [],
        }

        logger.info(f"  {name}: best_K={best_k}, NMI={best_nmi:.4f}, "
                     f"ARI={best_ari:.4f} ({elapsed:.1f}s)")

    # Check time budget before permutation tests
    elapsed_total = time.time() - t_start
    remaining_time = TOTAL_TIME_BUDGET_S - elapsed_total
    logger.info(f"  Time elapsed: {elapsed_total:.0f}s, remaining: {remaining_time:.0f}s")

    # Permutation tests - only for methods at their best K
    n_perm_actual = N_PERMUTATIONS
    if remaining_time < 600:
        n_perm_actual = min(100, N_PERMUTATIONS)
        logger.info(f"  Reduced permutations to {n_perm_actual} due to time pressure")
    elif remaining_time < 1200:
        n_perm_actual = min(200, N_PERMUTATIONS)
        logger.info(f"  Reduced permutations to {n_perm_actual}")

    for name, affinity_matrix in similarity_sources.items():
        if time.time() - t_start > TOTAL_TIME_BUDGET_S * 0.85:
            logger.warning(f"  Skipping permutation test for {name} (time pressure)")
            clustering_results[name]["perm_p_value"] = -1.0
            continue

        best_k = clustering_results[name]["best_k"]
        real_nmi = clustering_results[name]["best_nmi"]

        t_perm = time.time()
        p_val = permutation_test_nmi(
            affinity_matrix, true_labels, real_nmi, best_k,
            n_perm_actual, SEED,
        )
        clustering_results[name]["perm_p_value"] = p_val
        logger.info(f"  {name}: p-value={p_val:.4f} ({time.time() - t_perm:.1f}s, "
                     f"{n_perm_actual} perms)")

    # ==============================================================
    # PHASE F: FINGERPRINT STABILITY ANALYSIS
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE F: Fingerprint stability analysis")

    stability_results: dict[str, dict] = {}

    stability_sources = {
        "node_jaccard": node_jaccard_matrix,
        "edge_jaccard": edge_jaccard_matrix,
        "motif_count_ratio": motif_affinity,
        "graph_stats": graph_stats_affinity,
        "motif_plus_graph_stats": (motif_affinity + graph_stats_affinity) / 2,
        "all_three": (motif_affinity + graph_stats_affinity + edge_jaccard_matrix) / 3,
    }

    for name, sim_matrix in stability_sources.items():
        stab = compute_stability(sim_matrix, domain_labels)
        stability_results[name] = stab
        logger.info(f"  {name}: within={stab['mean_within']:.4f}, "
                     f"between={stab['mean_between']:.4f}, "
                     f"FDR={stab['fdr_discriminant']:.4f}, "
                     f"Cohen's d={stab['cohens_d']:.4f}")

    # ==============================================================
    # PHASE G: COMPLEMENTARITY ANALYSIS
    # ==============================================================
    logger.info("=" * 60)
    logger.info("PHASE G: Complementarity analysis")

    # Get cluster labels from each core method at K=8
    best_K_comp = 8
    comp_sources = {
        "motif": motif_affinity,
        "edge_overlap": edge_jaccard_matrix,
        "graph_stats": graph_stats_affinity,
    }

    cluster_labels_comp: dict[str, np.ndarray] = {}
    for name, aff in comp_sources.items():
        aff_clip = np.clip(aff, 0, None).copy()
        np.fill_diagonal(aff_clip, 1.0)
        try:
            if best_K_comp < n_total:
                labels = _safe_spectral_cluster(aff_clip, best_K_comp, SEED, n_init=10)
                cluster_labels_comp[name] = labels
        except Exception as e:
            logger.warning(f"  Complementarity clustering failed for {name}: {e}")

    # Pairwise NMI between clustering solutions
    complementarity_nmi: dict[str, float] = {}
    comp_names = sorted(cluster_labels_comp.keys())
    for a_name in comp_names:
        for b_name in comp_names:
            if a_name < b_name:
                nmi_ab = normalized_mutual_info_score(
                    cluster_labels_comp[a_name],
                    cluster_labels_comp[b_name],
                )
                key = f"{a_name}_vs_{b_name}"
                complementarity_nmi[key] = float(nmi_ab)
                logger.info(f"  {key}: NMI={nmi_ab:.4f}")

    # Check if combined beats individual
    individual_methods = ["motif_count_ratio", "edge_jaccard", "graph_stats"]
    combined_methods = ["motif_plus_graph_stats", "motif_plus_edge", "all_three"]

    best_individual_nmi = max(
        (clustering_results[m]["best_nmi"] for m in individual_methods
         if m in clustering_results),
        default=0.0,
    )
    best_combined_nmi = max(
        (clustering_results[m]["best_nmi"] for m in combined_methods
         if m in clustering_results),
        default=0.0,
    )
    combined_beats_individual = best_combined_nmi > best_individual_nmi

    logger.info(f"  Best individual NMI: {best_individual_nmi:.4f}")
    logger.info(f"  Best combined NMI: {best_combined_nmi:.4f}")
    logger.info(f"  Combined beats individual: {combined_beats_individual}")

    # ==============================================================
    # BUILD OUTPUT
    # ==============================================================
    logger.info("=" * 60)
    logger.info("BUILDING OUTPUT")

    # Per-motif universal overrepresentation summary
    universal_overrep: dict[str, dict] = {}
    for j, mid in enumerate(dag_valid_3):
        z_vals_per_domain: dict[str, float] = {}
        for d in domains_sorted:
            d_indices = [i for i, lab in enumerate(domain_labels) if lab == d]
            d_z = [motif_zscore_matrix[i, j] for i in d_indices if i in real_census_3]
            if d_z:
                z_vals_per_domain[d] = float(np.mean(d_z))
        all_z = [v for v in z_vals_per_domain.values()]
        n_domains_z_gt_2 = sum(1 for z in all_z if z > 2)
        universal_overrep[str(mid)] = {
            "man_label": names_3[mid],
            "mean_z": float(np.mean(all_z)) if all_z else 0.0,
            "std_z": float(np.std(all_z)) if all_z else 0.0,
            "n_domains_z_gt_2": n_domains_z_gt_2,
            "per_domain_mean_z": z_vals_per_domain,
        }

    # Per-graph total triads
    total_triads = []
    ffl_counts = []
    ffl_fractions = []
    ffl_id = [mid for mid in dag_valid_3 if names_3[mid] == "030T"]
    ffl_id = ffl_id[0] if ffl_id else dag_valid_3[-1]

    for gi in sorted(real_census_3.keys()):
        total = sum(real_census_3[gi][mid] for mid in dag_valid_3)
        total_triads.append(total)
        ffl_c = real_census_3[gi].get(ffl_id, 0)
        ffl_counts.append(ffl_c)
        ffl_fractions.append(ffl_c / total if total > 0 else 0.0)

    # Summary: find best method
    all_methods = list(clustering_results.keys())
    best_method = max(all_methods, key=lambda m: clustering_results[m]["best_nmi"])

    # Fingerprint best method
    stability_methods = list(stability_results.keys())
    fingerprint_best = max(
        stability_methods,
        key=lambda m: stability_results[m]["cohens_d"],
    )

    # Check if motif captures independent info from edge overlap
    motif_nmi_val = clustering_results.get("motif_count_ratio", {}).get("best_nmi", 0.0)
    edge_nmi_val = clustering_results.get("edge_jaccard", {}).get("best_nmi", 0.0)
    motif_edge_comp_nmi = complementarity_nmi.get("edge_overlap_vs_motif", -1.0)
    motif_captures_independent = motif_edge_comp_nmi < 0.5 if motif_edge_comp_nmi >= 0 else True

    # Aggregate result (stored in metadata)
    aggregate_results = {
        "metadata": {
            "n_graphs": n_total,
            "n_domains": len(domains_sorted),
            "prune_percentile": PRUNE_PERCENTILE,
            "n_null_models": n_null_actual,
            "n_permutations": n_perm_actual,
            "domains": domains_sorted,
            "domain_counts": domain_counts,
            "runtime_s": time.time() - t_start,
        },
        "phase_a_motif_census": {
            "dag_valid_3node_types": dag_valid_3,
            "man_labels": {str(k): v for k, v in names_3.items()},
            "per_graph_summary": {
                "mean_total_triads": float(np.mean(total_triads)) if total_triads else 0.0,
                "std_total_triads": float(np.std(total_triads)) if total_triads else 0.0,
                "mean_ffl_count": float(np.mean(ffl_counts)) if ffl_counts else 0.0,
                "mean_ffl_fraction": float(np.mean(ffl_fractions)) if ffl_fractions else 0.0,
            },
            "universal_overrepresentation": universal_overrep,
        },
        "phase_b_edge_overlap": {
            "node_jaccard_stats": node_jac_stats,
            "edge_jaccard_stats": edge_jac_stats,
            "weight_spearman_stats": weight_sp_stats,
        },
        "phase_e_clustering": {
            name: {
                "results_by_k": clustering_results[name]["results_by_k"],
                "best_k": clustering_results[name]["best_k"],
                "best_nmi": clustering_results[name]["best_nmi"],
                "best_ari": clustering_results[name]["best_ari"],
                "perm_p_value": clustering_results[name].get("perm_p_value", -1.0),
            }
            for name in clustering_results
        },
        "phase_f_fingerprint_stability": stability_results,
        "phase_g_complementarity": {
            "pairwise_cluster_nmi": complementarity_nmi,
            "combined_beats_individual": combined_beats_individual,
            "best_individual_nmi": best_individual_nmi,
            "best_combined_nmi": best_combined_nmi,
            "motif_is_identity_agnostic": True,
        },
        "summary": {
            "best_method": best_method,
            "best_nmi": clustering_results[best_method]["best_nmi"],
            "edge_overlap_nmi": edge_nmi_val,
            "motif_nmi": motif_nmi_val,
            "graph_stats_nmi": clustering_results.get("graph_stats", {}).get("best_nmi", 0.0),
            "motif_captures_independent_info": motif_captures_independent,
            "fingerprint_best_method": fingerprint_best,
        },
    }

    # ==============================================================
    # BUILD SCHEMA-COMPLIANT OUTPUT
    # ==============================================================
    # Get best method's predictions for per-example predict_ field
    best_pred_labels = clustering_results[best_method].get("pred_labels_best_k", [])
    motif_pred_labels = clustering_results.get("motif_count_ratio", {}).get(
        "pred_labels_best_k", []
    )
    edge_pred_labels = clustering_results.get("edge_jaccard", {}).get(
        "pred_labels_best_k", []
    )

    # Map cluster IDs to domain names
    if best_pred_labels:
        best_pred_domains = map_clusters_to_domains(best_pred_labels, domain_labels)
    else:
        best_pred_domains = ["unknown"] * n_total

    if motif_pred_labels:
        motif_pred_domains = map_clusters_to_domains(motif_pred_labels, domain_labels)
    else:
        motif_pred_domains = ["unknown"] * n_total

    if edge_pred_labels:
        edge_pred_domains = map_clusters_to_domains(edge_pred_labels, domain_labels)
    else:
        edge_pred_domains = ["unknown"] * n_total

    examples = []
    for i in range(n_total):
        # Per-graph output includes motif ratios, z-scores, graph stats
        per_graph_output = {
            "motif_count_ratios": motif_count_ratios[i].tolist(),
            "motif_z_scores": motif_zscore_matrix[i].tolist(),
            "graph_stats": graph_stats_matrix[i].tolist(),
            "domain": domain_labels[i],
        }

        examples.append({
            "input": all_records[i]["prompt"],
            "output": json.dumps(per_graph_output),
            "metadata_fold": domain_labels[i],
            "metadata_slug": all_records[i]["slug"],
            "metadata_model_correct": all_records[i]["model_correct"],
            "metadata_difficulty": all_records[i]["difficulty"],
            "predict_best_method": best_pred_domains[i],
            "predict_motif_cluster": motif_pred_domains[i],
            "predict_edge_overlap_cluster": edge_pred_domains[i],
        })

    method_out = {
        "metadata": aggregate_results,
        "datasets": [{
            "dataset": "neuronpedia_attribution_graphs_v3",
            "examples": examples,
        }],
    }

    # Write output
    OUTPUT_FILE.write_text(json.dumps(method_out, indent=2))
    logger.info(f"Output written to {OUTPUT_FILE}")
    logger.info(f"  File size: {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")

    total_runtime = time.time() - t_start
    logger.info(f"Total runtime: {total_runtime:.1f}s ({total_runtime / 60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
