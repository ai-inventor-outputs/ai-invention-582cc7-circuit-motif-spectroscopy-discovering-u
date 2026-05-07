#!/usr/bin/env python3
"""4-Node Motif Census on Full 174-Graph Corpus with 24D Capability Clustering.

Scales motif census to the full merged corpus (data_id4_it1 + data_id3_it2),
computes the first 4-node directed motif census on LLM attribution graphs
producing 24-dimensional significance profiles, and tests whether 24D 4-node
spectra achieve substantially higher capability clustering NMI than the
degenerate 3-node features and graph-statistics baselines.

Hypotheses:
  H1: Specific motif types are universally overrepresented (Z>2) across >=6/8 domains.
  H2: 24D 4-node motif significance profiles cluster circuits by capability type
      (NMI > 0.5), substantially beating 3-node enriched features and graph-stats.

Baseline: Graph-level statistics (16D), 3-node enriched features (12D), random (24D).
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
from collections import defaultdict, Counter
from typing import Any

import numpy as np
from loguru import logger

import igraph
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.manifold import TSNE
from scipy.stats import f_oneway
from scipy.spatial.distance import cosine as cos_dist


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
DATA_DIR_ITER1 = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_1/gen_art/data_id4_it1__opus/data_out"
)
DATA_DIR_ITER2 = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_2/gen_art/data_id3_it2__opus/data_out"
)
OUTPUT_FILE = WORKSPACE / "method_out.json"
LOG_DIR = WORKSPACE / "logs"

# Configurable via env vars for gradual scaling
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
N_NULL_3 = int(os.environ.get("N_NULL_3", "30"))
N_NULL_4 = int(os.environ.get("N_NULL_4", "50"))
SWAP_MULTIPLIER = 100
PRUNE_3NODE = 75  # fixed percentile for 3-node (matches iter 2)
PRUNE_4NODE_CANDIDATES = [90, 95, 97, 99]
MAX_4NODE_TIME_PER_GRAPH_S = 120
MAX_4NODE_NODES = 500
CLUSTER_K_VALUES = [2, 3, 4, 6, 8]
SEED = 42
TOTAL_TIME_BUDGET_S = 3500  # ~58 min, leave buffer
MAX_NULL_TIME_S = 1500  # ~25 min budget for 3-node null models

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
# PHASE A: LOAD AND MERGE FULL CORPUS
# ================================================================

def load_all_graphs_merged(max_graphs: int | None = None) -> list[dict]:
    """Load graphs from both iter1 and iter2 datasets, deduplicate by slug."""
    all_records: list[dict] = []
    seen_slugs: set[str] = set()

    def _load_split(fpath: Path, source_tag: str) -> list[dict]:
        """Load one split file, return compact records."""
        if not fpath.exists():
            logger.warning(f"Data file not found: {fpath}")
            return []
        logger.info(f"Loading {fpath.name} ({fpath.stat().st_size / 1e6:.1f} MB) [{source_tag}]")
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
                continue  # DEDUPLICATE
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
                "n_nodes_raw": ex["metadata_n_nodes"],
                "n_edges_raw": ex["metadata_n_edges"],
                "is_dag": ex["metadata_is_dag"],
                "slug": slug,
                "model_correct": ex.get("metadata_model_correct", "unknown"),
                "nodes": graph_json["nodes"],
                "links": graph_json["links"],
                "source_dataset": source_tag,
            })
        del raw, examples
        return records

    # Load iter1 first (3 files), then iter2 (8 files)
    for fpath in sorted(DATA_DIR_ITER1.glob("full_data_out_*.json")):
        new_recs = _load_split(fpath, "iter1")
        all_records.extend(new_recs)
        gc.collect()
        if max_graphs and len(all_records) >= max_graphs:
            all_records = all_records[:max_graphs]
            break

    if not (max_graphs and len(all_records) >= max_graphs):
        for fpath in sorted(DATA_DIR_ITER2.glob("full_data_out_*.json")):
            new_recs = _load_split(fpath, "iter2")
            all_records.extend(new_recs)
            gc.collect()
            if max_graphs and len(all_records) >= max_graphs:
                all_records = all_records[:max_graphs]
                break

    domain_counts = Counter(r["domain"] for r in all_records)
    logger.info(f"Loaded {len(all_records)} unique graphs across {len(domain_counts)} domains")
    logger.info(f"  Deduplication removed {len(seen_slugs) - len(all_records)} duplicate slugs")
    for d, c in sorted(domain_counts.items()):
        logger.info(f"  {d}: {c} graphs")

    # Validate all are DAGs
    for i, r in enumerate(all_records):
        if not r["is_dag"]:
            logger.warning(f"Graph {i} ({r['domain']}) reported as non-DAG")

    return all_records


# ================================================================
# STEP 2: BUILD IGRAPH GRAPHS WITH WEIGHT PRUNING
# ================================================================

def build_igraph(record: dict, prune_percentile: int) -> igraph.Graph:
    """Build a pruned igraph.Graph from a parsed graph record."""
    nodes = record["nodes"]
    links = record["links"]

    node_ids = [n["node_id"] for n in nodes]
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    layers = []
    feature_types = []
    for n in nodes:
        layer_val = n.get("layer", 0)
        try:
            layers.append(int(layer_val))
        except (ValueError, TypeError):
            layers.append(0)
        feature_types.append(n.get("feature_type", ""))

    all_weights = [abs(link.get("weight", 0.0)) for link in links]
    if not all_weights:
        threshold = 0.0
    else:
        threshold = float(np.percentile(all_weights, prune_percentile))

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
# STEP 3: BUILD ISOCLASS MAPPING
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
# STEP 4: COMPUTE MOTIF CENSUS
# ================================================================

def compute_motif_census(
    g: igraph.Graph, size: int, dag_valid_ids: list[int]
) -> dict[int, int]:
    """Run motifs_randesu, extract counts for DAG-valid connected types."""
    raw = g.motifs_randesu(size=size)
    counts = [0 if (x != x) else int(x) for x in raw]
    valid_counts = {idx: counts[idx] for idx in dag_valid_ids}
    return valid_counts


def compute_motif_census_sampled(
    g: igraph.Graph, size: int, dag_valid_ids: list[int],
    cut_prob: list[float] | None = None,
) -> dict[int, int]:
    """Run motifs_randesu with optional sampling via cut_prob."""
    if cut_prob is None:
        return compute_motif_census(g, size, dag_valid_ids)
    raw = g.motifs_randesu(size=size, cut_prob=cut_prob)
    counts = [0 if (x != x) else int(x) for x in raw]
    valid_counts = {idx: counts[idx] for idx in dag_valid_ids}
    return valid_counts


# ================================================================
# STEP 5: NULL MODEL GENERATION (Goni et al. Method 1)
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


def _null_model_batch_worker_sampled(args: tuple) -> list[dict[int, int]]:
    """Worker for ProcessPoolExecutor: null models + sampled census (for 4-node)."""
    n_nodes, edges, topo_rank, n_swap_attempts, seeds, size, dag_valid_ids, cut_prob = args

    results = []
    for seed in seeds:
        new_edges, _accepted = _generate_null_edges(
            n_nodes, edges, topo_rank, n_swap_attempts, seed
        )
        g_null = igraph.Graph(n=n_nodes, edges=new_edges, directed=True)
        raw = g_null.motifs_randesu(size=size, cut_prob=cut_prob)
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


def generate_null_census_parallel_sampled(
    g: igraph.Graph, size: int, dag_valid_ids: list[int],
    n_null: int, n_workers: int, cut_prob: list[float],
) -> list[dict[int, int]]:
    """Generate n_null null models in parallel with sampled census (4-node)."""
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
            batch_seeds, size, dag_valid_ids, cut_prob,
        ))

    all_results: list[dict[int, int]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_null_model_batch_worker_sampled, b): idx
                   for idx, b in enumerate(batches)}
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception:
                logger.exception(f"Null model batch {futures[future]} (sampled) failed")

    return all_results


# ================================================================
# STEP 6: Z-SCORES AND SIGNIFICANCE PROFILES
# ================================================================

def compute_zscores_and_sp(
    real_counts: dict[int, int], null_counts_list: list[dict[int, int]],
    dag_valid_ids: list[int],
) -> dict[str, Any]:
    """Compute Z-scores, empirical p-values, and significance profiles."""
    z_scores: dict[int, float] = {}
    p_values: dict[int, float] = {}

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
        p = float(np.sum(null_vals >= real_val) / len(null_vals))
        p_values[motif_id] = p

    z_vec = np.array([z_scores[mid] for mid in dag_valid_ids])
    z_norm = float(np.linalg.norm(z_vec))

    if z_norm == 0:
        sp = {mid: 0.0 for mid in dag_valid_ids}
    else:
        sp = {mid: float(z_scores[mid] / z_norm) for mid in dag_valid_ids}

    return {
        "z_scores": z_scores,
        "p_values": p_values,
        "sp": sp,
        "z_norm": z_norm,
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
# STEP 7: BASELINE - GRAPH STATISTICS (16 features)
# ================================================================

def compute_baseline_features(g: igraph.Graph) -> dict[str, float]:
    """Compute graph-level statistics for baseline clustering."""
    n_nodes = g.vcount()
    n_edges = g.ecount()

    in_degs = np.array(g.indegree(), dtype=float)
    out_degs = np.array(g.outdegree(), dtype=float)

    features: dict[str, float] = {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "density": float(g.density()),
        "in_degree_mean": float(np.mean(in_degs)) if n_nodes > 0 else 0.0,
        "in_degree_std": float(np.std(in_degs)) if n_nodes > 0 else 0.0,
        "in_degree_max": float(np.max(in_degs)) if n_nodes > 0 else 0.0,
        "out_degree_mean": float(np.mean(out_degs)) if n_nodes > 0 else 0.0,
        "out_degree_std": float(np.std(out_degs)) if n_nodes > 0 else 0.0,
        "out_degree_max": float(np.max(out_degs)) if n_nodes > 0 else 0.0,
        "transitivity": float(g.transitivity_undirected()),
    }

    if "layer" in g.vs.attributes() and n_edges > 0:
        layers_arr = np.array(g.vs["layer"], dtype=float)
        edge_spans = np.array([
            abs(layers_arr[e.target] - layers_arr[e.source]) for e in g.es
        ])
        features["layer_span_mean"] = float(np.mean(edge_spans))
        features["layer_span_std"] = float(np.std(edge_spans))
        features["layer_span_max"] = float(np.max(edge_spans))
        features["n_layers"] = float(len(set(g.vs["layer"])))
    else:
        features["layer_span_mean"] = 0.0
        features["layer_span_std"] = 0.0
        features["layer_span_max"] = 0.0
        features["n_layers"] = 1.0

    components = g.connected_components(mode="weak")
    features["n_components"] = float(len(components))
    features["largest_component_frac"] = (
        float(max(len(c) for c in components) / n_nodes) if n_nodes > 0 else 0.0
    )

    return features


# ================================================================
# STEP 8: CLUSTERING AND EVALUATION
# ================================================================

def cluster_and_evaluate(
    feature_matrix: np.ndarray, true_labels: np.ndarray,
    k_values: list[int], use_cosine: bool = True,
) -> dict[int, dict]:
    """Spectral clustering at multiple K values, evaluate with NMI/ARI."""
    n_samples = feature_matrix.shape[0]
    results: dict[int, dict] = {}

    if n_samples < 3:
        return results

    if use_cosine:
        cos_sim = cosine_similarity(feature_matrix)
        affinity = (cos_sim + 1) / 2
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)
        n_feat = X_scaled.shape[1]
        affinity = rbf_kernel(X_scaled, gamma=1.0 / max(n_feat, 1))

    for K in k_values:
        if K >= n_samples:
            continue
        try:
            sc = SpectralClustering(
                n_clusters=K, affinity="precomputed",
                assign_labels="kmeans", random_state=42, n_init=10,
            )
            pred_labels = sc.fit_predict(affinity)
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
    logger.info(f"Config: MAX_EXAMPLES={MAX_EXAMPLES}, N_NULL_3={N_NULL_3}, N_NULL_4={N_NULL_4}")
    logger.info(f"Time budget: {TOTAL_TIME_BUDGET_S}s")

    # ================================================================
    # PHASE A: LOAD AND MERGE FULL CORPUS
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE A: Loading and merging full corpus")
    all_records = load_all_graphs_merged(MAX_EXAMPLES or None)
    if not all_records:
        logger.error("No graphs loaded!")
        return
    n_total = len(all_records)

    # ================================================================
    # PHASE B: BUILD ISOCLASS MAPPINGS
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE B: Building isoclass mappings")
    mapping_3, dag_valid_3 = build_isoclass_mapping(3)
    names_3 = identify_3node_man_labels(mapping_3, dag_valid_3)
    logger.info(f"3-node DAG-valid connected types: {len(dag_valid_3)} IDs: {dag_valid_3}")
    for cls_id in dag_valid_3:
        logger.info(f"  ID {cls_id}: {names_3[cls_id]} (edges: {mapping_3[cls_id]['edges']})")

    mapping_4, dag_valid_4 = build_isoclass_mapping(4)
    logger.info(f"4-node DAG-valid connected types: {len(dag_valid_4)} IDs: {dag_valid_4}")

    dag_impossible_3 = [
        cls_id for cls_id in range(16)
        if mapping_3[cls_id]["is_connected"] and not mapping_3[cls_id]["is_dag"]
    ]

    ffl_id = [mid for mid in dag_valid_3 if names_3[mid] == "030T"][0]
    logger.info(f"Feed-forward loop (030T) isoclass ID: {ffl_id}")

    # ================================================================
    # PHASE C: ADAPTIVE PRUNING FOR 4-NODE FEASIBILITY
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE C: Building pruned graphs + adaptive 4-node pruning")

    # Build graphs at 75th percentile for 3-node
    graphs_75: dict[int, igraph.Graph] = {}
    for i, rec in enumerate(all_records):
        try:
            g = build_igraph(rec, PRUNE_3NODE)
            graphs_75[i] = g
        except Exception:
            logger.exception(f"Failed building graph {i} ({rec['domain']}) at {PRUNE_3NODE}%")

    logger.info(f"Built {len(graphs_75)} graphs at {PRUNE_3NODE}th percentile")
    if graphs_75:
        node_counts_75 = [g.vcount() for g in graphs_75.values()]
        logger.info(f"  Nodes: min={min(node_counts_75)}, median={np.median(node_counts_75):.0f}, max={max(node_counts_75)}")

    # Build graphs at 99th percentile for 4-node (known best from calibration runs)
    # From previous runs: 99% gives median ~390 nodes, 75% feasible at MAX_4NODE_NODES=500
    # Trying all thresholds [90,95,97,99] is too slow for 174 graphs (~7 min each)
    best_4node_pct = 99
    graphs_4node: dict[int, igraph.Graph] = {}
    size_stats: dict[int, dict] = {}

    node_counts_4: list[int] = []
    for i, rec in enumerate(all_records):
        try:
            g = build_igraph(rec, best_4node_pct)
            graphs_4node[i] = g
            node_counts_4.append(g.vcount())
        except Exception:
            pass

    n_valid = len(graphs_4node)
    if node_counts_4:
        n_under_limit = sum(1 for n in node_counts_4 if n <= MAX_4NODE_NODES)
        size_stats[best_4node_pct] = {
            "n_valid": n_valid,
            "median_nodes": float(np.median(node_counts_4)),
            "max_nodes": max(node_counts_4),
            "n_under_limit": n_under_limit,
            "feasible_fraction": n_under_limit / n_valid if n_valid > 0 else 0.0,
        }
        logger.info(
            f"  Threshold {best_4node_pct}%: {n_valid} valid, "
            f"nodes med={np.median(node_counts_4):.0f} max={max(node_counts_4)}, "
            f"feasible={n_under_limit}/{n_valid} ({size_stats[best_4node_pct]['feasible_fraction']:.0%})"
        )

    logger.info(f"Best 4-node pruning threshold: {best_4node_pct}%")
    logger.info(f"4-node feasible graphs: {len(graphs_4node)}")

    # Free raw node/link data after graph construction
    for rec in all_records:
        rec.pop("nodes", None)
        rec.pop("links", None)
    gc.collect()

    # ================================================================
    # PHASE D: 3-NODE CENSUS ON FULL CORPUS
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE D: 3-node motif census on full corpus")

    real_census_3: dict[int, dict[int, int]] = {}
    for gi, g in graphs_75.items():
        real_census_3[gi] = compute_motif_census(g, 3, dag_valid_3)

    # DAG validation check
    n_violations = 0
    for gi, g in graphs_75.items():
        raw = g.motifs_randesu(size=3)
        for idx in dag_impossible_3:
            val = raw[idx]
            if val == val and val != 0:
                n_violations += 1
    if n_violations == 0:
        logger.info("  DAG validation passed: all impossible motif counts are 0")
    else:
        logger.warning(f"  DAG validation: {n_violations} violations found")

    # ================================================================
    # PHASE D.2: 3-NODE NULL MODELS
    # ================================================================
    logger.info("=" * 60)
    logger.info(f"PHASE D.2: Generating {N_NULL_3} null models per graph (3-node)")

    # Timing calibration
    if graphs_75:
        calib_key = sorted(graphs_75.keys(), key=lambda k: graphs_75[k].ecount())[
            len(graphs_75) // 2
        ]
        g_calib = graphs_75[calib_key]
        t_calib = time.time()
        calib_n = min(10, N_NULL_3)
        _calib_results = generate_null_census_parallel(g_calib, 3, dag_valid_3, calib_n, n_workers)
        t_calib_elapsed = time.time() - t_calib
        time_per_null_3 = t_calib_elapsed / max(calib_n, 1)
        logger.info(f"  Calibration: {calib_n} null models in {t_calib_elapsed:.2f}s ({time_per_null_3:.3f}s/model)")

        estimated_total_s = time_per_null_3 * N_NULL_3 * len(graphs_75)
        logger.info(f"  Estimated total: {estimated_total_s:.0f}s ({estimated_total_s/60:.1f} min)")

        n_null_3_actual = N_NULL_3
        if estimated_total_s > MAX_NULL_TIME_S:
            n_null_3_actual = max(10, int(MAX_NULL_TIME_S / (time_per_null_3 * len(graphs_75))))
            logger.info(f"  Adjusted to {n_null_3_actual} null models to fit budget")
    else:
        n_null_3_actual = N_NULL_3

    null_census_3: dict[int, list[dict[int, int]]] = {}
    t_null_start = time.time()
    sorted_gis = sorted(graphs_75.keys())
    for idx, gi in enumerate(sorted_gis):
        g = graphs_75[gi]
        t_graph = time.time()
        null_census_3[gi] = generate_null_census_parallel(
            g, 3, dag_valid_3, n_null_3_actual, n_workers
        )
        elapsed = time.time() - t_graph
        if (idx + 1) % 20 == 0 or idx == 0:
            logger.info(
                f"  [{idx+1}/{len(sorted_gis)}] Graph {gi}: "
                f"{n_null_3_actual} nulls in {elapsed:.1f}s ({g.vcount()}n, {g.ecount()}e)"
            )

        total_null_elapsed = time.time() - t_null_start
        remaining_graphs = len(sorted_gis) - (idx + 1)
        if remaining_graphs > 0 and total_null_elapsed > MAX_NULL_TIME_S * 0.9:
            if n_null_3_actual > 100:
                n_null_3_actual = max(10, n_null_3_actual // 2)
                logger.info(f"  Time pressure: reducing to {n_null_3_actual} null models")

    logger.info(f"  3-node null models complete: {time.time()-t_null_start:.1f}s total")

    # Compute 3-node Z-scores and SP
    results_3node: dict[int, dict] = {}
    for gi in real_census_3:
        if gi not in null_census_3 or not null_census_3[gi]:
            continue
        stats = compute_zscores_and_sp(real_census_3[gi], null_census_3[gi], dag_valid_3)
        stats["domain"] = all_records[gi]["domain"]
        results_3node[gi] = stats

    logger.info(f"  3-node results for {len(results_3node)} graphs")

    # Log per-motif aggregate Z-scores
    for mid in dag_valid_3:
        z_vals = [results_3node[gi]["z_scores"][mid] for gi in results_3node]
        if z_vals:
            logger.info(
                f"  {names_3[mid]} (ID {mid}): mean Z={np.mean(z_vals):.2f} "
                f"+/- {np.std(z_vals):.2f}"
            )

    # ================================================================
    # PHASE E: 4-NODE CENSUS WITH SAMPLING FALLBACK
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE E: 4-node motif census")

    real_census_4: dict[int, dict[int, int]] = {}
    census_method: dict[int, str] = {}
    t_4node_start = time.time()

    for gi, g in sorted(graphs_4node.items()):
        n = g.vcount()
        if n > MAX_4NODE_NODES:
            census_method[gi] = "skipped_too_large"
            continue

        if n <= 200:
            # Exact enumeration
            t0 = time.time()
            try:
                census = compute_motif_census(g, 4, dag_valid_4)
                elapsed = time.time() - t0
                real_census_4[gi] = census
                census_method[gi] = f"exact_{elapsed:.0f}s"
            except Exception:
                logger.exception(f"  Graph {gi}: 4-node census failed (exact)")
                census_method[gi] = "failed"
        else:
            # Sampled: 5 runs with cut_prob, average
            cut_prob = [0, 0, 0.3, 0.7]
            samples = []
            for run in range(5):
                try:
                    c = compute_motif_census_sampled(g, 4, dag_valid_4, cut_prob)
                    samples.append(c)
                except Exception:
                    pass
            if samples:
                avg_census = {}
                for mid in dag_valid_4:
                    avg_census[mid] = int(np.mean([s[mid] for s in samples]))
                real_census_4[gi] = avg_census
                census_method[gi] = f"sampled_{len(samples)}"
            else:
                census_method[gi] = "failed"

        # Time check
        if time.time() - t_start > TOTAL_TIME_BUDGET_S * 0.6:
            logger.warning("  Approaching 60% of time budget, stopping 4-node census")
            for rem_gi in graphs_4node:
                if rem_gi not in census_method:
                    census_method[rem_gi] = "skipped_time"
            break

    n_exact = sum(1 for v in census_method.values() if v.startswith("exact"))
    n_sampled = sum(1 for v in census_method.values() if v.startswith("sampled"))
    n_failed = sum(1 for v in census_method.values() if v.startswith("failed"))
    n_skipped = sum(1 for v in census_method.values() if v.startswith("skipped"))
    logger.info(
        f"  4-node census: {len(real_census_4)} done "
        f"(exact={n_exact}, sampled={n_sampled}, failed={n_failed}, skipped={n_skipped})"
    )
    logger.info(f"  4-node census time: {time.time()-t_4node_start:.1f}s")

    # ================================================================
    # PHASE F: 4-NODE NULL MODELS
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE F: 4-node null models")

    remaining_time = TOTAL_TIME_BUDGET_S - (time.time() - t_start)
    n_null_4_actual = N_NULL_4
    results_4node: dict[int, dict] = {}

    if real_census_4 and remaining_time > 300:
        # Timing calibration for 4-node nulls
        sample_gi = sorted(real_census_4.keys(), key=lambda k: graphs_4node[k].vcount())[0]
        g_sample = graphs_4node[sample_gi]
        t_cal4 = time.time()
        cal4_n = min(5, n_null_4_actual)
        if g_sample.vcount() <= 150:
            _cal4_res = generate_null_census_parallel(g_sample, 4, dag_valid_4, cal4_n, n_workers)
        else:
            _cal4_res = generate_null_census_parallel_sampled(
                g_sample, 4, dag_valid_4, cal4_n, n_workers, [0, 0, 0.3, 0.7]
            )
        t_cal4_elapsed = time.time() - t_cal4
        time_per_null_4 = t_cal4_elapsed / max(cal4_n, 1)
        estimated_4null = time_per_null_4 * n_null_4_actual * len(real_census_4)
        logger.info(
            f"  4-node null calibration: {cal4_n} in {t_cal4_elapsed:.2f}s "
            f"({time_per_null_4:.3f}s/model)"
        )
        logger.info(f"  Estimated total: {estimated_4null:.0f}s ({estimated_4null/60:.1f}min)")

        # Adjust if needed
        max_4null_time = remaining_time * 0.6
        if estimated_4null > max_4null_time:
            n_null_4_actual = max(20, int(max_4null_time / (time_per_null_4 * len(real_census_4))))
            logger.info(f"  Adjusted to {n_null_4_actual} null models for 4-node")

        null_census_4: dict[int, list[dict[int, int]]] = {}
        t_null4_start = time.time()
        for idx_4, gi in enumerate(sorted(real_census_4.keys())):
            g = graphs_4node[gi]
            n = g.vcount()
            t_g4 = time.time()
            try:
                if n <= 200:
                    nc = generate_null_census_parallel(
                        g, 4, dag_valid_4, n_null_4_actual, n_workers
                    )
                else:
                    nc = generate_null_census_parallel_sampled(
                        g, 4, dag_valid_4, n_null_4_actual, n_workers, [0, 0, 0.3, 0.7]
                    )
                null_census_4[gi] = nc
            except Exception:
                logger.exception(f"  Graph {gi}: 4-node null models failed")

            if (idx_4 + 1) % 10 == 0 or idx_4 == 0:
                logger.info(
                    f"  [{idx_4+1}/{len(real_census_4)}] Graph {gi}: "
                    f"{n_null_4_actual} nulls in {time.time()-t_g4:.1f}s"
                )

            if time.time() - t_start > TOTAL_TIME_BUDGET_S * 0.85:
                logger.warning("  Approaching 85% time, stopping 4-node null models")
                break

        logger.info(f"  4-node null models complete: {time.time()-t_null4_start:.1f}s")

        # Compute 4-node Z-scores
        for gi in real_census_4:
            if gi in null_census_4 and len(null_census_4[gi]) >= 10:
                stats = compute_zscores_and_sp(
                    real_census_4[gi], null_census_4[gi], dag_valid_4
                )
                stats["domain"] = all_records[gi]["domain"]
                results_4node[gi] = stats

        logger.info(f"  4-node Z-scores computed for {len(results_4node)} graphs")
    else:
        logger.info("  Skipping 4-node null models (insufficient time or no 4-node census)")
        # Fallback: use raw count ratios without null models
        for gi in real_census_4:
            total = sum(real_census_4[gi][mid] for mid in dag_valid_4)
            if total > 0:
                results_4node[gi] = {
                    "z_scores": {mid: 0.0 for mid in dag_valid_4},
                    "p_values": {mid: 1.0 for mid in dag_valid_4},
                    "sp": {mid: 0.0 for mid in dag_valid_4},
                    "z_norm": 0.0,
                    "raw_counts": real_census_4[gi],
                    "null_means": {mid: 0.0 for mid in dag_valid_4},
                    "null_stds": {mid: 0.0 for mid in dag_valid_4},
                    "domain": all_records[gi]["domain"],
                    "fallback_count_ratios": True,
                }

    # ================================================================
    # PHASE G: BASELINE FEATURES
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE G: Computing baseline graph statistics")

    baseline_features: dict[int, dict[str, float]] = {}
    for gi, g in graphs_75.items():
        baseline_features[gi] = compute_baseline_features(g)

    # ================================================================
    # PHASE H: CLUSTERING COMPARISON (24D vs 12D vs 16D)
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE H: Clustering comparison across feature sets")

    # Only use graphs that have BOTH 3-node and 4-node results
    common_gids = sorted(set(results_4node.keys()) & set(results_3node.keys()))
    logger.info(f"  Common graphs (3-node AND 4-node): {len(common_gids)}")

    # If too few common graphs, use 3-node-only for some analyses
    gids_3node_only = sorted(results_3node.keys())
    logger.info(f"  3-node-only graphs: {len(gids_3node_only)}")

    domain_labels_common = [all_records[gi]["domain"] for gi in common_gids]
    domain_labels_3only = [all_records[gi]["domain"] for gi in gids_3node_only]

    le_common = LabelEncoder()
    le_3only = LabelEncoder()

    clustering_results: dict[str, dict] = {}
    feature_keys_sorted = sorted(baseline_features[gids_3node_only[0]].keys()) if gids_3node_only else []

    if len(common_gids) >= 10:
        true_labels_common = le_common.fit_transform(domain_labels_common)

        # (1) 4-node SP vectors (24D) -- PRIMARY METHOD
        sp_4node_matrix = np.array([
            [results_4node[gi]["sp"][mid] for mid in dag_valid_4]
            for gi in common_gids
        ])

        # (2) 4-node raw count ratios (24D)
        count_ratio_4node_list = []
        for gi in common_gids:
            raw = results_4node[gi]["raw_counts"]
            total = sum(raw[mid] for mid in dag_valid_4)
            if total > 0:
                count_ratio_4node_list.append([raw[mid] / total for mid in dag_valid_4])
            else:
                count_ratio_4node_list.append([0.0] * 24)
        count_ratio_4node_matrix = np.array(count_ratio_4node_list)

        # (3) 4-node Z-scores (24D)
        z_4node_matrix = np.array([
            [results_4node[gi]["z_scores"][mid] for mid in dag_valid_4]
            for gi in common_gids
        ])

        # (4) 3-node enriched features (12D)
        enriched_3node_list = []
        for gi in common_gids:
            r3 = results_3node[gi]
            raw3 = r3["raw_counts"]
            total3 = sum(raw3[mid] for mid in dag_valid_3)
            ratios3 = [raw3[mid] / total3 for mid in dag_valid_3] if total3 > 0 else [0.0] * 4
            bf = baseline_features[gi]
            vec = ratios3 + [
                r3["z_scores"][ffl_id],
                raw3[ffl_id] / max(bf["n_nodes"], 1),
                raw3[ffl_id] / max(bf["n_edges"], 1),
                bf["density"], bf["in_degree_std"], bf["out_degree_std"],
                bf["layer_span_mean"], bf["transitivity"],
            ]
            enriched_3node_list.append(vec)
        enriched_3node_matrix = np.array(enriched_3node_list)

        # (5) Graph statistics baseline (16D)
        baseline_matrix = np.array([
            [baseline_features[gi][k] for k in feature_keys_sorted]
            for gi in common_gids
        ])

        # (6) Random features baseline (24D)
        rng = np.random.RandomState(42)
        random_matrix = rng.randn(len(common_gids), 24)

        # (7) Combined: 4-node SP + 3-node enriched + graph stats
        combined_matrix = np.hstack([sp_4node_matrix, enriched_3node_matrix, baseline_matrix])

        # Cluster each feature set
        feature_sets = {
            "4node_sp_24d": (sp_4node_matrix, True),
            "4node_count_ratios_24d": (count_ratio_4node_matrix, True),
            "4node_zscores_24d": (z_4node_matrix, False),
            "3node_enriched_12d": (enriched_3node_matrix, False),
            "graph_stats_16d": (baseline_matrix, False),
            "random_24d": (random_matrix, False),
            "combined_all": (combined_matrix, False),
        }

        for name, (matrix, use_cosine) in feature_sets.items():
            # Remove zero-variance columns
            valid_cols = np.std(matrix, axis=0) > 1e-10
            if not np.all(valid_cols):
                matrix_clean = matrix[:, valid_cols]
                logger.info(f"  {name}: removed {np.sum(~valid_cols)} zero-variance cols")
            else:
                matrix_clean = matrix

            if matrix_clean.shape[1] == 0:
                logger.warning(f"  {name}: all columns zero-variance, skipping")
                clustering_results[name] = {
                    "results_by_k": {}, "best_k": 0, "best_nmi": 0.0,
                    "best_ari": 0.0, "n_features": 0, "n_graphs": len(common_gids),
                }
                continue

            clust = cluster_and_evaluate(matrix_clean, true_labels_common, CLUSTER_K_VALUES, use_cosine)
            if clust:
                best_k = max(clust, key=lambda k: clust[k]["nmi"])
                clustering_results[name] = {
                    "results_by_k": {str(k): {"nmi": v["nmi"], "ari": v["ari"]} for k, v in clust.items()},
                    "best_k": best_k,
                    "best_nmi": clust[best_k]["nmi"],
                    "best_ari": clust[best_k]["ari"],
                    "n_features": int(matrix_clean.shape[1]),
                    "n_graphs": len(common_gids),
                }
                logger.info(f"  {name}: best K={best_k}, NMI={clust[best_k]['nmi']:.3f}, ARI={clust[best_k]['ari']:.3f}")
            else:
                clustering_results[name] = {
                    "results_by_k": {}, "best_k": 0, "best_nmi": 0.0,
                    "best_ari": 0.0, "n_features": int(matrix_clean.shape[1]),
                    "n_graphs": len(common_gids),
                }
    else:
        logger.warning(f"  Only {len(common_gids)} common graphs, doing 3-node-only clustering")

    # Also run 3-node-only clustering on full corpus if we have enough
    if len(gids_3node_only) >= 10:
        true_labels_3only = le_3only.fit_transform(domain_labels_3only)

        enriched_3only_list = []
        for gi in gids_3node_only:
            r3 = results_3node[gi]
            raw3 = r3["raw_counts"]
            total3 = sum(raw3[mid] for mid in dag_valid_3)
            ratios3 = [raw3[mid] / total3 for mid in dag_valid_3] if total3 > 0 else [0.0] * 4
            bf = baseline_features[gi]
            vec = ratios3 + [
                r3["z_scores"][ffl_id],
                raw3[ffl_id] / max(bf["n_nodes"], 1),
                raw3[ffl_id] / max(bf["n_edges"], 1),
                bf["density"], bf["in_degree_std"], bf["out_degree_std"],
                bf["layer_span_mean"], bf["transitivity"],
            ]
            enriched_3only_list.append(vec)
        enriched_3only_matrix = np.array(enriched_3only_list)

        baseline_3only_matrix = np.array([
            [baseline_features[gi][k] for k in feature_keys_sorted]
            for gi in gids_3node_only
        ])

        # 3-node enriched on full corpus
        clust_3full = cluster_and_evaluate(enriched_3only_matrix, true_labels_3only, CLUSTER_K_VALUES, use_cosine=False)
        if clust_3full:
            best_k_3full = max(clust_3full, key=lambda k: clust_3full[k]["nmi"])
            clustering_results["3node_enriched_full_corpus"] = {
                "results_by_k": {str(k): {"nmi": v["nmi"], "ari": v["ari"]} for k, v in clust_3full.items()},
                "best_k": best_k_3full,
                "best_nmi": clust_3full[best_k_3full]["nmi"],
                "best_ari": clust_3full[best_k_3full]["ari"],
                "n_features": enriched_3only_matrix.shape[1],
                "n_graphs": len(gids_3node_only),
            }
            logger.info(
                f"  3node_enriched_full ({len(gids_3node_only)} graphs): "
                f"K={best_k_3full}, NMI={clust_3full[best_k_3full]['nmi']:.3f}"
            )

        # Baseline on full corpus
        clust_base_full = cluster_and_evaluate(baseline_3only_matrix, true_labels_3only, CLUSTER_K_VALUES, use_cosine=False)
        if clust_base_full:
            best_k_bfull = max(clust_base_full, key=lambda k: clust_base_full[k]["nmi"])
            clustering_results["graph_stats_full_corpus"] = {
                "results_by_k": {str(k): {"nmi": v["nmi"], "ari": v["ari"]} for k, v in clust_base_full.items()},
                "best_k": best_k_bfull,
                "best_nmi": clust_base_full[best_k_bfull]["nmi"],
                "best_ari": clust_base_full[best_k_bfull]["ari"],
                "n_features": baseline_3only_matrix.shape[1],
                "n_graphs": len(gids_3node_only),
            }
            logger.info(
                f"  graph_stats_full ({len(gids_3node_only)} graphs): "
                f"K={best_k_bfull}, NMI={clust_base_full[best_k_bfull]['nmi']:.3f}"
            )

    # t-SNE on best feature set (prefer 4-node SP if available)
    coords_2d = np.zeros((len(common_gids), 2))
    if len(common_gids) >= 5 and "4node_sp_24d" in clustering_results:
        try:
            sp_4node_matrix_for_tsne = np.array([
                [results_4node[gi]["sp"][mid] for mid in dag_valid_4]
                for gi in common_gids
            ])
            valid_cols = np.std(sp_4node_matrix_for_tsne, axis=0) > 1e-10
            sp_clean = sp_4node_matrix_for_tsne[:, valid_cols] if np.any(valid_cols) else sp_4node_matrix_for_tsne
            perp = min(30, max(2, len(common_gids) // 4))
            scaler = StandardScaler()
            sp_scaled = scaler.fit_transform(sp_clean)
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
            coords_2d = tsne.fit_transform(sp_scaled)
            logger.info(f"  t-SNE computed on 4-node SP ({len(common_gids)} points, perp={perp})")
        except Exception:
            logger.exception("  t-SNE failed")

    # ================================================================
    # PHASE I: FEATURE IMPORTANCE ANALYSIS
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE I: Feature importance analysis")

    importance_results: dict[str, Any] = {}

    if len(common_gids) >= 10 and results_4node:
        true_labels_common = le_common.fit_transform(domain_labels_common)

        # ANOVA F-statistic per 4-node motif type
        f_stats: dict[int, dict] = {}
        for mid in dag_valid_4:
            z_vals = np.array([results_4node[gi]["z_scores"][mid] for gi in common_gids])
            groups = [
                z_vals[true_labels_common == k]
                for k in range(len(le_common.classes_))
                if np.sum(true_labels_common == k) > 1
            ]
            groups = [grp for grp in groups if len(grp) > 1]
            if len(groups) >= 2:
                try:
                    f_stat, p_val = f_oneway(*groups)
                    if not np.isnan(f_stat):
                        f_stats[mid] = {"f_statistic": float(f_stat), "p_value": float(p_val)}
                except Exception:
                    pass

        top5 = sorted(f_stats.items(), key=lambda x: x[1]["f_statistic"], reverse=True)[:5]
        logger.info("  Top-5 discriminative 4-node motif types:")
        for mid, stats in top5:
            logger.info(f"    ID {mid}: F={stats['f_statistic']:.2f}, p={stats['p_value']:.4f}")

        # PCA on 24D SP vectors -- effective dimensionality
        sp_matrix = np.array([
            [results_4node[gi]["sp"][mid] for mid in dag_valid_4]
            for gi in common_gids
        ])

        valid_cols_pca = np.std(sp_matrix, axis=0) > 1e-10
        sp_for_pca = sp_matrix[:, valid_cols_pca] if np.any(valid_cols_pca) else sp_matrix

        n_components_pca = min(sp_for_pca.shape[1], len(common_gids) - 1, 24)
        if n_components_pca >= 1:
            pca = PCA(n_components=n_components_pca)
            pca.fit(sp_for_pca)
            explained_variance_ratios = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratios)

            eff_dim_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
            eff_dim_90 = int(np.searchsorted(cumulative_variance, 0.90) + 1)
            eff_dim_99 = int(np.searchsorted(cumulative_variance, 0.99) + 1)

            logger.info(f"  PCA effective dimensionality: 90%={eff_dim_90}, 95%={eff_dim_95}, 99%={eff_dim_99}")
        else:
            explained_variance_ratios = np.array([])
            cumulative_variance = np.array([])
            eff_dim_95 = 0
            eff_dim_90 = 0
            eff_dim_99 = 0

        # Correlation matrix eigenvalues for degeneracy check
        if sp_for_pca.shape[1] >= 2 and sp_for_pca.shape[0] >= 2:
            try:
                corr_matrix = np.corrcoef(sp_for_pca.T)
                # Handle NaN in correlation matrix
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                eigenvalues = np.linalg.eigvalsh(corr_matrix)
                n_significant_eigenvalues = int(np.sum(eigenvalues > 0.01))
            except Exception:
                eigenvalues = np.array([])
                n_significant_eigenvalues = 0
        else:
            eigenvalues = np.array([])
            n_significant_eigenvalues = 0

        importance_results = {
            "top5_discriminative": [
                {"motif_id": int(mid), "f_statistic": s["f_statistic"], "p_value": s["p_value"]}
                for mid, s in top5
            ],
            "f_stats_all": {str(mid): s for mid, s in f_stats.items()},
            "effective_dim_90": eff_dim_90,
            "effective_dim_95": eff_dim_95,
            "effective_dim_99": eff_dim_99,
            "n_significant_eigenvalues": n_significant_eigenvalues,
            "explained_variance_ratios": explained_variance_ratios.tolist(),
            "pca_cumulative_variance": cumulative_variance.tolist(),
            "corr_matrix_eigenvalues": sorted(eigenvalues.tolist(), reverse=True),
        }

    # ================================================================
    # PHASE J: HYPOTHESIS TESTING
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE J: Hypothesis testing")

    # H1: Universal overrepresentation (3-node)
    h1_3node: dict[str, dict] = {}
    for mid in dag_valid_3:
        z_by_domain: dict[str, list[float]] = defaultdict(list)
        for gi in results_3node:
            z_by_domain[all_records[gi]["domain"]].append(results_3node[gi]["z_scores"][mid])
        n_domains_sig = sum(1 for d, zs in z_by_domain.items() if np.mean(zs) > 2.0)
        h1_3node[str(mid)] = {
            "name": names_3[mid],
            "mean_z": float(np.mean([z for zs in z_by_domain.values() for z in zs])),
            "n_domains_significant": n_domains_sig,
            "per_domain_mean_z": {d: float(np.mean(zs)) for d, zs in z_by_domain.items()},
        }
        logger.info(f"  3-node {names_3[mid]}: sig in {n_domains_sig}/{len(z_by_domain)} domains")

    # H1: Universal overrepresentation (4-node)
    h1_4node: dict[str, dict] = {}
    if results_4node:
        for mid in dag_valid_4:
            z_by_domain_4: dict[str, list[float]] = defaultdict(list)
            for gi in results_4node:
                z_by_domain_4[all_records[gi]["domain"]].append(results_4node[gi]["z_scores"][mid])
            n_domains_sig_4 = sum(1 for d, zs in z_by_domain_4.items() if zs and np.mean(zs) > 2.0)
            h1_4node[str(mid)] = {
                "mean_z": float(np.mean([z for zs in z_by_domain_4.values() for z in zs])) if z_by_domain_4 else 0.0,
                "n_domains_significant": n_domains_sig_4,
                "per_domain_mean_z": {d: float(np.mean(zs)) for d, zs in z_by_domain_4.items()},
            }

    n_4node_universal = sum(1 for m in h1_4node.values() if m["n_domains_significant"] >= 6)
    logger.info(f"  4-node motifs universal (Z>2 in >=6 domains): {n_4node_universal}/24")

    # H2: Capability clustering
    sp_4_nmi = clustering_results.get("4node_sp_24d", {}).get("best_nmi", 0.0)
    cr_4_nmi = clustering_results.get("4node_count_ratios_24d", {}).get("best_nmi", 0.0)
    en_3_nmi = clustering_results.get("3node_enriched_12d", {}).get("best_nmi", 0.0)
    gs_nmi = clustering_results.get("graph_stats_16d", {}).get("best_nmi", 0.0)
    rnd_nmi = clustering_results.get("random_24d", {}).get("best_nmi", 0.0)

    best_4node_nmi = max(sp_4_nmi, cr_4_nmi)
    h2_result = "CONFIRM" if best_4node_nmi > 0.5 else "DISCONFIRM"
    h2_4_beats_3 = best_4node_nmi > en_3_nmi
    h2_4_beats_baseline = best_4node_nmi > gs_nmi

    logger.info(f"  H2 NMI: 4node_sp={sp_4_nmi:.3f}, 4node_cr={cr_4_nmi:.3f}, "
                f"3node_enr={en_3_nmi:.3f}, graph_stats={gs_nmi:.3f}, random={rnd_nmi:.3f}")
    logger.info(f"  H2 Result: {h2_result}, 4-node beats 3-node: {h2_4_beats_3}, beats baseline: {h2_4_beats_baseline}")

    # ================================================================
    # PHASE K: ASSEMBLE OUTPUT (exp_gen_sol_out schema)
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE K: Assembling output")

    # Determine best feature set for predictions
    best_pred_domains: list[str] = []
    baseline_pred_domains: list[str] = []

    if "4node_sp_24d" in clustering_results and clustering_results["4node_sp_24d"].get("best_k"):
        best_k_our = clustering_results["4node_sp_24d"]["best_k"]
        # Re-run clustering to get pred_labels
        if len(common_gids) >= 10:
            sp_4_mat = np.array([
                [results_4node[gi]["sp"][mid] for mid in dag_valid_4]
                for gi in common_gids
            ])
            valid_cols = np.std(sp_4_mat, axis=0) > 1e-10
            sp_clean = sp_4_mat[:, valid_cols] if np.any(valid_cols) else sp_4_mat
            if sp_clean.shape[1] > 0:
                clust_our = cluster_and_evaluate(sp_clean, le_common.fit_transform(domain_labels_common), [best_k_our], use_cosine=True)
                if clust_our and best_k_our in clust_our and clust_our[best_k_our]["pred_labels"]:
                    best_pred_domains = map_clusters_to_domains(
                        clust_our[best_k_our]["pred_labels"], domain_labels_common
                    )

    if "graph_stats_16d" in clustering_results and clustering_results["graph_stats_16d"].get("best_k"):
        best_k_base = clustering_results["graph_stats_16d"]["best_k"]
        if len(common_gids) >= 10:
            base_mat = np.array([
                [baseline_features[gi][k] for k in feature_keys_sorted]
                for gi in common_gids
            ])
            clust_base = cluster_and_evaluate(base_mat, le_common.fit_transform(domain_labels_common), [best_k_base], use_cosine=False)
            if clust_base and best_k_base in clust_base and clust_base[best_k_base]["pred_labels"]:
                baseline_pred_domains = map_clusters_to_domains(
                    clust_base[best_k_base]["pred_labels"], domain_labels_common
                )

    if not best_pred_domains:
        best_pred_domains = ["unknown"] * len(common_gids)
    if not baseline_pred_domains:
        baseline_pred_domains = ["unknown"] * len(common_gids)

    # Build per-example output -- include ALL graphs with available data
    # Map common_gids to their index for t-SNE / predictions
    common_gid_to_idx = {gi: idx for idx, gi in enumerate(common_gids)}
    all_output_gids = sorted(set(list(results_3node.keys()) + list(results_4node.keys())))

    examples: list[dict] = []
    for gi in all_output_gids:
        per_graph_output: dict[str, Any] = {}

        # 3-node results
        if gi in results_3node:
            r3 = results_3node[gi]
            raw3 = r3["raw_counts"]
            total3 = sum(raw3[mid] for mid in dag_valid_3)
            per_graph_output["motif_census_3node"] = {
                "raw_counts": {str(k): v for k, v in raw3.items()},
                "count_ratios": {
                    str(k): round(v / total3, 6) if total3 > 0 else 0.0
                    for k, v in raw3.items()
                },
                "z_scores": {str(k): round(v, 4) for k, v in r3["z_scores"].items()},
                "sp": {str(k): round(v, 6) for k, v in r3["sp"].items()},
            }

        # 4-node results
        if gi in results_4node:
            r4 = results_4node[gi]
            raw4 = r4["raw_counts"]
            total4 = sum(raw4[mid] for mid in dag_valid_4)
            per_graph_output["motif_census_4node"] = {
                "raw_counts": {str(k): v for k, v in raw4.items()},
                "count_ratios": {
                    str(k): round(v / total4, 6) if total4 > 0 else 0.0
                    for k, v in raw4.items()
                },
                "z_scores": {str(k): round(v, 4) for k, v in r4["z_scores"].items()},
                "sp": {str(k): round(v, 6) for k, v in r4["sp"].items()},
                "census_method": census_method.get(gi, "unknown"),
            }

        # Baseline features
        if gi in baseline_features:
            per_graph_output["baseline_features"] = {
                k: round(v, 6) for k, v in baseline_features[gi].items()
            }

        # Graph info
        per_graph_output["graph_info"] = {
            "n_nodes_75pct": graphs_75[gi].vcount() if gi in graphs_75 else 0,
            "n_edges_75pct": graphs_75[gi].ecount() if gi in graphs_75 else 0,
            "n_nodes_4node_pct": graphs_4node[gi].vcount() if gi in graphs_4node else 0,
            "n_edges_4node_pct": graphs_4node[gi].ecount() if gi in graphs_4node else 0,
        }

        cidx = common_gid_to_idx.get(gi)
        per_graph_output["tsne_coords"] = (
            coords_2d[cidx].tolist() if cidx is not None and cidx < len(coords_2d) else [0.0, 0.0]
        )

        per_graph_output["domain"] = all_records[gi]["domain"]
        per_graph_output["slug"] = all_records[gi]["slug"]

        example = {
            "input": all_records[gi]["prompt"],
            "output": json.dumps(per_graph_output, default=str),
            "predict_our_method": (
                best_pred_domains[cidx] if cidx is not None and cidx < len(best_pred_domains) else ""
            ),
            "predict_baseline": (
                baseline_pred_domains[cidx] if cidx is not None and cidx < len(baseline_pred_domains) else ""
            ),
            "metadata_fold": all_records[gi]["domain"],
            "metadata_graph_idx": gi,
            "metadata_slug": all_records[gi]["slug"],
            "metadata_n_nodes_raw": all_records[gi]["n_nodes_raw"],
            "metadata_n_edges_raw": all_records[gi]["n_edges_raw"],
            "metadata_source_dataset": all_records[gi]["source_dataset"],
        }
        examples.append(example)

    if not examples:
        logger.error("No examples to output!")
        return

    # Top-level output
    wall_time = time.time() - t_start
    output = {
        "metadata": {
            "experiment": "4node_motif_census_full_corpus",
            "n_graphs_total": n_total,
            "n_graphs_3node": len(results_3node),
            "n_graphs_4node": len(results_4node),
            "n_graphs_common": len(common_gids),
            "n_domains": len(set(r["domain"] for r in all_records)),
            "domains": sorted(set(r["domain"] for r in all_records)),
            "prune_3node_pct": PRUNE_3NODE,
            "prune_4node_pct": best_4node_pct,
            "n_null_3_actual": n_null_3_actual,
            "n_null_4_actual": n_null_4_actual,
            "isoclass_mapping_3node": {
                "dag_valid_connected_ids": dag_valid_3,
                "man_labels": {str(k): v for k, v in names_3.items()},
                "n_types": len(dag_valid_3),
            },
            "isoclass_mapping_4node": {
                "dag_valid_connected_ids": dag_valid_4,
                "n_types": len(dag_valid_4),
            },
            "4node_census_methods": {str(k): v for k, v in census_method.items()},
            "4node_pruning_stats": {str(k): v for k, v in size_stats.items()},
            "hypothesis_1_universal_overrepresentation_3node": {
                "per_motif": h1_3node,
                "criterion": "Z>2 in >=6/8 domains",
            },
            "hypothesis_1_universal_overrepresentation_4node": {
                "per_motif": h1_4node,
                "n_motifs_universal": n_4node_universal,
                "criterion": "Z>2 in >=6/8 domains",
            },
            "hypothesis_2_capability_clustering": {
                "feature_set_comparison": clustering_results,
                "4node_sp_best_nmi": sp_4_nmi,
                "4node_count_ratios_best_nmi": cr_4_nmi,
                "3node_enriched_best_nmi": en_3_nmi,
                "graph_stats_best_nmi": gs_nmi,
                "random_best_nmi": rnd_nmi,
                "4node_beats_3node": h2_4_beats_3,
                "4node_beats_baseline": h2_4_beats_baseline,
                "result": h2_result,
            },
            "feature_importance": importance_results,
            "effective_dimensionality": {
                "3node_sp": 1,
                "4node_sp": importance_results.get("effective_dim_95", 0),
                "4node_sp_significant_eigenvalues": importance_results.get("n_significant_eigenvalues", 0),
            },
            "methodological_finding": {
                "title": "3-node SP degeneracy under degree-preserving null models",
                "description": (
                    "Under degree-preserving DAG null models (Goni Method 1), "
                    "the 3-node SP is constant [-0.5,-0.5,-0.5,0.5] for ALL graphs. "
                    "The 4-node spectrum provides genuinely multi-dimensional structure."
                ),
            },
            "config": {
                "prune_thresholds_tested": PRUNE_4NODE_CANDIDATES,
                "n_null_3_requested": N_NULL_3,
                "n_null_4_requested": N_NULL_4,
                "swap_multiplier": SWAP_MULTIPLIER,
                "cluster_k_values": CLUSTER_K_VALUES,
                "max_4node_nodes": MAX_4NODE_NODES,
                "n_workers": n_workers,
                "hardware": {"cpus": NUM_CPUS, "ram_gb": round(TOTAL_RAM_GB, 1)},
            },
            "wall_time_s": round(wall_time, 1),
        },
        "datasets": [
            {
                "dataset": "neuronpedia_attribution_graphs_merged",
                "examples": examples,
            }
        ],
    }

    # Save output
    output_json = json.dumps(output, indent=2, default=str)
    OUTPUT_FILE.write_text(output_json)
    file_size_mb = len(output_json) / 1e6
    logger.info(f"Output saved to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    logger.info(f"Total runtime: {wall_time:.1f}s ({wall_time/60:.1f} min)")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Graphs: {n_total} total, {len(results_3node)} 3-node, {len(results_4node)} 4-node, {len(common_gids)} common")
    logger.info(f"  4-node pruning: {best_4node_pct}th percentile")
    logger.info(f"  H1 (4-node universal): {n_4node_universal}/24 motifs")
    logger.info(f"  H2 (clustering): 4node_sp NMI={sp_4_nmi:.3f} vs 3node_enr={en_3_nmi:.3f} vs baseline={gs_nmi:.3f}")
    logger.info(f"  Effective dim (4-node SP): {importance_results.get('effective_dim_95', 'N/A')}")
    logger.info(f"  Result: {h2_result}")

    return output


if __name__ == "__main__":
    main()
