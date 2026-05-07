#!/usr/bin/env python3
"""Circuit Motif Spectroscopy: 3-node & 4-node motif census with DAG null models,
Z-scores, and capability clustering on Neuronpedia attribution graphs.

Hypotheses:
  H1: Specific motif types are universally overrepresented (Z>2) across >=6/8 domains.
  H2: Motif significance profiles cluster circuits by capability type (NMI > 0.5).

Baseline: Graph-level statistics (density, degree distribution, layer spans) for clustering.
"""

import argparse
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
DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_1/gen_art/data_id4_it1__opus/data_out"
)
DATA_FILES = ["full_data_out_1.json", "full_data_out_2.json", "full_data_out_3.json"]
OUTPUT_FILE = WORKSPACE / "method_out.json"
LOG_DIR = WORKSPACE / "logs"

SWAP_MULTIPLIER = 100
CLUSTER_K_VALUES = [2, 3, 4, 6, 8]
TSNE_PERPLEXITY = 8
MAX_4NODE_NODES = 400
MAX_4NODE_TIME_S = 120

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
# STEP 1: DATA LOADING
# ================================================================

def load_all_graphs(max_graphs: int | None = None) -> list[dict]:
    """Load attribution graphs from split JSON data files.

    Loads one file at a time, extracts compact records, releases raw JSON.
    """
    all_graphs: list[dict] = []
    for data_file in DATA_FILES:
        fpath = DATA_DIR / data_file
        if not fpath.exists():
            logger.warning(f"Data file not found: {fpath}")
            continue
        logger.info(f"Loading {fpath.name} ({fpath.stat().st_size / 1e6:.1f} MB)")
        try:
            raw = json.loads(fpath.read_text())
        except json.JSONDecodeError:
            logger.exception(f"Invalid JSON: {fpath}")
            continue
        examples = raw["datasets"][0]["examples"]
        for ex in examples:
            try:
                graph_json = json.loads(ex["output"])
            except json.JSONDecodeError:
                logger.exception(f"Invalid graph JSON in example {len(all_graphs)}")
                continue
            record = {
                "prompt": ex["input"],
                "domain": ex["metadata_fold"],
                "n_nodes_raw": ex["metadata_n_nodes"],
                "n_edges_raw": ex["metadata_n_edges"],
                "is_dag": ex["metadata_is_dag"],
                "nodes": graph_json["nodes"],
                "links": graph_json["links"],
            }
            all_graphs.append(record)
            if max_graphs and len(all_graphs) >= max_graphs:
                break
        del raw, examples
        gc.collect()
        if max_graphs and len(all_graphs) >= max_graphs:
            break

    domain_counts = Counter(r["domain"] for r in all_graphs)
    logger.info(f"Loaded {len(all_graphs)} graphs across {len(domain_counts)} domains")
    for d, c in sorted(domain_counts.items()):
        logger.info(f"  {d}: {c} graphs")

    for i, r in enumerate(all_graphs):
        if not r["is_dag"]:
            logger.warning(f"Graph {i} ({r['domain']}) reported as non-DAG")

    return all_graphs


# ================================================================
# STEP 2: BUILD IGRAPH GRAPHS WITH WEIGHT PRUNING
# ================================================================

def build_igraph(record: dict, prune_percentile: int) -> igraph.Graph:
    """Build a pruned igraph.Graph from a parsed graph record.

    Prunes edges below the given percentile of abs(weight), simplifies,
    removes isolated nodes, and validates DAG property.
    """
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

    # Compute weight threshold using abs(weight)
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

    # Simplify: remove multi-edges (keep max weight), remove self-loops
    g = g.simplify(multiple=True, loops=True, combine_edges="max")

    # Remove isolated nodes (degree 0 after pruning)
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
    """Build mapping from igraph isoclass ID to graph properties.

    Returns (mapping_dict, dag_valid_connected_ids).
    """
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
            # Only 3-edge connected DAG triad: feed-forward loop
            names[cls_id] = "030T"
        elif n_edges == 2:
            if max(out_degs) == 2:
                names[cls_id] = "021D"  # out-star: one node with out-degree 2
            elif max(in_degs) == 2:
                names[cls_id] = "021U"  # in-star: one node with in-degree 2
            else:
                names[cls_id] = "021C"  # chain: max degrees are 1
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
    # NaN check: NaN != NaN in Python
    counts = [0 if (x != x) else int(x) for x in raw]
    valid_counts = {idx: counts[idx] for idx in dag_valid_ids}
    return valid_counts


def compute_motif_census_sampled(
    g: igraph.Graph, size: int, dag_valid_ids: list[int],
    cut_prob: list[float] | None = None,
) -> dict[int, int]:
    """Run motifs_randesu with optional sampling via cut_prob."""
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
    """Generate one DAG-preserving random graph via degree-preserving edge swaps.

    Uses Goni et al. Method 1: preserves in-degree and out-degree of each node.
    Uses topological ordering for O(1) acyclicity check per swap (conservative).
    Returns (new_edge_list, n_accepted_swaps).
    """
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

        # Skip trivial swaps
        if u1 == u2 or v1 == v2:
            continue

        new_e1 = (u1, v2)
        new_e2 = (u2, v1)

        # Check multi-edge
        if new_e1 in adj_set or new_e2 in adj_set:
            continue

        # Acyclicity check via topological ordering (conservative)
        if topo_rank[u1] >= topo_rank[v2] or topo_rank[u2] >= topo_rank[v1]:
            continue

        # Accept the swap
        adj_set.discard((u1, v1))
        adj_set.discard((u2, v2))
        adj_set.add(new_e1)
        adj_set.add(new_e2)
        edge_list[i1] = new_e1
        edge_list[i2] = new_e2
        accepted += 1

    return edge_list, accepted


def _null_model_batch_worker(args: tuple) -> list[dict[int, int]]:
    """Worker for ProcessPoolExecutor: generate a batch of null models + census.

    Receives serialized graph data (not igraph object) to avoid pickle issues.
    Computes topological ordering once, then generates multiple null models.
    """
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

    # Compute topological ordering once
    topo_order = g.topological_sorting()
    topo_rank = [0] * n_nodes
    for rank, node in enumerate(topo_order):
        topo_rank[node] = rank

    n_swap_attempts = SWAP_MULTIPLIER * len(edges)

    # Split seeds into batches, one per worker
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

    # Significance Profile: SP_i = Z_i / ||Z||
    z_vec = np.array([z_scores[mid] for mid in dag_valid_ids])
    z_norm = float(np.linalg.norm(z_vec))

    if z_norm == 0:
        sp = {mid: 0.0 for mid in dag_valid_ids}
        is_random_like = True
    else:
        sp = {mid: float(z_scores[mid] / z_norm) for mid in dag_valid_ids}
        is_random_like = False

    return {
        "z_scores": z_scores,
        "p_values": p_values,
        "sp": sp,
        "z_norm": z_norm,
        "is_random_like": is_random_like,
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
# STEP 7: BASELINE - GRAPH STATISTICS
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

    # Layer span statistics
    if "layer" in g.vs.attributes() and n_edges > 0:
        layers_arr = np.array(g.vs["layer"], dtype=float)
        edge_spans = []
        for e in g.es:
            edge_spans.append(abs(layers_arr[e.target] - layers_arr[e.source]))
        edge_spans = np.array(edge_spans)
        features["layer_span_mean"] = float(np.mean(edge_spans))
        features["layer_span_std"] = float(np.std(edge_spans))
        features["layer_span_max"] = float(np.max(edge_spans))
        features["n_layers"] = float(len(set(g.vs["layer"])))
    else:
        features["layer_span_mean"] = 0.0
        features["layer_span_std"] = 0.0
        features["layer_span_max"] = 0.0
        features["n_layers"] = 1.0

    # Connected components
    components = g.connected_components(mode="weak")
    features["n_components"] = float(len(components))
    features["largest_component_frac"] = (
        float(max(len(c) for c in components) / n_nodes) if n_nodes > 0 else 0.0
    )

    # Degree assortativity
    try:
        features["assortativity"] = float(g.assortativity_degree(directed=True))
    except Exception:
        features["assortativity"] = 0.0

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

    if use_cosine:
        cos_sim = cosine_similarity(feature_matrix)
        affinity = (cos_sim + 1) / 2  # shift to [0, 1]
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
    parser = argparse.ArgumentParser(description="Circuit Motif Spectroscopy")
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Limit number of graphs (for testing)")
    parser.add_argument("--n-null-3", type=int, default=500,
                        help="Number of 3-node null models per graph per threshold")
    parser.add_argument("--n-null-4", type=int, default=100,
                        help="Number of 4-node null models per graph")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[50, 75, 90],
                        help="Pruning thresholds (percentiles)")
    parser.add_argument("--primary-threshold", type=int, default=75,
                        help="Primary threshold for clustering analysis")
    parser.add_argument("--skip-4node", action="store_true",
                        help="Skip 4-node motif analysis")
    args = parser.parse_args()

    t_start = time.time()
    thresholds = args.thresholds
    primary_pct = args.primary_threshold
    n_workers = max(1, NUM_CPUS - 1)

    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
    logger.info(f"Workers: {n_workers}, Thresholds: {thresholds}, Primary: {primary_pct}")
    logger.info(f"Null models: 3-node={args.n_null_3}, 4-node={args.n_null_4}")

    # ---- STEP 1: Load graphs ----
    logger.info("=" * 60)
    logger.info("STEP 1: Loading attribution graphs")
    all_graphs = load_all_graphs(args.max_graphs)
    if not all_graphs:
        logger.error("No graphs loaded!")
        return

    # ---- STEP 2: Build isoclass mappings ----
    logger.info("=" * 60)
    logger.info("STEP 2: Building isoclass mappings")
    mapping_3, dag_valid_3 = build_isoclass_mapping(3)
    names_3 = identify_3node_man_labels(mapping_3, dag_valid_3)
    logger.info(f"3-node DAG-valid connected types: {len(dag_valid_3)} IDs: {dag_valid_3}")
    for cls_id in dag_valid_3:
        logger.info(f"  ID {cls_id}: {names_3[cls_id]} (edges: {mapping_3[cls_id]['edges']})")
    if len(dag_valid_3) != 4:
        logger.error(f"Expected 4 DAG-valid 3-node types, got {len(dag_valid_3)}")

    mapping_4, dag_valid_4 = build_isoclass_mapping(4)
    logger.info(f"4-node DAG-valid connected types: {len(dag_valid_4)}")
    if len(dag_valid_4) != 24:
        logger.warning(f"Expected 24 DAG-valid 4-node types, got {len(dag_valid_4)}")

    # Identify DAG-impossible connected types for validation
    dag_impossible_3 = [
        cls_id for cls_id in range(16)
        if mapping_3[cls_id]["is_connected"] and not mapping_3[cls_id]["is_dag"]
    ]
    logger.info(f"3-node DAG-impossible connected types: {len(dag_impossible_3)} IDs: {dag_impossible_3}")

    # ---- STEP 3: Build pruned igraph graphs ----
    logger.info("=" * 60)
    logger.info("STEP 3: Building pruned igraph graphs")
    pruned_graphs: dict[tuple[int, int], igraph.Graph] = {}
    for i, record in enumerate(all_graphs):
        for pct in thresholds:
            try:
                g = build_igraph(record, pct)
                pruned_graphs[(i, pct)] = g
                logger.debug(
                    f"Graph {i} ({record['domain']}), prune={pct}%: "
                    f"{g.vcount()} nodes, {g.ecount()} edges"
                )
            except Exception:
                logger.exception(f"Failed building graph {i} at {pct}%")
        # Release raw node/link data to save memory
        record.pop("nodes", None)
        record.pop("links", None)
    gc.collect()
    logger.info(f"Built {len(pruned_graphs)} pruned graphs")

    for pct in thresholds:
        sizes = [
            (pruned_graphs[(i, pct)].vcount(), pruned_graphs[(i, pct)].ecount())
            for i in range(len(all_graphs))
            if (i, pct) in pruned_graphs
        ]
        if sizes:
            nodes_list = [s[0] for s in sizes]
            edges_list = [s[1] for s in sizes]
            logger.info(
                f"  Threshold {pct}%: nodes {min(nodes_list)}-{max(nodes_list)} "
                f"(median {np.median(nodes_list):.0f}), "
                f"edges {min(edges_list)}-{max(edges_list)} "
                f"(median {np.median(edges_list):.0f})"
            )

    # ---- STEP 4: Compute real 3-node motif census ----
    logger.info("=" * 60)
    logger.info("STEP 4: Computing real 3-node motif census")
    real_census_3: dict[tuple[int, int], dict[int, int]] = {}
    for key, g in pruned_graphs.items():
        real_census_3[key] = compute_motif_census(g, 3, dag_valid_3)
    logger.info(f"Computed 3-node census for {len(real_census_3)} graphs")

    # Validation: DAG-impossible connected types must be 0
    n_violations = 0
    for key, g in pruned_graphs.items():
        raw = g.motifs_randesu(size=3)
        for idx in dag_impossible_3:
            val = raw[idx]
            if val == val and val != 0:  # not NaN and not 0
                logger.warning(
                    f"DAG-impossible motif {idx} has count {int(val)} in graph {key}"
                )
                n_violations += 1
    if n_violations == 0:
        logger.info("  DAG validation passed: all impossible motif counts are 0")
    else:
        logger.warning(f"  DAG validation: {n_violations} violations found")

    # ---- STEP 5: Timing calibration ----
    logger.info("=" * 60)
    logger.info("STEP 5: Timing calibration for null model generation")

    # Pick a medium-sized graph at primary threshold
    valid_keys = [
        (i, primary_pct) for i in range(len(all_graphs))
        if (i, primary_pct) in pruned_graphs
    ]
    if not valid_keys:
        logger.error("No valid graphs at primary threshold!")
        return

    calib_key = sorted(valid_keys, key=lambda k: pruned_graphs[k].ecount())[
        len(valid_keys) // 2
    ]
    g_calib = pruned_graphs[calib_key]
    logger.info(
        f"Calibration graph: {calib_key}, "
        f"{g_calib.vcount()} nodes, {g_calib.ecount()} edges"
    )

    t_calib = time.time()
    calib_n = min(10, args.n_null_3)
    _calib_results = generate_null_census_parallel(
        g_calib, 3, dag_valid_3, calib_n, n_workers
    )
    t_calib_elapsed = time.time() - t_calib
    time_per_null = t_calib_elapsed / max(calib_n, 1)
    logger.info(
        f"Calibration: {calib_n} null models in {t_calib_elapsed:.2f}s "
        f"({time_per_null:.3f}s/model)"
    )

    # Estimate total time
    n_combos = len(pruned_graphs)
    estimated_total_s = time_per_null * args.n_null_3 * n_combos / max(n_workers, 1)
    logger.info(
        f"Estimated total null model time: {estimated_total_s:.0f}s "
        f"({estimated_total_s / 60:.1f} min)"
    )

    # Adjust null model count if needed
    n_null_3 = args.n_null_3
    max_null_time_s = 2400  # 40 minutes budget for null models
    if estimated_total_s > max_null_time_s:
        n_null_3 = max(
            50,
            int(max_null_time_s / (time_per_null * n_combos / max(n_workers, 1))),
        )
        logger.info(f"Adjusted null model count to {n_null_3} to fit time budget")

    # ---- STEP 6: Generate null models ----
    logger.info("=" * 60)
    logger.info(f"STEP 6: Generating {n_null_3} null models per graph-threshold combo")

    null_census_3: dict[tuple[int, int], list[dict[int, int]]] = {}
    t_null_start = time.time()
    for idx, ((gi, pct), g) in enumerate(sorted(pruned_graphs.items())):
        t_graph = time.time()
        null_census_3[(gi, pct)] = generate_null_census_parallel(
            g, 3, dag_valid_3, n_null_3, n_workers
        )
        elapsed = time.time() - t_graph
        logger.info(
            f"  [{idx + 1}/{n_combos}] Graph {gi} prune={pct}%: "
            f"{n_null_3} null models in {elapsed:.1f}s "
            f"({g.vcount()}n, {g.ecount()}e)"
        )

        # Time check
        total_null_elapsed = time.time() - t_null_start
        remaining = n_combos - (idx + 1)
        if remaining > 0 and total_null_elapsed > max_null_time_s * 0.9:
            logger.warning(
                f"Approaching time budget ({total_null_elapsed:.0f}s / "
                f"{max_null_time_s}s). {remaining} combos remaining."
            )
            # Reduce null models for remaining combos
            if n_null_3 > 100:
                n_null_3 = max(50, n_null_3 // 2)
                logger.info(f"Reducing null models to {n_null_3} for remaining combos")

    logger.info(
        f"Null model generation complete: {time.time() - t_null_start:.1f}s total"
    )

    # ---- STEP 7: Compute Z-scores and significance profiles ----
    logger.info("=" * 60)
    logger.info("STEP 7: Computing Z-scores and significance profiles")

    results_3node: dict[tuple[int, int], dict] = {}
    for key in real_census_3:
        if key not in null_census_3 or not null_census_3[key]:
            logger.warning(f"No null models for {key}, skipping Z-scores")
            continue
        gi, pct = key
        stats = compute_zscores_and_sp(
            real_census_3[key], null_census_3[key], dag_valid_3
        )
        stats["raw_counts"] = real_census_3[key]
        stats["domain"] = all_graphs[gi]["domain"]
        stats["prompt"] = all_graphs[gi]["prompt"]
        stats["n_nodes"] = pruned_graphs[key].vcount()
        stats["n_edges"] = pruned_graphs[key].ecount()
        results_3node[key] = stats

    # Log per-motif aggregate Z-scores at primary threshold
    logger.info(f"Per-motif Z-scores at threshold {primary_pct}%:")
    for mid in dag_valid_3:
        z_vals = [
            results_3node[(gi, primary_pct)]["z_scores"][mid]
            for gi in range(len(all_graphs))
            if (gi, primary_pct) in results_3node
        ]
        if z_vals:
            logger.info(
                f"  {names_3[mid]} (ID {mid}): mean Z = {np.mean(z_vals):.2f} "
                f"+/- {np.std(z_vals):.2f}, range [{min(z_vals):.1f}, {max(z_vals):.1f}]"
            )

    # ---- STEP 8: Baseline features ----
    logger.info("=" * 60)
    logger.info("STEP 8: Computing baseline graph statistics")

    baseline_features: dict[tuple[int, int], dict[str, float]] = {}
    for key, g in pruned_graphs.items():
        baseline_features[key] = compute_baseline_features(g)

    # ---- STEP 9: Clustering ----
    logger.info("=" * 60)
    logger.info("STEP 9: Spectral clustering analysis")

    # IMPORTANT: Under degree-preserving null models, the 3-node motif changes
    # satisfy: delta(021U) = delta(021C) = delta(021D) = -delta(030T).
    # This is a mathematical identity (degree seq. preserves C(k_out,2), C(k_in,2),
    # k_in*k_out per node). Consequently, the SP is CONSTANT [-0.5,-0.5,-0.5,0.5]
    # for ALL graphs. SP-based clustering is therefore uninformative.
    #
    # Solution: use ENRICHED motif features for clustering:
    #   - Raw motif count ratios (distribution of types among connected triads)
    #   - Z_030T magnitude (quantifies FFL overrepresentation strength)
    #   - FFL density (030T count per node)
    #   - Graph statistics as supplement
    logger.info("NOTE: 3-node SP is degenerate under degree-preserving null models")
    logger.info("  Using enriched motif features (count ratios + Z-magnitude + stats)")

    # Build enriched motif feature vectors at primary threshold
    motif_vectors: list[list[float]] = []
    domain_labels: list[str] = []
    graph_indices: list[int] = []
    motif_count_ratios_all: list[dict[int, float]] = []

    for gi in range(len(all_graphs)):
        key = (gi, primary_pct)
        if key not in results_3node or key not in baseline_features:
            continue

        r = results_3node[key]
        raw = r["raw_counts"]
        total_connected = sum(raw[mid] for mid in dag_valid_3)

        # Motif count ratios
        if total_connected > 0:
            ratios = {mid: raw[mid] / total_connected for mid in dag_valid_3}
        else:
            ratios = {mid: 0.0 for mid in dag_valid_3}
        motif_count_ratios_all.append(ratios)

        # Find the 030T isoclass ID (the one with 3 edges)
        ffl_id = [mid for mid in dag_valid_3 if names_3[mid] == "030T"][0]

        # Enriched feature vector for our method:
        # [ratio_021U, ratio_021C, ratio_021D, ratio_030T,
        #  Z_030T, FFL_density, FFL_per_edge]
        bf = baseline_features[key]
        n_nodes_g = bf["n_nodes"]
        n_edges_g = bf["n_edges"]
        feat_vec = [ratios[mid] for mid in dag_valid_3]  # 4 ratios
        feat_vec.append(r["z_scores"][ffl_id])  # Z_030T
        feat_vec.append(raw[ffl_id] / max(n_nodes_g, 1))  # FFL per node
        feat_vec.append(raw[ffl_id] / max(n_edges_g, 1))  # FFL per edge
        feat_vec.append(bf["density"])
        feat_vec.append(bf["in_degree_std"])
        feat_vec.append(bf["out_degree_std"])
        feat_vec.append(bf["layer_span_mean"])
        feat_vec.append(bf["transitivity"])

        motif_vectors.append(feat_vec)
        domain_labels.append(all_graphs[gi]["domain"])
        graph_indices.append(gi)

    if len(motif_vectors) < 3:
        logger.error(f"Too few valid vectors ({len(motif_vectors)}), cannot cluster")
        return

    motif_matrix = np.array(motif_vectors)
    le = LabelEncoder()
    true_labels = le.fit_transform(domain_labels)

    logger.info(
        f"Enriched motif matrix shape: {motif_matrix.shape}, "
        f"{len(set(domain_labels))} domains, {len(motif_vectors)} graphs"
    )

    # Our method: clustering on enriched motif features (RBF kernel)
    our_clustering = cluster_and_evaluate(
        motif_matrix, true_labels, CLUSTER_K_VALUES, use_cosine=False
    )
    if our_clustering:
        best_k_our = max(our_clustering, key=lambda k: our_clustering[k]["nmi"])
        logger.info(
            f"Our method best K={best_k_our}: "
            f"NMI={our_clustering[best_k_our]['nmi']:.3f}, "
            f"ARI={our_clustering[best_k_our]['ari']:.3f}"
        )
    else:
        best_k_our = 2
        logger.warning("Our clustering failed for all K values")

    # Baseline: clustering on graph statistics ONLY (no motif information)
    feature_keys_sorted = sorted(baseline_features[graph_indices[0], primary_pct].keys())
    baseline_matrix = []
    for gi in graph_indices:
        key = (gi, primary_pct)
        feats = baseline_features[key]
        baseline_matrix.append([feats[k] for k in feature_keys_sorted])
    baseline_matrix_np = np.array(baseline_matrix)

    baseline_clustering = cluster_and_evaluate(
        baseline_matrix_np, true_labels, CLUSTER_K_VALUES, use_cosine=False
    )
    if baseline_clustering:
        best_k_base = max(baseline_clustering, key=lambda k: baseline_clustering[k]["nmi"])
        logger.info(
            f"Baseline best K={best_k_base}: "
            f"NMI={baseline_clustering[best_k_base]['nmi']:.3f}, "
            f"ARI={baseline_clustering[best_k_base]['ari']:.3f}"
        )
    else:
        best_k_base = 2
        logger.warning("Baseline clustering failed for all K values")

    # t-SNE visualization on enriched features
    perp = min(TSNE_PERPLEXITY, max(1, len(motif_vectors) - 1))
    try:
        scaler_tsne = StandardScaler()
        motif_scaled = scaler_tsne.fit_transform(motif_matrix)
        tsne = TSNE(
            n_components=2, perplexity=perp, random_state=42, metric="euclidean"
        )
        coords_2d = tsne.fit_transform(motif_scaled)
    except Exception:
        logger.exception("t-SNE failed, using zeros")
        coords_2d = np.zeros((len(motif_vectors), 2))

    # Map clusters to predicted domains
    our_pred_domains: list[str] = []
    baseline_pred_domains: list[str] = []
    if our_clustering and best_k_our in our_clustering and our_clustering[best_k_our]["pred_labels"]:
        our_pred_domains = map_clusters_to_domains(
            our_clustering[best_k_our]["pred_labels"], domain_labels
        )
    else:
        our_pred_domains = ["unknown"] * len(domain_labels)

    if baseline_clustering and best_k_base in baseline_clustering and baseline_clustering[best_k_base]["pred_labels"]:
        baseline_pred_domains = map_clusters_to_domains(
            baseline_clustering[best_k_base]["pred_labels"], domain_labels
        )
    else:
        baseline_pred_domains = ["unknown"] * len(domain_labels)

    # ---- STEP 10: Hypothesis testing ----
    logger.info("=" * 60)
    logger.info("STEP 10: Hypothesis testing")

    # H1: Universal overrepresentation
    h1_per_motif: dict[str, dict] = {}
    for mid in dag_valid_3:
        z_by_domain: dict[str, list[float]] = defaultdict(list)
        for gi in range(len(all_graphs)):
            key = (gi, primary_pct)
            if key in results_3node:
                z_by_domain[all_graphs[gi]["domain"]].append(
                    results_3node[key]["z_scores"][mid]
                )

        all_z = [z for zs in z_by_domain.values() for z in zs]
        n_domains_sig = sum(
            1 for d, zs in z_by_domain.items() if np.mean(zs) > 2.0
        )
        h1_per_motif[str(mid)] = {
            "name": names_3[mid],
            "mean_z": float(np.mean(all_z)) if all_z else 0.0,
            "std_z": float(np.std(all_z)) if all_z else 0.0,
            "n_domains_significant": n_domains_sig,
            "n_domains_total": len(z_by_domain),
            "per_domain_mean_z": {
                d: float(np.mean(zs)) for d, zs in z_by_domain.items()
            },
        }
        logger.info(
            f"  {names_3[mid]}: mean Z={h1_per_motif[str(mid)]['mean_z']:.2f}, "
            f"significant in {n_domains_sig}/{len(z_by_domain)} domains"
        )

    n_motifs_universal = sum(
        1 for m in h1_per_motif.values() if m["n_domains_significant"] >= 6
    )
    h1_result = "CONFIRM" if n_motifs_universal >= 3 else "DISCONFIRM"
    logger.info(
        f"H1 Result: {n_motifs_universal} motif types with Z>2 in >=6/8 domains "
        f"-> {h1_result}"
    )

    # H2: Capability clustering
    best_nmi_our = our_clustering[best_k_our]["nmi"] if our_clustering and best_k_our in our_clustering else 0.0
    h2_result = "CONFIRM" if best_nmi_our > 0.5 else "DISCONFIRM"
    logger.info(f"H2 Result: Best NMI={best_nmi_our:.3f} -> {h2_result}")

    # Discriminative motifs (F-statistic on count ratios, not degenerate SP)
    discriminative_motifs: dict[str, dict] = {}
    for j, mid in enumerate(dag_valid_3):
        # Use motif count ratios (column j of motif_matrix) for F-test
        ratio_vals = motif_matrix[:, j]  # first 4 cols are ratios
        groups = [
            ratio_vals[true_labels == k]
            for k in range(len(le.classes_))
            if np.sum(true_labels == k) > 1  # need >= 2 samples per group
        ]
        groups = [g_arr for g_arr in groups if len(g_arr) > 1]
        if len(groups) >= 2:
            try:
                f_stat, f_pval = f_oneway(*groups)
                if not np.isnan(f_stat):
                    discriminative_motifs[str(mid)] = {
                        "name": names_3[mid],
                        "f_statistic": float(f_stat),
                        "p_value": float(f_pval),
                    }
                    logger.info(
                        f"  {names_3[mid]} ratio: F={f_stat:.2f}, p={f_pval:.4f}"
                    )
            except Exception:
                pass

    # ---- STEP 11: Threshold robustness ----
    logger.info("=" * 60)
    logger.info("STEP 11: Threshold robustness analysis")

    # Use motif count RATIOS for threshold robustness (SP is degenerate)
    threshold_stability: dict[int, dict] = {}
    for gi in range(len(all_graphs)):
        ratio_at_thresholds = []
        for pct in thresholds:
            key = (gi, pct)
            if key in results_3node:
                raw = results_3node[key]["raw_counts"]
                total = sum(raw[mid] for mid in dag_valid_3)
                if total > 0:
                    ratio_at_thresholds.append(
                        [raw[mid] / total for mid in dag_valid_3]
                    )

        if len(ratio_at_thresholds) == len(thresholds) and len(thresholds) >= 2:
            stability: dict[str, float] = {}
            for t_idx in range(len(thresholds) - 1):
                t1, t2 = thresholds[t_idx], thresholds[t_idx + 1]
                try:
                    sim = 1 - cos_dist(
                        ratio_at_thresholds[t_idx], ratio_at_thresholds[t_idx + 1]
                    )
                except Exception:
                    sim = 0.0
                stability[f"sim_{t1}_{t2}"] = float(sim)
            threshold_stability[gi] = stability

    if threshold_stability:
        all_stability_keys = list(list(threshold_stability.values())[0].keys())
        for sk in all_stability_keys:
            vals = [s[sk] for s in threshold_stability.values() if sk in s]
            if vals:
                logger.info(
                    f"  {sk}: mean={np.mean(vals):.3f}, "
                    f"std={np.std(vals):.3f}, min={min(vals):.3f}"
                )

    # Clustering at each threshold using motif count ratios + Z + stats
    clustering_by_threshold: dict[str, dict] = {}
    for pct in thresholds:
        feat_vecs_t: list[list[float]] = []
        labels_t: list[str] = []
        ffl_id = [mid for mid in dag_valid_3 if names_3[mid] == "030T"][0]
        for gi in range(len(all_graphs)):
            key = (gi, pct)
            if key in results_3node and key in baseline_features:
                r = results_3node[key]
                raw = r["raw_counts"]
                total = sum(raw[mid] for mid in dag_valid_3)
                if total > 0:
                    ratios = [raw[mid] / total for mid in dag_valid_3]
                else:
                    ratios = [0.0] * len(dag_valid_3)
                bf = baseline_features[key]
                vec = ratios + [
                    r["z_scores"][ffl_id],
                    raw[ffl_id] / max(bf["n_nodes"], 1),
                    bf["density"], bf["in_degree_std"],
                    bf["out_degree_std"], bf["layer_span_mean"],
                    bf["transitivity"],
                ]
                feat_vecs_t.append(vec)
                labels_t.append(all_graphs[gi]["domain"])
        if len(feat_vecs_t) >= 3:
            mat = np.array(feat_vecs_t)
            tl = LabelEncoder().fit_transform(labels_t)
            clust = cluster_and_evaluate(mat, tl, CLUSTER_K_VALUES, use_cosine=False)
            if clust:
                best_k_t = max(clust, key=lambda k: clust[k]["nmi"])
                clustering_by_threshold[str(pct)] = {
                    "best_k": best_k_t,
                    "best_nmi": clust[best_k_t]["nmi"],
                    "best_ari": clust[best_k_t]["ari"],
                }
                logger.info(
                    f"  Threshold {pct}%: best K={best_k_t}, "
                    f"NMI={clust[best_k_t]['nmi']:.3f}, ARI={clust[best_k_t]['ari']:.3f}"
                )

    # ---- STEP 12: 4-node analysis (if time permits) ----
    results_4node: dict[int, dict[int, int]] = {}
    null_census_4: dict[int, list[dict[int, int]]] = {}
    results_4node_stats: dict[int, dict] = {}

    remaining_time = 3600 - (time.time() - t_start)
    if not args.skip_4node and remaining_time > 300:
        logger.info("=" * 60)
        logger.info("STEP 12: Attempting 4-node motif census")

        for gi in range(len(all_graphs)):
            key = (gi, primary_pct)
            if key not in pruned_graphs:
                continue
            g = pruned_graphs[key]
            if g.vcount() > MAX_4NODE_NODES:
                logger.info(
                    f"  Skipping graph {gi}: {g.vcount()} nodes > {MAX_4NODE_NODES}"
                )
                continue

            t4 = time.time()
            try:
                cut_prob = [0, 0, 0.3, 0.6] if g.vcount() > 200 else None
                census = compute_motif_census_sampled(
                    g, 4, dag_valid_4, cut_prob
                )
                elapsed = time.time() - t4
                results_4node[gi] = census
                logger.info(
                    f"  Graph {gi}: 4-node census in {elapsed:.1f}s "
                    f"({g.vcount()}n, {g.ecount()}e)"
                )
            except Exception:
                logger.exception(f"  Graph {gi}: 4-node census failed")

            if time.time() - t_start > 3300:
                logger.warning("Time budget nearing limit, stopping 4-node")
                break

        # Generate null models for 4-node (fewer)
        if results_4node:
            logger.info(
                f"  Generating {args.n_null_4} null models for "
                f"{len(results_4node)} graphs (4-node)"
            )
            for gi in results_4node:
                key = (gi, primary_pct)
                g = pruned_graphs[key]
                t4n = time.time()
                try:
                    nc = generate_null_census_parallel(
                        g, 4, dag_valid_4, args.n_null_4, n_workers
                    )
                    null_census_4[gi] = nc
                    elapsed = time.time() - t4n
                    logger.info(
                        f"  Graph {gi}: {args.n_null_4} 4-node null models "
                        f"in {elapsed:.1f}s"
                    )
                except Exception:
                    logger.exception(f"  Graph {gi}: 4-node null models failed")

                if time.time() - t_start > 3400:
                    logger.warning("Time budget limit, stopping 4-node null models")
                    break

            # Compute 4-node Z-scores
            for gi in results_4node:
                if gi in null_census_4 and null_census_4[gi]:
                    stats = compute_zscores_and_sp(
                        results_4node[gi], null_census_4[gi], dag_valid_4
                    )
                    results_4node_stats[gi] = stats
    else:
        if args.skip_4node:
            logger.info("STEP 12: 4-node analysis skipped (--skip-4node)")
        else:
            logger.info("STEP 12: 4-node analysis skipped (insufficient time)")

    # ---- STEP 13: Assemble output ----
    logger.info("=" * 60)
    logger.info("STEP 13: Assembling output")

    # Domain aggregate stats (use count ratios instead of degenerate SP)
    domain_agg: dict[str, dict] = defaultdict(
        lambda: {"n_graphs": 0, "z_scores": defaultdict(list),
                 "count_ratios": defaultdict(list)}
    )
    for gi in range(len(all_graphs)):
        key = (gi, primary_pct)
        if key in results_3node:
            d = all_graphs[gi]["domain"]
            domain_agg[d]["n_graphs"] += 1
            raw = results_3node[key]["raw_counts"]
            total = sum(raw[mid] for mid in dag_valid_3)
            for mid in dag_valid_3:
                domain_agg[d]["z_scores"][mid].append(
                    results_3node[key]["z_scores"][mid]
                )
                domain_agg[d]["count_ratios"][mid].append(
                    raw[mid] / total if total > 0 else 0.0
                )

    domain_aggregate_stats = {}
    for d, agg in domain_agg.items():
        domain_aggregate_stats[d] = {
            "n_graphs": agg["n_graphs"],
            "mean_z_scores": {
                str(mid): float(np.mean(agg["z_scores"][mid]))
                for mid in dag_valid_3
            },
            "std_z_scores": {
                str(mid): float(np.std(agg["z_scores"][mid]))
                for mid in dag_valid_3
            },
            "mean_count_ratios": {
                str(mid): float(np.mean(agg["count_ratios"][mid]))
                for mid in dag_valid_3
            },
        }

    # Build per-example output entries
    examples: list[dict] = []
    for idx, gi in enumerate(graph_indices):
        per_graph_output: dict[str, Any] = {
            "graph_stats": {},
            "motif_census_3node": {},
            "baseline_features": {},
        }

        for pct in thresholds:
            pkey = (gi, pct)
            if pkey in pruned_graphs:
                per_graph_output["graph_stats"][str(pct)] = {
                    "n_nodes": pruned_graphs[pkey].vcount(),
                    "n_edges": pruned_graphs[pkey].ecount(),
                    "density": float(pruned_graphs[pkey].density()),
                }
            if pkey in results_3node:
                r = results_3node[pkey]
                raw_c = r["raw_counts"]
                total_c = sum(raw_c[mid] for mid in dag_valid_3)
                per_graph_output["motif_census_3node"][str(pct)] = {
                    "raw_counts": {str(k): v for k, v in raw_c.items()},
                    "count_ratios": {
                        str(k): round(v / total_c, 6) if total_c > 0 else 0.0
                        for k, v in raw_c.items()
                    },
                    "z_scores": {str(k): round(v, 4) for k, v in r["z_scores"].items()},
                    "p_values": {str(k): round(v, 4) for k, v in r["p_values"].items()},
                    "sp": {str(k): round(v, 6) for k, v in r["sp"].items()},
                    "is_random_like": r["is_random_like"],
                    "z_norm": round(r["z_norm"], 4),
                }

        prim_key = (gi, primary_pct)
        if prim_key in baseline_features:
            per_graph_output["baseline_features"] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in baseline_features[prim_key].items()
            }

        if gi in results_4node_stats:
            r4 = results_4node_stats[gi]
            per_graph_output["motif_census_4node"] = {
                "raw_counts": {str(k): v for k, v in results_4node[gi].items()},
                "z_scores": {str(k): round(v, 4) for k, v in r4["z_scores"].items()},
                "sp": {str(k): round(v, 6) for k, v in r4["sp"].items()},
            }

        if gi in threshold_stability:
            per_graph_output["threshold_robustness"] = threshold_stability[gi]

        per_graph_output["tsne_coords"] = (
            coords_2d[idx].tolist() if idx < len(coords_2d) else [0.0, 0.0]
        )

        example = {
            "input": all_graphs[gi]["prompt"],
            "output": json.dumps(per_graph_output, default=str),
            "predict_our_method": (
                our_pred_domains[idx] if idx < len(our_pred_domains) else ""
            ),
            "predict_baseline": (
                baseline_pred_domains[idx] if idx < len(baseline_pred_domains) else ""
            ),
            "metadata_fold": all_graphs[gi]["domain"],
            "metadata_graph_idx": gi,
            "metadata_n_nodes_raw": all_graphs[gi]["n_nodes_raw"],
            "metadata_n_edges_raw": all_graphs[gi]["n_edges_raw"],
        }
        examples.append(example)

    # Top-level output
    output = {
        "metadata": {
            "experiment": "circuit_motif_spectroscopy",
            "n_graphs": len(all_graphs),
            "n_domains": len(set(r["domain"] for r in all_graphs)),
            "domains": sorted(set(r["domain"] for r in all_graphs)),
            "isoclass_mapping_3node": {
                "dag_valid_connected_ids": dag_valid_3,
                "man_labels": {str(k): v for k, v in names_3.items()},
            },
            "isoclass_mapping_4node": {
                "dag_valid_connected_ids": dag_valid_4,
                "n_types": len(dag_valid_4),
            },
            "methodological_finding": {
                "title": "3-node SP degeneracy under degree-preserving null models",
                "description": (
                    "Under degree-preserving DAG null models (Goni Method 1), "
                    "the change in motif counts satisfies the identity: "
                    "delta(021U) = delta(021C) = delta(021D) = -delta(030T). "
                    "This follows from degree preservation of C(k_out,2), "
                    "C(k_in,2), and k_in*k_out per node. Consequently, "
                    "Z-scores are identical for all 2-edge types and the "
                    "Significance Profile (SP) is constant [-0.5,-0.5,-0.5,0.5] "
                    "for ALL graphs, making SP-based clustering uninformative. "
                    "We instead use motif count ratios (distribution of types "
                    "among connected triads) combined with Z-magnitude and "
                    "graph statistics for clustering."
                ),
                "implication": (
                    "The effective dimensionality of the 3-node DAG motif "
                    "spectrum under degree-preserving null models is 1, not 4. "
                    "The 4-node spectrum (24D) is critical for richer analysis."
                ),
            },
            "hypothesis_1_universal_overrepresentation": {
                "description": (
                    "For each motif type, across how many domains is mean Z > 2.0? "
                    "Note: due to the mathematical identity above, only 030T "
                    "(FFL overrepresentation) is independently testable."
                ),
                "per_motif": h1_per_motif,
                "criterion": ">=3 motif types with Z>2.0 in >=6/8 domains",
                "n_motifs_universal": n_motifs_universal,
                "result": h1_result,
            },
            "hypothesis_2_capability_clustering": {
                "our_method_nmi_by_k": {
                    str(k): v["nmi"] for k, v in our_clustering.items()
                },
                "our_method_ari_by_k": {
                    str(k): v["ari"] for k, v in our_clustering.items()
                },
                "baseline_nmi_by_k": {
                    str(k): v["nmi"] for k, v in baseline_clustering.items()
                },
                "baseline_ari_by_k": {
                    str(k): v["ari"] for k, v in baseline_clustering.items()
                },
                "our_method_best_k": best_k_our,
                "our_method_best_nmi": (
                    our_clustering[best_k_our]["nmi"]
                    if best_k_our in our_clustering else 0.0
                ),
                "baseline_best_k": best_k_base,
                "baseline_best_nmi": (
                    baseline_clustering[best_k_base]["nmi"]
                    if best_k_base in baseline_clustering else 0.0
                ),
                "criterion": "NMI > 0.5",
                "result": h2_result,
                "discriminative_motifs": discriminative_motifs,
            },
            "threshold_robustness": {
                "clustering_by_threshold": clustering_by_threshold,
                "per_graph_stability_summary": {
                    sk: {
                        "mean": float(np.mean([
                            s[sk] for s in threshold_stability.values() if sk in s
                        ])),
                        "std": float(np.std([
                            s[sk] for s in threshold_stability.values() if sk in s
                        ])),
                    }
                    for sk in (
                        list(list(threshold_stability.values())[0].keys())
                        if threshold_stability else []
                    )
                },
            },
            "domain_aggregate_stats": domain_aggregate_stats,
            "config": {
                "prune_thresholds": thresholds,
                "primary_threshold": primary_pct,
                "n_null_models_3node_actual": n_null_3,
                "n_null_models_4node": args.n_null_4,
                "swap_multiplier": SWAP_MULTIPLIER,
                "cluster_k_values": CLUSTER_K_VALUES,
                "n_workers": n_workers,
                "hardware": {"cpus": NUM_CPUS, "ram_gb": round(TOTAL_RAM_GB, 1)},
            },
            "runtime_seconds": round(time.time() - t_start, 1),
        },
        "datasets": [
            {
                "dataset": "neuronpedia_attribution_graphs",
                "examples": examples,
            }
        ],
    }

    # Save output
    output_json = json.dumps(output, indent=2, default=str)
    OUTPUT_FILE.write_text(output_json)
    file_size_mb = len(output_json) / 1e6
    logger.info(f"Output saved to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    logger.info(f"Total runtime: {time.time() - t_start:.1f}s")

    return output


if __name__ == "__main__":
    main()
