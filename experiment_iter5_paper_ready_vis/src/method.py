#!/usr/bin/env python3
"""Paper-Ready Visualization Data: Embeddings, Heatmaps, Z-Scores, Ablation,
Confusion Matrices, and Motif Diagrams for 200 Attribution Graphs.

Recompute motif census, weighted features, graph statistics, and null-model
Z-scores from 200 attribution graphs. Produce structured JSON with:
  (A) t-SNE/UMAP 2D embedding coordinates
  (B) Domain-level motif spectrum heatmap matrices
  (C) Per-domain Z-score box plot statistics
  (D) Ablation impact comparison data with bootstrap CIs
  (E) Clustering confusion matrices with Hungarian alignment
  (F) Network motif diagram specifications with representative instances

Baseline: random clustering assignment for domain prediction.
Our method: spectral clustering on combined motif+graph features.
"""

import json
import sys
import os
import gc
import math
import random
import time
import resource
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from typing import Any

import numpy as np
from loguru import logger
import igraph
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats as scipy_stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ================================================================
# HARDWARE DETECTION (cgroup-aware, from aii_use_hardware skill)
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
    for p in ["/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
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
FEAT_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_2/gen_art/data_id4_it2__opus"
)
OUTPUT_FILE = WORKSPACE / "method_out.json"
LOG_DIR = WORKSPACE / "logs"

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))
N_NULL_MODELS = int(os.environ.get("N_NULL_MODELS", "30"))
SWAP_MULTIPLIER = 100
PRUNE_PERCENTILE = 75
MIN_NODES = 30
SEED = 42
TOTAL_TIME_BUDGET_S = int(os.environ.get("TIME_BUDGET_S", "3400"))
MAX_NULL_TIME_S = 2400
ABLATION_GRAPHS = int(os.environ.get("ABLATION_GRAPHS", "50"))
N_BOOTSTRAP = 2000

DOMAINS = ["antonym", "arithmetic", "code_completion", "country_capital",
           "multi_hop_reasoning", "rhyme", "sentiment", "translation"]

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

_RAM_BUDGET_BYTES = int(min(TOTAL_RAM_GB * 0.75, 22) * 1e9)
try:
    resource.setrlimit(resource.RLIMIT_AS, (_RAM_BUDGET_BYTES * 3, _RAM_BUDGET_BYTES * 3))
except ValueError:
    pass
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))


# ================================================================
# JSON SANITIZER
# ================================================================

def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
    return obj


# ================================================================
# DATA LOADING (proven pattern from exp_id4_it4)
# ================================================================

def load_all_graphs(max_graphs: int | None = None) -> list[dict]:
    """Load graphs from data_id5_it3 split files, deduplicate by slug."""
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
                "n_nodes_raw": ex.get("metadata_n_nodes", 0),
                "n_edges_raw": ex.get("metadata_n_edges", 0),
            })
        del raw, examples
        return records

    mini_mode = os.environ.get("MINI_MODE", "0") == "1"
    if mini_mode:
        new_recs = _load_split(DATA_DIR / "mini_data_out.json")
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
    correct_counts = Counter(r["model_correct"] for r in all_records)
    logger.info(f"Loaded {len(all_records)} unique graphs, {len(domain_counts)} domains")
    for d, c in sorted(domain_counts.items()):
        logger.info(f"  {d}: {c}")
    logger.info(f"Correctness: {dict(correct_counts)}")
    return all_records


# ================================================================
# GRAPH CONSTRUCTION (with signed weights preserved)
# ================================================================

def build_igraph(record: dict, prune_percentile: int) -> igraph.Graph:
    """Build a pruned igraph.Graph, preserving signed_weight attribute."""
    nodes = record["nodes"]
    links = record["links"]

    node_ids = [n["node_id"] for n in nodes]
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    layers, features, feature_types = [], [], []
    for n in nodes:
        try:
            layers.append(int(n.get("layer", 0)))
        except (ValueError, TypeError):
            layers.append(0)
        try:
            features.append(int(n.get("feature", 0)))
        except (ValueError, TypeError):
            features.append(0)
        feature_types.append(n.get("feature_type", ""))

    all_abs_weights = [abs(link.get("weight", 0.0)) for link in links]
    threshold = float(np.percentile(all_abs_weights, prune_percentile)) if all_abs_weights else 0.0

    edges, edge_weights, signed_weights = [], [], []
    for link in links:
        raw_w = link.get("weight", 0.0)
        w = abs(raw_w)
        if w >= threshold:
            src = node_id_to_idx.get(link.get("source"))
            tgt = node_id_to_idx.get(link.get("target"))
            if src is not None and tgt is not None and src != tgt:
                edges.append((src, tgt))
                edge_weights.append(w)
                signed_weights.append(raw_w)

    g = igraph.Graph(n=len(node_ids), edges=edges, directed=True)
    g.vs["node_id"] = node_ids
    g.vs["layer"] = layers
    g.vs["feature"] = features
    g.vs["feature_type"] = feature_types
    if edge_weights:
        g.es["weight"] = edge_weights
        g.es["signed_weight"] = signed_weights

    g = g.simplify(multiple=True, loops=True, combine_edges={"weight": "max", "signed_weight": "max"})

    isolated = [v.index for v in g.vs if g.degree(v) == 0]
    if isolated:
        g.delete_vertices(isolated)

    if not g.is_dag():
        raise ValueError("Graph is not DAG after pruning")
    return g


# ================================================================
# ISOCLASS MAPPING (3-node)
# ================================================================

def build_isoclass_mapping() -> tuple[dict, list[int], dict[int, str]]:
    """Build mapping from igraph isoclass IDs to MAN labels for 3-node triads."""
    mapping: dict[int, dict] = {}
    dag_valid: list[int] = []

    for cls_id in range(16):
        g = igraph.Graph.Isoclass(n=3, cls=cls_id, directed=True)
        edge_list = g.get_edgelist()
        mapping[cls_id] = {
            "edges": edge_list,
            "n_edges": len(edge_list),
            "is_connected": g.is_connected(mode="weak"),
            "is_dag": g.is_dag(),
        }
        if g.is_connected(mode="weak") and g.is_dag():
            dag_valid.append(cls_id)

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

    return mapping, dag_valid, names


# ================================================================
# MOTIF CENSUS
# ================================================================

def compute_motif_census(g: igraph.Graph, dag_valid_ids: list[int]) -> dict[int, int]:
    raw = g.motifs_randesu(size=3)
    counts = [0 if (x != x) else int(x) for x in raw]
    return {idx: counts[idx] for idx in dag_valid_ids}


# ================================================================
# NULL MODEL (degree-preserving DAG edge swaps)
# ================================================================

def _generate_null_edges(
    n_nodes: int, edges: list[tuple[int, int]], topo_rank: list[int],
    n_swap_attempts: int, seed: int,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edge_list = list(edges)
    n_edges = len(edge_list)
    if n_edges < 2:
        return edge_list
    adj_set = set(edge_list)
    for _ in range(n_swap_attempts):
        i1 = rng.randint(0, n_edges - 1)
        i2 = rng.randint(0, n_edges - 1)
        if i1 == i2:
            continue
        u1, v1 = edge_list[i1]
        u2, v2 = edge_list[i2]
        if u1 == u2 or v1 == v2:
            continue
        new_e1, new_e2 = (u1, v2), (u2, v1)
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
    return edge_list


def _null_batch_worker(args: tuple) -> list[dict[int, int]]:
    n_nodes, edges, topo_rank, n_swap, seeds, dag_valid_ids = args
    results = []
    for seed in seeds:
        new_edges = _generate_null_edges(n_nodes, edges, topo_rank, n_swap, seed)
        g_null = igraph.Graph(n=n_nodes, edges=new_edges, directed=True)
        raw = g_null.motifs_randesu(size=3)
        counts = [0 if (x != x) else int(x) for x in raw]
        results.append({idx: counts[idx] for idx in dag_valid_ids})
        del g_null
    return results


def generate_null_census(
    g: igraph.Graph, dag_valid_ids: list[int],
    n_null: int, n_workers: int,
) -> list[dict[int, int]]:
    n_nodes = g.vcount()
    edges = [tuple(e.tuple) for e in g.es]
    topo_order = g.topological_sorting()
    topo_rank = [0] * n_nodes
    for rank, node in enumerate(topo_order):
        topo_rank[node] = rank
    n_swap = SWAP_MULTIPLIER * len(edges)

    all_seeds = list(range(n_null))
    batch_size = max(1, math.ceil(n_null / n_workers))
    batches = []
    for i in range(0, n_null, batch_size):
        batches.append((
            n_nodes, edges, topo_rank, n_swap,
            all_seeds[i:i + batch_size], dag_valid_ids,
        ))

    all_results: list[dict[int, int]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_null_batch_worker, b): idx for idx, b in enumerate(batches)}
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception:
                logger.exception(f"Null batch {futures[future]} failed")
    return all_results


def compute_zscores(
    real_counts: dict[int, int], null_list: list[dict[int, int]],
    dag_valid_ids: list[int],
) -> dict[str, Any]:
    z_scores, null_means, null_stds = {}, {}, {}
    for mid in dag_valid_ids:
        real_val = real_counts[mid]
        nulls = np.array([nc[mid] for nc in null_list], dtype=float)
        mu, sigma = float(np.mean(nulls)), float(np.std(nulls))
        null_means[mid] = mu
        null_stds[mid] = sigma
        if sigma == 0:
            z_scores[mid] = 0.0 if real_val == mu else (10.0 if real_val > mu else -10.0)
        else:
            z_scores[mid] = float((real_val - mu) / sigma)
    return {"z_scores": z_scores, "null_means": null_means, "null_stds": null_stds}


# ================================================================
# WEIGHTED MOTIF FEATURES (FFL / 030T enumeration)
# ================================================================

def enumerate_ffls(g: igraph.Graph) -> list[dict]:
    """Enumerate all feed-forward loop (030T) instances: A->B, A->C, B->C."""
    adj_set: set[tuple[int, int]] = set()
    weight_map: dict[tuple[int, int], float] = {}
    signed_map: dict[tuple[int, int], float] = {}

    has_weight = "weight" in g.es.attributes() if g.ecount() > 0 else False
    has_signed = "signed_weight" in g.es.attributes() if g.ecount() > 0 else False

    for e in g.es:
        edge = (e.source, e.target)
        adj_set.add(edge)
        weight_map[edge] = e["weight"] if has_weight else 1.0
        signed_map[edge] = e["signed_weight"] if has_signed else weight_map[edge]

    successors: dict[int, list[int]] = {}
    for s, t in adj_set:
        successors.setdefault(s, []).append(t)

    ffls = []
    for a, succs in successors.items():
        if len(succs) < 2:
            continue
        for i in range(len(succs)):
            for j in range(len(succs)):
                if i == j:
                    continue
                b, c = succs[i], succs[j]
                if (b, c) in adj_set:
                    w_ab = weight_map[(a, b)]
                    w_ac = weight_map[(a, c)]
                    w_bc = weight_map[(b, c)]
                    sw_ab = signed_map[(a, b)]
                    sw_ac = signed_map[(a, c)]
                    sw_bc = signed_map[(b, c)]
                    intensity = (w_ab * w_ac * w_bc) ** (1.0 / 3.0)
                    denom_pd = w_ab * w_bc
                    path_dom = w_ac / denom_pd if denom_pd > 1e-12 else 0.0
                    is_coherent = (1 if sw_ab > 0 else -1) * (1 if sw_ac > 0 else -1) * (1 if sw_bc > 0 else -1) > 0
                    denom_wa = w_ab + w_bc
                    w_asym = abs(w_ab - w_bc) / denom_wa if denom_wa > 1e-12 else 0.0
                    ffls.append({
                        "a": a, "b": b, "c": c,
                        "w_ab": w_ab, "w_ac": w_ac, "w_bc": w_bc,
                        "intensity": intensity,
                        "path_dominance": path_dom,
                        "is_coherent": is_coherent,
                        "weight_asymmetry": w_asym,
                    })
    return ffls


def compute_weighted_features(ffls: list[dict], motif_ratios: np.ndarray) -> np.ndarray:
    """Compute ~12D weighted feature vector from FFL instances + motif ratios."""
    if not ffls:
        return np.concatenate([np.zeros(8), motif_ratios])

    intensities = np.array([f["intensity"] for f in ffls])
    path_doms = np.array([f["path_dominance"] for f in ffls])
    coherences = np.array([f["is_coherent"] for f in ffls], dtype=float)
    asymmetries = np.array([f["weight_asymmetry"] for f in ffls])

    weighted_feats = np.array([
        float(np.mean(intensities)),
        float(np.median(intensities)),
        float(np.std(intensities)),
        float(np.mean(coherences)),
        float(np.mean(path_doms)),
        float(np.mean(asymmetries)),
        float(len(ffls)),
        float(len(ffls)) / max(1.0, float(sum(motif_ratios > 0))),
    ])
    return np.concatenate([weighted_feats, motif_ratios])


# ================================================================
# GRAPH STATISTICS (10D)
# ================================================================

def compute_graph_stats(g: igraph.Graph) -> np.ndarray:
    n = g.vcount()
    m = g.ecount()
    in_degs = np.array(g.indegree(), dtype=float) if n > 0 else np.array([0.0])
    out_degs = np.array(g.outdegree(), dtype=float) if n > 0 else np.array([0.0])
    layers = set(g.vs["layer"]) if "layer" in g.vs.attributes() else {0}
    has_w = "weight" in g.es.attributes() if m > 0 else False
    weights = np.array(g.es["weight"], dtype=float) if has_w else np.array([0.0])
    try:
        comps = g.connected_components(mode="weak")
        largest = comps.giant()
        diam = largest.diameter(directed=True) if largest.vcount() > 1 else 0
    except Exception:
        diam = 0
    return np.array([
        n, m, g.density() if n > 1 else 0.0,
        float(np.mean(in_degs)), float(np.mean(out_degs)),
        float(np.max(out_degs)) if n > 0 else 0.0,
        len(layers), diam,
        float(np.mean(weights)), float(np.std(weights)),
    ])


# ================================================================
# SAFE SPECTRAL CLUSTERING (dense LAPACK, no ARPACK)
# ================================================================

def _safe_spectral_cluster(affinity: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    n = affinity.shape[0]
    degrees = affinity.sum(axis=1)
    degrees = np.maximum(degrees, 1e-10)
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    L_norm = np.eye(n) - (d_inv_sqrt[:, None] * affinity * d_inv_sqrt[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    embedding = eigenvectors[:, :n_clusters].copy()
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding = embedding / norms
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
    return km.fit_predict(embedding)


# ================================================================
# CONFUSION MATRIX WITH HUNGARIAN ALIGNMENT
# ================================================================

def build_confusion_with_hungarian(
    true_labels: np.ndarray, pred_labels: np.ndarray,
    n_true_classes: int, n_pred_clusters: int,
) -> tuple[np.ndarray, list[int]]:
    """Build confusion matrix and align clusters to true classes via Hungarian."""
    cm = np.zeros((n_true_classes, n_pred_clusters), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)
    aligned_order = list(col_ind)
    # Reorder columns by alignment
    cm_aligned = cm[:, aligned_order]
    return cm_aligned, aligned_order


# ================================================================
# ABLATION ANALYSIS
# ================================================================

def compute_ablation_data(
    graphs: list[igraph.Graph], graph_indices: list[int],
    all_ffls: list[list[dict]], domains: list[str],
) -> dict:
    """Compute ablation impact for hub vs control nodes."""
    hub_impacts: list[float] = []
    control_degree: list[float] = []
    control_random: list[float] = []
    control_layer: list[float] = []
    mpi_impact_pairs: list[tuple[float, float]] = []

    for idx, gi in enumerate(graph_indices):
        g = graphs[gi]
        ffls = all_ffls[gi]
        if not ffls or g.vcount() < MIN_NODES:
            continue

        total_weight = sum(g.es["weight"]) if g.ecount() > 0 else 1.0
        if total_weight < 1e-12:
            continue

        # Compute MPI
        mpi: dict[int, int] = {}
        for ffl in ffls:
            for node_key in ["a", "b", "c"]:
                nid = ffl[node_key]
                mpi[nid] = mpi.get(nid, 0) + 1

        if not mpi:
            continue

        sorted_nodes = sorted(mpi.keys(), key=lambda x: mpi[x], reverse=True)
        n_hub = max(1, len(sorted_nodes) // 10)
        hub_nodes = sorted_nodes[:n_hub]

        # Store MPI-impact pairs for dose-response
        for nid in mpi:
            if nid >= g.vcount():
                continue
            incident = g.incident(nid, mode="all")
            node_impact = sum(g.es[e]["weight"] for e in incident) / total_weight
            mpi_impact_pairs.append((float(mpi[nid]), float(node_impact)))

        # Hub ablation
        for hub in hub_nodes:
            if hub >= g.vcount():
                continue
            incident = g.incident(hub, mode="all")
            hub_impact = sum(g.es[e]["weight"] for e in incident) / total_weight
            hub_impacts.append(hub_impact)

            hub_deg = g.degree(hub)
            hub_layer = g.vs[hub]["layer"]

            # Control: degree-matched
            candidates = [v for v in range(g.vcount())
                          if v != hub and v not in hub_nodes
                          and abs(g.degree(v) - hub_deg) <= 2]
            if candidates:
                cv = random.choice(candidates)
                inc = g.incident(cv, mode="all")
                control_degree.append(sum(g.es[e]["weight"] for e in inc) / total_weight)

            # Control: layer-matched
            layer_cands = [v for v in range(g.vcount())
                           if v != hub and v not in hub_nodes
                           and g.vs[v]["layer"] == hub_layer]
            if layer_cands:
                cv = random.choice(layer_cands)
                inc = g.incident(cv, mode="all")
                control_layer.append(sum(g.es[e]["weight"] for e in inc) / total_weight)

            # Control: random
            rand_cands = [v for v in range(g.vcount()) if v != hub and v not in hub_nodes]
            if rand_cands:
                cv = random.choice(rand_cands)
                inc = g.incident(cv, mode="all")
                control_random.append(sum(g.es[e]["weight"] for e in inc) / total_weight)

    # Bootstrap CIs
    def bootstrap_ci(vals: list[float], n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
        if not vals:
            return (0.0, 0.0)
        arr = np.array(vals)
        medians = []
        rng = np.random.RandomState(SEED)
        for _ in range(n_boot):
            sample = rng.choice(arr, size=len(arr), replace=True)
            medians.append(float(np.median(sample)))
        return (float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5)))

    hub_med = float(np.median(hub_impacts)) if hub_impacts else 0.0
    hub_ci = bootstrap_ci(hub_impacts)

    bar_chart = {"control_types": []}
    for name, vals in [("degree_matched", control_degree),
                       ("layer_matched", control_layer),
                       ("random", control_random)]:
        c_med = float(np.median(vals)) if vals else 0.0
        c_ci = bootstrap_ci(vals)
        ratio = hub_med / c_med if c_med > 1e-12 else 0.0
        bar_chart["control_types"].append({
            "name": name,
            "hub_median": hub_med,
            "control_median": c_med,
            "ratio": ratio,
            "ci_95_lo": c_ci[0],
            "ci_95_hi": c_ci[1],
            "hub_ci_95_lo": hub_ci[0],
            "hub_ci_95_hi": hub_ci[1],
            "n_comparisons": len(vals),
        })

    # Dose-response: bin by log(MPI+1)
    dose_response = {"bins": []}
    if mpi_impact_pairs:
        log_mpis = [math.log(m + 1) for m, _ in mpi_impact_pairs]
        impacts = [imp for _, imp in mpi_impact_pairs]
        n_bins = 8
        bin_edges = np.linspace(min(log_mpis), max(log_mpis) + 1e-9, n_bins + 1)
        for bi in range(n_bins):
            lo, hi = bin_edges[bi], bin_edges[bi + 1]
            in_bin = [impacts[k] for k in range(len(log_mpis))
                      if lo <= log_mpis[k] < hi]
            if in_bin:
                dose_response["bins"].append({
                    "mpi_bin_center": float((lo + hi) / 2),
                    "mean_impact": float(np.mean(in_bin)),
                    "std_impact": float(np.std(in_bin)),
                    "n_points": len(in_bin),
                })

    return {
        "bar_chart": bar_chart,
        "dose_response": dose_response,
        "n_graphs_used": len(graph_indices),
        "n_hub_total": len(hub_impacts),
    }


# ================================================================
# FEATURE EXPLANATION LOOKUP
# ================================================================

def load_feature_explanations() -> dict[str, str]:
    """Load (layer_feature) -> explanation lookup from data_id4_it2."""
    lookup: dict[str, str] = {}
    feat_file = FEAT_DIR / "full_data_out.json"
    if not feat_file.exists():
        logger.warning(f"Feature explanation file not found: {feat_file}")
        return lookup
    try:
        raw = json.loads(feat_file.read_text())
        for ex in raw["datasets"][0]["examples"]:
            try:
                out = json.loads(ex["output"])
                key = f"{out['layer_num']}_{out['feature_index']}"
                lookup[key] = out.get("explanation", "N/A")
            except (json.JSONDecodeError, KeyError):
                continue
        logger.info(f"Loaded {len(lookup)} feature explanations")
    except Exception:
        logger.exception("Failed to load feature explanations")
    return lookup


# ================================================================
# MOTIF DIAGRAM SPECIFICATIONS
# ================================================================

def build_motif_diagrams(
    dag_valid_ids: list[int], names: dict[int, str],
    mapping: dict, z_data: list[dict],
    graphs: list[igraph.Graph], valid_indices: list[int],
    feat_lookup: dict[str, str],
) -> list[dict]:
    """Build motif diagram specs for each 3-node DAG-valid type."""
    diagrams = []
    for cls_id in dag_valid_ids:
        name = names[cls_id]
        edges = mapping[cls_id]["edges"]
        adj = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for s, t in edges:
            adj[s][t] = 1

        # Canonical labels
        g_temp = igraph.Graph(n=3, edges=edges, directed=True)
        labels = []
        for v in range(3):
            role = []
            if g_temp.indegree(v) == 0:
                role.append("source")
            elif g_temp.outdegree(v) == 0:
                role.append("target")
            else:
                role.append("relay")
            labels.append(f"{chr(65 + v)} ({', '.join(role)})")

        # Collect Z-scores for this motif type
        zs = [zd["z_scores"].get(cls_id, 0.0) for zd in z_data if zd]
        mean_z = float(np.mean(zs)) if zs else 0.0

        # Universality
        domain_zs: dict[str, list[float]] = {}
        for i, zd in enumerate(z_data):
            if not zd or i >= len(valid_indices):
                continue
            # We need domain info - pass it separately
        universality = 0  # Computed later

        # Representative instance: graph with median Z for this motif
        rep = {"node_layers": [], "edge_weights": [], "node_explanations": []}
        if zs:
            sorted_zs = sorted(enumerate(zs), key=lambda x: abs(x[1] - np.median(zs)))
            if sorted_zs:
                best_idx = sorted_zs[0][0]
                if best_idx < len(valid_indices):
                    gi = valid_indices[best_idx]
                    g_rep = graphs[gi]
                    # Find first FFL instance in this graph
                    if g_rep.vcount() >= 3 and g_rep.ecount() >= 3:
                        for e in g_rep.es[:min(20, g_rep.ecount())]:
                            src_v = g_rep.vs[e.source]
                            tgt_v = g_rep.vs[e.target]
                            rep["node_layers"] = [
                                src_v["layer"], tgt_v["layer"]
                            ]
                            rep["edge_weights"] = [
                                e["weight"] if "weight" in e.attributes() else 0.0
                            ]
                            key_s = f"{src_v['layer']}_{src_v['feature']}"
                            key_t = f"{tgt_v['layer']}_{tgt_v['feature']}"
                            rep["node_explanations"] = [
                                feat_lookup.get(key_s, "explanation_not_available"),
                                feat_lookup.get(key_t, "explanation_not_available"),
                            ]
                            break

        diagrams.append({
            "name": name,
            "isoclass_id": cls_id,
            "adjacency_3x3": adj,
            "canonical_labels": labels,
            "mean_z_score": mean_z,
            "universality_score": universality,
            "representative": rep,
        })
    return diagrams


# ================================================================
# EMBEDDING COMPUTATION
# ================================================================

def compute_embeddings(
    feature_matrices: dict[str, np.ndarray],
    domains: list[str], slugs: list[str], correctness: list[str],
) -> tuple[list[dict], dict]:
    """Compute t-SNE and UMAP embeddings for multiple feature sets."""
    embeddings_list = []
    best_overall = {"method": "", "feature_set": "", "hyperparams": {}, "silhouette": -1.0}
    le = LabelEncoder()
    domain_encoded = le.fit_transform(domains)

    tsne_perplexities = [15, 30, 50]
    umap_neighbors = [10, 15, 30]

    for feat_name, X in feature_matrices.items():
        n = X.shape[0]
        # t-SNE
        for perp in tsne_perplexities:
            effective_perp = min(perp, n - 1)
            if effective_perp < 2:
                continue
            try:
                emb = TSNE(
                    n_components=2, perplexity=effective_perp,
                    random_state=SEED, max_iter=1000, init="pca",
                    learning_rate="auto",
                ).fit_transform(X)
                points = []
                for i in range(n):
                    points.append({
                        "x": float(emb[i, 0]),
                        "y": float(emb[i, 1]),
                        "domain": domains[i],
                        "slug": slugs[i],
                        "correctness": correctness[i],
                    })
                sil = -1.0
                if len(set(domains)) > 1:
                    try:
                        sil = float(silhouette_score(emb, domain_encoded))
                    except Exception:
                        pass
                rec = {
                    "method": "tsne",
                    "feature_set": feat_name,
                    "hyperparams": {"perplexity": effective_perp},
                    "points": points,
                    "silhouette_score": sil,
                    "is_recommended": False,
                }
                embeddings_list.append(rec)
                if sil > best_overall["silhouette"]:
                    best_overall = {
                        "method": "tsne", "feature_set": feat_name,
                        "hyperparams": {"perplexity": effective_perp},
                        "silhouette": sil,
                    }
            except Exception:
                logger.exception(f"t-SNE failed: {feat_name} perp={effective_perp}")

        # UMAP
        if HAS_UMAP:
            for nn in umap_neighbors:
                effective_nn = min(nn, n - 1)
                if effective_nn < 2:
                    continue
                try:
                    emb = umap.UMAP(
                        n_components=2, n_neighbors=effective_nn,
                        min_dist=0.1, random_state=SEED,
                        metric="euclidean",
                    ).fit_transform(X)
                    points = []
                    for i in range(n):
                        points.append({
                            "x": float(emb[i, 0]),
                            "y": float(emb[i, 1]),
                            "domain": domains[i],
                            "slug": slugs[i],
                            "correctness": correctness[i],
                        })
                    sil = -1.0
                    if len(set(domains)) > 1:
                        try:
                            sil = float(silhouette_score(emb, domain_encoded))
                        except Exception:
                            pass
                    rec = {
                        "method": "umap",
                        "feature_set": feat_name,
                        "hyperparams": {"n_neighbors": effective_nn},
                        "points": points,
                        "silhouette_score": sil,
                        "is_recommended": False,
                    }
                    embeddings_list.append(rec)
                    if sil > best_overall["silhouette"]:
                        best_overall = {
                            "method": "umap", "feature_set": feat_name,
                            "hyperparams": {"n_neighbors": effective_nn},
                            "silhouette": sil,
                        }
                except Exception:
                    logger.exception(f"UMAP failed: {feat_name} nn={effective_nn}")

    # Mark recommended
    for rec in embeddings_list:
        if (rec["method"] == best_overall["method"]
                and rec["feature_set"] == best_overall["feature_set"]
                and rec["hyperparams"] == best_overall.get("hyperparams", {})):
            rec["is_recommended"] = True

    return embeddings_list, best_overall


# ================================================================
# HEATMAP DATA
# ================================================================

def compute_heatmap_data(
    motif_ratios: np.ndarray, weighted_feats: np.ndarray,
    domain_labels: list[str], motif_names: list[str],
    weighted_feat_names: list[str],
) -> dict:
    """Compute domain-level heatmap matrices."""
    domains_sorted = sorted(set(domain_labels))
    n_domains = len(domains_sorted)

    # 8A: Count-ratio heatmap
    domain_mean_ratios = np.zeros((n_domains, motif_ratios.shape[1]))
    domain_std_ratios = np.zeros((n_domains, motif_ratios.shape[1]))
    for di, d in enumerate(domains_sorted):
        mask = [i for i, dl in enumerate(domain_labels) if dl == d]
        if mask:
            domain_mean_ratios[di] = np.mean(motif_ratios[mask], axis=0)
            domain_std_ratios[di] = np.std(motif_ratios[mask], axis=0)

    # Z-normalize columns for heatmap
    col_means = domain_mean_ratios.mean(axis=0)
    col_stds = domain_mean_ratios.std(axis=0)
    col_stds[col_stds == 0] = 1.0
    heatmap_z = (domain_mean_ratios - col_means) / col_stds

    # 8B: Domain similarity (cosine)
    sim_matrix = cosine_similarity(domain_mean_ratios)

    # 8C: Weighted feature heatmap
    domain_mean_wf = np.zeros((n_domains, weighted_feats.shape[1]))
    for di, d in enumerate(domains_sorted):
        mask = [i for i, dl in enumerate(domain_labels) if dl == d]
        if mask:
            domain_mean_wf[di] = np.mean(weighted_feats[mask], axis=0)

    wf_col_means = domain_mean_wf.mean(axis=0)
    wf_col_stds = domain_mean_wf.std(axis=0)
    wf_col_stds[wf_col_stds == 0] = 1.0
    wf_heatmap_z = (domain_mean_wf - wf_col_means) / wf_col_stds

    # 8D: Most discriminative features
    discriminative = {}
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            d1, d2 = domains_sorted[i], domains_sorted[j]
            mask1 = [k for k, dl in enumerate(domain_labels) if dl == d1]
            mask2 = [k for k, dl in enumerate(domain_labels) if dl == d2]
            if len(mask1) < 2 or len(mask2) < 2:
                continue
            t_stats = []
            all_feats = np.hstack([motif_ratios, weighted_feats])
            all_names = motif_names + weighted_feat_names
            for fi in range(all_feats.shape[1]):
                v1 = all_feats[mask1, fi]
                v2 = all_feats[mask2, fi]
                try:
                    t, p = scipy_stats.ttest_ind(v1, v2, equal_var=False)
                    t_stats.append((all_names[fi] if fi < len(all_names) else f"feat_{fi}",
                                    float(abs(t)), float(p)))
                except Exception:
                    pass
            t_stats.sort(key=lambda x: x[1], reverse=True)
            discriminative[f"{d1}_vs_{d2}"] = [
                {"feature": ts[0], "abs_t": ts[1], "p_value": ts[2]}
                for ts in t_stats[:3]
            ]

    return {
        "count_ratio_heatmap": {
            "matrix": heatmap_z.tolist(),
            "raw_matrix": domain_mean_ratios.tolist(),
            "std_matrix": domain_std_ratios.tolist(),
            "row_labels": domains_sorted,
            "col_labels": motif_names,
        },
        "weighted_feature_heatmap": {
            "matrix": wf_heatmap_z.tolist(),
            "row_labels": domains_sorted,
            "col_labels": weighted_feat_names,
        },
        "domain_similarity_matrix": {
            "matrix": sim_matrix.tolist(),
            "labels": domains_sorted,
        },
        "discriminative_features": discriminative,
    }


# ================================================================
# Z-SCORE BOX PLOT DATA
# ================================================================

def compute_zscore_boxplot(
    z_data: list[dict], domain_labels: list[str],
    dag_valid_ids: list[int], names: dict[int, str],
    valid_indices: list[int],
) -> dict:
    motif_types = {}
    for cls_id in dag_valid_ids:
        mname = names[cls_id]
        per_domain: dict[str, dict] = {}
        domains_sorted = sorted(set(domain_labels))
        for d in domains_sorted:
            zs = []
            for k, gi in enumerate(valid_indices):
                if domain_labels[gi] == d and k < len(z_data) and z_data[k]:
                    z_val = z_data[k]["z_scores"].get(cls_id, 0.0)
                    zs.append(z_val)
            if not zs:
                per_domain[d] = {
                    "median": 0.0, "q1": 0.0, "q3": 0.0,
                    "whisker_lo": 0.0, "whisker_hi": 0.0,
                    "min": 0.0, "max": 0.0, "outliers": [], "n": 0,
                }
                continue
            arr = np.array(zs)
            q1, med, q3 = float(np.percentile(arr, 25)), float(np.median(arr)), float(np.percentile(arr, 75))
            iqr = q3 - q1
            wlo = max(float(np.min(arr)), q1 - 1.5 * iqr)
            whi = min(float(np.max(arr)), q3 + 1.5 * iqr)
            outliers = [float(v) for v in arr if v < wlo or v > whi]
            per_domain[d] = {
                "median": med, "q1": q1, "q3": q3,
                "whisker_lo": wlo, "whisker_hi": whi,
                "min": float(np.min(arr)), "max": float(np.max(arr)),
                "outliers": outliers, "n": len(zs),
            }
        motif_types[mname] = {"per_domain": per_domain}
    return motif_types


# ================================================================
# CLUSTERING CONFUSION MATRICES
# ================================================================

def compute_confusion_matrices(
    feature_matrices: dict[str, np.ndarray],
    true_labels: np.ndarray, domain_names: list[str],
) -> dict:
    domains_sorted = sorted(set(domain_names))
    n_classes = len(domains_sorted)
    results = {}

    for feat_name, X in feature_matrices.items():
        try:
            sim = cosine_similarity(X)
            affinity = (sim + 1.0) / 2.0
            np.fill_diagonal(affinity, 1.0)
            affinity = np.clip(affinity, 0, None)

            pred = _safe_spectral_cluster(affinity, n_classes, SEED)
            nmi = float(normalized_mutual_info_score(true_labels, pred))
            ari = float(adjusted_rand_score(true_labels, pred))

            cm_aligned, aligned_order = build_confusion_with_hungarian(
                true_labels, pred, n_classes, n_classes)

            # Per-domain precision/recall
            precision, recall = [], []
            for di in range(n_classes):
                col_sum = cm_aligned[:, di].sum()
                row_sum = cm_aligned[di, :].sum()
                prec = float(cm_aligned[di, di] / col_sum) if col_sum > 0 else 0.0
                rec = float(cm_aligned[di, di] / row_sum) if row_sum > 0 else 0.0
                precision.append(prec)
                recall.append(rec)

            results[feat_name] = {
                "matrix": cm_aligned.tolist(),
                "domain_order": domains_sorted,
                "cluster_order": aligned_order,
                "nmi": nmi,
                "ari": ari,
                "precision": precision,
                "recall": recall,
            }
        except Exception:
            logger.exception(f"Confusion matrix failed for {feat_name}")
            results[feat_name] = {
                "matrix": [], "domain_order": domains_sorted,
                "cluster_order": [], "nmi": 0.0, "ari": 0.0,
                "precision": [], "recall": [],
            }
    return results


# ================================================================
# BASELINE: RANDOM CLUSTERING
# ================================================================

def random_clustering_baseline(
    n_samples: int, n_classes: int, true_labels: np.ndarray,
    domain_names: list[str],
) -> dict:
    """Baseline: random cluster assignment."""
    rng = np.random.RandomState(SEED)
    pred = rng.randint(0, n_classes, size=n_samples)
    nmi = float(normalized_mutual_info_score(true_labels, pred))
    ari = float(adjusted_rand_score(true_labels, pred))
    return {
        "method": "random_assignment",
        "nmi": nmi,
        "ari": ari,
        "pred_labels": pred.tolist(),
    }


# ================================================================
# MAIN PIPELINE
# ================================================================

@logger.catch
def main():
    t_start = time.time()
    n_workers = max(1, NUM_CPUS - 1)
    random.seed(SEED)
    np.random.seed(SEED)

    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
    logger.info(f"Workers: {n_workers}, UMAP: {HAS_UMAP}")
    logger.info(f"Config: MAX_EXAMPLES={MAX_EXAMPLES}, N_NULL={N_NULL_MODELS}, "
                f"ABLATION_GRAPHS={ABLATION_GRAPHS}")

    # ==============================================================
    # STEP 1: LOAD DATA
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 1: LOADING DATA")
    all_records = load_all_graphs(MAX_EXAMPLES or None)
    if not all_records:
        logger.error("No graphs loaded!")
        return
    n_total = len(all_records)
    domain_labels = [r["domain"] for r in all_records]
    slugs = [r["slug"] for r in all_records]
    correctness_list = [r["model_correct"] for r in all_records]
    domains_sorted = sorted(set(domain_labels))
    domain_counts = dict(Counter(domain_labels))
    correct_counts = dict(Counter(correctness_list))
    le = LabelEncoder()
    le.fit(domains_sorted)
    true_labels = le.transform(domain_labels)
    n_classes = len(domains_sorted)

    # ==============================================================
    # STEP 2: BUILD ISOCLASS MAPPING
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 2: BUILD ISOCLASS MAPPING")
    mapping, dag_valid_ids, names = build_isoclass_mapping()
    motif_names = [names[cid] for cid in dag_valid_ids]
    logger.info(f"DAG-valid 3-node types: {dag_valid_ids} -> {motif_names}")

    # ==============================================================
    # STEP 3: BUILD PRUNED GRAPHS
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 3: BUILD PRUNED GRAPHS")
    pruned_graphs: list[igraph.Graph | None] = [None] * n_total
    valid_indices: list[int] = []

    for i, rec in enumerate(all_records):
        try:
            g = build_igraph(rec, PRUNE_PERCENTILE)
            if g.vcount() >= MIN_NODES:
                pruned_graphs[i] = g
                valid_indices.append(i)
            else:
                logger.debug(f"Graph {i} too small: {g.vcount()} nodes")
        except Exception:
            logger.exception(f"Failed graph {i} ({rec['domain']})")
        # Free raw graph data
        rec["nodes"] = []
        rec["links"] = []

    gc.collect()
    logger.info(f"Valid graphs: {len(valid_indices)}/{n_total}")
    if valid_indices:
        nc = [pruned_graphs[i].vcount() for i in valid_indices]
        logger.info(f"  Nodes: min={min(nc)}, med={np.median(nc):.0f}, max={max(nc)}")

    # ==============================================================
    # STEP 4: MOTIF CENSUS + NULL MODEL Z-SCORES
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 4: MOTIF CENSUS + Z-SCORES")

    motif_count_ratios = np.zeros((n_total, len(dag_valid_ids)))
    real_census: dict[int, dict[int, int]] = {}
    z_data: list[dict] = [{}] * n_total
    all_ffls: list[list[dict]] = [[] for _ in range(n_total)]

    # 4A: Real census
    t_census_start = time.time()
    for gi in valid_indices:
        g = pruned_graphs[gi]
        try:
            census = compute_motif_census(g, dag_valid_ids)
            real_census[gi] = census
            total = sum(census[mid] for mid in dag_valid_ids)
            if total > 0:
                motif_count_ratios[gi] = [census[mid] / total for mid in dag_valid_ids]
        except Exception:
            logger.exception(f"Census failed graph {gi}")
    logger.info(f"Real census done in {time.time() - t_census_start:.1f}s")

    # 4B: Calibrate null model timing
    t_calib_start = time.time()
    calib_n = 5
    mid_idx = valid_indices[len(valid_indices) // 2]
    calib_nulls = generate_null_census(
        pruned_graphs[mid_idx], dag_valid_ids, calib_n, n_workers)
    t_calib = time.time() - t_calib_start
    time_per_null = t_calib / max(calib_n, 1)
    est_total = time_per_null * N_NULL_MODELS * len(valid_indices)
    logger.info(f"Calibration: {t_calib:.1f}s for {calib_n} nulls, "
                f"est total={est_total:.0f}s for {N_NULL_MODELS} nulls x {len(valid_indices)} graphs")

    actual_n_null = N_NULL_MODELS
    if est_total > MAX_NULL_TIME_S:
        actual_n_null = max(10, int(MAX_NULL_TIME_S / (time_per_null * len(valid_indices))))
        logger.warning(f"Reducing N_NULL from {N_NULL_MODELS} to {actual_n_null}")

    # 4C: Generate null models + Z-scores in batches
    BATCH_SIZE = 20
    t_null_start = time.time()
    for batch_start in range(0, len(valid_indices), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(valid_indices))
        batch_idxs = valid_indices[batch_start:batch_end]

        elapsed = time.time() - t_null_start
        if elapsed > MAX_NULL_TIME_S and batch_start > 0:
            logger.warning(f"Null model time budget exceeded at batch {batch_start}, stopping")
            break

        for k, gi in enumerate(batch_idxs):
            g = pruned_graphs[gi]
            try:
                null_list = generate_null_census(g, dag_valid_ids, actual_n_null, n_workers)
                z_result = compute_zscores(real_census[gi], null_list, dag_valid_ids)
                z_data[gi] = z_result
                del null_list
            except Exception:
                logger.exception(f"Null model failed graph {gi}")

        gc.collect()
        elapsed_now = time.time() - t_null_start
        pct = (batch_end / len(valid_indices)) * 100
        logger.info(f"  Null models: {batch_end}/{len(valid_indices)} ({pct:.0f}%) "
                     f"in {elapsed_now:.0f}s")

    logger.info(f"Z-scores completed in {time.time() - t_null_start:.1f}s")

    # ==============================================================
    # STEP 5: WEIGHTED MOTIF FEATURES (FFL enumeration)
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 5: WEIGHTED MOTIF FEATURES")
    t_ffl = time.time()
    weighted_features = np.zeros((n_total, 8 + len(dag_valid_ids)))

    for gi in valid_indices:
        g = pruned_graphs[gi]
        try:
            ffls = enumerate_ffls(g)
            all_ffls[gi] = ffls
            weighted_features[gi] = compute_weighted_features(ffls, motif_count_ratios[gi])
        except Exception:
            logger.exception(f"FFL failed graph {gi}")
            weighted_features[gi] = compute_weighted_features([], motif_count_ratios[gi])

    logger.info(f"FFL enumeration done in {time.time() - t_ffl:.1f}s")
    ffl_counts = [len(all_ffls[gi]) for gi in valid_indices]
    logger.info(f"  FFLs per graph: min={min(ffl_counts)}, med={np.median(ffl_counts):.0f}, "
                f"max={max(ffl_counts)}")

    # ==============================================================
    # STEP 6: GRAPH STATISTICS (10D)
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 6: GRAPH STATISTICS")
    graph_stats = np.zeros((n_total, 10))
    for gi in valid_indices:
        try:
            graph_stats[gi] = compute_graph_stats(pruned_graphs[gi])
        except Exception:
            logger.exception(f"Stats failed graph {gi}")
    logger.info("Graph statistics computed")

    # ==============================================================
    # STEP 7: FEATURE MATRIX CONSTRUCTION
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 7: FEATURE MATRICES")

    # Use only valid indices for all downstream computation
    vi = np.array(valid_indices)
    motif_only = motif_count_ratios[vi]
    graph_only = graph_stats[vi]
    weighted_only = weighted_features[vi]
    all_combined = np.hstack([motif_only, graph_only, weighted_only])

    # Standardize
    scaler_m = StandardScaler().fit(motif_only)
    scaler_g = StandardScaler().fit(graph_only)
    scaler_w = StandardScaler().fit(weighted_only)
    scaler_a = StandardScaler().fit(all_combined)

    motif_scaled = scaler_m.transform(motif_only)
    graph_scaled = scaler_g.transform(graph_only)
    weighted_scaled = scaler_w.transform(weighted_only)
    all_scaled = scaler_a.transform(all_combined)

    # Replace any NaN with 0
    for arr in [motif_scaled, graph_scaled, weighted_scaled, all_scaled]:
        arr[np.isnan(arr)] = 0.0

    feature_matrices = {
        "motif_only": motif_scaled,
        "graph_stats_only": graph_scaled,
        "weighted_motif_only": weighted_scaled,
        "all_combined": all_scaled,
    }

    feat_name_map = {
        "motif_only": motif_names,
        "graph_stats_only": ["n_nodes", "n_edges", "density", "mean_in_deg",
                             "mean_out_deg", "max_out_deg", "n_layers", "diameter",
                             "mean_weight", "std_weight"],
        "weighted_motif_only": ["ffl_intensity_mean", "ffl_intensity_median",
                                "ffl_intensity_std", "ffl_coherence_frac",
                                "ffl_path_dominance_mean", "ffl_weight_asymmetry_mean",
                                "ffl_count", "ffl_density"] + motif_names,
        "all_combined": [],
    }
    feat_name_map["all_combined"] = (feat_name_map["motif_only"]
                                     + feat_name_map["graph_stats_only"]
                                     + feat_name_map["weighted_motif_only"])

    logger.info(f"Feature shapes: motif={motif_scaled.shape}, graph={graph_scaled.shape}, "
                f"weighted={weighted_scaled.shape}, combined={all_scaled.shape}")

    # Filter domain labels/slugs/correctness to valid indices only
    valid_domains = [domain_labels[i] for i in valid_indices]
    valid_slugs = [slugs[i] for i in valid_indices]
    valid_correct = [correctness_list[i] for i in valid_indices]
    valid_true_labels = le.transform(valid_domains)

    # ==============================================================
    # STEP 8: EMBEDDINGS (t-SNE + UMAP)
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 8: EMBEDDINGS")
    t_emb = time.time()
    embeddings_list, best_emb = compute_embeddings(
        feature_matrices, valid_domains, valid_slugs, valid_correct)
    logger.info(f"Embeddings: {len(embeddings_list)} computed in {time.time() - t_emb:.1f}s")
    logger.info(f"  Best: {best_emb['method']} / {best_emb['feature_set']} / "
                f"sil={best_emb['silhouette']:.3f}")

    # ==============================================================
    # STEP 9: HEATMAP DATA
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 9: HEATMAP DATA")
    heatmap_data = compute_heatmap_data(
        motif_only, weighted_only, valid_domains, motif_names,
        feat_name_map["weighted_motif_only"])
    logger.info("Heatmap data computed")

    # ==============================================================
    # STEP 10: Z-SCORE BOX PLOTS
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 10: Z-SCORE BOX PLOTS")
    zscore_valid = [z_data[gi] for gi in valid_indices]
    zscore_boxplot = compute_zscore_boxplot(
        zscore_valid, valid_domains, dag_valid_ids, names, list(range(len(valid_indices))))
    logger.info("Z-score box plot data computed")

    # ==============================================================
    # STEP 11: ABLATION
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 11: ABLATION")
    t_abl = time.time()

    # Stratified subset
    n_per_domain = max(2, ABLATION_GRAPHS // n_classes)
    ablation_indices = []
    for d in domains_sorted:
        d_indices = [gi for gi in valid_indices if domain_labels[gi] == d]
        ablation_indices.extend(d_indices[:n_per_domain])

    ablation_result = compute_ablation_data(
        pruned_graphs, ablation_indices, all_ffls, domain_labels)
    logger.info(f"Ablation done in {time.time() - t_abl:.1f}s, "
                f"used {ablation_result['n_graphs_used']} graphs")

    # ==============================================================
    # STEP 12: CONFUSION MATRICES
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 12: CONFUSION MATRICES")
    confusion_results = compute_confusion_matrices(
        feature_matrices, valid_true_labels, valid_domains)

    best_nmi_feat = ""
    best_nmi_val = -1.0
    for fn, cr in confusion_results.items():
        logger.info(f"  {fn}: NMI={cr['nmi']:.3f}, ARI={cr['ari']:.3f}")
        if cr["nmi"] > best_nmi_val:
            best_nmi_val = cr["nmi"]
            best_nmi_feat = fn

    # Baseline: random clustering
    baseline = random_clustering_baseline(
        len(valid_indices), n_classes, valid_true_labels, valid_domains)
    logger.info(f"  Baseline random: NMI={baseline['nmi']:.3f}, ARI={baseline['ari']:.3f}")

    # ==============================================================
    # STEP 13: MOTIF DIAGRAMS + FEATURE EXPLANATIONS
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 13: MOTIF DIAGRAMS")
    feat_lookup = load_feature_explanations()
    diagrams = build_motif_diagrams(
        dag_valid_ids, names, mapping, zscore_valid,
        pruned_graphs, valid_indices, feat_lookup)

    # Compute universality scores now that we have domain info
    for diag in diagrams:
        cls_id = diag["isoclass_id"]
        domain_mean_zs: dict[str, float] = {}
        for k, gi in enumerate(valid_indices):
            d = domain_labels[gi]
            if k < len(zscore_valid) and zscore_valid[k]:
                z = zscore_valid[k]["z_scores"].get(cls_id, 0.0)
                domain_mean_zs.setdefault(d, [])
                domain_mean_zs[d].append(z)
        univ = sum(1 for d, zs in domain_mean_zs.items()
                   if np.mean(zs) > 2.0) if domain_mean_zs else 0
        diag["universality_score"] = univ

    logger.info(f"Diagrams built for {len(diagrams)} motif types")
    for diag in diagrams:
        logger.info(f"  {diag['name']}: mean_z={diag['mean_z_score']:.2f}, "
                     f"universality={diag['universality_score']}/{n_classes}")

    # ==============================================================
    # STEP 14: ASSEMBLE OUTPUT (exp_gen_sol_out schema)
    # ==============================================================
    logger.info("=" * 60)
    logger.info("STEP 14: ASSEMBLE OUTPUT")
    total_time = time.time() - t_start

    # Compute FFL universality / mean Z
    ffl_cls_id = [cid for cid in dag_valid_ids if names[cid] == "030T"]
    ffl_zscores = []
    if ffl_cls_id:
        ffl_id = ffl_cls_id[0]
        for k, gi in enumerate(valid_indices):
            if k < len(zscore_valid) and zscore_valid[k]:
                ffl_zscores.append(zscore_valid[k]["z_scores"].get(ffl_id, 0.0))
    mean_ffl_z = float(np.mean(ffl_zscores)) if ffl_zscores else 0.0

    # Get predicted domain labels from best clustering for predict_our_method
    best_confusion = confusion_results.get(best_nmi_feat, {})
    best_pred = None
    if best_confusion.get("matrix"):
        # Re-run clustering for predictions
        X_best = feature_matrices[best_nmi_feat]
        sim = cosine_similarity(X_best)
        affinity = np.clip((sim + 1.0) / 2.0, 0, None)
        np.fill_diagonal(affinity, 1.0)
        try:
            pred_labels = _safe_spectral_cluster(affinity, n_classes, SEED)
            # Map to domain names using majority vote
            cluster_to_domain: dict[int, str] = {}
            for cl_id in set(pred_labels):
                indices = [i for i, p in enumerate(pred_labels) if p == cl_id]
                doms = [valid_domains[i] for i in indices]
                cluster_to_domain[cl_id] = Counter(doms).most_common(1)[0][0]
            best_pred = [cluster_to_domain[p] for p in pred_labels]
        except Exception:
            logger.exception("Failed to produce final predictions")

    # Build per-example records in exp_gen_sol_out schema
    examples = []
    for k, gi in enumerate(valid_indices):
        rec = all_records[gi]

        # Compact graph summary (NOT full raw JSON — that's 5MB+ per graph)
        g_now = pruned_graphs[gi]
        output_summary = json.dumps({
            "graph_summary": {
                "n_nodes_raw": rec.get("n_nodes_raw", 0),
                "n_edges_raw": rec.get("n_edges_raw", 0),
                "n_nodes_pruned": g_now.vcount() if g_now else 0,
                "n_edges_pruned": g_now.ecount() if g_now else 0,
                "domain": rec["domain"],
                "slug": rec["slug"],
                "source": "data_id5_it3__opus",
            }
        })

        # Compact per-graph analysis
        named_ratios = {names[cid]: round(float(motif_count_ratios[gi][j]), 6)
                        for j, cid in enumerate(dag_valid_ids)}
        named_z = {}
        zd = z_data[gi]
        if zd and "z_scores" in zd:
            for cid in dag_valid_ids:
                named_z[names[cid]] = round(zd["z_scores"].get(cid, 0.0), 3)

        our_prediction = best_pred[k] if best_pred and k < len(best_pred) else "unknown"
        baseline_prediction = domains_sorted[baseline["pred_labels"][k]] if k < len(baseline["pred_labels"]) else "unknown"

        examples.append({
            "input": rec["prompt"],
            "output": output_summary,
            "predict_our_method": json.dumps({
                "predicted_domain": our_prediction,
                "motif_ratios": named_ratios,
                "z_scores": named_z,
                "n_ffls": len(all_ffls[gi]),
                "ffl_coherence": round(float(weighted_features[gi][3]), 4) if weighted_features[gi][3] else 0.0,
            }),
            "predict_baseline": json.dumps({
                "predicted_domain": baseline_prediction,
                "method": "random_assignment",
            }),
            "metadata_fold": rec["domain"],
            "metadata_slug": rec["slug"],
            "metadata_model_correct": rec["model_correct"],
            "metadata_difficulty": rec["difficulty"],
            "metadata_n_nodes_pruned": int(g_now.vcount()) if g_now else 0,
            "metadata_n_edges_pruned": int(g_now.ecount()) if g_now else 0,
            "metadata_n_ffls": len(all_ffls[gi]),
        })

    # Figures metadata (top-level)
    figures_data = {
        "fig_tsne_umap_embeddings": {
            "description": "2D embedding coordinates from t-SNE and UMAP across multiple feature sets and hyperparameters",
            "embeddings": sanitize_for_json(embeddings_list),
            "recommended_default": sanitize_for_json(best_emb),
            "n_embeddings": len(embeddings_list),
        },
        "fig_motif_heatmap": {
            "description": "Domain-level motif spectrum and weighted feature heatmaps with domain similarity",
            **sanitize_for_json(heatmap_data),
        },
        "fig_zscore_boxplot": {
            "description": "Per-domain Z-score distributions for each motif type (box plot data)",
            "motif_types": sanitize_for_json(zscore_boxplot),
        },
        "fig_ablation_comparison": {
            "description": "Ablation impact comparison: FFL hub nodes vs matched controls with bootstrap CIs",
            **sanitize_for_json(ablation_result),
        },
        "fig_confusion_matrices": {
            "description": "Clustering confusion matrices with Hungarian alignment for 4 feature sets",
            "feature_sets": sanitize_for_json(confusion_results),
            "baseline": sanitize_for_json(baseline),
        },
        "fig_motif_diagrams": {
            "description": "Network motif diagram specifications with representative real instances",
            "motifs_3node": sanitize_for_json(diagrams),
        },
    }

    summary_stats = {
        "n_graphs_processed": n_total,
        "n_valid_after_pruning": len(valid_indices),
        "domain_counts": domain_counts,
        "correctness_counts": correct_counts,
        "best_clustering_nmi": best_nmi_val,
        "best_clustering_feature_set": best_nmi_feat,
        "baseline_nmi": baseline["nmi"],
        "ffl_universality_score": diagrams[-1]["universality_score"] if diagrams else 0,
        "mean_ffl_zscore": mean_ffl_z,
        "actual_n_null_models": actual_n_null,
        "compute_time_s": total_time,
        "feature_dimensions": {fn: X.shape[1] for fn, X in feature_matrices.items()},
    }

    output = {
        "metadata": {
            "method_name": "motif_viz_pipeline",
            "description": "Paper-ready visualization data for 200 attribution graphs",
            "run_date": "2026-03-19",
            "prune_percentile": PRUNE_PERCENTILE,
            "n_null_models": actual_n_null,
            "seed": SEED,
            "figures": sanitize_for_json(figures_data),
            "summary_statistics": sanitize_for_json(summary_stats),
        },
        "datasets": [{
            "dataset": "neuronpedia_attribution_graphs_viz",
            "examples": examples,
        }],
    }

    output = sanitize_for_json(output)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    fsize = OUTPUT_FILE.stat().st_size / 1e6
    logger.info(f"Output written: {OUTPUT_FILE} ({fsize:.1f} MB)")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Best clustering: {best_nmi_feat} NMI={best_nmi_val:.3f}")
    logger.info(f"Baseline NMI: {baseline['nmi']:.3f}")
    logger.info(f"Mean FFL Z-score: {mean_ffl_z:.2f}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
