#!/usr/bin/env python3
"""Motif vs. Graph-Stats Unique Information Decomposition.

Determines whether motif-level features carry unique structural information
about capability domains beyond what graph-level statistics already capture.
Builds on iter_4 results (weighted_motif NMI=0.705, graph_stats NMI=0.677,
combined NMI=0.844) and asks: does the NMI=0.844 combined score reflect
genuinely orthogonal information from motifs, or confounded redundancy?

Phases:
  A: Load graphs & compute two feature blocks (motif 16D, graph-stats 10D)
  B: Variance decomposition (McFadden pseudo-R² with bootstrap CIs)
  C: Residualized clustering with permutation tests
  D: Domain-normalized weighted intensity
  E: Conditional mutual information
  F: Canonical correlation analysis
  G: Output (method_out.json)
"""

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import igraph
import numpy as np
import scipy.stats
from loguru import logger
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress convergence warnings during bootstrap
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONSTANTS & PATHS
# ============================================================
WORKSPACE = Path(__file__).parent
DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability"
    "/3_invention_loop/iter_3/gen_art/data_id5_it3__opus/data_out"
)
OUTPUT_FILE = WORKSPACE / "method_out.json"

# Configurable via env vars for gradual scaling
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "1000"))
N_PERMUTATIONS = int(os.environ.get("N_PERMUTATIONS", "1000"))
PRUNE_PERCENTILE = 75
MIN_NODES = 30
SEED = 42
CLUSTER_K_VALUES = [2, 4, 6, 8]

# ============================================================
# LOGGING
# ============================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WORKSPACE / "logs" / "run.log", rotation="30 MB", level="DEBUG")

# ============================================================
# HARDWARE DETECTION (cgroup-aware)
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

# Set memory limits: use ~75% of container RAM
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.75 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")
logger.info(f"Config: MAX_EXAMPLES={MAX_EXAMPLES}, N_BOOTSTRAP={N_BOOTSTRAP}, "
            f"N_PERMUTATIONS={N_PERMUTATIONS}")

# ============================================================
# PHASE A: LOAD GRAPHS & COMPUTE TWO FEATURE BLOCKS
# ============================================================


def parse_layer(layer_str: str) -> int:
    """Parse layer string to integer."""
    try:
        return int(layer_str)
    except (ValueError, TypeError):
        return 0


def load_all_graphs_with_weights() -> list[dict]:
    """Load from data_id5_it3, 12 split files, preserve signed weights.

    Always loads ALL files first for balanced domains, then subsamples
    if MAX_EXAMPLES is set (stratified by domain).
    """
    all_graphs = []
    split_files = sorted(DATA_DIR.glob("full_data_out_*.json"))
    logger.info(f"Found {len(split_files)} split files in {DATA_DIR}")

    for fpath in split_files:
        logger.debug(f"Loading {fpath.name}...")
        raw = json.loads(fpath.read_text())
        examples = raw["datasets"][0]["examples"]

        for ex in examples:
            try:
                graph_json = json.loads(ex["output"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping example with invalid JSON output")
                continue

            nodes = graph_json["nodes"]
            links = graph_json["links"]

            # Build node_id -> idx mapping
            node_id_to_idx = {n["node_id"]: i for i, n in enumerate(nodes)}

            # Collect edge weights (signed) and compute |w| threshold
            all_abs_w = [abs(lk.get("weight", 0.0)) for lk in links]
            if not all_abs_w:
                continue
            threshold = np.percentile(all_abs_w, PRUNE_PERCENTILE)

            # Build edge list preserving BOTH abs and signed weights
            edges, abs_weights, signed_weights = [], [], []
            signed_weight_dict = {}

            for lk in links:
                raw_w = lk.get("weight", 0.0)
                if abs(raw_w) >= threshold:
                    src = node_id_to_idx.get(lk["source"])
                    tgt = node_id_to_idx.get(lk["target"])
                    if src is not None and tgt is not None and src != tgt:
                        edges.append((src, tgt))
                        abs_weights.append(abs(raw_w))
                        signed_weights.append(float(raw_w))
                        signed_weight_dict[(lk["source"], lk["target"])] = float(raw_w)

            if not edges:
                continue

            # Build igraph with both weight attributes
            g = igraph.Graph(n=len(nodes), edges=edges, directed=True)
            g.vs["node_id"] = [n["node_id"] for n in nodes]
            g.vs["layer"] = [parse_layer(n.get("layer", "0")) for n in nodes]
            g.vs["feature_type"] = [n.get("feature_type", "") for n in nodes]
            g.es["weight"] = abs_weights
            g.es["signed_weight"] = signed_weights

            # Store signed weights before simplify
            edge_signed_before = {}
            for e in g.es:
                key = (e.source, e.target)
                if key not in edge_signed_before or abs(e["signed_weight"]) > abs(edge_signed_before[key]):
                    edge_signed_before[key] = e["signed_weight"]

            g.simplify(multiple=True, loops=True, combine_edges={"weight": "max"})

            # Restore signed weights after simplify
            for e in g.es:
                src_nid = g.vs[e.source]["node_id"]
                tgt_nid = g.vs[e.target]["node_id"]
                e["signed_weight"] = signed_weight_dict.get(
                    (src_nid, tgt_nid),
                    edge_signed_before.get((e.source, e.target), e["weight"])
                )

            # Remove isolated vertices
            isolated = [v.index for v in g.vs if g.degree(v) == 0]
            if isolated:
                g.delete_vertices(isolated)

            if g.vcount() < MIN_NODES or not g.is_dag():
                continue

            all_graphs.append({
                "graph": g,
                "domain": ex["metadata_fold"],
                "slug": ex.get("metadata_slug", ""),
                "model_correct": ex.get("metadata_model_correct", "unknown"),
                "prompt": ex["input"],
            })

        del raw
        gc.collect()

    logger.info(f"Loaded {len(all_graphs)} valid graphs (before subsampling)")

    # Subsample with domain stratification if MAX_EXAMPLES is set
    if MAX_EXAMPLES > 0 and len(all_graphs) > MAX_EXAMPLES:
        rng = np.random.RandomState(SEED)
        domain_groups: dict[str, list[int]] = {}
        for i, gr in enumerate(all_graphs):
            domain_groups.setdefault(gr["domain"], []).append(i)

        n_domains = len(domain_groups)
        per_domain = max(1, MAX_EXAMPLES // n_domains)
        selected_indices = []
        for domain, indices in sorted(domain_groups.items()):
            n_take = min(per_domain, len(indices))
            chosen = rng.choice(indices, n_take, replace=False).tolist()
            selected_indices.extend(chosen)

        remaining = MAX_EXAMPLES - len(selected_indices)
        if remaining > 0:
            leftover = [i for i in range(len(all_graphs)) if i not in set(selected_indices)]
            extra = rng.choice(leftover, min(remaining, len(leftover)), replace=False).tolist()
            selected_indices.extend(extra)

        selected_indices = sorted(selected_indices[:MAX_EXAMPLES])
        all_graphs = [all_graphs[i] for i in selected_indices]
        logger.info(f"Subsampled to {len(all_graphs)} graphs (stratified by domain)")

    logger.info(f"Final: {len(all_graphs)} graphs")
    return all_graphs


# ============================================================
# STEP A2: MOTIF ENUMERATION & FEATURE COMPUTATION (reuse from iter_4)
# ============================================================


def build_adjacency_and_weight_lookups(g: igraph.Graph) -> tuple[set, dict]:
    """Build O(1) edge-existence set and weight lookup dict."""
    adj_set = set()
    weight_lookup = {}
    for e in g.es:
        adj_set.add((e.source, e.target))
        weight_lookup[(e.source, e.target)] = {
            "abs": float(e["weight"]),
            "signed": float(e["signed_weight"]),
        }
    return adj_set, weight_lookup


def enumerate_ffls(g: igraph.Graph, adj_set: set, weight_lookup: dict) -> list[dict]:
    """Enumerate FFL (030T) instances: A->B, A->C, B->C."""
    ffls = []
    for a in range(g.vcount()):
        succs = g.successors(a)
        if len(succs) < 2:
            continue
        for i, b in enumerate(succs):
            for c in succs[i + 1:]:
                if (b, c) in adj_set:
                    ffls.append({
                        "a": a, "b": b, "c": c,
                        "w_ab": weight_lookup[(a, b)],
                        "w_ac": weight_lookup[(a, c)],
                        "w_bc": weight_lookup[(b, c)],
                    })
                if (c, b) in adj_set:
                    ffls.append({
                        "a": a, "b": c, "c": b,
                        "w_ab": weight_lookup[(a, c)],
                        "w_ac": weight_lookup[(a, b)],
                        "w_bc": weight_lookup[(c, b)],
                    })
    return ffls


def enumerate_chains(g: igraph.Graph, adj_set: set, weight_lookup: dict) -> list[dict]:
    """Enumerate 021C (chain) instances: A->B, B->C (no A->C or C->A edge)."""
    chains = []
    for b in range(g.vcount()):
        preds = g.predecessors(b)
        succs = g.successors(b)
        for a in preds:
            for c in succs:
                if a != c and (a, c) not in adj_set and (c, a) not in adj_set:
                    chains.append({
                        "a": a, "b": b, "c": c,
                        "w_ab": weight_lookup[(a, b)],
                        "w_bc": weight_lookup[(b, c)],
                    })
    return chains


def enumerate_fanouts(g: igraph.Graph, adj_set: set, weight_lookup: dict) -> list[dict]:
    """Enumerate 021D (fan-out): A->B, A->C (no B<->C edge)."""
    fanouts = []
    for a in range(g.vcount()):
        succs = g.successors(a)
        if len(succs) < 2:
            continue
        for i, b in enumerate(succs):
            for c in succs[i + 1:]:
                if (b, c) not in adj_set and (c, b) not in adj_set:
                    fanouts.append({
                        "a": a, "b": b, "c": c,
                        "w_ab": weight_lookup[(a, b)],
                        "w_ac": weight_lookup[(a, c)],
                    })
    return fanouts


def enumerate_fanins(g: igraph.Graph, adj_set: set, weight_lookup: dict) -> list[dict]:
    """Enumerate 021U (fan-in): B->A, C->A (no B<->C edge)."""
    fanins = []
    for a in range(g.vcount()):
        preds = g.predecessors(a)
        if len(preds) < 2:
            continue
        for i, b in enumerate(preds):
            for c in preds[i + 1:]:
                if (b, c) not in adj_set and (c, b) not in adj_set:
                    fanins.append({
                        "a": a, "b": b, "c": c,
                        "w_ba": weight_lookup[(b, a)],
                        "w_ca": weight_lookup[(c, a)],
                    })
    return fanins


def compute_ffl_weighted_features(ffl: dict) -> dict:
    """For one FFL instance, compute weighted features."""
    w_ab_abs = ffl["w_ab"]["abs"]
    w_ac_abs = ffl["w_ac"]["abs"]
    w_bc_abs = ffl["w_bc"]["abs"]
    w_ab_s = ffl["w_ab"]["signed"]
    w_ac_s = ffl["w_ac"]["signed"]
    w_bc_s = ffl["w_bc"]["signed"]

    intensity = (w_ab_abs * w_ac_abs * w_bc_abs) ** (1.0 / 3.0)
    indirect = w_ab_abs * w_bc_abs
    path_dominance = w_ac_abs / indirect if indirect > 1e-12 else 100.0
    sign_product = np.sign(w_ab_s) * np.sign(w_ac_s) * np.sign(w_bc_s)
    is_coherent = sign_product > 0
    denom = w_ab_abs + w_bc_abs
    weight_asymmetry = abs(w_ab_abs - w_bc_abs) / denom if denom > 1e-12 else 0.0
    arith_mean = (w_ab_abs + w_ac_abs + w_bc_abs) / 3.0
    coherence_onnela = intensity / arith_mean if arith_mean > 1e-12 else 0.0

    return {
        "intensity": intensity,
        "path_dominance": path_dominance,
        "is_coherent": is_coherent,
        "weight_asymmetry": weight_asymmetry,
        "coherence_onnela": coherence_onnela,
    }


def compute_binary_motif_ratios(g: igraph.Graph) -> dict:
    """Compute ratios of 4 DAG-valid 3-node motif types using igraph triad_census."""
    tc = g.triad_census()
    count_021D = tc[6]
    count_021U = tc[5]
    count_021C = tc[7]
    count_030T = tc[9]

    total = count_021D + count_021U + count_021C + count_030T
    if total == 0:
        total = 1

    return {
        "ratio_021U": count_021U / total,
        "ratio_021C": count_021C / total,
        "ratio_021D": count_021D / total,
        "ratio_030T": count_030T / total,
    }


def process_one_graph(graph_record: dict, graph_idx: int) -> dict:
    """Process one graph: enumerate motifs, compute both feature blocks.

    Returns dict with 'motif_features' (16D) and 'graph_stat_features' (10D).
    """
    g = graph_record["graph"]
    t0 = time.time()

    adj_set, wl = build_adjacency_and_weight_lookups(g)

    # --- Enumerate all 3-node motif types ---
    ffls = enumerate_ffls(g, adj_set, wl)
    chains = enumerate_chains(g, adj_set, wl)
    fanouts = enumerate_fanouts(g, adj_set, wl)
    fanins = enumerate_fanins(g, adj_set, wl)

    # --- FFL weighted features (streaming aggregation) ---
    ffl_intensities = []
    ffl_path_doms = []
    ffl_asymmetries = []
    ffl_coherences_onnela = []
    n_coherent = 0
    n_ffls = len(ffls)

    for ffl in ffls:
        feats = compute_ffl_weighted_features(ffl)
        ffl_intensities.append(feats["intensity"])
        pd_val = feats["path_dominance"]
        if pd_val < 100:
            ffl_path_doms.append(pd_val)
        ffl_asymmetries.append(feats["weight_asymmetry"])
        ffl_coherences_onnela.append(feats["coherence_onnela"])
        if feats["is_coherent"]:
            n_coherent += 1

    fi_arr = np.array(ffl_intensities) if ffl_intensities else np.array([0.0])
    fpd_arr = np.array(ffl_path_doms) if ffl_path_doms else np.array([0.0])
    fa_arr = np.array(ffl_asymmetries) if ffl_asymmetries else np.array([0.0])
    fco_arr = np.array(ffl_coherences_onnela) if ffl_coherences_onnela else np.array([0.0])

    # Chain features
    chain_intensities = []
    chain_sign_agrees = []
    for c in chains:
        w_ab = c["w_ab"]["abs"]
        w_bc = c["w_bc"]["abs"]
        chain_intensities.append((w_ab * w_bc) ** 0.5)
        chain_sign_agrees.append(int(np.sign(c["w_ab"]["signed"]) == np.sign(c["w_bc"]["signed"])))

    # Fan-out features
    fanout_intensities = []
    for fo in fanouts:
        w_ab = fo["w_ab"]["abs"]
        w_ac = fo["w_ac"]["abs"]
        fanout_intensities.append((w_ab * w_ac) ** 0.5)

    # Fan-in features
    fanin_intensities = []
    for fi_inst in fanins:
        w_ba = fi_inst["w_ba"]["abs"]
        w_ca = fi_inst["w_ca"]["abs"]
        fanin_intensities.append((w_ba * w_ca) ** 0.5)

    # --- Binary motif ratios via triad census ---
    ratios = compute_binary_motif_ratios(g)

    # --- MOTIF FEATURE BLOCK (16D) ---
    motif_features = {
        # Binary motif ratios (4D)
        "ratio_021U": ratios["ratio_021U"],
        "ratio_021C": ratios["ratio_021C"],
        "ratio_021D": ratios["ratio_021D"],
        "ratio_030T": ratios["ratio_030T"],
        # FFL weighted features (8D)
        "ffl_intensity_mean": float(np.mean(fi_arr)),
        "ffl_intensity_median": float(np.median(fi_arr)),
        "ffl_intensity_std": float(np.std(fi_arr)),
        "ffl_coherent_frac": n_coherent / max(n_ffls, 1),
        "ffl_path_dom_mean": float(np.mean(fpd_arr)),
        "ffl_asymmetry_mean": float(np.mean(fa_arr)),
        "ffl_coherence_onnela_mean": float(np.mean(fco_arr)),
        "ffl_coherence_onnela_std": float(np.std(fco_arr)),
        # Other motif weighted features (4D)
        "chain_intensity_mean": float(np.mean(chain_intensities)) if chain_intensities else 0.0,
        "chain_sign_agree_frac": float(np.mean(chain_sign_agrees)) if chain_sign_agrees else 0.0,
        "fanout_intensity_mean": float(np.mean(fanout_intensities)) if fanout_intensities else 0.0,
        "fanin_intensity_mean": float(np.mean(fanin_intensities)) if fanin_intensities else 0.0,
    }

    # --- GRAPH-STATS BLOCK (10D) ---
    n_nodes = g.vcount()
    n_edges = g.ecount()
    density = 2.0 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    in_degrees = g.indegree()
    out_degrees = g.outdegree()
    all_degrees = g.degree()
    layers = g.vs["layer"]
    edge_weights = np.array(g.es["weight"])

    # Approximate diameter
    try:
        if n_nodes < 500:
            diameter_val = g.diameter(directed=True)
        else:
            sampled = np.random.choice(n_nodes, min(10, n_nodes), replace=False)
            max_d = 0
            for s in sampled:
                dists = g.distances(source=int(s), mode="out")[0]
                valid = [d for d in dists if d < float('inf')]
                if valid:
                    max_d = max(max_d, max(valid))
            diameter_val = max_d
    except Exception:
        diameter_val = 0

    graph_stat_features = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "mean_in_deg": float(np.mean(in_degrees)),
        "mean_out_deg": float(np.mean(out_degrees)),
        "max_degree": max(all_degrees) if all_degrees else 0,
        "diameter": diameter_val,
        "n_layers": len(set(layers)),
        "mean_abs_ew": float(np.mean(edge_weights)),
        "edge_weight_std": float(np.std(edge_weights)),
    }

    elapsed = time.time() - t0
    if graph_idx % 10 == 0:
        logger.info(
            f"  Graph {graph_idx}: {n_nodes} nodes, {n_edges} edges, "
            f"{n_ffls} FFLs, {len(chains)} chains, {elapsed:.1f}s"
        )

    return {
        "motif_features": motif_features,
        "graph_stat_features": graph_stat_features,
        "domain": graph_record["domain"],
        "slug": graph_record["slug"],
        "prompt": graph_record["prompt"],
        "model_correct": graph_record["model_correct"],
    }


# ============================================================
# FEATURE NAME CONSTANTS
# ============================================================
MOTIF_NAMES = [
    "ratio_021U", "ratio_021C", "ratio_021D", "ratio_030T",
    "ffl_intensity_mean", "ffl_intensity_median", "ffl_intensity_std",
    "ffl_coherent_frac", "ffl_path_dom_mean", "ffl_asymmetry_mean",
    "ffl_coherence_onnela_mean", "ffl_coherence_onnela_std",
    "chain_intensity_mean", "chain_sign_agree_frac",
    "fanout_intensity_mean", "fanin_intensity_mean",
]

GSTAT_NAMES = [
    "n_nodes", "n_edges", "density", "mean_in_deg", "mean_out_deg",
    "max_degree", "diameter", "n_layers", "mean_abs_ew", "edge_weight_std",
]


def build_feature_matrices(
    processed: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build X_motif, X_gstat, X_combined, y arrays from processed graph results."""
    N = len(processed)
    X_motif = np.zeros((N, len(MOTIF_NAMES)), dtype=np.float64)
    X_gstat = np.zeros((N, len(GSTAT_NAMES)), dtype=np.float64)
    domains = []

    for i, rec in enumerate(processed):
        for j, feat_name in enumerate(MOTIF_NAMES):
            X_motif[i, j] = rec["motif_features"][feat_name]
        for j, feat_name in enumerate(GSTAT_NAMES):
            X_gstat[i, j] = rec["graph_stat_features"][feat_name]
        domains.append(rec["domain"])

    le = LabelEncoder()
    y = le.fit_transform(domains)
    domain_names = le.classes_.tolist()

    # Replace NaN/inf with 0
    X_motif = np.nan_to_num(X_motif, nan=0.0, posinf=0.0, neginf=0.0)
    X_gstat = np.nan_to_num(X_gstat, nan=0.0, posinf=0.0, neginf=0.0)

    X_combined = np.hstack([X_motif, X_gstat])

    logger.info(f"Feature matrices: X_motif={X_motif.shape}, X_gstat={X_gstat.shape}, "
                f"X_combined={X_combined.shape}, y={y.shape}")
    logger.info(f"Domains: {domain_names}")
    logger.info(f"Domain counts: {dict(Counter(domains))}")

    return X_motif, X_gstat, X_combined, y, domain_names


# ============================================================
# PHASE B: VARIANCE DECOMPOSITION (McFadden pseudo-R²)
# ============================================================


def mcfadden_r2(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    """Compute McFadden pseudo-R² = 1 - (LL_model / LL_null)."""
    log_proba = model.predict_log_proba(X)
    ll_model = sum(log_proba[i, y[i]] for i in range(len(y)))
    n_classes = len(np.unique(y))
    ll_null = len(y) * np.log(1.0 / n_classes)
    r2 = 1.0 - (ll_model / ll_null)
    return float(r2)


def fit_best_logreg(
    X: np.ndarray,
    y: np.ndarray,
    C_candidates: list[float] | None = None,
    max_iter: int = 2000,
) -> LogisticRegression:
    """Fit logistic regression with CV-tuned regularization C."""
    if C_candidates is None:
        C_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError("Need >= 2 classes for logistic regression")

    # Determine number of CV folds (adapt to small datasets)
    min_class_count = min(Counter(y).values())
    n_folds = min(5, min_class_count)
    n_folds = max(2, n_folds)

    best_C = 1.0
    best_score = -np.inf

    for C in C_candidates:
        try:
            scores = cross_val_score(
                LogisticRegression(
                    C=C, solver="lbfgs",
                    max_iter=max_iter, random_state=SEED
                ),
                X, y,
                cv=StratifiedKFold(n_folds, shuffle=True, random_state=SEED),
                scoring="neg_log_loss",
            )
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
        except Exception:
            continue

    model = LogisticRegression(
        C=best_C, solver="lbfgs",
        max_iter=max_iter, random_state=SEED,
    )
    model.fit(X, y)
    return model


def run_variance_decomposition(
    X_motif_s: np.ndarray,
    X_gstat_s: np.ndarray,
    X_combined_s: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Phase B: Compute McFadden R² decomposition with bootstrap CIs."""
    logger.info("=== PHASE B: Variance Decomposition ===")
    t0 = time.time()

    # Fit 3 models on full data
    logger.info("Fitting logistic regression models...")
    model_motif = fit_best_logreg(X_motif_s, y)
    model_gstat = fit_best_logreg(X_gstat_s, y)
    model_combined = fit_best_logreg(X_combined_s, y)

    R2_motif = mcfadden_r2(model_motif, X_motif_s, y)
    R2_gstat = mcfadden_r2(model_gstat, X_gstat_s, y)
    R2_combined = mcfadden_r2(model_combined, X_combined_s, y)

    unique_motif = R2_combined - R2_gstat
    unique_gstat = R2_combined - R2_motif
    shared = R2_motif + R2_gstat - R2_combined

    logger.info(f"R2_motif={R2_motif:.4f}, R2_gstat={R2_gstat:.4f}, R2_combined={R2_combined:.4f}")
    logger.info(f"unique_motif={unique_motif:.4f}, unique_gstat={unique_gstat:.4f}, shared={shared:.4f}")

    # Bootstrap for CIs
    logger.info(f"Starting {N_BOOTSTRAP} bootstrap iterations...")
    rng = np.random.RandomState(SEED)
    N = len(y)

    boot_results = {k: [] for k in [
        "R2_motif", "R2_gstat", "R2_combined",
        "unique_motif", "unique_gstat", "shared",
    ]}

    # Use reduced C candidates for speed during bootstrap
    C_boot = [0.1, 1.0, 10.0]
    failed_boots = 0

    for b in range(N_BOOTSTRAP):
        try:
            indices = rng.choice(N, N, replace=True)
            X_m_b = X_motif_s[indices]
            X_g_b = X_gstat_s[indices]
            X_c_b = X_combined_s[indices]
            y_b = y[indices]

            # Check we have at least 2 classes
            if len(np.unique(y_b)) < 2:
                failed_boots += 1
                continue

            m_m = fit_best_logreg(X_m_b, y_b, C_candidates=C_boot, max_iter=3000)
            m_g = fit_best_logreg(X_g_b, y_b, C_candidates=C_boot, max_iter=3000)
            m_c = fit_best_logreg(X_c_b, y_b, C_candidates=C_boot, max_iter=3000)

            r2_m = mcfadden_r2(m_m, X_m_b, y_b)
            r2_g = mcfadden_r2(m_g, X_g_b, y_b)
            r2_c = mcfadden_r2(m_c, X_c_b, y_b)

            boot_results["R2_motif"].append(r2_m)
            boot_results["R2_gstat"].append(r2_g)
            boot_results["R2_combined"].append(r2_c)
            boot_results["unique_motif"].append(r2_c - r2_g)
            boot_results["unique_gstat"].append(r2_c - r2_m)
            boot_results["shared"].append(r2_m + r2_g - r2_c)

        except Exception as e:
            failed_boots += 1
            if failed_boots <= 5:
                logger.debug(f"Bootstrap {b} failed: {e}")
            continue

        if (b + 1) % 100 == 0:
            logger.info(f"  Bootstrap {b + 1}/{N_BOOTSTRAP} done ({failed_boots} failed)")

    logger.info(f"Bootstrap complete: {N_BOOTSTRAP - failed_boots} successful, {failed_boots} failed")

    # Compute CIs
    result = {}
    for key in boot_results:
        vals = np.array(boot_results[key])
        if len(vals) >= 10:
            ci_lo = float(np.percentile(vals, 2.5))
            ci_hi = float(np.percentile(vals, 97.5))
        else:
            ci_lo = ci_hi = float("nan")
        result[key] = {
            "value": locals().get(key, 0.0) if key in ["R2_motif", "R2_gstat", "R2_combined",
                                                         "unique_motif", "unique_gstat", "shared"] else 0.0,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }

    # Set point estimates
    result["R2_motif"]["value"] = R2_motif
    result["R2_gstat"]["value"] = R2_gstat
    result["R2_combined"]["value"] = R2_combined
    result["unique_motif"]["value"] = unique_motif
    result["unique_gstat"]["value"] = unique_gstat
    result["shared"]["value"] = shared

    # Key test: CI of unique_motif excludes zero
    unique_motif_significant = result["unique_motif"]["ci_lower"] > 0
    result["unique_motif_significant"] = unique_motif_significant
    result["n_bootstrap_successful"] = N_BOOTSTRAP - failed_boots
    result["n_bootstrap_failed"] = failed_boots

    elapsed = time.time() - t0
    logger.info(f"Phase B complete in {elapsed:.1f}s. unique_motif_significant={unique_motif_significant}")

    return result


# ============================================================
# PHASE C: RESIDUALIZED CLUSTERING
# ============================================================


def residualize_features(
    X_target: np.ndarray,
    X_predictor: np.ndarray,
    target_names: list[str],
    predictor_names: list[str],
    label: str = "target",
) -> tuple[np.ndarray, dict]:
    """Residualize target features against predictor features using Ridge regression.

    Returns residualized features and dict of Ridge R² per target feature.
    """
    logger.info(f"Residualizing {label}: {X_target.shape[1]} features against {X_predictor.shape[1]} predictors")

    X_resid = np.zeros_like(X_target)
    ridge_r2s = {}

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    # Determine number of CV folds based on sample size
    n_folds = min(5, X_target.shape[0] // 2)
    n_folds = max(2, n_folds)

    for j in range(X_target.shape[1]):
        feat_name = target_names[j] if j < len(target_names) else f"feat_{j}"

        best_alpha = 1.0
        best_score = -np.inf

        for alpha in alphas:
            try:
                scores = cross_val_score(
                    Ridge(alpha=alpha),
                    X_predictor, X_target[:, j],
                    cv=n_folds,
                    scoring="r2",
                )
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
            except Exception:
                continue

        ridge = Ridge(alpha=best_alpha).fit(X_predictor, X_target[:, j])
        predicted = ridge.predict(X_predictor)
        X_resid[:, j] = X_target[:, j] - predicted

        # Compute R² of ridge regression
        ss_res = np.sum((X_target[:, j] - predicted) ** 2)
        ss_tot = np.sum((X_target[:, j] - np.mean(X_target[:, j])) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        ridge_r2s[feat_name] = round(float(r2), 4)

        logger.debug(f"  {feat_name}: Ridge R²={r2:.4f} (alpha={best_alpha})")

    # Verify residuals have mean ≈ 0
    mean_resid = np.mean(X_resid, axis=0)
    assert np.allclose(mean_resid, 0, atol=0.1), f"Residual means not near zero: max={np.max(np.abs(mean_resid)):.4f}"

    return X_resid, ridge_r2s


def cluster_with_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    n_perms: int,
    label: str = "features",
) -> dict:
    """Cluster on features, compute NMI/ARI at each K, permutation test at best K."""
    logger.info(f"Clustering on {label} ({X.shape[1]}D) with K={k_values}")

    nmi_by_k = {}
    ari_by_k = {}
    best_pred = None
    best_nmi = -1.0
    best_k = k_values[0]

    for K in k_values:
        try:
            sc = SpectralClustering(
                n_clusters=K, affinity="rbf", random_state=SEED, n_init=10,
            )
            pred = sc.fit_predict(X)
            nmi = float(normalized_mutual_info_score(y, pred))
            ari = float(adjusted_rand_score(y, pred))
        except Exception as e:
            logger.warning(f"SpectralClustering failed at K={K}: {e}, trying KMeans")
            try:
                km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
                pred = km.fit_predict(X)
                nmi = float(normalized_mutual_info_score(y, pred))
                ari = float(adjusted_rand_score(y, pred))
            except Exception as e2:
                logger.warning(f"KMeans also failed at K={K}: {e2}")
                nmi, ari = 0.0, 0.0
                pred = np.zeros(len(y), dtype=int)

        nmi_by_k[str(K)] = round(nmi, 4)
        ari_by_k[str(K)] = round(ari, 4)

        if nmi > best_nmi:
            best_nmi = nmi
            best_k = K
            best_pred = pred.copy()

    logger.info(f"  Best K={best_k}, NMI={best_nmi:.4f}")

    # Permutation test at best K
    logger.info(f"  Running {n_perms} permutations for p-value...")
    rng = np.random.RandomState(SEED)
    count_exceed = 0

    for p_idx in range(n_perms):
        shuffled = rng.permutation(y)
        try:
            sc = SpectralClustering(
                n_clusters=best_k, affinity="rbf", random_state=SEED + p_idx + 1, n_init=5,
            )
            perm_pred = sc.fit_predict(X)
            nmi_perm = normalized_mutual_info_score(shuffled, perm_pred)
        except Exception:
            try:
                km = KMeans(n_clusters=best_k, random_state=SEED + p_idx + 1, n_init=5)
                perm_pred = km.fit_predict(X)
                nmi_perm = normalized_mutual_info_score(shuffled, perm_pred)
            except Exception:
                nmi_perm = 0.0

        if nmi_perm >= best_nmi:
            count_exceed += 1

        if (p_idx + 1) % 100 == 0:
            logger.info(f"    Permutation {p_idx + 1}/{n_perms}")

    p_value = (count_exceed + 1) / (n_perms + 1)
    logger.info(f"  Permutation p-value={p_value:.4f} (exceeded {count_exceed}/{n_perms})")

    return {
        "nmi_by_k": nmi_by_k,
        "ari_by_k": ari_by_k,
        "best_nmi": round(best_nmi, 4),
        "best_k": best_k,
        "perm_p_value": round(p_value, 4),
        "best_predictions": best_pred.tolist() if best_pred is not None else [],
    }


def run_residualized_clustering(
    X_motif_s: np.ndarray,
    X_gstat_s: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Phase C: Residualize each feature block against the other, then cluster."""
    logger.info("=== PHASE C: Residualized Clustering ===")
    t0 = time.time()

    # C1: Residualize motif features against graph stats
    X_motif_resid, ridge_r2_motif = residualize_features(
        X_motif_s, X_gstat_s, MOTIF_NAMES, GSTAT_NAMES, label="motif→gstats"
    )

    # C2: Cluster on residualized motif features
    motif_resid_results = cluster_with_permutation_test(
        X_motif_resid, y, CLUSTER_K_VALUES, N_PERMUTATIONS,
        label="motif_resid_on_gstats",
    )

    # C4: Reverse direction — residualize graph-stats against motifs
    X_gstat_resid, ridge_r2_gstat = residualize_features(
        X_gstat_s, X_motif_s, GSTAT_NAMES, MOTIF_NAMES, label="gstats→motifs"
    )

    # Cluster on residualized graph-stats
    gstat_resid_results = cluster_with_permutation_test(
        X_gstat_resid, y, CLUSTER_K_VALUES, N_PERMUTATIONS,
        label="gstat_resid_on_motifs",
    )

    elapsed = time.time() - t0
    logger.info(f"Phase C complete in {elapsed:.1f}s")

    return {
        "motif_resid_on_gstats": motif_resid_results,
        "gstat_resid_on_motifs": gstat_resid_results,
        "ridge_r2_per_motif_feature": ridge_r2_motif,
        "ridge_r2_per_gstat_feature": ridge_r2_gstat,
        "X_motif_resid": X_motif_resid,
        "X_gstat_resid": X_gstat_resid,
    }


# ============================================================
# PHASE D: DOMAIN-NORMALIZED WEIGHTED INTENSITY
# ============================================================


def run_domain_normalized_analysis(
    processed: list[dict],
    X_motif_s: np.ndarray,
    y: np.ndarray,
    domain_names: list[str],
) -> dict:
    """Phase D: Recompute motif features with domain-normalized edge weights."""
    logger.info("=== PHASE D: Domain-Normalized Weighted Intensity ===")
    t0 = time.time()

    # D1: Compute per-domain weight stats
    domain_weights: dict[str, list[float]] = {}
    for rec in processed:
        domain = rec["domain"]
        domain_weights.setdefault(domain, [])
        for feat_name in ["mean_abs_ew"]:
            domain_weights[domain].append(rec["graph_stat_features"][feat_name])

    # Collect all edge weights per domain from the graphs
    # Since we already processed graphs, use stored features instead
    # We'll use graph_stat_features mean_abs_ew and edge_weight_std

    domain_weight_stats = {}
    for domain in sorted(set(r["domain"] for r in processed)):
        domain_recs = [r for r in processed if r["domain"] == domain]
        ew_means = [r["graph_stat_features"]["mean_abs_ew"] for r in domain_recs]
        ew_stds = [r["graph_stat_features"]["edge_weight_std"] for r in domain_recs]
        domain_weight_stats[domain] = {
            "mean": round(float(np.mean(ew_means)), 6),
            "std": round(float(np.mean(ew_stds)), 6),
        }

    logger.info(f"Domain weight stats: {domain_weight_stats}")

    # D2/D3: Normalize motif intensity features by domain mean/std of edge weights
    # Intensity features that are confounded with edge weight scale:
    # ffl_intensity_mean (col 4), ffl_intensity_median (col 5), ffl_intensity_std (col 6)
    # chain_intensity_mean (col 12), fanout_intensity_mean (col 14), fanin_intensity_mean (col 15)
    intensity_cols = [4, 5, 6, 12, 14, 15]

    X_motif_normed = X_motif_s.copy()
    domains_list = [r["domain"] for r in processed]

    for domain in sorted(set(domains_list)):
        d_mean = domain_weight_stats[domain]["mean"]
        d_std = domain_weight_stats[domain]["std"]
        eps = 1e-8
        mask = np.array([d == domain for d in domains_list])

        for col in intensity_cols:
            if d_std > eps:
                X_motif_normed[mask, col] = (X_motif_normed[mask, col] - d_mean) / (d_std + eps)
            else:
                # Use median absolute deviation
                vals = X_motif_normed[mask, col]
                mad = np.median(np.abs(vals - np.median(vals)))
                if mad > eps:
                    X_motif_normed[mask, col] = (vals - np.median(vals)) / (mad * 1.4826 + eps)

    # Re-standardize
    scaler_normed = StandardScaler()
    X_motif_normed_s = scaler_normed.fit_transform(X_motif_normed)

    # D4: Cluster on domain-normalized motif features at K=8
    try:
        sc = SpectralClustering(n_clusters=8, affinity="rbf", random_state=SEED, n_init=10)
        pred_normed = sc.fit_predict(X_motif_normed_s)
        nmi_normed_k8 = float(normalized_mutual_info_score(y, pred_normed))
    except Exception:
        km = KMeans(n_clusters=8, random_state=SEED, n_init=10)
        pred_normed = km.fit_predict(X_motif_normed_s)
        nmi_normed_k8 = float(normalized_mutual_info_score(y, pred_normed))

    # Also cluster on raw motif features at K=8 for comparison
    try:
        sc_raw = SpectralClustering(n_clusters=8, affinity="rbf", random_state=SEED, n_init=10)
        pred_raw = sc_raw.fit_predict(X_motif_s)
        nmi_raw_k8 = float(normalized_mutual_info_score(y, pred_raw))
    except Exception:
        km_raw = KMeans(n_clusters=8, random_state=SEED, n_init=10)
        pred_raw = km_raw.fit_predict(X_motif_s)
        nmi_raw_k8 = float(normalized_mutual_info_score(y, pred_raw))

    # Also cluster on domain-normalized RESIDUALIZED features for strongest test
    # (residualize normed motif against gstats)
    # We'll do this inline with a simple ridge residualization
    X_gstat_for_normed = StandardScaler().fit_transform(
        np.array([[r["graph_stat_features"][f] for f in GSTAT_NAMES] for r in processed])
    )
    X_normed_resid = np.zeros_like(X_motif_normed_s)
    for j in range(X_motif_normed_s.shape[1]):
        ridge = Ridge(alpha=1.0).fit(X_gstat_for_normed, X_motif_normed_s[:, j])
        X_normed_resid[:, j] = X_motif_normed_s[:, j] - ridge.predict(X_gstat_for_normed)

    try:
        sc_nresid = SpectralClustering(n_clusters=8, affinity="rbf", random_state=SEED, n_init=10)
        pred_nresid = sc_nresid.fit_predict(X_normed_resid)
        nmi_normed_resid_k8 = float(normalized_mutual_info_score(y, pred_nresid))
    except Exception:
        km_nresid = KMeans(n_clusters=8, random_state=SEED, n_init=10)
        pred_nresid = km_nresid.fit_predict(X_normed_resid)
        nmi_normed_resid_k8 = float(normalized_mutual_info_score(y, pred_nresid))

    elapsed = time.time() - t0
    logger.info(f"Phase D complete in {elapsed:.1f}s")
    logger.info(f"  NMI raw motif K=8: {nmi_raw_k8:.4f}")
    logger.info(f"  NMI normalized motif K=8: {nmi_normed_k8:.4f}")
    logger.info(f"  NMI normalized+residualized K=8: {nmi_normed_resid_k8:.4f}")

    return {
        "nmi_raw_motif_k8": round(nmi_raw_k8, 4),
        "nmi_normalized_motif_k8": round(nmi_normed_k8, 4),
        "nmi_normalized_resid_k8": round(nmi_normed_resid_k8, 4),
        "domain_weight_stats": domain_weight_stats,
    }


# ============================================================
# PHASE E: CONDITIONAL MUTUAL INFORMATION
# ============================================================


def run_conditional_mi(
    X_motif_s: np.ndarray,
    X_gstat_s: np.ndarray,
    X_motif_resid: np.ndarray,
    X_gstat_resid: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Phase E: Compute raw and residualized mutual information."""
    logger.info("=== PHASE E: Conditional Mutual Information ===")
    t0 = time.time()

    # E1: Raw MI
    mi_motif_raw = mutual_info_classif(
        X_motif_s, y, discrete_features=False, n_neighbors=5, random_state=SEED
    )
    mi_gstat_raw = mutual_info_classif(
        X_gstat_s, y, discrete_features=False, n_neighbors=5, random_state=SEED
    )

    # E2: Residualized MI
    mi_motif_resid = mutual_info_classif(
        X_motif_resid, y, discrete_features=False, n_neighbors=5, random_state=SEED
    )
    mi_gstat_resid = mutual_info_classif(
        X_gstat_resid, y, discrete_features=False, n_neighbors=5, random_state=SEED
    )

    mi_motif_raw_total = float(np.sum(mi_motif_raw))
    mi_gstat_raw_total = float(np.sum(mi_gstat_raw))
    mi_motif_resid_total = float(np.sum(mi_motif_resid))
    mi_gstat_resid_total = float(np.sum(mi_gstat_resid))

    logger.info(f"MI raw: motif_total={mi_motif_raw_total:.4f}, gstat_total={mi_gstat_raw_total:.4f}")
    logger.info(f"MI resid: motif_total={mi_motif_resid_total:.4f}, gstat_total={mi_gstat_resid_total:.4f}")

    # E3: Per-feature table
    per_feature = []

    for j, name in enumerate(MOTIF_NAMES):
        raw_mi = float(mi_motif_raw[j])
        resid_mi = float(mi_motif_resid[j])
        retained = resid_mi / raw_mi if raw_mi > 1e-8 else 0.0
        per_feature.append({
            "name": name,
            "block": "motif",
            "raw_mi": round(raw_mi, 4),
            "resid_mi": round(resid_mi, 4),
            "retained_frac": round(retained, 4),
        })

    for j, name in enumerate(GSTAT_NAMES):
        raw_mi = float(mi_gstat_raw[j])
        resid_mi = float(mi_gstat_resid[j])
        retained = resid_mi / raw_mi if raw_mi > 1e-8 else 0.0
        per_feature.append({
            "name": name,
            "block": "gstat",
            "raw_mi": round(raw_mi, 4),
            "resid_mi": round(resid_mi, 4),
            "retained_frac": round(retained, 4),
        })

    elapsed = time.time() - t0
    logger.info(f"Phase E complete in {elapsed:.1f}s")

    return {
        "mi_motif_raw_total": round(mi_motif_raw_total, 4),
        "mi_gstat_raw_total": round(mi_gstat_raw_total, 4),
        "mi_motif_resid_total": round(mi_motif_resid_total, 4),
        "mi_gstat_resid_total": round(mi_gstat_resid_total, 4),
        "per_feature": per_feature,
    }


# ============================================================
# PHASE F: CANONICAL CORRELATION ANALYSIS
# ============================================================


def run_cca_analysis(
    X_motif_s: np.ndarray,
    X_gstat_s: np.ndarray,
) -> dict:
    """Phase F: CCA between motif and graph-stats feature blocks."""
    logger.info("=== PHASE F: Canonical Correlation Analysis ===")
    t0 = time.time()

    N = X_motif_s.shape[0]
    p1 = X_motif_s.shape[1]
    p2 = X_gstat_s.shape[1]
    n_components = min(p1, p2)  # = 10

    try:
        cca = CCA(n_components=n_components, max_iter=1000)
        X_motif_c, X_gstat_c = cca.fit_transform(X_motif_s, X_gstat_s)

        # Compute canonical correlations
        canonical_corrs = []
        for i in range(n_components):
            r = float(np.corrcoef(X_motif_c[:, i], X_gstat_c[:, i])[0, 1])
            if np.isnan(r):
                r = 0.0
            canonical_corrs.append(round(r, 4))

        logger.info(f"Canonical correlations: {canonical_corrs}")

        # Wilks' Lambda significance test
        wilks_results = []
        for k in range(n_components):
            remaining_corrs = canonical_corrs[k:]
            lambda_k = float(np.prod([1 - r ** 2 for r in remaining_corrs]))

            if lambda_k <= 0 or lambda_k >= 1:
                wilks_results.append({
                    "dim": k + 1, "r": canonical_corrs[k],
                    "lambda": round(lambda_k, 6),
                    "chi2": 0.0, "df": 0, "p": 1.0,
                })
                continue

            chi2 = -(N - 1 - (p1 + p2 + 1) / 2) * np.log(lambda_k)
            df = (p1 - k) * (p2 - k)
            if df <= 0 or chi2 < 0:
                p_val = 1.0
            else:
                p_val = float(1 - scipy.stats.chi2.cdf(chi2, df))

            wilks_results.append({
                "dim": k + 1,
                "r": canonical_corrs[k],
                "lambda": round(float(lambda_k), 6),
                "chi2": round(float(chi2), 4),
                "df": int(df),
                "p": round(float(p_val), 6),
            })

        n_significant = sum(1 for w in wilks_results if w["p"] < 0.05)

        if n_significant >= n_components - 1:
            interpretation = (
                f"All {n_significant}/{n_components} canonical dimensions significant: "
                "substantial linear redundancy between motif and graph-stats feature blocks."
            )
        elif n_significant >= n_components // 2:
            interpretation = (
                f"{n_significant}/{n_components} canonical dimensions significant: "
                "moderate linear overlap, but some orthogonal dimensions remain."
            )
        else:
            interpretation = (
                f"Only {n_significant}/{n_components} canonical dimensions significant: "
                "feature blocks are largely orthogonal — motifs capture genuinely different information."
            )

        logger.info(f"CCA: {n_significant}/{n_components} significant dimensions")

    except Exception as e:
        logger.warning(f"CCA failed: {e}. Skipping Phase F.")
        canonical_corrs = []
        wilks_results = []
        n_significant = 0
        interpretation = f"CCA failed ({e}). Phase F skipped."

    elapsed = time.time() - t0
    logger.info(f"Phase F complete in {elapsed:.1f}s")

    return {
        "canonical_correlations": canonical_corrs,
        "wilks_lambda_tests": wilks_results,
        "n_significant_dims": n_significant,
        "n_total_dims": n_components,
        "interpretation": interpretation,
    }


# ============================================================
# PHASE G: BUILD OUTPUT
# ============================================================


def build_output(
    processed: list[dict],
    variance_decomp: dict,
    resid_clustering: dict,
    domain_normalized: dict,
    cond_mi: dict,
    cca_result: dict,
    domain_names: list[str],
    total_runtime: float,
) -> dict:
    """Build output JSON conforming to exp_gen_sol_out schema."""
    logger.info("=== PHASE G: Building Output ===")

    # Determine verdict
    unique_motif_val = variance_decomp["unique_motif"]["value"]
    unique_motif_sig = variance_decomp.get("unique_motif_significant", False)
    resid_nmi = resid_clustering["motif_resid_on_gstats"]["best_nmi"]
    resid_pval = resid_clustering["motif_resid_on_gstats"]["perm_p_value"]
    mi_retained = (
        cond_mi["mi_motif_resid_total"] / cond_mi["mi_motif_raw_total"]
        if cond_mi["mi_motif_raw_total"] > 1e-8 else 0.0
    )

    motif_unique_value = (
        unique_motif_sig
        and resid_pval < 0.05
        and resid_nmi > 0.05
    )

    evidence_parts = []
    if unique_motif_sig:
        evidence_parts.append(
            f"Unique motif R²={unique_motif_val:.4f} "
            f"(CI: [{variance_decomp['unique_motif']['ci_lower']:.4f}, "
            f"{variance_decomp['unique_motif']['ci_upper']:.4f}]) — CI excludes 0."
        )
    else:
        evidence_parts.append(
            f"Unique motif R²={unique_motif_val:.4f} "
            f"(CI includes 0 — not significant)."
        )

    evidence_parts.append(
        f"Residualized motif NMI={resid_nmi:.4f} (p={resid_pval:.4f})."
    )
    evidence_parts.append(
        f"MI retained after residualization: {mi_retained:.1%}."
    )
    evidence_parts.append(
        f"CCA: {cca_result['n_significant_dims']}/{cca_result.get('n_total_dims', 10)} dimensions significant."
    )
    evidence_parts.append(
        f"Domain-normalized motif NMI K=8: {domain_normalized['nmi_normalized_motif_k8']:.4f}."
    )

    verdict = {
        "motif_unique_value": motif_unique_value,
        "evidence_summary": " ".join(evidence_parts),
        "unique_motif_R2": round(unique_motif_val, 4),
        "residualized_nmi": round(resid_nmi, 4),
        "conditional_mi_retained": round(mi_retained, 4),
    }

    logger.info(f"Verdict: motif_unique_value={motif_unique_value}")
    logger.info(f"Evidence: {verdict['evidence_summary']}")

    # Build per-graph examples
    # Get cluster predictions from residualized clustering
    motif_resid_preds = resid_clustering["motif_resid_on_gstats"]["best_predictions"]
    gstat_resid_preds = resid_clustering["gstat_resid_on_motifs"]["best_predictions"]

    examples = []
    for i, rec in enumerate(processed):
        output_data = {
            "motif_features": {k: round(v, 6) if isinstance(v, float) else v
                               for k, v in rec["motif_features"].items()},
            "graph_stat_features": {k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in rec["graph_stat_features"].items()},
        }

        example = {
            "input": rec["prompt"],
            "output": json.dumps(output_data),
            "metadata_fold": rec["domain"],
            "metadata_slug": rec["slug"],
            "metadata_model_correct": rec["model_correct"],
        }

        if i < len(motif_resid_preds):
            example["predict_motif_resid_cluster"] = f"cluster_{motif_resid_preds[i]}"
        if i < len(gstat_resid_preds):
            example["predict_gstat_resid_cluster"] = f"cluster_{gstat_resid_preds[i]}"

        examples.append(example)

    # Remove internal numpy arrays from resid_clustering before serialization
    resid_clustering_out = {
        k: v for k, v in resid_clustering.items()
        if k not in ("X_motif_resid", "X_gstat_resid")
    }

    output = {
        "metadata": {
            "method_name": "motif_vs_graphstats_unique_information_decomposition",
            "description": (
                "Determines whether motif-level features carry unique structural "
                "information about capability domains beyond graph-level statistics. "
                "Uses McFadden R² decomposition, residualized clustering, conditional MI, "
                "domain normalization, and CCA."
            ),
            "parameters": {
                "prune_percentile": PRUNE_PERCENTILE,
                "min_nodes": MIN_NODES,
                "n_bootstrap": N_BOOTSTRAP,
                "n_permutations": N_PERMUTATIONS,
                "seed": SEED,
                "cluster_k_values": CLUSTER_K_VALUES,
            },
            "n_graphs_processed": len(processed),
            "n_domains": len(domain_names),
            "domain_counts": dict(Counter(r["domain"] for r in processed)),
            "domain_names": domain_names,
            "total_runtime_seconds": round(total_runtime, 1),
            "variance_decomposition": variance_decomp,
            "residualized_clustering": resid_clustering_out,
            "domain_normalized": domain_normalized,
            "conditional_mutual_info": cond_mi,
            "cca_analysis": cca_result,
            "verdict": verdict,
        },
        "datasets": [{
            "dataset": "neuronpedia_attribution_graphs_v3",
            "examples": examples,
        }],
    }

    return output


# ============================================================
# MAIN
# ============================================================


@logger.catch
def main():
    """Run the full unique information decomposition pipeline."""
    t_start = time.time()

    # PHASE A: Load graphs and extract features
    logger.info("=== PHASE A: Loading Graphs & Computing Features ===")
    graphs = load_all_graphs_with_weights()

    if len(graphs) < 8:
        logger.error(f"Only {len(graphs)} graphs loaded — need at least 8 (1 per domain)")
        sys.exit(1)

    # Process each graph sequentially (igraph objects not picklable)
    processed = []
    for i, gr in enumerate(graphs):
        try:
            result = process_one_graph(gr, i)
            processed.append(result)
        except Exception:
            logger.exception(f"Failed on graph {i}")
            continue

    # Free graph objects to reclaim memory
    for gr in graphs:
        del gr["graph"]
    del graphs
    gc.collect()

    logger.info(f"Successfully processed {len(processed)}/{len(processed)} graphs")

    # Build feature matrices
    X_motif, X_gstat, X_combined, y, domain_names = build_feature_matrices(processed)

    # Standardize
    scaler_motif = StandardScaler()
    scaler_gstat = StandardScaler()
    scaler_combined = StandardScaler()

    X_motif_s = scaler_motif.fit_transform(X_motif)
    X_gstat_s = scaler_gstat.fit_transform(X_gstat)
    X_combined_s = scaler_combined.fit_transform(X_combined)

    # PHASE B: Variance Decomposition
    variance_decomp = run_variance_decomposition(X_motif_s, X_gstat_s, X_combined_s, y)

    # PHASE C: Residualized Clustering
    resid_clustering = run_residualized_clustering(X_motif_s, X_gstat_s, y)

    # PHASE D: Domain-Normalized Analysis
    domain_normalized = run_domain_normalized_analysis(processed, X_motif_s, y, domain_names)

    # PHASE E: Conditional Mutual Information
    X_motif_resid = resid_clustering["X_motif_resid"]
    X_gstat_resid = resid_clustering["X_gstat_resid"]
    cond_mi = run_conditional_mi(X_motif_s, X_gstat_s, X_motif_resid, X_gstat_resid, y)

    # PHASE F: CCA
    cca_result = run_cca_analysis(X_motif_s, X_gstat_s)

    # PHASE G: Output
    total_runtime = time.time() - t_start
    output = build_output(
        processed, variance_decomp, resid_clustering,
        domain_normalized, cond_mi, cca_result,
        domain_names, total_runtime,
    )

    # Write output
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    logger.info(f"Output written to {OUTPUT_FILE}")
    logger.info(f"Total runtime: {total_runtime:.1f}s")

    # Verify output structure
    assert "datasets" in output, "Missing 'datasets' key"
    assert len(output["datasets"]) >= 1, "Empty datasets"
    assert len(output["datasets"][0]["examples"]) >= 1, "No examples"
    ex = output["datasets"][0]["examples"][0]
    assert "input" in ex, "Missing 'input' in example"
    assert "output" in ex, "Missing 'output' in example"
    logger.info("Output structure verified ✓")


if __name__ == "__main__":
    main()
