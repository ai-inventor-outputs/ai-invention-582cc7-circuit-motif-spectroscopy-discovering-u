#!/usr/bin/env python3
"""Weighted Motif Intensity & Sign-Coherence Features for Capability Clustering.

Computes Onnela-style weighted motif features (intensity, sign-pattern coherence,
path dominance, weight asymmetry) for all FFL instances and other 3-node DAG motif
types across 200 attribution graphs, aggregates into per-graph feature vectors,
and systematically compares 5 feature sets for spectral clustering quality (NMI/ARI)
against 8 capability domain labels.

Key baseline to beat: binary 3-node enriched features at NMI=0.742.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import igraph
import numpy as np
import scipy.stats
from loguru import logger
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

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
PRUNE_PERCENTILE = 75
MIN_NODES = 30
SEED = 42
N_PERMUTATIONS = int(os.environ.get("N_PERMUTATIONS", "1000"))
CLUSTER_K_VALUES = [2, 4, 6, 8]
BATCH_SIZE = 25

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

# Set memory limits: use ~75% of container RAM (leave room for OS + agent)
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.75 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 4, RAM_BUDGET_BYTES * 4))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

NUM_WORKERS = max(1, NUM_CPUS - 1)  # Leave 1 core for OS

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, workers={NUM_WORKERS}")
logger.info(f"RAM budget: {RAM_BUDGET_BYTES / 1e9:.1f} GB")
logger.info(f"Config: MAX_EXAMPLES={MAX_EXAMPLES}, PRUNE_PERCENTILE={PRUNE_PERCENTILE}, "
            f"N_PERMUTATIONS={N_PERMUTATIONS}")


# ============================================================
# PHASE A: LOAD GRAPHS PRESERVING SIGNED WEIGHTS
# ============================================================

def parse_layer(layer_str: str) -> int:
    """Parse layer string to integer."""
    try:
        return int(layer_str)
    except (ValueError, TypeError):
        return 0


def load_all_graphs_with_weights() -> list[dict]:
    """Load from data_id5_it3, 12 split files, preserve signed weights.

    Always loads ALL files first so domains are balanced, then subsamples
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
                logger.warning(f"Skipping example with invalid JSON output")
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
            signed_weight_dict = {}  # (src_nid, tgt_nid) -> signed_w

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
        # Group by domain
        domain_groups: dict[str, list[int]] = {}
        for i, gr in enumerate(all_graphs):
            domain_groups.setdefault(gr["domain"], []).append(i)

        # Allocate proportionally, at least 1 per domain
        n_domains = len(domain_groups)
        per_domain = max(1, MAX_EXAMPLES // n_domains)
        selected_indices = []
        for domain, indices in sorted(domain_groups.items()):
            n_take = min(per_domain, len(indices))
            chosen = rng.choice(indices, n_take, replace=False).tolist()
            selected_indices.extend(chosen)

        # Fill remaining slots if needed
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
# PHASE B: ENUMERATE ALL 3-NODE MOTIF INSTANCES WITH WEIGHTS
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


# ============================================================
# PHASE B2: COMPUTE WEIGHTED FEATURES PER MOTIF INSTANCE
# ============================================================

def compute_ffl_weighted_features(ffl: dict) -> dict:
    """For one FFL instance, compute weighted features."""
    w_ab_abs = ffl["w_ab"]["abs"]
    w_ac_abs = ffl["w_ac"]["abs"]
    w_bc_abs = ffl["w_bc"]["abs"]
    w_ab_s = ffl["w_ab"]["signed"]
    w_ac_s = ffl["w_ac"]["signed"]
    w_bc_s = ffl["w_bc"]["signed"]

    # 1. Onnela intensity: geometric mean of |weights|
    intensity = (w_ab_abs * w_ac_abs * w_bc_abs) ** (1.0 / 3.0)

    # 2. Path dominance ratio: direct / indirect path strength
    indirect = w_ab_abs * w_bc_abs
    path_dominance = w_ac_abs / indirect if indirect > 1e-12 else 100.0

    # 3. Sign coherence classification
    sign_product = np.sign(w_ab_s) * np.sign(w_ac_s) * np.sign(w_bc_s)
    is_coherent = sign_product > 0
    sign_pattern = (int(np.sign(w_ab_s)), int(np.sign(w_ac_s)), int(np.sign(w_bc_s)))

    # 4. Weight asymmetry: |w_AB - w_BC| / (w_AB + w_BC)
    denom = w_ab_abs + w_bc_abs
    weight_asymmetry = abs(w_ab_abs - w_bc_abs) / denom if denom > 1e-12 else 0.0

    # 5. Onnela coherence: geometric_mean / arithmetic_mean
    arith_mean = (w_ab_abs + w_ac_abs + w_bc_abs) / 3.0
    coherence_onnela = intensity / arith_mean if arith_mean > 1e-12 else 0.0

    return {
        "intensity": intensity,
        "path_dominance": path_dominance,
        "is_coherent": is_coherent,
        "sign_pattern": sign_pattern,
        "weight_asymmetry": weight_asymmetry,
        "coherence_onnela": coherence_onnela,
    }


def compute_chain_features(chain: dict) -> dict:
    """2-edge intensity, sign agreement, weight ratio."""
    w_ab = chain["w_ab"]["abs"]
    w_bc = chain["w_bc"]["abs"]
    intensity = (w_ab * w_bc) ** 0.5
    sign_agree = int(np.sign(chain["w_ab"]["signed"]) == np.sign(chain["w_bc"]["signed"]))
    weight_ratio = max(w_ab, w_bc) / (min(w_ab, w_bc) + 1e-12)
    return {"intensity": intensity, "sign_agree": sign_agree, "weight_ratio": weight_ratio}


def compute_fanout_features(fanout: dict) -> dict:
    """2-edge fan-out features."""
    w_ab = fanout["w_ab"]["abs"]
    w_ac = fanout["w_ac"]["abs"]
    intensity = (w_ab * w_ac) ** 0.5
    sign_agree = int(np.sign(fanout["w_ab"]["signed"]) == np.sign(fanout["w_ac"]["signed"]))
    weight_ratio = max(w_ab, w_ac) / (min(w_ab, w_ac) + 1e-12)
    return {"intensity": intensity, "sign_agree": sign_agree, "weight_ratio": weight_ratio}


def compute_fanin_features(fanin: dict) -> dict:
    """2-edge fan-in features."""
    w_ba = fanin["w_ba"]["abs"]
    w_ca = fanin["w_ca"]["abs"]
    intensity = (w_ba * w_ca) ** 0.5
    sign_agree = int(np.sign(fanin["w_ba"]["signed"]) == np.sign(fanin["w_ca"]["signed"]))
    weight_ratio = max(w_ba, w_ca) / (min(w_ba, w_ca) + 1e-12)
    return {"intensity": intensity, "sign_agree": sign_agree, "weight_ratio": weight_ratio}


# ============================================================
# PHASE C: PER-GRAPH FEATURE AGGREGATION
# ============================================================

def process_one_graph(graph_record: dict, graph_idx: int) -> dict:
    """Process one graph: enumerate motifs, compute features, aggregate."""
    g = graph_record["graph"]
    t0 = time.time()

    adj_set, wl = build_adjacency_and_weight_lookups(g)

    # --- Enumerate all 3-node motif types ---
    ffls = enumerate_ffls(g, adj_set, wl)
    chains = enumerate_chains(g, adj_set, wl)
    fanouts = enumerate_fanouts(g, adj_set, wl)
    fanins = enumerate_fanins(g, adj_set, wl)

    # --- Compute per-instance weighted features for FFLs ---
    # Streaming aggregation to avoid storing all instances
    ffl_intensities = []
    ffl_path_doms = []
    ffl_asymmetries = []
    ffl_coherences_onnela = []
    n_coherent = 0
    n_incoherent = 0
    sign_pattern_counts = Counter()

    for ffl in ffls:
        feats = compute_ffl_weighted_features(ffl)
        ffl_intensities.append(feats["intensity"])
        pd_val = feats["path_dominance"]
        if pd_val < 100:  # cap outliers for stability
            ffl_path_doms.append(pd_val)
        ffl_asymmetries.append(feats["weight_asymmetry"])
        ffl_coherences_onnela.append(feats["coherence_onnela"])
        if feats["is_coherent"]:
            n_coherent += 1
        else:
            n_incoherent += 1
        sp_key = str(feats["sign_pattern"])
        sign_pattern_counts[sp_key] += 1

    # Chain features (streaming)
    chain_intensities = []
    chain_sign_agrees = []
    for c in chains:
        cf = compute_chain_features(c)
        chain_intensities.append(cf["intensity"])
        chain_sign_agrees.append(cf["sign_agree"])

    # Fan-out features (streaming)
    fanout_intensities = []
    fanout_sign_agrees = []
    for fo in fanouts:
        fof = compute_fanout_features(fo)
        fanout_intensities.append(fof["intensity"])
        fanout_sign_agrees.append(fof["sign_agree"])

    # Fan-in features (streaming)
    fanin_intensities = []
    fanin_sign_agrees = []
    for fi_inst in fanins:
        fif = compute_fanin_features(fi_inst)
        fanin_intensities.append(fif["intensity"])
        fanin_sign_agrees.append(fif["sign_agree"])

    # --- Aggregate into per-graph features ---
    n_ffls = len(ffls)
    fi_arr = np.array(ffl_intensities) if ffl_intensities else np.array([0.0])
    fpd_arr = np.array(ffl_path_doms) if ffl_path_doms else np.array([0.0])
    fa_arr = np.array(ffl_asymmetries) if ffl_asymmetries else np.array([0.0])
    fco_arr = np.array(ffl_coherences_onnela) if ffl_coherences_onnela else np.array([0.0])

    # Graph-level statistics
    n_nodes = g.vcount()
    n_edges = g.ecount()
    density = 2.0 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    degrees = g.degree()
    mean_degree = float(np.mean(degrees))
    max_degree = max(degrees) if degrees else 0
    layers = g.vs["layer"]
    n_layers = len(set(layers))

    # Approximate diameter using BFS from a few nodes
    try:
        if n_nodes < 500:
            diameter_approx = g.diameter(directed=True)
        else:
            # Sample BFS from 10 nodes for speed
            sampled = np.random.choice(n_nodes, min(10, n_nodes), replace=False)
            max_d = 0
            for s in sampled:
                dists = g.distances(source=int(s), mode="out")[0]
                valid = [d for d in dists if d < float('inf')]
                if valid:
                    max_d = max(max_d, max(valid))
            diameter_approx = max_d
    except Exception:
        diameter_approx = 0

    edge_weights = np.array(g.es["weight"])
    signed_weights = np.array(g.es["signed_weight"])

    features = {
        # FFL weighted features (~12 features)
        "ffl_count": n_ffls,
        "ffl_intensity_mean": float(np.mean(fi_arr)),
        "ffl_intensity_median": float(np.median(fi_arr)),
        "ffl_intensity_std": float(np.std(fi_arr)),
        "ffl_intensity_q25": float(np.percentile(fi_arr, 25)),
        "ffl_intensity_q75": float(np.percentile(fi_arr, 75)),
        "ffl_coherent_frac": n_coherent / max(n_ffls, 1),
        "ffl_path_dom_mean": float(np.mean(fpd_arr)),
        "ffl_path_dom_std": float(np.std(fpd_arr)),
        "ffl_asymmetry_mean": float(np.mean(fa_arr)),
        "ffl_coherence_onnela_mean": float(np.mean(fco_arr)),
        "ffl_coherence_onnela_std": float(np.std(fco_arr)),

        # Chain weighted features
        "chain_count": len(chains),
        "chain_intensity_mean": float(np.mean(chain_intensities)) if chain_intensities else 0.0,
        "chain_sign_agree_frac": float(np.mean(chain_sign_agrees)) if chain_sign_agrees else 0.0,

        # Fan-out weighted features
        "fanout_count": len(fanouts),
        "fanout_intensity_mean": float(np.mean(fanout_intensities)) if fanout_intensities else 0.0,
        "fanout_sign_agree_frac": float(np.mean(fanout_sign_agrees)) if fanout_sign_agrees else 0.0,

        # Fan-in weighted features
        "fanin_count": len(fanins),
        "fanin_intensity_mean": float(np.mean(fanin_intensities)) if fanin_intensities else 0.0,
        "fanin_sign_agree_frac": float(np.mean(fanin_sign_agrees)) if fanin_sign_agrees else 0.0,

        # Edge weight distribution features
        "edge_weight_mean": float(np.mean(edge_weights)),
        "edge_weight_std": float(np.std(edge_weights)),
        "neg_edge_frac": float(np.mean(signed_weights < 0)),
        "edge_weight_kurtosis": float(scipy.stats.kurtosis(edge_weights))
        if len(edge_weights) > 3 else 0.0,

        # Graph structure features
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "diameter_approx": diameter_approx,
        "n_layers": n_layers,

        # Sign coherence details
        "sign_pattern_counts": dict(sign_pattern_counts),
    }

    elapsed = time.time() - t0
    if graph_idx % 10 == 0:
        logger.info(
            f"  Graph {graph_idx}: {n_nodes} nodes, {n_edges} edges, "
            f"{n_ffls} FFLs, {len(chains)} chains, {elapsed:.1f}s"
        )

    return features


# ============================================================
# PHASE C2: COMPUTE BINARY MOTIF RATIOS VIA TRIAD CENSUS
# ============================================================

def compute_binary_motif_ratios(g: igraph.Graph) -> dict:
    """Compute ratios of 4 DAG-valid 3-node motif types using igraph triad_census.

    igraph triad census indices (MAN notation):
      Index 6 = 021D (fan-out: A->B, A->C)
      Index 5 = 021U (fan-in: B->A, C->A)
      Index 7 = 021C (chain: A->B, B->C)
      Index 9 = 030T (FFL: A->B, A->C, B->C)
    """
    tc = g.triad_census()
    count_021D = tc[6]
    count_021U = tc[5]
    count_021C = tc[7]
    count_030T = tc[9]

    total = count_021D + count_021U + count_021C + count_030T
    if total == 0:
        total = 1  # avoid division by zero

    return {
        "ratio_030T": count_030T / total,
        "ratio_021D": count_021D / total,
        "ratio_021U": count_021U / total,
        "ratio_021C": count_021C / total,
        "triad_030T": count_030T,
        "triad_021D": count_021D,
        "triad_021U": count_021U,
        "triad_021C": count_021C,
    }


# ============================================================
# PHASE D: CLUSTERING COMPARISON
# ============================================================

FEATURE_SETS = {
    "weighted_motif_only": [
        "ffl_intensity_mean", "ffl_intensity_median", "ffl_intensity_std",
        "ffl_intensity_q25", "ffl_intensity_q75",
        "ffl_coherent_frac", "ffl_path_dom_mean", "ffl_path_dom_std",
        "ffl_asymmetry_mean", "ffl_coherence_onnela_mean", "ffl_coherence_onnela_std",
        "chain_intensity_mean", "chain_sign_agree_frac",
        "fanout_intensity_mean", "fanout_sign_agree_frac",
        "fanin_intensity_mean", "fanin_sign_agree_frac",
        "neg_edge_frac", "edge_weight_kurtosis",
    ],
    "binary_motif_only": [
        "ratio_030T", "ratio_021D", "ratio_021U", "ratio_021C",
    ],
    "graph_stats_only": [
        "n_nodes", "n_edges", "density", "mean_degree",
        "max_degree", "diameter_approx", "n_layers", "edge_weight_mean",
    ],
    "weighted_plus_binary": [
        "ffl_intensity_mean", "ffl_intensity_median", "ffl_intensity_std",
        "ffl_intensity_q25", "ffl_intensity_q75",
        "ffl_coherent_frac", "ffl_path_dom_mean", "ffl_path_dom_std",
        "ffl_asymmetry_mean", "ffl_coherence_onnela_mean", "ffl_coherence_onnela_std",
        "chain_intensity_mean", "chain_sign_agree_frac",
        "fanout_intensity_mean", "fanout_sign_agree_frac",
        "fanin_intensity_mean", "fanin_sign_agree_frac",
        "neg_edge_frac", "edge_weight_kurtosis",
        "ratio_030T", "ratio_021D", "ratio_021U", "ratio_021C",
    ],
    "all_combined": [
        "ffl_intensity_mean", "ffl_intensity_median", "ffl_intensity_std",
        "ffl_intensity_q25", "ffl_intensity_q75",
        "ffl_coherent_frac", "ffl_path_dom_mean", "ffl_path_dom_std",
        "ffl_asymmetry_mean", "ffl_coherence_onnela_mean", "ffl_coherence_onnela_std",
        "chain_intensity_mean", "chain_sign_agree_frac",
        "fanout_intensity_mean", "fanout_sign_agree_frac",
        "fanin_intensity_mean", "fanin_sign_agree_frac",
        "neg_edge_frac", "edge_weight_kurtosis",
        "ratio_030T", "ratio_021D", "ratio_021U", "ratio_021C",
        "n_nodes", "n_edges", "density", "mean_degree",
        "max_degree", "diameter_approx", "n_layers", "edge_weight_mean",
    ],
}


def cluster_and_nmi(X: np.ndarray, labels: np.ndarray, K: int) -> float:
    """Run spectral clustering and return NMI."""
    try:
        sc = SpectralClustering(
            n_clusters=K, affinity="rbf", random_state=SEED, n_init=10
        )
        pred = sc.fit_predict(X)
        return float(normalized_mutual_info_score(labels, pred))
    except Exception:
        return 0.0


def run_clustering_comparison(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    feature_set_name: str,
    k_values: list[int],
) -> dict:
    """Run spectral clustering for each K, compute NMI/ARI."""
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    results = {}
    predictions = {}
    for K in k_values:
        try:
            sc = SpectralClustering(
                n_clusters=K, affinity="rbf", random_state=SEED, n_init=10
            )
            pred = sc.fit_predict(X)
            nmi = float(normalized_mutual_info_score(labels, pred))
            ari = float(adjusted_rand_score(labels, pred))
            results[str(K)] = {"nmi": nmi, "ari": ari}
            predictions[str(K)] = pred.tolist()
        except Exception as e:
            logger.warning(f"Clustering failed for {feature_set_name} K={K}: {e}")
            results[str(K)] = {"nmi": 0.0, "ari": 0.0}
            predictions[str(K)] = [0] * len(labels)

    best_k = max(results, key=lambda k: results[k]["nmi"])
    return {
        "results_by_k": results,
        "best_k": int(best_k),
        "best_nmi": results[best_k]["nmi"],
        "best_ari": results[best_k]["ari"],
        "n_features": int(feature_matrix.shape[1]),
        "n_graphs": int(feature_matrix.shape[0]),
        "best_k_predictions": predictions[best_k],
    }


# ============================================================
# PHASE D2: PERMUTATION TEST FOR NMI DIFFERENCES
# ============================================================

def permutation_test_nmi(
    X1: np.ndarray,
    X2: np.ndarray,
    labels: np.ndarray,
    n_perms: int = 200,
    K: int = 8,
) -> dict:
    """Test if NMI(X1) > NMI(X2) is significant by shuffling labels."""
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X1s = scaler1.fit_transform(X1)
    X2s = scaler2.fit_transform(X2)

    nmi1_real = cluster_and_nmi(X1s, labels, K)
    nmi2_real = cluster_and_nmi(X2s, labels, K)
    observed_diff = nmi1_real - nmi2_real

    rng = np.random.RandomState(SEED)
    count_exceed = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(labels)
        nmi1_perm = cluster_and_nmi(X1s, shuffled, K)
        nmi2_perm = cluster_and_nmi(X2s, shuffled, K)
        if (nmi1_perm - nmi2_perm) >= observed_diff:
            count_exceed += 1

    p_value = (count_exceed + 1) / (n_perms + 1)
    return {
        "nmi_X1": nmi1_real,
        "nmi_X2": nmi2_real,
        "observed_diff": observed_diff,
        "p_value": p_value,
        "n_permutations": n_perms,
    }


# ============================================================
# PHASE E: DISCRIMINATIVE WEIGHT PATTERN ANALYSIS
# ============================================================

def discriminative_analysis(feature_dict_list: list[dict], labels: list[str], feature_names: list[str]) -> dict:
    """ANOVA F-test per feature; eta-squared effect size."""
    unique_domains = sorted(set(labels))
    labels_arr = np.array(labels)
    results = {}

    for feat_name in feature_names:
        try:
            vals = np.array([fd[feat_name] for fd in feature_dict_list], dtype=float)
        except (KeyError, TypeError):
            continue

        groups = [vals[labels_arr == d] for d in unique_domains]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue

        try:
            f_stat, p_val = scipy.stats.f_oneway(*groups)
        except Exception:
            continue

        if np.isnan(f_stat):
            continue

        # eta-squared = SS_between / SS_total
        grand_mean = np.mean(vals)
        ss_between = sum(len(g_arr) * (np.mean(g_arr) - grand_mean) ** 2 for g_arr in groups)
        ss_total = np.sum((vals - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        per_domain = {d: float(np.mean(vals[labels_arr == d]))
                      for d in unique_domains if np.sum(labels_arr == d) > 0}

        results[feat_name] = {
            "F_statistic": float(f_stat),
            "p_value": float(p_val),
            "eta_squared": float(eta_sq),
            "per_domain_means": per_domain,
        }

    ranked = sorted(results.items(), key=lambda x: x[1]["eta_squared"], reverse=True)
    return {"all_features": results, "top_5": dict(ranked[:5])}


# ============================================================
# PHASE F: WEIGHT-TOPOLOGY INTERACTION ANALYSIS
# ============================================================

def weight_topology_interaction(per_graph_features: list[dict], domain_labels: list[str]) -> dict:
    """Correlation between FFL count and mean FFL intensity per domain."""
    results = {}
    for domain in sorted(set(domain_labels)):
        mask = [i for i, d in enumerate(domain_labels) if d == domain]
        counts = [per_graph_features[i]["ffl_count"] for i in mask]
        intensities = [per_graph_features[i]["ffl_intensity_mean"] for i in mask]

        if len(counts) >= 5:
            try:
                r, p = scipy.stats.pearsonr(counts, intensities)
                results[domain] = {
                    "pearson_r": float(r),
                    "p_value": float(p),
                    "n": len(counts),
                    "interpretation": (
                        "complementary" if abs(r) < 0.3
                        else "correlated" if r > 0
                        else "anticorrelated"
                    ),
                }
            except Exception:
                results[domain] = {"pearson_r": 0.0, "p_value": 1.0, "n": len(counts),
                                   "interpretation": "insufficient_data"}

    # Overall correlation
    all_counts = [f["ffl_count"] for f in per_graph_features]
    all_intensities = [f["ffl_intensity_mean"] for f in per_graph_features]
    try:
        r_all, p_all = scipy.stats.pearsonr(all_counts, all_intensities)
        results["overall"] = {"pearson_r": float(r_all), "p_value": float(p_all)}
    except Exception:
        results["overall"] = {"pearson_r": 0.0, "p_value": 1.0}

    return results


# ============================================================
# PHASE G: SIGN COHERENCE ANALYSIS PER DOMAIN
# ============================================================

def sign_coherence_analysis(per_graph_features: list[dict], domain_labels: list[str]) -> dict:
    """Aggregate sign coherence patterns per domain."""
    results = {}
    for domain in sorted(set(domain_labels)):
        mask = [i for i, d in enumerate(domain_labels) if d == domain]
        coherent_fracs = [per_graph_features[i]["ffl_coherent_frac"] for i in mask]
        ffl_counts = [per_graph_features[i]["ffl_count"] for i in mask]

        # Merge sign pattern counts across domain
        merged_patterns = Counter()
        for i in mask:
            for pat, cnt in per_graph_features[i].get("sign_pattern_counts", {}).items():
                merged_patterns[pat] += cnt

        total_patterns = sum(merged_patterns.values())
        pattern_dist = {k: v / max(total_patterns, 1) for k, v in merged_patterns.most_common(8)}

        results[domain] = {
            "coherent_frac_mean": float(np.mean(coherent_fracs)) if coherent_fracs else 0.0,
            "coherent_frac_std": float(np.std(coherent_fracs)) if coherent_fracs else 0.0,
            "sign_pattern_distribution": pattern_dist,
            "n_ffls_total": int(sum(ffl_counts)),
            "n_graphs": len(mask),
        }

    return results


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main():
    t_start = time.time()
    np.random.seed(SEED)

    # ---- PHASE A: Load graphs ----
    logger.info("=== PHASE A: Loading graphs ===")
    all_graphs = load_all_graphs_with_weights()
    if not all_graphs:
        logger.error("No graphs loaded!")
        sys.exit(1)

    n_graphs = len(all_graphs)
    domains = [g["domain"] for g in all_graphs]
    domain_counts = Counter(domains)
    logger.info(f"Loaded {n_graphs} graphs across {len(domain_counts)} domains")
    logger.info(f"Domain distribution: {dict(domain_counts)}")

    # ---- PHASE B+C: Process all graphs ----
    logger.info("=== PHASE B+C: Processing graphs (motif enumeration + features) ===")
    per_graph_features = []
    per_graph_binary_ratios = []

    for idx, gr in enumerate(all_graphs):
        try:
            features = process_one_graph(gr, idx)
            per_graph_features.append(features)

            binary_ratios = compute_binary_motif_ratios(gr["graph"])
            per_graph_binary_ratios.append(binary_ratios)
        except MemoryError:
            logger.error(f"OOM on graph {idx}, stopping early")
            break
        except Exception:
            logger.exception(f"Error processing graph {idx}")
            # Append zeros as fallback
            per_graph_features.append({k: 0.0 for k in FEATURE_SETS["all_combined"]})
            per_graph_features[-1]["ffl_count"] = 0
            per_graph_features[-1]["sign_pattern_counts"] = {}
            per_graph_binary_ratios.append({
                "ratio_030T": 0.0, "ratio_021D": 0.0,
                "ratio_021U": 0.0, "ratio_021C": 0.0,
                "triad_030T": 0, "triad_021D": 0,
                "triad_021U": 0, "triad_021C": 0,
            })

        if (idx + 1) % 20 == 0:
            gc.collect()

    # Trim to match processed count
    n_processed = len(per_graph_features)
    domains = domains[:n_processed]
    all_graphs = all_graphs[:n_processed]
    logger.info(f"Processed {n_processed}/{n_graphs} graphs")

    t_processing = time.time() - t_start
    logger.info(f"Processing time: {t_processing:.1f}s")

    # FREE graph objects to reclaim memory before clustering/perm tests
    for gr in all_graphs:
        gr["graph"] = None  # Drop igraph object, keep metadata
    gc.collect()
    logger.info("Freed graph objects to reclaim memory")

    # ---- Merge binary ratios into feature dicts ----
    for i in range(n_processed):
        per_graph_features[i].update(per_graph_binary_ratios[i])

    # ---- PHASE D: Clustering comparison ----
    logger.info("=== PHASE D: Clustering comparison ===")
    labels_arr = np.array(domains)
    clustering_results = {}

    for fs_name, fs_cols in FEATURE_SETS.items():
        try:
            X = np.array([[fd.get(c, 0.0) for c in fs_cols] for fd in per_graph_features])
            # Replace NaN/inf
            X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
            result = run_clustering_comparison(X, labels_arr, fs_name, CLUSTER_K_VALUES)
            clustering_results[fs_name] = result
            logger.info(f"  {fs_name}: best_k={result['best_k']}, "
                        f"NMI={result['best_nmi']:.4f}, ARI={result['best_ari']:.4f}")
        except Exception:
            logger.exception(f"Clustering failed for {fs_name}")
            clustering_results[fs_name] = {
                "results_by_k": {}, "best_k": 0, "best_nmi": 0.0, "best_ari": 0.0,
                "n_features": len(fs_cols), "n_graphs": n_processed,
            }

    # ---- PHASE D2: Permutation tests ----
    logger.info("=== PHASE D2: Permutation tests ===")
    # Determine actual permutation count (fallback to fewer if needed)
    actual_perms = min(N_PERMUTATIONS, 200) if n_processed < 50 else N_PERMUTATIONS
    # Use best K from the best feature set
    best_overall_k = 8  # default
    if clustering_results.get("weighted_motif_only", {}).get("best_k"):
        best_overall_k = clustering_results["weighted_motif_only"]["best_k"]

    perm_results = {}
    comparisons = [
        ("weighted_vs_binary", "weighted_motif_only", "binary_motif_only"),
        ("weighted_vs_graph_stats", "weighted_motif_only", "graph_stats_only"),
        ("combined_vs_best_single", "all_combined", "weighted_motif_only"),
        ("weighted_plus_binary_vs_binary", "weighted_plus_binary", "binary_motif_only"),
    ]

    for comp_name, fs1_name, fs2_name in comparisons:
        try:
            fs1_cols = FEATURE_SETS[fs1_name]
            fs2_cols = FEATURE_SETS[fs2_name]
            X1 = np.array([[fd.get(c, 0.0) for c in fs1_cols] for fd in per_graph_features])
            X2 = np.array([[fd.get(c, 0.0) for c in fs2_cols] for fd in per_graph_features])
            X1 = np.nan_to_num(X1, nan=0.0, posinf=100.0, neginf=-100.0)
            X2 = np.nan_to_num(X2, nan=0.0, posinf=100.0, neginf=-100.0)

            result = permutation_test_nmi(X1, X2, labels_arr, n_perms=actual_perms, K=best_overall_k)
            perm_results[comp_name] = result
            logger.info(f"  {comp_name}: diff={result['observed_diff']:.4f}, p={result['p_value']:.4f}")
        except Exception:
            logger.exception(f"Permutation test failed for {comp_name}")
            perm_results[comp_name] = {
                "nmi_X1": 0.0, "nmi_X2": 0.0,
                "observed_diff": 0.0, "p_value": 1.0, "n_permutations": 0,
            }
        gc.collect()  # Free memory between permutation tests

    # ---- PHASE E: Discriminative analysis ----
    logger.info("=== PHASE E: Discriminative feature analysis ===")
    all_feature_names = FEATURE_SETS["all_combined"]
    disc_results = discriminative_analysis(per_graph_features, domains, all_feature_names)
    if disc_results.get("top_5"):
        for fname, fdata in disc_results["top_5"].items():
            logger.info(f"  Top feature: {fname}, eta^2={fdata['eta_squared']:.4f}, "
                        f"F={fdata['F_statistic']:.2f}, p={fdata['p_value']:.2e}")

    # ---- PHASE F: Weight-topology interaction ----
    logger.info("=== PHASE F: Weight-topology interaction ===")
    wt_interaction = weight_topology_interaction(per_graph_features, domains)
    if "overall" in wt_interaction:
        logger.info(f"  Overall count-intensity correlation: "
                    f"r={wt_interaction['overall']['pearson_r']:.4f}, "
                    f"p={wt_interaction['overall']['p_value']:.4f}")

    # ---- PHASE G: Sign coherence analysis ----
    logger.info("=== PHASE G: Sign coherence analysis ===")
    sign_coh = sign_coherence_analysis(per_graph_features, domains)
    for domain, data in sign_coh.items():
        logger.info(f"  {domain}: coherent_frac={data['coherent_frac_mean']:.4f}, "
                    f"n_ffls={data['n_ffls_total']}")

    # ---- Per-domain aggregated stats ----
    logger.info("=== Computing per-domain aggregated stats ===")
    per_domain_agg = {}
    for domain in sorted(set(domains)):
        mask = [i for i, d in enumerate(domains) if d == domain]
        domain_feats = [per_graph_features[i] for i in mask]
        agg = {}
        for feat_name in all_feature_names:
            vals = [df.get(feat_name, 0.0) for df in domain_feats]
            vals = [v for v in vals if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
            if vals:
                agg[feat_name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        per_domain_agg[domain] = agg

    # ============================================================
    # OUTPUT: method_out.json (exp_gen_sol_out schema)
    # ============================================================
    logger.info("=== Building output ===")
    t_total = time.time() - t_start

    # Build per-graph examples for schema compliance
    examples = []
    for i in range(n_processed):
        gr = all_graphs[i]
        feats = per_graph_features[i]

        # Clean sign_pattern_counts for JSON serialization
        clean_feats = {}
        for k, v in feats.items():
            if k == "sign_pattern_counts":
                clean_feats[k] = {str(pk): pv for pk, pv in v.items()}
            elif isinstance(v, (int, float, str, bool)):
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    clean_feats[k] = 0.0
                else:
                    clean_feats[k] = v
            else:
                clean_feats[k] = v

        output_record = {
            "per_graph_features": clean_feats,
            "binary_motif_ratios": {
                k: per_graph_binary_ratios[i][k]
                for k in ["ratio_030T", "ratio_021D", "ratio_021U", "ratio_021C",
                           "triad_030T", "triad_021D", "triad_021U", "triad_021C"]
            },
        }

        # Build predict_ fields from clustering results (best-K assignments)
        predict_weighted = "cluster_" + str(
            clustering_results.get("weighted_motif_only", {}).get("best_k_predictions", [0]*n_processed)[i]
        )
        predict_binary = "cluster_" + str(
            clustering_results.get("binary_motif_only", {}).get("best_k_predictions", [0]*n_processed)[i]
        )
        predict_combined = "cluster_" + str(
            clustering_results.get("all_combined", {}).get("best_k_predictions", [0]*n_processed)[i]
        )

        example = {
            "input": gr["prompt"],
            "output": json.dumps(output_record),
            "metadata_fold": gr["domain"],
            "metadata_slug": gr["slug"],
            "metadata_model_correct": gr["model_correct"],
            "predict_weighted_motif": predict_weighted,
            "predict_binary_baseline": predict_binary,
            "predict_all_combined": predict_combined,
        }
        examples.append(example)

    output = {
        "metadata": {
            "method_name": "weighted_motif_intensity_sign_coherence",
            "description": (
                "Weighted motif features (Onnela intensity, sign coherence, path dominance, "
                "weight asymmetry) for FFL and other 3-node DAG motifs across attribution graphs, "
                "with spectral clustering comparison against capability domain labels."
            ),
            "parameters": {
                "prune_percentile": PRUNE_PERCENTILE,
                "min_nodes": MIN_NODES,
                "seed": SEED,
                "n_permutations": actual_perms,
                "cluster_k_values": CLUSTER_K_VALUES,
            },
            "n_graphs_processed": n_processed,
            "n_domains": len(set(domains)),
            "domain_counts": dict(Counter(domains)),
            "total_runtime_seconds": round(t_total, 1),
            "processing_time_seconds": round(t_processing, 1),
            "clustering_comparison": clustering_results,
            "permutation_tests": perm_results,
            "discriminative_features": disc_results,
            "weight_topology_interaction": wt_interaction,
            "sign_coherence_analysis": sign_coh,
            "per_domain_aggregated": per_domain_agg,
            "feature_set_definitions": {
                k: v for k, v in FEATURE_SETS.items()
            },
        },
        "datasets": [
            {
                "dataset": "neuronpedia_attribution_graphs_v3",
                "examples": examples,
            }
        ],
    }

    # Write output
    output_json = json.dumps(output, indent=2, default=str)
    OUTPUT_FILE.write_text(output_json)
    logger.info(f"Output written to {OUTPUT_FILE} ({len(output_json) / 1e6:.1f} MB)")
    logger.info(f"Total runtime: {t_total:.1f}s")

    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Graphs: {n_processed}")
    for fs_name, res in clustering_results.items():
        logger.info(f"  {fs_name}: NMI={res['best_nmi']:.4f} (K={res['best_k']})")
    if perm_results:
        for comp_name, res in perm_results.items():
            logger.info(f"  Perm {comp_name}: diff={res['observed_diff']:.4f}, p={res['p_value']:.4f}")


if __name__ == "__main__":
    main()
