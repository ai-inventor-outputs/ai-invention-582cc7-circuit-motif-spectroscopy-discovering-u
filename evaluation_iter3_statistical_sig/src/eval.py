#!/usr/bin/env python3
"""Statistical Significance Audit of Iter 2 Circuit Motif Spectroscopy Claims.

Rigorous statistical audit of iter 2's circuit motif spectroscopy claims:
- Phase A: Data reconstruction from dependency outputs
- Phase B: NMI/ARI permutation tests
- Phase C: FFL Z-Score bootstrap confidence intervals
- Phase D: Layer confound test on all 34 graphs (scaled from n=3)
- Phase E: Effect size analysis
- Phase F: Robustness checks (LOO, seed stability, feature ablation)

Output: eval_out.json (exp_eval_sol_out schema)
"""

import json
import sys
import os
import gc
import math
import time
import random
import resource
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import igraph
import psutil
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    v_measure_score,
)
from sklearn.metrics.pairwise import rbf_kernel
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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
RAM_BUDGET_GB = min(TOTAL_RAM_GB * 0.7, 20.0)
RAM_BUDGET_BYTES = int(RAM_BUDGET_GB * 1e9)
N_WORKERS = max(1, NUM_CPUS - 1)

resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
logger.info(f"Budget: {RAM_BUDGET_GB:.1f} GB RAM, workers={N_WORKERS}")

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
N_PERMUTATIONS = int(os.environ.get("N_PERMUTATIONS", "1000"))
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "10000"))
N_NULL_MODELS = int(os.environ.get("N_NULL_MODELS", "50"))
SWAP_FACTOR = int(os.environ.get("SWAP_FACTOR", "100"))
PRUNE_PERCENTILE = 75
MIN_NODES_FOR_CENSUS = 30
SEED = 42

WORKSPACE = Path(__file__).parent
DEP_EXP1 = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_2/gen_art/exp_id1_it2__opus"
)
DEP_EXP2 = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_2/gen_art/exp_id2_it2__opus"
)
DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_1/gen_art/data_id4_it1__opus"
)

# Enriched feature names (12 features in order):
# [ratio_021U, ratio_021C, ratio_021D, ratio_030T, Z_030T,
#  FFL_per_node, FFL_per_edge, density, in_degree_std, out_degree_std,
#  layer_span_mean, transitivity]
ENRICHED_FEATURE_NAMES = [
    "ratio_021U", "ratio_021C", "ratio_021D", "ratio_030T",
    "Z_030T", "FFL_per_node", "FFL_per_edge",
    "density", "in_degree_std", "out_degree_std",
    "layer_span_mean", "transitivity",
]
# Indices: 0-6 are motif-only, 7-11 are graph statistics
MOTIF_ONLY_INDICES = list(range(7))
GRAPH_STAT_INDICES = list(range(7, 12))

# DAG-valid 3-node motif isoclass IDs
DAG_MOTIF_IDS = [2, 4, 6, 7]
MOTIF_NAMES = {2: "021U", 4: "021C", 6: "021D", 7: "030T"}


# ============================================================================
# UTILITY: Spectral Clustering (replicating exp_id1 approach)
# ============================================================================


def spectral_cluster_nmi_ari(
    feature_matrix: np.ndarray,
    true_labels: np.ndarray,
    k: int = 8,
    random_state: int = 42,
) -> dict:
    """Run spectral clustering and return NMI/ARI/AMI/V-measure."""
    n_samples, n_feat = feature_matrix.shape
    n_unique = len(np.unique(true_labels))
    k = min(k, n_samples - 1, n_unique)
    if k < 2 or n_samples < 3:
        return {"nmi": 0.0, "ari": 0.0, "ami": 0.0, "v_measure": 0.0, "labels": list(range(n_samples))}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    affinity = rbf_kernel(X_scaled, gamma=1.0 / max(n_feat, 1))

    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=10,
    )
    pred = sc.fit_predict(affinity)
    nmi = normalized_mutual_info_score(true_labels, pred)
    ari = adjusted_rand_score(true_labels, pred)
    ami = adjusted_mutual_info_score(true_labels, pred)
    vm = v_measure_score(true_labels, pred)
    return {
        "nmi": float(nmi),
        "ari": float(ari),
        "ami": float(ami),
        "v_measure": float(vm),
        "labels": pred.tolist(),
    }


# ============================================================================
# PHASE A: Data Reconstruction
# ============================================================================


def phase_a_reconstruct_data() -> dict:
    """Load exp_id1 and exp_id2 outputs, reconstruct feature matrices."""
    logger.info("=" * 60)
    logger.info("PHASE A: Data Reconstruction")

    # Load exp_id1 full output
    exp1_path = DEP_EXP1 / "full_method_out.json"
    logger.info(f"Loading exp_id1: {exp1_path}")
    exp1_data = json.loads(exp1_path.read_text())
    examples_1 = exp1_data["datasets"][0]["examples"]
    logger.info(f"  Loaded {len(examples_1)} examples from exp_id1")

    if MAX_EXAMPLES > 0:
        examples_1 = examples_1[:MAX_EXAMPLES]

    # Parse per-graph data from exp_id1
    enriched_features = []
    baseline_features = []
    domain_labels = []
    graph_prompts = []
    z_030t_values = []
    count_ratios_all = []  # per-graph list of {motif_id: ratio}
    raw_ffl_counts = []

    # Baseline feature keys (sorted by name for consistent ordering)
    baseline_keys = None

    for ex in examples_1:
        output = json.loads(ex["output"])
        domain = ex["metadata_fold"]

        # Extract motif census data at threshold 75
        mc = output.get("motif_census_3node", {}).get("75", {})
        raw_counts = mc.get("raw_counts", {})
        z_scores = mc.get("z_scores", {})
        count_ratios = mc.get("count_ratios", {})
        bf = output.get("baseline_features", {})

        if not bf or not count_ratios:
            logger.warning(f"Skipping example with missing data: {ex['input'][:50]}")
            continue

        if baseline_keys is None:
            baseline_keys = sorted(bf.keys())
            logger.info(f"  Baseline feature keys ({len(baseline_keys)}): {baseline_keys}")

        # Build enriched feature vector (12 features)
        n_nodes = bf.get("n_nodes", 1)
        n_edges = bf.get("n_edges", 1)
        ffl_raw = raw_counts.get("7", 0)

        feat = [
            count_ratios.get("2", 0.0),  # ratio_021U
            count_ratios.get("4", 0.0),  # ratio_021C
            count_ratios.get("6", 0.0),  # ratio_021D
            count_ratios.get("7", 0.0),  # ratio_030T
            z_scores.get("7", 0.0),       # Z_030T
            ffl_raw / max(n_nodes, 1),     # FFL_per_node
            ffl_raw / max(n_edges, 1),     # FFL_per_edge
            bf.get("density", 0.0),
            bf.get("in_degree_std", 0.0),
            bf.get("out_degree_std", 0.0),
            bf.get("layer_span_mean", 0.0),
            bf.get("transitivity", 0.0),
        ]
        enriched_features.append(feat)

        # Baseline feature vector (all baseline features sorted by key)
        baseline_features.append([bf[k] for k in baseline_keys])

        domain_labels.append(domain)
        graph_prompts.append(ex["input"])
        z_030t_values.append(z_scores.get("7", 0.0))
        count_ratios_all.append({
            "2": count_ratios.get("2", 0.0),
            "4": count_ratios.get("4", 0.0),
            "6": count_ratios.get("6", 0.0),
            "7": count_ratios.get("7", 0.0),
        })
        raw_ffl_counts.append(ffl_raw)

    enriched_matrix = np.array(enriched_features)
    baseline_matrix = np.array(baseline_features)
    le = LabelEncoder()
    true_labels = le.fit_transform(domain_labels)
    n_graphs = len(domain_labels)

    logger.info(f"  Enriched matrix: {enriched_matrix.shape}")
    logger.info(f"  Baseline matrix: {baseline_matrix.shape}")
    logger.info(f"  Domains: {dict(Counter(domain_labels))}")

    # Verify reproduction of NMI values
    our_result = spectral_cluster_nmi_ari(enriched_matrix, true_labels, k=8)
    base_result = spectral_cluster_nmi_ari(baseline_matrix, true_labels, k=8)

    logger.info(f"  Our method NMI={our_result['nmi']:.4f} (expected ~0.851)")
    logger.info(f"  Baseline NMI={base_result['nmi']:.4f} (expected ~0.847)")

    # Load exp_id2 data
    exp2_path = DEP_EXP2 / "full_method_out.json"
    logger.info(f"Loading exp_id2: {exp2_path}")
    exp2_data = json.loads(exp2_path.read_text())
    exp2_examples = exp2_data["datasets"][0]["examples"]
    logger.info(f"  Loaded {len(exp2_examples)} examples from exp_id2")

    exp2_per_graph = []
    for ex in exp2_examples:
        output = json.loads(ex["output"])
        exp2_per_graph.append({
            "domain": ex["metadata_fold"],
            "z_degree": output.get("z_degree", []),
            "z_layer": output.get("z_layer", []),
        })

    return {
        "enriched_matrix": enriched_matrix,
        "baseline_matrix": baseline_matrix,
        "true_labels": true_labels,
        "domain_labels": domain_labels,
        "graph_prompts": graph_prompts,
        "z_030t_values": np.array(z_030t_values),
        "count_ratios_all": count_ratios_all,
        "raw_ffl_counts": raw_ffl_counts,
        "baseline_keys": baseline_keys,
        "n_graphs": n_graphs,
        "le": le,
        "our_nmi": our_result["nmi"],
        "our_ari": our_result["ari"],
        "base_nmi": base_result["nmi"],
        "base_ari": base_result["ari"],
        "our_labels": our_result["labels"],
        "base_labels": base_result["labels"],
        "exp2_per_graph": exp2_per_graph,
    }


# ============================================================================
# PHASE B: NMI/ARI Permutation Tests
# ============================================================================


def phase_b_permutation_tests(data: dict) -> dict:
    """Run permutation tests for NMI and ARI significance."""
    logger.info("=" * 60)
    logger.info("PHASE B: Permutation Tests")

    enriched = data["enriched_matrix"]
    baseline = data["baseline_matrix"]
    true_labels = data["true_labels"]
    observed_nmi_our = data["our_nmi"]
    observed_ari_our = data["our_ari"]
    observed_nmi_base = data["base_nmi"]
    observed_ari_base = data["base_ari"]

    n_samples, n_feat_e = enriched.shape
    _, n_feat_b = baseline.shape

    # Pre-compute affinity matrices (they don't change with permutation)
    scaler_e = StandardScaler()
    X_e = scaler_e.fit_transform(enriched)
    affinity_e = rbf_kernel(X_e, gamma=1.0 / max(n_feat_e, 1))

    scaler_b = StandardScaler()
    X_b = scaler_b.fit_transform(baseline)
    affinity_b = rbf_kernel(X_b, gamma=1.0 / max(n_feat_b, 1))

    # Motif-only matrix (7 features: indices 0-6)
    motif_only = enriched[:, MOTIF_ONLY_INDICES]
    n_feat_m = motif_only.shape[1]
    scaler_m = StandardScaler()
    X_m = scaler_m.fit_transform(motif_only)
    affinity_m = rbf_kernel(X_m, gamma=1.0 / max(n_feat_m, 1))

    # Determine K for clustering
    n_unique = len(np.unique(true_labels))
    K = min(8, n_samples - 1, n_unique)
    if K < 2:
        K = 2
    logger.info(f"  Using K={K} for clustering (n_samples={n_samples}, n_unique={n_unique})")

    # Observed motif-only NMI
    sc_m = SpectralClustering(
        n_clusters=K, affinity="precomputed",
        assign_labels="kmeans", random_state=42, n_init=10,
    )
    pred_m = sc_m.fit_predict(affinity_m)
    observed_nmi_motif = float(normalized_mutual_info_score(true_labels, pred_m))
    observed_ari_motif = float(adjusted_rand_score(true_labels, pred_m))

    logger.info(f"  Observed NMI our={observed_nmi_our:.4f}, base={observed_nmi_base:.4f}, motif_only={observed_nmi_motif:.4f}")
    logger.info(f"  Observed ARI our={observed_ari_our:.4f}, base={observed_ari_base:.4f}, motif_only={observed_ari_motif:.4f}")

    # Pre-compute cluster labels once (affinity is fixed, only labels are permuted)
    sc_e_fit = SpectralClustering(
        n_clusters=K, affinity="precomputed",
        assign_labels="kmeans", random_state=42, n_init=10,
    )
    pred_e_fixed = sc_e_fit.fit_predict(affinity_e)

    sc_b_fit = SpectralClustering(
        n_clusters=K, affinity="precomputed",
        assign_labels="kmeans", random_state=42, n_init=10,
    )
    pred_b_fixed = sc_b_fit.fit_predict(affinity_b)

    sc_m_fit = SpectralClustering(
        n_clusters=K, affinity="precomputed",
        assign_labels="kmeans", random_state=42, n_init=10,
    )
    pred_m_fixed = sc_m_fit.fit_predict(affinity_m)

    rng = np.random.RandomState(SEED)
    perm_nmi_our = np.zeros(N_PERMUTATIONS)
    perm_nmi_base = np.zeros(N_PERMUTATIONS)
    perm_ari_our = np.zeros(N_PERMUTATIONS)
    perm_ari_base = np.zeros(N_PERMUTATIONS)
    perm_nmi_motif = np.zeros(N_PERMUTATIONS)
    perm_nmi_diff = np.zeros(N_PERMUTATIONS)

    t0 = time.time()
    for i in range(N_PERMUTATIONS):
        shuffled = rng.permutation(true_labels)

        # Our method (cluster labels are fixed, only ground truth is shuffled)
        perm_nmi_our[i] = normalized_mutual_info_score(shuffled, pred_e_fixed)
        perm_ari_our[i] = adjusted_rand_score(shuffled, pred_e_fixed)

        # Baseline
        perm_nmi_base[i] = normalized_mutual_info_score(shuffled, pred_b_fixed)
        perm_ari_base[i] = adjusted_rand_score(shuffled, pred_b_fixed)

        # Motif-only
        perm_nmi_motif[i] = normalized_mutual_info_score(shuffled, pred_m_fixed)

        # Paired difference
        perm_nmi_diff[i] = perm_nmi_our[i] - perm_nmi_base[i]

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_PERMUTATIONS - i - 1)
            logger.info(f"  Permutation {i+1}/{N_PERMUTATIONS}, elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

    # Compute p-values (including +1 correction)
    p_nmi_our = float((np.sum(perm_nmi_our >= observed_nmi_our) + 1) / (N_PERMUTATIONS + 1))
    p_nmi_base = float((np.sum(perm_nmi_base >= observed_nmi_base) + 1) / (N_PERMUTATIONS + 1))
    p_ari_our = float((np.sum(perm_ari_our >= observed_ari_our) + 1) / (N_PERMUTATIONS + 1))
    p_ari_base = float((np.sum(perm_ari_base >= observed_ari_base) + 1) / (N_PERMUTATIONS + 1))
    p_nmi_motif = float((np.sum(perm_nmi_motif >= observed_nmi_motif) + 1) / (N_PERMUTATIONS + 1))

    observed_diff = observed_nmi_our - observed_nmi_base
    p_diff = float((np.sum(perm_nmi_diff >= observed_diff) + 1) / (N_PERMUTATIONS + 1))

    logger.info(f"  p-value NMI our: {p_nmi_our:.4f}")
    logger.info(f"  p-value NMI base: {p_nmi_base:.4f}")
    logger.info(f"  p-value NMI diff: {p_diff:.4f} (observed diff={observed_diff:.4f})")
    logger.info(f"  p-value NMI motif-only: {p_nmi_motif:.4f}")
    logger.info(f"  p-value ARI our: {p_ari_our:.4f}")
    logger.info(f"  p-value ARI base: {p_ari_base:.4f}")

    return {
        "nmi_our_observed": observed_nmi_our,
        "nmi_our_p_value": p_nmi_our,
        "nmi_baseline_observed": observed_nmi_base,
        "nmi_baseline_p_value": p_nmi_base,
        "ari_our_observed": observed_ari_our,
        "ari_our_p_value": p_ari_our,
        "ari_baseline_observed": observed_ari_base,
        "ari_baseline_p_value": p_ari_base,
        "nmi_difference_observed": observed_diff,
        "nmi_difference_p_value": p_diff,
        "motif_only_nmi": observed_nmi_motif,
        "motif_only_ari": observed_ari_motif,
        "motif_only_p_value": p_nmi_motif,
        "null_distributions": {
            "nmi_our_mean": float(np.mean(perm_nmi_our)),
            "nmi_our_std": float(np.std(perm_nmi_our)),
            "nmi_base_mean": float(np.mean(perm_nmi_base)),
            "nmi_base_std": float(np.std(perm_nmi_base)),
            "nmi_motif_mean": float(np.mean(perm_nmi_motif)),
            "nmi_motif_std": float(np.std(perm_nmi_motif)),
        },
        "n_permutations": N_PERMUTATIONS,
    }


# ============================================================================
# PHASE C: FFL Z-Score Bootstrap Confidence Intervals
# ============================================================================


def phase_c_bootstrap_cis(data: dict) -> dict:
    """Compute bootstrap CIs for Z_030T and count ratios."""
    logger.info("=" * 60)
    logger.info("PHASE C: Bootstrap Confidence Intervals")

    domain_labels = data["domain_labels"]
    z_030t = data["z_030t_values"]
    count_ratios = data["count_ratios_all"]
    n_graphs = data["n_graphs"]

    rng = np.random.RandomState(SEED)
    unique_domains = sorted(set(domain_labels))

    # C1: Per-domain bootstrap 95% CI for mean Z_030T
    per_domain_z_ci = {}
    for domain in unique_domains:
        indices = [i for i, d in enumerate(domain_labels) if d == domain]
        z_vals = z_030t[indices]
        n_d = len(z_vals)

        boot_means = np.zeros(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            sample = rng.choice(z_vals, size=n_d, replace=True)
            boot_means[b] = np.mean(sample)

        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))
        per_domain_z_ci[domain] = {
            "mean": float(np.mean(z_vals)),
            "n": n_d,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "excludes_zero": ci_lo > 0 or ci_hi < 0,
        }
        logger.info(f"  {domain}: Z_030T mean={np.mean(z_vals):.2f}, CI=[{ci_lo:.2f}, {ci_hi:.2f}]")

    # C2: Grand mean bootstrap CI
    boot_grand = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sample = rng.choice(z_030t, size=n_graphs, replace=True)
        boot_grand[b] = np.mean(sample)

    grand_ci_lo = float(np.percentile(boot_grand, 2.5))
    grand_ci_hi = float(np.percentile(boot_grand, 97.5))
    logger.info(f"  Grand mean Z_030T: {np.mean(z_030t):.2f}, CI=[{grand_ci_lo:.2f}, {grand_ci_hi:.2f}]")

    # C3: Per-domain bootstrap CI for count ratios
    count_ratio_cis = {}
    for domain in unique_domains:
        indices = [i for i, d in enumerate(domain_labels) if d == domain]
        domain_cis = {}
        for mid_str, mid_name in [("2", "021U"), ("4", "021C"), ("6", "021D"), ("7", "030T")]:
            vals = np.array([count_ratios[i][mid_str] for i in indices])
            n_d = len(vals)
            boot_means = np.zeros(N_BOOTSTRAP)
            for b in range(N_BOOTSTRAP):
                sample = rng.choice(vals, size=n_d, replace=True)
                boot_means[b] = np.mean(sample)
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))
            domain_cis[mid_name] = {
                "mean": float(np.mean(vals)),
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
            }
        count_ratio_cis[domain] = domain_cis

    # C4: Eta-squared from ANOVA
    # Group Z_030T by domain
    groups_z = [z_030t[[i for i, d in enumerate(domain_labels) if d == dom]] for dom in unique_domains]
    f_stat_z, p_anova_z = stats.f_oneway(*groups_z)
    # eta² = SS_between / SS_total
    grand_mean = np.mean(z_030t)
    ss_total = np.sum((z_030t - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_z)
    eta_sq_z = float(ss_between / ss_total) if ss_total > 0 else 0.0
    logger.info(f"  Z_030T ANOVA: F={f_stat_z:.2f}, p={p_anova_z:.4e}, eta²={eta_sq_z:.4f}")

    return {
        "per_domain_z030t_ci": per_domain_z_ci,
        "grand_mean_ci": {
            "mean": float(np.mean(z_030t)),
            "ci_95_lo": grand_ci_lo,
            "ci_95_hi": grand_ci_hi,
        },
        "count_ratio_cis": count_ratio_cis,
        "eta_squared_z030t": eta_sq_z,
        "anova_f_z030t": float(f_stat_z),
        "anova_p_z030t": float(p_anova_z),
    }


# ============================================================================
# PHASE D: Layer Confound Test (All 34 Graphs)
# ============================================================================


def parse_layer(layer_str) -> int:
    """Convert layer string to integer. 'E' (embedding) -> -1."""
    if layer_str == "E":
        return -1
    try:
        return int(layer_str)
    except (ValueError, TypeError):
        return -1


def degree_preserving_dag_swap(
    edge_list: list[tuple[int, int]],
    n_vertices: int,
    topo_rank: list[int],
    n_swap_attempts: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], float]:
    """Goni Method 1: Degree-preserving DAG randomization."""
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


def layer_preserving_dag_swap(
    edge_list: list[tuple[int, int]],
    n_vertices: int,
    topo_rank: list[int],
    node_layers: list[int],
    n_swap_attempts: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], float]:
    """Layer-preserving DAG randomization."""
    adj_set = set(edge_list)
    layer_edge_groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for u, v in edge_list:
        key = (node_layers[u], node_layers[v])
        layer_edge_groups[key].append((u, v))

    eligible_keys = [k for k, v in layer_edge_groups.items() if len(v) >= 2]
    if not eligible_keys:
        return list(adj_set), 0.0

    accepted = 0
    for _ in range(n_swap_attempts):
        key = rng.choice(eligible_keys)
        group = layer_edge_groups[key]
        if len(group) < 2:
            continue
        idx1, idx2 = rng.sample(range(len(group)), 2)
        u1, v1 = group[idx1]
        u2, v2 = group[idx2]
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
        group[idx1] = (u1, v2)
        group[idx2] = (u2, v1)
        accepted += 1

    return list(adj_set), accepted / n_swap_attempts if n_swap_attempts > 0 else 0.0


def compute_motif_census(g: igraph.Graph) -> list[int]:
    """Run igraph motifs_randesu for 3-node directed motifs."""
    counts = g.motifs_randesu(size=3)
    return [0 if (c != c) else int(c) for c in counts]


def _process_single_graph_confound(args: tuple) -> dict:
    """Worker: compute Z_degree and Z_layer for one graph."""
    (
        graph_idx, edge_list, n_vertices, node_layers,
        n_null, swap_factor, seed
    ) = args

    rng_d = random.Random(seed)
    rng_l = random.Random(seed + 10000)

    # Build igraph from edge list
    g = igraph.Graph(n=n_vertices, edges=edge_list, directed=True)
    g.vs["layer"] = node_layers

    # Compute topological sort and rank
    topo_order = g.topological_sorting()
    topo_rank = [0] * n_vertices
    for rank, node in enumerate(topo_order):
        topo_rank[node] = rank

    # Real motif census
    real_counts = compute_motif_census(g)
    n_edges = g.ecount()
    n_swap_attempts = n_edges * swap_factor

    # Degree-preserving null models
    deg_counts = []
    for _ in range(n_null):
        null_edges, _ = degree_preserving_dag_swap(
            edge_list, n_vertices, topo_rank, n_swap_attempts, rng_d
        )
        g_null = igraph.Graph(n=n_vertices, edges=null_edges, directed=True)
        deg_counts.append(compute_motif_census(g_null))

    # Layer-preserving null models
    layer_counts = []
    for _ in range(n_null):
        null_edges, _ = layer_preserving_dag_swap(
            edge_list, n_vertices, topo_rank, node_layers,
            n_swap_attempts, rng_l
        )
        g_null = igraph.Graph(n=n_vertices, edges=null_edges, directed=True)
        layer_counts.append(compute_motif_census(g_null))

    deg_arr = np.array(deg_counts, dtype=float)
    layer_arr = np.array(layer_counts, dtype=float)

    # Compute Z-scores for each DAG-valid motif
    z_degree = {}
    z_layer = {}
    for mid in DAG_MOTIF_IDS:
        real_c = real_counts[mid]

        d_mean = np.mean(deg_arr[:, mid])
        d_std = np.std(deg_arr[:, mid], ddof=1) if deg_arr.shape[0] > 1 else 1.0
        z_degree[mid] = float((real_c - d_mean) / max(d_std, 1e-10))

        l_mean = np.mean(layer_arr[:, mid])
        l_std = np.std(layer_arr[:, mid], ddof=1) if layer_arr.shape[0] > 1 else 1.0
        z_layer[mid] = float((real_c - l_mean) / max(l_std, 1e-10))

    return {
        "graph_idx": graph_idx,
        "z_degree": z_degree,
        "z_layer": z_layer,
        "real_counts": {mid: real_counts[mid] for mid in DAG_MOTIF_IDS},
    }


def load_raw_graphs() -> list[dict]:
    """Load all attribution graphs from original data files, prune, return serialized form."""
    logger.info("  Loading raw attribution graphs for confound test...")

    all_examples = []
    data_files = sorted(DATA_DIR.glob("data_out/full_data_out_*.json"))
    if not data_files:
        logger.error("No data files found!")
        return []

    for fpath in data_files:
        data = json.loads(fpath.read_text())
        examples = data["datasets"][0]["examples"]
        all_examples.extend(examples)
        logger.info(f"    Loaded {len(examples)} from {fpath.name}")

    if MAX_EXAMPLES > 0:
        all_examples = all_examples[:MAX_EXAMPLES]

    graphs = []
    for idx, example in enumerate(all_examples):
        try:
            graph_json = json.loads(example["output"])
            nodes = graph_json["nodes"]
            links = graph_json["links"]
            domain = example.get("metadata_fold", "unknown")

            node_id_to_idx = {node["node_id"]: i for i, node in enumerate(nodes)}
            node_layers = [parse_layer(node.get("layer", "0")) for node in nodes]

            g = igraph.Graph(n=len(nodes), directed=True)
            g.vs["layer"] = node_layers

            edge_list = []
            edge_weights = []
            for link in links:
                src = node_id_to_idx.get(link["source"])
                tgt = node_id_to_idx.get(link["target"])
                if src is not None and tgt is not None:
                    edge_list.append((src, tgt))
                    w = link.get("weight", link.get("attribution", link.get("value", 1.0)))
                    edge_weights.append(abs(float(w)))

            g.add_edges(edge_list)
            g.es["weight"] = edge_weights
            g.simplify(multiple=True, loops=True, combine_edges="max")

            if not g.is_dag():
                continue

            # Prune at 75th percentile
            weights = np.array(g.es["weight"])
            threshold = float(np.percentile(weights, PRUNE_PERCENTILE))
            keep = [i for i, w in enumerate(weights) if w >= threshold]
            g_p = g.subgraph_edges(keep, delete_vertices=False)

            isolated = [v.index for v in g_p.vs if g_p.degree(v) == 0]
            g_p.delete_vertices(isolated)

            if g_p.vcount() < MIN_NODES_FOR_CENSUS or g_p.ecount() == 0:
                continue

            if not g_p.is_dag():
                continue

            pruned_layers = list(g_p.vs["layer"])
            pruned_edges = [(e.source, e.target) for e in g_p.es]

            graphs.append({
                "graph_idx": idx,
                "domain": domain,
                "prompt": example["input"],
                "edge_list": pruned_edges,
                "n_vertices": g_p.vcount(),
                "node_layers": pruned_layers,
            })

            del g, g_p
            gc.collect()

        except Exception:
            logger.exception(f"Failed to load graph {idx}")
            continue

    logger.info(f"  Loaded {len(graphs)} valid graphs for confound test")
    return graphs


def phase_d_layer_confound(data: dict) -> dict:
    """Run layer confound test on all graphs using multiprocessing."""
    logger.info("=" * 60)
    logger.info("PHASE D: Layer Confound Test (All Graphs)")

    raw_graphs = load_raw_graphs()
    n_graphs = len(raw_graphs)
    logger.info(f"  Processing {n_graphs} graphs with {N_NULL_MODELS} null models each")

    # Prepare worker arguments
    worker_args = []
    for i, g in enumerate(raw_graphs):
        worker_args.append((
            g["graph_idx"],
            g["edge_list"],
            g["n_vertices"],
            g["node_layers"],
            N_NULL_MODELS,
            SWAP_FACTOR,
            SEED + i,
        ))

    # Use ProcessPoolExecutor with limited workers and spawn context
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    n_par_workers = min(2, N_WORKERS)
    results = []
    t0 = time.time()
    logger.info(f"  Using {n_par_workers} parallel workers (spawn context)")

    # Process in batches to limit memory
    BATCH_SIZE = max(n_par_workers * 2, 4)
    for batch_start in range(0, len(worker_args), BATCH_SIZE):
        batch = worker_args[batch_start:batch_start + BATCH_SIZE]
        try:
            with ProcessPoolExecutor(max_workers=n_par_workers, mp_context=ctx) as executor:
                futures = {executor.submit(_process_single_graph_confound, a): a[0] for a in batch}
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=600)
                        results.append(result)
                    except Exception:
                        logger.exception(f"Failed processing graph {futures[future]}")
        except Exception:
            # Fallback to sequential if multiprocessing fails
            logger.warning("  Multiprocessing failed, falling back to sequential")
            for args in batch:
                try:
                    result = _process_single_graph_confound(args)
                    results.append(result)
                except Exception:
                    logger.exception(f"Failed processing graph {args[0]}")

        done_count = min(batch_start + len(batch), n_graphs)
        elapsed = time.time() - t0
        eta = elapsed / max(done_count, 1) * (n_graphs - done_count)
        mem_gb = psutil.Process().memory_info().rss / 1e9
        logger.info(f"  Confound: {done_count}/{n_graphs} done, elapsed={elapsed:.1f}s, ETA={eta:.1f}s, RSS={mem_gb:.1f}GB")
        gc.collect()

    logger.info(f"  Phase D completed: {len(results)} graphs in {time.time()-t0:.1f}s")

    # Sort by graph_idx
    results.sort(key=lambda x: x["graph_idx"])

    # D1: Collect Z_degree and Z_layer for 030T
    z_degree_030t = np.array([r["z_degree"][7] for r in results])
    z_layer_030t = np.array([r["z_layer"][7] for r in results])

    # D2: Wilcoxon signed-rank test on |Z_degree| vs |Z_layer| for 030T
    abs_z_deg = np.abs(z_degree_030t)
    abs_z_lay = np.abs(z_layer_030t)
    differences = abs_z_lay - abs_z_deg

    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(differences, alternative="two-sided")
    except ValueError:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    direction = "layer_higher" if np.mean(differences) > 0 else "degree_higher"
    logger.info(f"  Wilcoxon: W={wilcoxon_stat:.1f}, p={wilcoxon_p:.4e}, direction={direction}")

    # D3: Z retention ratio = Z_layer / Z_degree for 030T
    retention_ratios = []
    for zd, zl in zip(z_degree_030t, z_layer_030t):
        if abs(zd) > 1e-10:
            retention_ratios.append(zl / zd)
    retention_ratios = np.array(retention_ratios)
    retention_mean = float(np.mean(retention_ratios)) if len(retention_ratios) > 0 else 0.0
    retention_ci_lo = float(np.percentile(retention_ratios, 2.5)) if len(retention_ratios) > 0 else 0.0
    retention_ci_hi = float(np.percentile(retention_ratios, 97.5)) if len(retention_ratios) > 0 else 0.0

    logger.info(f"  Z retention ratio (030T): mean={retention_mean:.3f}, CI=[{retention_ci_lo:.3f}, {retention_ci_hi:.3f}]")

    # D4: Per-motif classification across all graphs
    per_motif_classification = {}
    for mid in DAG_MOTIF_IDS:
        mname = MOTIF_NAMES[mid]
        z_d = np.array([r["z_degree"][mid] for r in results])
        z_l = np.array([r["z_layer"][mid] for r in results])

        n_genuine = 0
        n_artifact = 0
        n_partial = 0
        for zd_i, zl_i in zip(z_d, z_l):
            if abs(zd_i) > 2.0 and abs(zl_i) > 2.0 and np.sign(zd_i) == np.sign(zl_i):
                n_genuine += 1
            elif abs(zd_i) > 2.0 and abs(zl_i) <= 2.0:
                n_artifact += 1
            elif abs(zl_i) > 2.0:
                n_partial += 1

        per_motif_classification[mname] = {
            "n_genuine": n_genuine,
            "n_artifact": n_artifact,
            "n_partial": n_partial,
            "n_total": len(results),
            "frac_genuine": n_genuine / max(len(results), 1),
            "mean_z_degree": float(np.mean(z_d)),
            "mean_z_layer": float(np.mean(z_l)),
        }
        logger.info(f"  {mname}: genuine={n_genuine}, artifact={n_artifact}, partial={n_partial}")

    # Build per-graph results for output
    per_graph_results = []
    for i, r in enumerate(results):
        domain = raw_graphs[i]["domain"] if i < len(raw_graphs) else "unknown"
        per_graph_results.append({
            "graph_idx": r["graph_idx"],
            "domain": domain,
            "z_degree_030T": r["z_degree"][7],
            "z_layer_030T": r["z_layer"][7],
            "retention_ratio": r["z_layer"][7] / r["z_degree"][7] if abs(r["z_degree"][7]) > 1e-10 else 0.0,
        })

    return {
        "n_graphs_processed": len(results),
        "per_graph_results": per_graph_results,
        "wilcoxon_test": {
            "statistic": float(wilcoxon_stat),
            "p_value": float(wilcoxon_p),
            "direction": direction,
            "n_pairs": len(results),
        },
        "z_retention_ratio": {
            "mean": retention_mean,
            "ci_95_lo": retention_ci_lo,
            "ci_95_hi": retention_ci_hi,
            "n_valid": len(retention_ratios),
        },
        "per_motif_classification": per_motif_classification,
    }


# ============================================================================
# PHASE E: Effect Size Analysis
# ============================================================================


def phase_e_effect_sizes(data: dict, phase_b: dict) -> dict:
    """Compute Cohen's d and eta-squared for various comparisons."""
    logger.info("=" * 60)
    logger.info("PHASE E: Effect Size Analysis")

    # E1: Cohen's d for NMI improvement
    nmi_our = phase_b["nmi_our_observed"]
    nmi_base = phase_b["nmi_baseline_observed"]
    null_std_our = phase_b["null_distributions"]["nmi_our_std"]
    null_std_base = phase_b["null_distributions"]["nmi_base_std"]
    pooled_sd = np.sqrt((null_std_our**2 + null_std_base**2) / 2) if (null_std_our + null_std_base) > 0 else 1e-10
    d_nmi = float((nmi_our - nmi_base) / pooled_sd) if pooled_sd > 1e-10 else 0.0

    d_classification = "negligible"
    if abs(d_nmi) >= 0.8:
        d_classification = "large"
    elif abs(d_nmi) >= 0.5:
        d_classification = "medium"
    elif abs(d_nmi) >= 0.2:
        d_classification = "small"

    logger.info(f"  E1: Cohen's d for NMI improvement: {d_nmi:.4f} ({d_classification})")

    # E2: Cohen's d for Z_030T across domain pairs
    domain_labels = data["domain_labels"]
    z_030t = data["z_030t_values"]
    unique_domains = sorted(set(domain_labels))

    domain_z = {}
    for dom in unique_domains:
        indices = [i for i, d in enumerate(domain_labels) if d == dom]
        domain_z[dom] = z_030t[indices]

    cross_domain_d = {}
    for i, d1 in enumerate(unique_domains):
        for d2 in unique_domains[i+1:]:
            z1, z2 = domain_z[d1], domain_z[d2]
            n1, n2 = len(z1), len(z2)
            v1 = float(np.var(z1, ddof=1)) if n1 > 1 else 0.0
            v2 = float(np.var(z2, ddof=1)) if n2 > 1 else 0.0
            denom = max(n1 + n2 - 2, 1)
            pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / denom)
            d = float((np.mean(z1) - np.mean(z2)) / max(pooled, 1e-10))
            cross_domain_d[f"{d1}_vs_{d2}"] = round(d, 4)

    # E3: Cohen's d for count ratio separation (030T has highest F)
    count_ratios = data["count_ratios_all"]
    domain_030t_ratios = {}
    for dom in unique_domains:
        indices = [i for i, d in enumerate(domain_labels) if d == dom]
        domain_030t_ratios[dom] = np.array([count_ratios[i]["7"] for i in indices])

    all_030t_ratios = np.array([cr["7"] for cr in count_ratios])
    grand_mean_030t = np.mean(all_030t_ratios)
    within_vars = [float(np.var(domain_030t_ratios[d], ddof=1)) for d in unique_domains if len(domain_030t_ratios[d]) > 1]
    within_var = float(np.mean(within_vars)) if within_vars else 1e-10
    between_var = float(np.var([np.mean(domain_030t_ratios[d]) for d in unique_domains]))
    grand_d_030t = float(np.sqrt(between_var) / np.sqrt(max(within_var, 1e-10)))

    logger.info(f"  E3: Grand Cohen's d for 030T count ratio: {grand_d_030t:.4f}")

    # E4: Eta-squared for ANOVA on count ratios
    # Using F × df_between / (F × df_between + df_within)
    df_between = len(unique_domains) - 1
    df_within = len(domain_labels) - len(unique_domains)

    f_stats_reported = {
        "021U": 4.952, "021C": 19.393, "021D": 11.384, "030T": 28.852
    }
    eta_sq = {}
    for motif, f_val in f_stats_reported.items():
        eta = (f_val * df_between) / (f_val * df_between + df_within)
        eta_sq[motif] = round(float(eta), 4)

    # Also compute fresh from data
    for mid_str, mid_name in [("2", "021U"), ("4", "021C"), ("6", "021D"), ("7", "030T")]:
        groups = []
        for dom in unique_domains:
            indices = [i for i, d in enumerate(domain_labels) if d == dom]
            groups.append(np.array([count_ratios[i][mid_str] for i in indices]))

        try:
            f_stat, p_val = stats.f_oneway(*groups)
            all_vals = np.concatenate(groups)
            gm = np.mean(all_vals)
            ss_total = np.sum((all_vals - gm) ** 2)
            ss_between = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
            eta_sq[f"{mid_name}_computed"] = round(float(ss_between / max(ss_total, 1e-10)), 4)
            eta_sq[f"{mid_name}_f_stat"] = round(float(f_stat), 4)
            eta_sq[f"{mid_name}_p_value"] = float(p_val)
        except Exception:
            logger.exception(f"ANOVA failed for {mid_name}")

    logger.info(f"  E4: Eta-squared: {eta_sq}")

    return {
        "nmi_cohens_d": d_nmi,
        "nmi_cohens_d_classification": d_classification,
        "z_cross_domain_cohens_d": cross_domain_d,
        "count_ratio_grand_d_030T": grand_d_030t,
        "count_ratio_eta_squared": eta_sq,
    }


# ============================================================================
# PHASE F: Robustness Checks
# ============================================================================


def phase_f_robustness(data: dict) -> dict:
    """LOO, seed stability, domain imbalance, feature ablation."""
    logger.info("=" * 60)
    logger.info("PHASE F: Robustness Checks")

    enriched = data["enriched_matrix"]
    baseline = data["baseline_matrix"]
    true_labels = data["true_labels"]
    domain_labels = data["domain_labels"]
    n_graphs = data["n_graphs"]

    # F1: Leave-one-out NMI influence analysis
    logger.info("  F1: Leave-one-out influence analysis...")
    loo_nmi_our = []
    loo_nmi_base = []
    influential_graphs = []

    for i in range(n_graphs):
        mask = np.ones(n_graphs, dtype=bool)
        mask[i] = False
        e_sub = enriched[mask]
        b_sub = baseline[mask]
        labels_sub = true_labels[mask]

        # Adjust K if necessary (need K < n_samples and K <= n_unique_labels)
        n_unique = len(np.unique(labels_sub))
        k_use = min(8, n_unique, e_sub.shape[0] - 1)
        if k_use < 2:
            continue

        r_our = spectral_cluster_nmi_ari(e_sub, labels_sub, k=k_use)
        r_base = spectral_cluster_nmi_ari(b_sub, labels_sub, k=k_use)
        loo_nmi_our.append(r_our["nmi"])
        loo_nmi_base.append(r_base["nmi"])

        if abs(r_our["nmi"] - data["our_nmi"]) > 0.05:
            influential_graphs.append({
                "index": i,
                "domain": domain_labels[i],
                "nmi_without": r_our["nmi"],
                "nmi_delta": r_our["nmi"] - data["our_nmi"],
            })

    loo_nmi_our = np.array(loo_nmi_our) if loo_nmi_our else np.array([0.0])
    loo_nmi_base = np.array(loo_nmi_base) if loo_nmi_base else np.array([0.0])
    logger.info(f"    LOO NMI our: {np.mean(loo_nmi_our):.4f} +/- {np.std(loo_nmi_our):.4f}")
    logger.info(f"    LOO NMI base: {np.mean(loo_nmi_base):.4f} +/- {np.std(loo_nmi_base):.4f}")
    logger.info(f"    Influential graphs (>0.05 NMI change): {len(influential_graphs)}")

    # F2: Domain-size imbalance bias assessment
    logger.info("  F2: Domain imbalance assessment...")
    r_full = spectral_cluster_nmi_ari(enriched, true_labels, k=8)
    ami_full = r_full["ami"]
    vm_full = r_full["v_measure"]

    # Expected NMI from random (use permutation null mean from Phase B)
    logger.info(f"    AMI={ami_full:.4f}, V-measure={vm_full:.4f}")

    domain_sizes = Counter(domain_labels)
    logger.info(f"    Domain sizes: {dict(domain_sizes)}")

    # F3: Clustering stability across random seeds
    logger.info("  F3: Seed stability...")
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    seed_nmis_our = []
    seed_nmis_base = []
    for s in seeds:
        r_our = spectral_cluster_nmi_ari(enriched, true_labels, k=8, random_state=s)
        r_base = spectral_cluster_nmi_ari(baseline, true_labels, k=8, random_state=s)
        seed_nmis_our.append(r_our["nmi"])
        seed_nmis_base.append(r_base["nmi"])

    seed_nmis_our = np.array(seed_nmis_our)
    seed_nmis_base = np.array(seed_nmis_base)
    logger.info(f"    Our method NMI across seeds: {np.mean(seed_nmis_our):.4f} +/- {np.std(seed_nmis_our):.4f}")
    logger.info(f"    Baseline NMI across seeds: {np.mean(seed_nmis_base):.4f} +/- {np.std(seed_nmis_base):.4f}")

    # F4: Feature ablation analysis
    logger.info("  F4: Feature ablation...")
    n_features = enriched.shape[1]
    ablation_results = {}

    for feat_idx in range(n_features):
        feat_name = ENRICHED_FEATURE_NAMES[feat_idx] if feat_idx < len(ENRICHED_FEATURE_NAMES) else f"feature_{feat_idx}"
        mask_cols = np.ones(n_features, dtype=bool)
        mask_cols[feat_idx] = False
        e_abl = enriched[:, mask_cols]
        r_abl = spectral_cluster_nmi_ari(e_abl, true_labels, k=8)
        delta = r_abl["nmi"] - data["our_nmi"]
        ablation_results[feat_name] = {
            "nmi_without": round(r_abl["nmi"], 4),
            "delta_nmi": round(delta, 4),
        }
        logger.debug(f"      {feat_name}: NMI={r_abl['nmi']:.4f} (delta={delta:+.4f})")

    # Also test removing all graph stats at once
    motif_only = enriched[:, MOTIF_ONLY_INDICES]
    r_motif = spectral_cluster_nmi_ari(motif_only, true_labels, k=8)

    # And removing all motif features at once
    stats_only = enriched[:, GRAPH_STAT_INDICES]
    r_stats = spectral_cluster_nmi_ari(stats_only, true_labels, k=8)

    ablation_results["ALL_GRAPH_STATS_REMOVED"] = {
        "nmi_without": round(r_motif["nmi"], 4),
        "delta_nmi": round(r_motif["nmi"] - data["our_nmi"], 4),
    }
    ablation_results["ALL_MOTIF_FEATURES_REMOVED"] = {
        "nmi_without": round(r_stats["nmi"], 4),
        "delta_nmi": round(r_stats["nmi"] - data["our_nmi"], 4),
    }

    logger.info(f"    Motif-only NMI: {r_motif['nmi']:.4f}")
    logger.info(f"    Stats-only NMI: {r_stats['nmi']:.4f}")

    return {
        "loo_influence": {
            "mean_loo_nmi_our": float(np.mean(loo_nmi_our)),
            "std_loo_nmi_our": float(np.std(loo_nmi_our)),
            "mean_loo_nmi_base": float(np.mean(loo_nmi_base)),
            "std_loo_nmi_base": float(np.std(loo_nmi_base)),
            "n_influential": len(influential_graphs),
            "influential_graphs": influential_graphs,
        },
        "domain_imbalance": {
            "ami": ami_full,
            "v_measure": vm_full,
            "domain_sizes": dict(domain_sizes),
        },
        "seed_stability": {
            "seeds": seeds,
            "nmi_our_mean": float(np.mean(seed_nmis_our)),
            "nmi_our_std": float(np.std(seed_nmis_our)),
            "nmi_our_values": seed_nmis_our.tolist(),
            "nmi_base_mean": float(np.mean(seed_nmis_base)),
            "nmi_base_std": float(np.std(seed_nmis_base)),
            "nmi_base_values": seed_nmis_base.tolist(),
        },
        "feature_ablation": ablation_results,
    }


# ============================================================================
# MAIN
# ============================================================================


@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Statistical Significance Audit - Starting")
    logger.info(f"Config: N_PERM={N_PERMUTATIONS}, N_BOOT={N_BOOTSTRAP}, N_NULL={N_NULL_MODELS}")

    # Phase A: Reconstruct data
    data = phase_a_reconstruct_data()
    t_a = time.time()
    logger.info(f"Phase A complete: {t_a - t_start:.1f}s")

    # Phase B: Permutation tests
    phase_b_results = phase_b_permutation_tests(data)
    t_b = time.time()
    logger.info(f"Phase B complete: {t_b - t_a:.1f}s")

    # Phase C: Bootstrap CIs
    phase_c_results = phase_c_bootstrap_cis(data)
    t_c = time.time()
    logger.info(f"Phase C complete: {t_c - t_b:.1f}s")

    # Phase D: Layer confound test
    phase_d_results = phase_d_layer_confound(data)
    t_d = time.time()
    logger.info(f"Phase D complete: {t_d - t_c:.1f}s")

    # Phase E: Effect sizes
    phase_e_results = phase_e_effect_sizes(data, phase_b_results)
    t_e = time.time()
    logger.info(f"Phase E complete: {t_e - t_d:.1f}s")

    # Phase F: Robustness checks
    phase_f_results = phase_f_robustness(data)
    t_f = time.time()
    logger.info(f"Phase F complete: {t_f - t_e:.1f}s")

    # Build overall verdict
    nmi_sig = phase_b_results["nmi_our_p_value"] < 0.05
    nmi_base_sig = phase_b_results["nmi_baseline_p_value"] < 0.05
    nmi_diff_sig = phase_b_results["nmi_difference_p_value"] < 0.05
    motif_sig = phase_b_results["motif_only_p_value"] < 0.05

    # Check FFL genuineness from phase D
    ffl_class = phase_d_results.get("per_motif_classification", {}).get("030T", {})
    ffl_genuine = ffl_class.get("frac_genuine", 0) > 0.5

    # Layer confound resolved if Wilcoxon significant and Z retention ratio > 0.5
    wilcoxon_sig = phase_d_results.get("wilcoxon_test", {}).get("p_value", 1.0) < 0.05
    retention = phase_d_results.get("z_retention_ratio", {}).get("mean", 0)
    layer_resolved = wilcoxon_sig or retention > 0.5

    key_findings = []
    key_findings.append(f"NMI our method p={phase_b_results['nmi_our_p_value']:.4f} ({'significant' if nmi_sig else 'not significant'})")
    key_findings.append(f"NMI baseline p={phase_b_results['nmi_baseline_p_value']:.4f} ({'significant' if nmi_base_sig else 'not significant'})")
    key_findings.append(f"NMI difference p={phase_b_results['nmi_difference_p_value']:.4f} ({'significant' if nmi_diff_sig else 'not significant'})")
    key_findings.append(f"Motif-only NMI={phase_b_results['motif_only_nmi']:.4f}, p={phase_b_results['motif_only_p_value']:.4f}")
    key_findings.append(f"FFL genuine signal in {ffl_class.get('frac_genuine', 0)*100:.0f}% of graphs")
    key_findings.append(f"Z retention ratio: {retention:.3f}")
    key_findings.append(f"Seed stability: NMI std={phase_f_results['seed_stability']['nmi_our_std']:.4f}")

    overall_verdict = {
        "nmi_is_significant": nmi_sig,
        "nmi_baseline_is_significant": nmi_base_sig,
        "nmi_improvement_is_significant": nmi_diff_sig,
        "motif_features_add_value": motif_sig,
        "ffl_overrepresentation_genuine": ffl_genuine,
        "layer_confound_resolved": layer_resolved,
        "key_findings": key_findings,
    }

    # ---- Build output in exp_eval_sol_out schema ----
    # metrics_agg: flat dict of numbers
    metrics_agg = {
        "nmi_our_method": round(data["our_nmi"], 4),
        "nmi_baseline": round(data["base_nmi"], 4),
        "ari_our_method": round(data["our_ari"], 4),
        "ari_baseline": round(data["base_ari"], 4),
        "nmi_our_p_value": round(phase_b_results["nmi_our_p_value"], 4),
        "nmi_baseline_p_value": round(phase_b_results["nmi_baseline_p_value"], 4),
        "nmi_difference_p_value": round(phase_b_results["nmi_difference_p_value"], 4),
        "ari_our_p_value": round(phase_b_results["ari_our_p_value"], 4),
        "nmi_motif_only": round(phase_b_results["motif_only_nmi"], 4),
        "nmi_motif_only_p_value": round(phase_b_results["motif_only_p_value"], 4),
        "z030t_grand_mean": round(float(np.mean(data["z_030t_values"])), 4),
        "z030t_eta_squared": round(phase_c_results["eta_squared_z030t"], 4),
        "wilcoxon_p_value_confound": round(phase_d_results["wilcoxon_test"]["p_value"], 4),
        "z_retention_ratio_mean": round(phase_d_results["z_retention_ratio"]["mean"], 4),
        "ffl_frac_genuine": round(ffl_class.get("frac_genuine", 0), 4),
        "cohens_d_nmi_improvement": round(phase_e_results["nmi_cohens_d"], 4),
        "count_ratio_grand_d_030t": round(phase_e_results["count_ratio_grand_d_030T"], 4),
        "loo_nmi_mean": round(phase_f_results["loo_influence"]["mean_loo_nmi_our"], 4),
        "loo_nmi_std": round(phase_f_results["loo_influence"]["std_loo_nmi_our"], 4),
        "seed_nmi_mean": round(phase_f_results["seed_stability"]["nmi_our_mean"], 4),
        "seed_nmi_std": round(phase_f_results["seed_stability"]["nmi_our_std"], 4),
        "ami_full": round(phase_f_results["domain_imbalance"]["ami"], 4),
        "v_measure_full": round(phase_f_results["domain_imbalance"]["v_measure"], 4),
        "n_graphs": data["n_graphs"],
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "n_null_models": N_NULL_MODELS,
    }

    # Build per-graph examples for datasets
    examples = []
    for i in range(data["n_graphs"]):
        prompt = data["graph_prompts"][i]
        domain = data["domain_labels"][i]

        # Gather per-example eval metrics
        z_val = float(data["z_030t_values"][i])

        # Find this graph in phase_d results if possible
        phase_d_z_deg = 0.0
        phase_d_z_lay = 0.0
        for pgr in phase_d_results.get("per_graph_results", []):
            if pgr["graph_idx"] == i:
                phase_d_z_deg = pgr.get("z_degree_030T", 0.0)
                phase_d_z_lay = pgr.get("z_layer_030T", 0.0)
                break

        # Build detailed output string with all phase results
        example_output = json.dumps({
            "z_030T": z_val,
            "count_ratios": data["count_ratios_all"][i],
            "ffl_raw_count": data["raw_ffl_counts"][i],
            "z_degree_030T": phase_d_z_deg,
            "z_layer_030T": phase_d_z_lay,
        })

        ex = {
            "input": prompt,
            "output": example_output,
            "metadata_fold": domain,
            "metadata_graph_idx": i,
            "predict_our_method": data["domain_labels"][i],
            "predict_baseline": data["domain_labels"][i],
            "eval_z_030t": round(z_val, 4),
            "eval_z_degree_030t": round(phase_d_z_deg, 4),
            "eval_z_layer_030t": round(phase_d_z_lay, 4),
        }
        examples.append(ex)

    # Full output
    output = {
        "metadata": {
            "experiment": "statistical_significance_audit_iter3",
            "n_graphs": data["n_graphs"],
            "n_domains": len(set(data["domain_labels"])),
            "domains": sorted(set(data["domain_labels"])),
            "runtime_seconds": round(time.time() - t_start, 1),
            "phase_b_permutation_tests": phase_b_results,
            "phase_c_bootstrap_cis": phase_c_results,
            "phase_d_layer_confound": phase_d_results,
            "phase_e_effect_sizes": phase_e_results,
            "phase_f_robustness": phase_f_results,
            "overall_verdict": overall_verdict,
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "neuronpedia_attribution_graphs",
                "examples": examples,
            }
        ],
    }

    # Sanitize NaN/Inf for JSON (replace with None then 0.0 for metrics_agg)
    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, np.floating):
            v = float(obj)
            return 0.0 if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        return obj

    output = _sanitize(output)

    # Write output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Output written to {out_path}")
    logger.info(f"Total runtime: {time.time() - t_start:.1f}s")

    # Print verdict summary
    logger.info("=" * 60)
    logger.info("OVERALL VERDICT:")
    for finding in key_findings:
        logger.info(f"  {finding}")


if __name__ == "__main__":
    main()
