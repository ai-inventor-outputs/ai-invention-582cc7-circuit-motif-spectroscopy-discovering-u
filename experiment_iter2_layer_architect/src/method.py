#!/usr/bin/env python3
"""Layer-Architecture Confound Test for Attribution Graph Motif Analysis.

Compares degree-preserving vs layer-preserving DAG null models to determine
whether motif overrepresentation in LLM attribution graphs is a genuine
computational signal or an artifact of layered transformer architecture.

Uses all 34 graphs from the Neuronpedia dependency dataset, 75th percentile
edge weight pruning, configurable randomizations per null model type, and
statistical comparison via Wilcoxon signed-rank tests.

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
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import igraph
from scipy import stats
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

# Set resource limits (virtual address space — 3x budget for RSS headroom)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

# ============================================================================
# CONSTANTS (configurable via env vars for gradual scaling)
# ============================================================================

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
N_NULL_MODELS = int(os.environ.get("N_NULL_MODELS", "300"))
SWAP_FACTOR = int(os.environ.get("SWAP_FACTOR", "100"))
PRUNE_PERCENTILE = 75
MIN_NODES_FOR_CENSUS = 30
N_WORKERS = max(1, NUM_CPUS)
LAYER_BASELINE_N = 100
SEED = 42

# ============================================================================
# PATHS
# ============================================================================

WORKSPACE = Path(__file__).parent
DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_1/gen_art/data_id4_it1__opus"
)


# ============================================================================
# PHASE 0: Build Isoclass-to-MAN Mapping
# ============================================================================


def build_isoclass_mapping() -> tuple[dict[int, str], dict[str, int], list[int]]:
    """Build mapping from igraph isoclass IDs to MAN labels for 3-node triads.

    Creates all 16 3-node directed triad types from known edge lists,
    queries igraph for their isoclass IDs, and identifies the 4 DAG-possible
    weakly-connected types (021D, 021U, 021C, 030T).

    Returns:
        (isoclass_to_man, man_to_isoclass, dag_possible_ids)
    """
    # All 16 3-node directed triads defined by edge structure
    triads: dict[str, list[tuple[int, int]]] = {
        "003": [],
        "012": [(0, 1)],
        "102": [(0, 1), (1, 0)],
        "021D": [(1, 0), (1, 2)],  # out-star from node 1
        "021U": [(0, 1), (2, 1)],  # in-star to node 1
        "021C": [(0, 1), (1, 2)],  # chain 0->1->2
        "111D": [(0, 1), (1, 0), (2, 1)],  # mutual 0<->1, edge 2->1
        "111U": [(0, 1), (1, 0), (1, 2)],  # mutual 0<->1, edge 1->2
        "030T": [(0, 1), (0, 2), (1, 2)],  # feed-forward loop
        "030C": [(0, 1), (1, 2), (2, 0)],  # directed 3-cycle
        "201": [(0, 1), (1, 0), (0, 2), (2, 0)],  # two mutual pairs
        "120D": [(1, 2), (2, 1), (1, 0), (2, 0)],  # mutual + diverging
        "120U": [(1, 2), (2, 1), (0, 1), (0, 2)],  # mutual + converging
        "120C": [(0, 1), (1, 0), (1, 2), (2, 0)],  # mutual + cyclic
        "210": [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)],  # 5 edges
        "300": [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)],  # complete
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

    assert len(isoclass_to_man) == 16, (
        f"Expected 16 isoclasses, got {len(isoclass_to_man)}"
    )

    # Identify the 4 DAG-possible weakly-connected 3-node motif types
    dag_possible_labels = ["021D", "021U", "021C", "030T"]
    dag_possible_ids = sorted(man_to_isoclass[l] for l in dag_possible_labels)

    # Validate: each DAG-possible triad must be a DAG and weakly connected
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
# PHASE A: Graph Loading and Layer Characterization
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
    """Load all attribution graphs from dependency data files.

    Parses JSON graphs, builds igraph objects, applies 75th percentile
    edge weight pruning, removes isolated vertices, and computes layer
    characterization statistics.

    Args:
        max_examples: Maximum number of examples to load (0 = all).

    Returns:
        List of graph info dicts with igraph Graph objects and metadata.
    """
    logger.info("Loading attribution graphs from dependency data...")

    # Collect examples from all data files
    all_examples: list[dict] = []
    data_files = sorted(DATA_DIR.glob("data_out/full_data_out_*.json"))

    if not data_files:
        # Fall back to mini data for testing
        mini_path = DATA_DIR / "mini_data_out.json"
        logger.warning(f"No full data files found, using mini: {mini_path}")
        data_files = [mini_path]

    for fpath in data_files:
        logger.info(f"  Loading {fpath.name}...")
        data = json.loads(fpath.read_text())
        examples = data["datasets"][0]["examples"]
        all_examples.extend(examples)
        logger.info(f"    -> {len(examples)} examples")

    if max_examples > 0:
        all_examples = all_examples[:max_examples]

    logger.info(f"Total examples to process: {len(all_examples)}")

    # Build graph objects from each example
    all_graphs: list[dict] = []
    for idx, example in enumerate(all_examples):
        try:
            prompt = example["input"]
            domain = example.get("metadata_fold", "unknown")
            graph_json = json.loads(example["output"])
            nodes = graph_json["nodes"]
            links = graph_json["links"]

            # Build node_id -> index mapping
            node_id_to_idx = {
                node["node_id"]: i for i, node in enumerate(nodes)
            }

            # Extract integer layers for each node
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
                    w = link.get(
                        "weight",
                        link.get("attribution", link.get("value", 1.0)),
                    )
                    edge_weights.append(abs(float(w)))

            g.add_edges(edge_list)
            g.es["weight"] = edge_weights

            # Simplify: remove multi-edges and self-loops
            g.simplify(multiple=True, loops=True, combine_edges="max")

            # Verify DAG property
            if not g.is_dag():
                logger.warning(
                    f"Graph {idx} ({domain}) is NOT a DAG — skipping"
                )
                continue

            # 75th percentile edge weight pruning (keep top 25%)
            weights = np.array(g.es["weight"])
            threshold = float(np.percentile(weights, PRUNE_PERCENTILE))
            edges_to_keep = [
                i for i, w in enumerate(weights) if w >= threshold
            ]
            g_pruned = g.subgraph_edges(edges_to_keep, delete_vertices=False)

            # Remove isolated vertices (no edges after pruning)
            isolated = [
                v.index for v in g_pruned.vs if g_pruned.degree(v) == 0
            ]
            g_pruned.delete_vertices(isolated)

            if g_pruned.vcount() < MIN_NODES_FOR_CENSUS:
                logger.warning(
                    f"Graph {idx} ({domain}): {g_pruned.vcount()} nodes "
                    f"after pruning (< {MIN_NODES_FOR_CENSUS}), skipping"
                )
                continue

            if g_pruned.ecount() == 0:
                logger.warning(
                    f"Graph {idx} ({domain}): 0 edges after pruning, skipping"
                )
                continue

            # Verify DAG property preserved after pruning
            assert g_pruned.is_dag(), (
                f"Pruned graph {idx} ({domain}) is not a DAG"
            )

            # Compute layer characterization
            pruned_layers = list(g_pruned.vs["layer"])
            unique_layers = sorted(set(pruned_layers))
            nodes_per_layer = dict(Counter(pruned_layers))

            layer_distances: list[int] = []
            layer_pair_counts: Counter = Counter()
            for edge in g_pruned.es:
                src_layer = pruned_layers[edge.source]
                tgt_layer = pruned_layers[edge.target]
                dist = tgt_layer - src_layer
                layer_distances.append(dist)
                layer_pair_counts[(src_layer, tgt_layer)] += 1

            n_edges_pruned = len(layer_distances)
            frac_span_1 = (
                sum(1 for d in layer_distances if d == 1) / n_edges_pruned
                if n_edges_pruned > 0
                else 0.0
            )
            frac_span_2plus = (
                sum(1 for d in layer_distances if d >= 2) / n_edges_pruned
                if n_edges_pruned > 0
                else 0.0
            )

            all_graphs.append(
                {
                    "graph": g_pruned,
                    "domain": domain,
                    "prompt": prompt,
                    "n_nodes": g_pruned.vcount(),
                    "n_edges": g_pruned.ecount(),
                    "n_layers": len(unique_layers),
                    "nodes_per_layer": nodes_per_layer,
                    "layer_pair_counts": dict(layer_pair_counts),
                    "frac_span_1_layer": frac_span_1,
                    "frac_span_2plus_layers": frac_span_2plus,
                    "layer_distance_mean": float(np.mean(layer_distances))
                    if layer_distances
                    else 0.0,
                    "layer_distance_std": float(np.std(layer_distances))
                    if layer_distances
                    else 0.0,
                }
            )

            logger.debug(
                f"Graph {idx} ({domain}): {g_pruned.vcount()} nodes, "
                f"{g_pruned.ecount()} edges, {len(unique_layers)} layers"
            )

            # Free original graph memory
            del g
            gc.collect()

        except Exception:
            logger.exception(f"Failed to process graph {idx}")
            continue

    logger.info(f"Loaded {len(all_graphs)} graphs after pruning")
    return all_graphs


# ============================================================================
# PHASE B: Null Model Implementations
# ============================================================================


def degree_preserving_dag_swap(
    edge_list: list[tuple[int, int]],
    n_vertices: int,
    topo_rank: list[int],
    n_swap_attempts: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], float]:
    """Goni et al. Method 1 (DD): Degree-preserving DAG randomization.

    Preserves directed degree sequence + acyclicity.
    Uses topological ordering for O(1) acyclicity check per swap.

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

        # Skip if any endpoints overlap
        if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
            continue

        # Skip if new edges already exist (prevent multi-edges)
        if (u1, v2) in adj_set or (u2, v1) in adj_set:
            continue

        # Acyclicity check: new edges must respect topological ordering
        if topo_rank[u1] >= topo_rank[v2] or topo_rank[u2] >= topo_rank[v1]:
            continue

        # Accept swap: update both adj_set and edges list
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
    """Layer-preserving DAG randomization.

    Extended null model: Preserves degree sequence + acyclicity + layer
    endpoints. Swaps (u1->v1, u2->v2) only if layer(u1)==layer(u2) AND
    layer(v1)==layer(v2), automatically satisfied by swapping within
    same (source_layer, target_layer) groups.

    Returns:
        (new_edge_list, acceptance_rate)
    """
    adj_set = set(edge_list)

    # Group edges by (source_layer, target_layer) for constrained swapping
    layer_edge_groups: dict[tuple[int, int], list[tuple[int, int]]] = (
        defaultdict(list)
    )
    for u, v in edge_list:
        key = (node_layers[u], node_layers[v])
        layer_edge_groups[key].append((u, v))

    # Only groups with >= 2 edges can participate in swaps
    eligible_keys = [
        k for k, v in layer_edge_groups.items() if len(v) >= 2
    ]

    if not eligible_keys:
        return list(adj_set), 0.0

    accepted = 0
    for _ in range(n_swap_attempts):
        # Pick random layer-pair group
        key = rng.choice(eligible_keys)
        group = layer_edge_groups[key]

        if len(group) < 2:
            continue

        # Pick 2 distinct random edges from this group
        idx1, idx2 = rng.sample(range(len(group)), 2)
        u1, v1 = group[idx1]
        u2, v2 = group[idx2]

        # Skip if endpoints overlap
        if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
            continue

        # Skip if new edges already exist
        if (u1, v2) in adj_set or (u2, v1) in adj_set:
            continue

        # Acyclicity check via topological ordering
        if topo_rank[u1] >= topo_rank[v2] or topo_rank[u2] >= topo_rank[v1]:
            continue

        # Accept swap
        adj_set.discard((u1, v1))
        adj_set.discard((u2, v2))
        adj_set.add((u1, v2))
        adj_set.add((u2, v1))

        # Update group in-place (new edges stay in same layer-pair group)
        group[idx1] = (u1, v2)
        group[idx2] = (u2, v1)

        accepted += 1

    return list(adj_set), accepted / n_swap_attempts if n_swap_attempts > 0 else 0.0


# ============================================================================
# PHASE C: Motif Census and Per-Graph Processing
# ============================================================================


def compute_motif_census(g: igraph.Graph) -> list[int]:
    """Run igraph motifs_randesu for 3-node directed motifs.

    Returns:
        List of 16 counts (NaN entries replaced with 0).
    """
    counts = g.motifs_randesu(size=3)
    # NaN check: c != c is True only for NaN
    return [0 if (c != c) else int(c) for c in counts]


def process_single_graph(args: tuple) -> dict:
    """Process one graph: real census + both null model ensembles.

    Worker function for multiprocessing. Receives serialized graph data,
    generates degree-preserving and layer-preserving null models, computes
    motif censuses, Z-scores, and empirical p-values.

    Args:
        args: (graph_data_dict, n_null_models, swap_factor, worker_seed)

    Returns:
        Dict with complete per-graph results.
    """
    graph_data, n_null, swap_factor, worker_seed = args
    rng = random.Random(worker_seed)

    # Reconstruct igraph Graph from serialized data
    g = igraph.Graph(
        n=graph_data["n_vertices"],
        edges=graph_data["edges"],
        directed=True,
    )
    g.vs["layer"] = graph_data["vertex_layers"]

    node_layers = graph_data["vertex_layers"]
    n_edges = g.ecount()
    n_swap_attempts = swap_factor * n_edges

    # Pre-compute topological ordering for O(1) acyclicity checks
    topo_order = g.topological_sorting()
    topo_rank = [0] * g.vcount()
    for rank, vid in enumerate(topo_order):
        topo_rank[vid] = rank

    edge_list = g.get_edgelist()

    # Real motif census
    real_counts = compute_motif_census(g)

    t0 = time.time()

    # --- Degree-preserving null models ---
    degree_null_counts: list[list[int]] = []
    degree_acceptance_rates: list[float] = []
    for i in range(n_null):
        null_edges, acc_rate = degree_preserving_dag_swap(
            edge_list, g.vcount(), topo_rank, n_swap_attempts, rng,
        )
        g_null = igraph.Graph(
            n=g.vcount(), edges=null_edges, directed=True,
        )
        null_counts = compute_motif_census(g_null)
        degree_null_counts.append(null_counts)
        degree_acceptance_rates.append(acc_rate)
        del g_null

    t1 = time.time()

    # --- Layer-preserving null models ---
    layer_null_counts: list[list[int]] = []
    layer_acceptance_rates: list[float] = []
    for i in range(n_null):
        null_edges, acc_rate = layer_preserving_dag_swap(
            edge_list, g.vcount(), topo_rank, node_layers,
            n_swap_attempts, rng,
        )
        g_null = igraph.Graph(
            n=g.vcount(), edges=null_edges, directed=True,
        )
        null_counts = compute_motif_census(g_null)
        layer_null_counts.append(null_counts)
        layer_acceptance_rates.append(acc_rate)
        del g_null

    t2 = time.time()

    # --- Compute Z-scores and p-values for all 16 motif types ---
    n_motif_types = 16
    z_degree = [0.0] * n_motif_types
    z_layer = [0.0] * n_motif_types
    p_degree = [1.0] * n_motif_types
    p_layer = [1.0] * n_motif_types

    for m in range(n_motif_types):
        real_val = real_counts[m]

        # Degree-preserving Z-score
        null_vals = [nc[m] for nc in degree_null_counts]
        mean_null = float(np.mean(null_vals))
        std_null = float(np.std(null_vals))
        if std_null > 0:
            z_degree[m] = (real_val - mean_null) / std_null
        elif real_val > mean_null:
            z_degree[m] = 10.0  # Cap per research guidance
        else:
            z_degree[m] = 0.0
        # Empirical p-value (one-sided: fraction of nulls >= real)
        p_degree[m] = sum(1 for v in null_vals if v >= real_val) / len(null_vals)

        # Layer-preserving Z-score
        null_vals_l = [nc[m] for nc in layer_null_counts]
        mean_null_l = float(np.mean(null_vals_l))
        std_null_l = float(np.std(null_vals_l))
        if std_null_l > 0:
            z_layer[m] = (real_val - mean_null_l) / std_null_l
        elif real_val > mean_null_l:
            z_layer[m] = 10.0
        else:
            z_layer[m] = 0.0
        p_layer[m] = (
            sum(1 for v in null_vals_l if v >= real_val) / len(null_vals_l)
        )

    # Compute Significance Profile (SP) vectors
    sp_degree = compute_significance_profile(z_degree)
    sp_layer = compute_significance_profile(z_layer)

    return {
        "domain": graph_data["domain"],
        "prompt": graph_data["prompt"],
        "n_nodes": graph_data["n_vertices"],
        "n_edges": n_edges,
        "real_counts": real_counts,
        "z_degree": z_degree,
        "z_layer": z_layer,
        "p_degree": p_degree,
        "p_layer": p_layer,
        "sp_degree": sp_degree,
        "sp_layer": sp_layer,
        "degree_acceptance_rate_mean": float(np.mean(degree_acceptance_rates)),
        "degree_acceptance_rate_std": float(np.std(degree_acceptance_rates)),
        "layer_acceptance_rate_mean": float(np.mean(layer_acceptance_rates)),
        "layer_acceptance_rate_std": float(np.std(layer_acceptance_rates)),
        "layer_stats": graph_data["layer_stats"],
        "timing": {
            "degree_null_seconds": round(t1 - t0, 2),
            "layer_null_seconds": round(t2 - t1, 2),
        },
    }


def compute_significance_profile(z_scores: list[float]) -> list[float]:
    """Compute Milo et al. (2004) Significance Profile: SP_i = Z_i / ||Z||.

    Normalizes Z-score vector to unit Euclidean length for cross-network
    comparison. Returns zero vector if all Z-scores are zero.
    """
    norm = math.sqrt(sum(z * z for z in z_scores))
    if norm > 0:
        return [z / norm for z in z_scores]
    return [0.0] * len(z_scores)


# ============================================================================
# PHASE D: Comparison Analysis
# ============================================================================


def run_comparison_analysis(
    results: list[dict],
    isoclass_to_man: dict[int, str],
    dag_possible_ids: list[int],
) -> dict:
    """Compare Z-scores from degree-preserving vs layer-preserving null models.

    For each DAG-possible motif type, computes paired Wilcoxon signed-rank
    test and classifies the outcome as one of:
        - genuine_signal: Z_layer ≈ Z_degree (layer doesn't explain it)
        - architectural_artifact: Z_layer ≈ 0 (layer fully explains it)
        - partial_explanation: 0 < Z_layer < Z_degree
        - not_overrepresented: Z_degree too small to analyse
    """
    comparison: dict[str, dict] = {}

    for motif_id in dag_possible_ids:
        man_label = isoclass_to_man[motif_id]
        z_deg_vals = [r["z_degree"][motif_id] for r in results]
        z_lay_vals = [r["z_layer"][motif_id] for r in results]
        differences = [d - l for d, l in zip(z_deg_vals, z_lay_vals)]

        # Wilcoxon signed-rank test on paired differences
        non_zero_diffs = [d for d in differences if abs(d) > 1e-10]
        if len(non_zero_diffs) >= 3:
            try:
                stat_val, p_wilcoxon = stats.wilcoxon(z_deg_vals, z_lay_vals)
                stat_val = float(stat_val)
                p_wilcoxon = float(p_wilcoxon)
            except ValueError:
                stat_val, p_wilcoxon = 0.0, 1.0
        else:
            stat_val, p_wilcoxon = 0.0, 1.0

        # Classification based on Z-score retention ratio
        mean_z_deg = float(np.mean(z_deg_vals))
        mean_z_lay = float(np.mean(z_lay_vals))

        if abs(mean_z_deg) > 0.5:
            ratio = mean_z_lay / mean_z_deg
        else:
            ratio = float("nan")

        if abs(mean_z_deg) < 1.5:
            classification = "not_overrepresented"
        elif not math.isnan(ratio) and ratio > 0.7:
            classification = "genuine_signal"
        elif not math.isnan(ratio) and ratio < 0.3:
            classification = "architectural_artifact"
        else:
            classification = "partial_explanation"

        comparison[man_label] = {
            "isoclass_id": motif_id,
            "mean_z_degree": round(mean_z_deg, 4),
            "mean_z_layer": round(mean_z_lay, 4),
            "mean_difference": round(float(np.mean(differences)), 4),
            "std_difference": round(float(np.std(differences)), 4),
            "wilcoxon_statistic": round(stat_val, 4),
            "wilcoxon_p_value": round(p_wilcoxon, 6),
            "z_retention_ratio": (
                round(ratio, 4) if not math.isnan(ratio) else None
            ),
            "classification": classification,
            "per_graph_z_degree": [round(v, 4) for v in z_deg_vals],
            "per_graph_z_layer": [round(v, 4) for v in z_lay_vals],
        }

        logger.info(
            f"Motif {man_label}: z_deg={mean_z_deg:.2f}, "
            f"z_lay={mean_z_lay:.2f}, "
            f"ratio={'N/A' if math.isnan(ratio) else f'{ratio:.2f}'}, "
            f"class={classification}, Wilcoxon p={p_wilcoxon:.4f}"
        )

    return comparison


# ============================================================================
# PHASE E: Layer-Only Baseline
# ============================================================================


def generate_layer_only_random_dag(
    nodes_per_layer: dict[int, int],
    edges_per_layer_pair: dict[tuple[int, int], int],
    rng: random.Random,
) -> igraph.Graph:
    """Generate random DAG matching ONLY the layer structure.

    Preserves: #nodes per layer, #edges per layer-pair.
    Randomizes: all within-pair wiring.
    Uses layer ordering for DAG guarantee (edges go from lower to higher
    node IDs, which are assigned in layer order).
    """
    # Assign node IDs in sorted layer order
    layer_node_map: dict[int, list[int]] = {}
    vid = 0
    total_nodes = 0
    for layer_idx in sorted(nodes_per_layer.keys()):
        count = nodes_per_layer[layer_idx]
        layer_node_map[layer_idx] = list(range(vid, vid + count))
        vid += count
        total_nodes += count

    g = igraph.Graph(n=total_nodes, directed=True)
    layer_arr = [0] * total_nodes
    for layer_idx, node_ids in layer_node_map.items():
        for nid in node_ids:
            layer_arr[nid] = layer_idx
    g.vs["layer"] = layer_arr

    # For each layer pair, add random edges
    all_edges: set[tuple[int, int]] = set()
    for (src_layer, tgt_layer), n_edges in edges_per_layer_pair.items():
        src_nodes = layer_node_map.get(src_layer, [])
        tgt_nodes = layer_node_map.get(tgt_layer, [])
        if not src_nodes or not tgt_nodes:
            continue

        # For within-layer edges (src_layer == tgt_layer), only allow s < t
        # to maintain DAG. For cross-layer forward edges, s < t is automatic
        # from our ID assignment. For backward edges (src > tgt layer),
        # skip since they can't maintain DAG with our ordering.
        if src_layer > tgt_layer:
            continue  # Skip backward layer edges for DAG safety

        max_possible = len(src_nodes) * len(tgt_nodes)
        if src_layer == tgt_layer:
            # Within-layer: only s < t pairs
            max_possible = len(src_nodes) * (len(src_nodes) - 1) // 2
        n_to_add = min(n_edges, max_possible)

        attempts = 0
        added_count = 0
        while added_count < n_to_add and attempts < n_to_add * 15:
            s = rng.choice(src_nodes)
            t = rng.choice(tgt_nodes)
            if s != t and s < t and (s, t) not in all_edges:
                all_edges.add((s, t))
                added_count += 1
            attempts += 1

    g.add_edges(list(all_edges))
    return g


def compute_layer_baseline(
    all_graphs: list[dict],
    isoclass_to_man: dict[int, str],
    dag_possible_ids: list[int],
    n_baseline: int = LAYER_BASELINE_N,
) -> dict:
    """Compute layer-only baseline: Z-scores of real vs layer-structure-only DAGs.

    For each graph, generates n_baseline random DAGs matching only the layer
    structure, computes motif census on each, and calculates Z-scores comparing
    the real graph to the layer-only ensemble.
    """
    logger.info(
        f"Computing layer-only baseline ({n_baseline} DAGs per graph)..."
    )
    rng = random.Random(SEED + 7777)
    baseline_results: dict[str, dict] = {}

    for gi, graph_info in enumerate(all_graphs):
        g = graph_info["graph"]
        real_counts = compute_motif_census(g)

        # Generate baseline DAGs
        baselines: list[list[int]] = []
        for _ in range(n_baseline):
            g_bl = generate_layer_only_random_dag(
                graph_info["nodes_per_layer"],
                graph_info["layer_pair_counts"],
                rng,
            )
            counts = compute_motif_census(g_bl)
            baselines.append(counts)
            del g_bl

        # Compute Z-scores for DAG-possible motifs
        z_layer_only: dict[str, float] = {}
        for motif_id in dag_possible_ids:
            man_label = isoclass_to_man[motif_id]
            real_val = real_counts[motif_id]
            null_vals = [b[motif_id] for b in baselines]
            mean_n = float(np.mean(null_vals))
            std_n = float(np.std(null_vals))
            if std_n > 0:
                z = (real_val - mean_n) / std_n
            elif real_val > mean_n:
                z = 10.0
            else:
                z = 0.0
            z_layer_only[man_label] = round(float(z), 4)

        key = f"{graph_info['domain']}_{gi}"
        baseline_results[key] = {
            "domain": graph_info["domain"],
            "z_layer_only": z_layer_only,
        }

        logger.debug(
            f"  Graph {gi} ({graph_info['domain']}): "
            f"z_layer_only={z_layer_only}"
        )

    logger.info("Layer-only baseline complete")
    return baseline_results


# ============================================================================
# OUTPUT CONSTRUCTION
# ============================================================================


def determine_conclusion(comparison: dict) -> str:
    """Determine overall conclusion from per-motif classifications."""
    classifications = [v["classification"] for v in comparison.values()]

    genuine = classifications.count("genuine_signal")
    artifact = classifications.count("architectural_artifact")
    partial = classifications.count("partial_explanation")
    not_over = classifications.count("not_overrepresented")

    if genuine > artifact and genuine > partial:
        return "genuine_signal"
    elif artifact > genuine and artifact > partial:
        return "architectural_artifact"
    elif not_over == len(classifications):
        return "no_motif_overrepresentation"
    else:
        return "mixed"


def build_output(
    results: list[dict],
    comparison: dict,
    layer_baseline: dict,
    isoclass_to_man: dict[int, str],
    dag_possible_ids: list[int],
    runtime_seconds: float,
) -> dict:
    """Build output dict conforming to exp_gen_sol_out.json schema."""

    conclusion = determine_conclusion(comparison)

    # Build per-graph examples
    examples: list[dict] = []
    for result in results:
        # Per-motif Z-score dicts for predict fields
        degree_z_dict: dict[str, float] = {}
        layer_z_dict: dict[str, float] = {}
        classification_dict: dict[str, str] = {}

        for mid in dag_possible_ids:
            man = isoclass_to_man[mid]
            degree_z_dict[man] = round(result["z_degree"][mid], 4)
            layer_z_dict[man] = round(result["z_layer"][mid], 4)
            if man in comparison:
                classification_dict[man] = comparison[man]["classification"]

        # Full per-graph output as JSON string
        output_data = {
            "real_counts": result["real_counts"],
            "z_degree": [round(v, 4) for v in result["z_degree"]],
            "z_layer": [round(v, 4) for v in result["z_layer"]],
            "p_degree": [round(v, 4) for v in result["p_degree"]],
            "p_layer": [round(v, 4) for v in result["p_layer"]],
            "sp_degree": [round(v, 4) for v in result["sp_degree"]],
            "sp_layer": [round(v, 4) for v in result["sp_layer"]],
            "degree_acceptance_rate": round(
                result["degree_acceptance_rate_mean"], 4
            ),
            "layer_acceptance_rate": round(
                result["layer_acceptance_rate_mean"], 4
            ),
            "layer_stats": result["layer_stats"],
            "timing": result["timing"],
        }

        example = {
            "input": result["prompt"],
            "output": json.dumps(output_data),
            "metadata_fold": result["domain"],
            "metadata_n_nodes_pruned": result["n_nodes"],
            "metadata_n_edges_pruned": result["n_edges"],
            "metadata_n_layers": result["layer_stats"]["n_layers"],
            "predict_degree_z": json.dumps(degree_z_dict),
            "predict_layer_z": json.dumps(layer_z_dict),
            "predict_motif_classification": json.dumps(classification_dict),
        }
        examples.append(example)

    output = {
        "metadata": {
            "experiment": "layer_architecture_confound_test",
            "description": (
                "Compares degree-preserving vs layer-preserving DAG null "
                "models to determine whether motif overrepresentation in "
                "LLM attribution graphs is a genuine computational signal "
                "or an artifact of layered transformer architecture."
            ),
            "parameters": {
                "n_null_models": N_NULL_MODELS,
                "swap_factor": SWAP_FACTOR,
                "prune_percentile": PRUNE_PERCENTILE,
                "min_nodes_for_census": MIN_NODES_FOR_CENSUS,
                "n_graphs_analyzed": len(results),
                "n_workers": N_WORKERS,
                "layer_baseline_n": LAYER_BASELINE_N,
                "seed": SEED,
            },
            "isoclass_mapping": {
                str(k): v for k, v in isoclass_to_man.items()
            },
            "dag_possible_motif_ids": dag_possible_ids,
            "comparison_analysis": comparison,
            "layer_baseline_results": layer_baseline,
            "summary": {
                "n_graphs": len(results),
                "domains": sorted(set(r["domain"] for r in results)),
                "mean_degree_acceptance_rate": round(
                    float(
                        np.mean(
                            [
                                r["degree_acceptance_rate_mean"]
                                for r in results
                            ]
                        )
                    ),
                    4,
                ),
                "mean_layer_acceptance_rate": round(
                    float(
                        np.mean(
                            [
                                r["layer_acceptance_rate_mean"]
                                for r in results
                            ]
                        )
                    ),
                    4,
                ),
                "overall_conclusion": conclusion,
            },
            "runtime_seconds": round(runtime_seconds, 1),
        },
        "datasets": [
            {
                "dataset": "neuronpedia_attribution_graphs",
                "examples": examples,
            }
        ],
    }

    return output


# ============================================================================
# MAIN
# ============================================================================


@logger.catch
def main():
    """Run the Layer-Architecture Confound Test experiment."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Layer-Architecture Confound Test")
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM")
    logger.info(
        f"Parameters: N_NULL={N_NULL_MODELS}, SWAP_FACTOR={SWAP_FACTOR}, "
        f"MAX_EXAMPLES={MAX_EXAMPLES}"
    )
    logger.info(f"Workers: {N_WORKERS}")
    logger.info("=" * 60)

    # Phase 0: Build isoclass mapping
    isoclass_to_man, man_to_isoclass, dag_possible_ids = (
        build_isoclass_mapping()
    )

    # Phase A: Load and characterize graphs
    all_graphs = load_graphs(max_examples=MAX_EXAMPLES)

    if not all_graphs:
        logger.error("No graphs loaded — exiting")
        return

    # Log graph statistics
    for gi, ginfo in enumerate(all_graphs):
        logger.info(
            f"  Graph {gi}: {ginfo['domain']}, "
            f"{ginfo['n_nodes']} nodes, {ginfo['n_edges']} edges, "
            f"{ginfo['n_layers']} layers, "
            f"span1={ginfo['frac_span_1_layer']:.2f}"
        )

    # Serialize graph data for multiprocessing workers
    graph_args: list[tuple] = []
    for gi, graph_info in enumerate(all_graphs):
        g = graph_info["graph"]
        serialized = {
            "n_vertices": g.vcount(),
            "edges": g.get_edgelist(),
            "vertex_layers": list(g.vs["layer"]),
            "domain": graph_info["domain"],
            "prompt": graph_info["prompt"],
            "layer_stats": {
                "n_layers": graph_info["n_layers"],
                "frac_span_1": round(graph_info["frac_span_1_layer"], 4),
                "frac_span_2plus": round(
                    graph_info["frac_span_2plus_layers"], 4
                ),
                "layer_distance_mean": round(
                    graph_info["layer_distance_mean"], 4
                ),
            },
        }
        worker_seed = SEED + gi * 1000
        graph_args.append((serialized, N_NULL_MODELS, SWAP_FACTOR, worker_seed))

    # Phase C: Process graphs (motif census under both null models)
    logger.info(
        f"Processing {len(graph_args)} graphs with {N_WORKERS} workers, "
        f"{N_NULL_MODELS} null models each..."
    )

    results: list[dict] = []
    if N_WORKERS > 1 and len(graph_args) > 1:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(process_single_graph, args): i
                for i, args in enumerate(graph_args)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    elapsed = time.time() - start_time
                    logger.info(
                        f"  Graph {idx} ({result['domain']}) complete: "
                        f"deg_acc={result['degree_acceptance_rate_mean']:.3f}, "
                        f"lay_acc={result['layer_acceptance_rate_mean']:.3f}, "
                        f"deg_t={result['timing']['degree_null_seconds']:.0f}s, "
                        f"lay_t={result['timing']['layer_null_seconds']:.0f}s, "
                        f"total_elapsed={elapsed:.0f}s"
                    )
                except Exception:
                    logger.exception(f"Failed processing graph {idx}")
    else:
        # Sequential processing (for debugging or single graph)
        for i, args in enumerate(graph_args):
            try:
                result = process_single_graph(args)
                results.append(result)
                elapsed = time.time() - start_time
                logger.info(
                    f"  Graph {i} ({result['domain']}) complete: "
                    f"deg_acc={result['degree_acceptance_rate_mean']:.3f}, "
                    f"lay_acc={result['layer_acceptance_rate_mean']:.3f}, "
                    f"elapsed={elapsed:.0f}s"
                )
            except Exception:
                logger.exception(f"Failed processing graph {i}")

    if not results:
        logger.error("No results produced — exiting")
        return

    logger.info(f"Completed {len(results)}/{len(graph_args)} graphs")

    # Validate: DAG-impossible motif counts should all be zero
    dag_impossible_ids = [
        i for i in range(16) if i not in dag_possible_ids
        and isoclass_to_man.get(i, "003") not in ("003", "012")
    ]
    for result in results:
        for mid in dag_impossible_ids:
            if result["real_counts"][mid] != 0:
                logger.warning(
                    f"DAG-impossible motif {isoclass_to_man.get(mid, mid)} "
                    f"has count {result['real_counts'][mid]} in "
                    f"{result['domain']} — data integrity issue!"
                )

    # Phase D: Comparison analysis
    logger.info("Running comparison analysis...")
    comparison = run_comparison_analysis(
        results, isoclass_to_man, dag_possible_ids,
    )

    # Phase E: Layer-only baseline
    layer_baseline = compute_layer_baseline(
        all_graphs, isoclass_to_man, dag_possible_ids,
    )

    runtime = time.time() - start_time

    # Build output
    output = build_output(
        results, comparison, layer_baseline,
        isoclass_to_man, dag_possible_ids, runtime,
    )

    # Save output
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output saved to {out_path}")
    logger.info(f"Total runtime: {runtime:.1f}s")

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    conclusion = output["metadata"]["summary"]["overall_conclusion"]
    logger.info(f"Overall conclusion: {conclusion}")
    for man_label, comp in comparison.items():
        logger.info(
            f"  {man_label}: z_deg={comp['mean_z_degree']:.2f}, "
            f"z_lay={comp['mean_z_layer']:.2f}, "
            f"retention={comp['z_retention_ratio']}, "
            f"class={comp['classification']}"
        )
    logger.info(
        f"Mean acceptance rates: "
        f"degree={output['metadata']['summary']['mean_degree_acceptance_rate']:.3f}, "
        f"layer={output['metadata']['summary']['mean_layer_acceptance_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
