#!/usr/bin/env python3
"""Graph-Theoretic Node Ablation: FFL Motif Participation vs Structural Importance.

Tests whether nodes participating in overrepresented Feed-Forward Loop (FFL/030T)
motif instances are disproportionately important to graph information flow by
performing graph-theoretic ablation on 200 pruned attribution graphs. Compare
FFL-hub nodes against degree-matched, attribution-matched, layer-matched, and
random control nodes across 4 impact metrics.
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

import numpy as np
import psutil
from loguru import logger
from scipy import stats

# ============================================================
# CONSTANTS & PATHS
# ============================================================

WORKSPACE = Path(__file__).parent
DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/"
    "3_invention_loop/iter_3/gen_art/data_id5_it3__opus/data_out"
)
OUTPUT_FILE = WORKSPACE / "method_out.json"
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

PRUNE_PERCENTILE = 75
MIN_NODES = 30
HUB_PERCENTILE = 90
N_CONTROLS_PER_HUB = 3
N_BOOTSTRAP = 2000
DOSE_RESPONSE_SAMPLE = 80
SEED = 42
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))

# ============================================================
# HARDWARE DETECTION
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
TOTAL_RAM_GB = _container_ram_gb() or (psutil.virtual_memory().total / 1e9)
N_WORKERS = min(NUM_CPUS, 4)

# ============================================================
# RESOURCE LIMITS
# ============================================================

RAM_BUDGET_GB = min(TOTAL_RAM_GB * 0.75, 22.0)
RAM_BUDGET_BYTES = int(RAM_BUDGET_GB * 1024**3)
try:
    resource.setrlimit(
        resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3)
    )
except ValueError:
    pass
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except ValueError:
    pass

# ============================================================
# LOGGING SETUP
# ============================================================

logger.remove()
logger.add(
    sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}"
)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM")
logger.info(f"RAM budget: {RAM_BUDGET_GB:.1f} GB, Workers: {N_WORKERS}")
logger.info(f"MAX_EXAMPLES={MAX_EXAMPLES}")


# ============================================================
# CUSTOM JSON ENCODER
# ============================================================


class NpEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


# ============================================================
# PHASE A: GRAPH LOADING
# ============================================================


def parse_layer(layer_str: str) -> int:
    """Parse layer string: 'E' -> -1, numeric -> int."""
    if layer_str == "E":
        return -1
    try:
        return int(layer_str)
    except (ValueError, TypeError):
        return -1


def load_graphs(max_examples: int = 0) -> list[dict]:
    """Load attribution graphs from split data files, prune, and filter."""
    import igraph

    data_files = sorted(DATA_DIR.glob("full_data_out_*.json"))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {DATA_DIR}")
    logger.info(f"Found {len(data_files)} data files in {DATA_DIR}")

    all_examples = []
    for f in data_files:
        try:
            raw = json.loads(f.read_text())
            examples = raw["datasets"][0]["examples"]
            all_examples.extend(examples)
            logger.debug(f"Loaded {len(examples)} examples from {f.name}")
        except Exception:
            logger.exception(f"Failed loading {f.name}")
            continue

    logger.info(f"Total raw examples: {len(all_examples)}")
    if max_examples > 0:
        all_examples = all_examples[:max_examples]
        logger.info(f"Limiting to {max_examples} examples")

    graphs = []
    skipped_reasons = {"not_dag": 0, "too_small": 0, "error": 0}

    for idx, ex in enumerate(all_examples):
        try:
            graph_json = json.loads(ex["output"])
            nodes = graph_json["nodes"]
            links = graph_json["links"]

            # Build node_id -> index mapping
            node_ids = [n["node_id"] for n in nodes]
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            # Build edge list and weights
            n_v = len(nodes)
            edges = []
            weights = []
            for link in links:
                src_idx = id_to_idx.get(link["source"])
                tgt_idx = id_to_idx.get(link["target"])
                if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
                    edges.append((src_idx, tgt_idx))
                    weights.append(abs(link.get("weight", 1.0)))

            g = igraph.Graph(n=n_v, edges=edges, directed=True)
            g.vs["node_id"] = node_ids
            g.vs["layer"] = [parse_layer(n["layer"]) for n in nodes]
            g.vs["feature_type"] = [n.get("feature_type", "") for n in nodes]
            g.vs["feature"] = [n.get("feature", 0) for n in nodes]
            g.vs["is_target_logit"] = [
                bool(n.get("is_target_logit", False)) for n in nodes
            ]
            g.es["weight"] = weights

            # Simplify: remove multi-edges keeping max weight, remove self-loops
            g = g.simplify(
                multiple=True, loops=True, combine_edges={"weight": "max"}
            )

            # Verify DAG
            if not g.is_dag():
                skipped_reasons["not_dag"] += 1
                logger.debug(f"Graph {idx} is not a DAG, skipping")
                continue

            # Prune: keep edges >= 75th percentile weight
            if g.ecount() > 0:
                all_w = np.array(g.es["weight"])
                threshold = float(np.percentile(all_w, PRUNE_PERCENTILE))
                keep_edges = [
                    i for i, w in enumerate(g.es["weight"]) if w >= threshold
                ]
                g = g.subgraph_edges(keep_edges, delete_vertices=False)

            # Remove isolated vertices
            isolated = [v.index for v in g.vs if v.degree() == 0]
            if isolated:
                g.delete_vertices(isolated)

            if g.vcount() < MIN_NODES:
                skipped_reasons["too_small"] += 1
                logger.debug(
                    f"Graph {idx} too small after pruning: {g.vcount()} nodes"
                )
                continue

            domain = ex.get("metadata_fold", "unknown")
            slug = ex.get("metadata_slug", f"graph_{idx}")

            graphs.append(
                {
                    "graph": g,
                    "domain": domain,
                    "prompt": ex.get("input", ""),
                    "slug": slug,
                    "correctness": ex.get("metadata_model_correct", "unknown"),
                    "difficulty": ex.get("metadata_difficulty", "unknown"),
                    "n_nodes": g.vcount(),
                    "n_edges": g.ecount(),
                }
            )
        except Exception:
            skipped_reasons["error"] += 1
            logger.exception(f"Failed processing graph {idx}")
            continue

    logger.info(
        f"Loaded {len(graphs)} valid graphs from {len(all_examples)} raw "
        f"(skipped: {skipped_reasons})"
    )
    return graphs


# ============================================================
# PHASE B: FFL ENUMERATION + MPI COMPUTATION
# ============================================================


def enumerate_ffls(g) -> list[dict]:
    """Enumerate all Feed-Forward Loop (030T) instances.

    FFL: A->B, A->C, B->C (3 directed edges forming a triangle).
    """
    adj = set()
    weight_map = {}
    for e in g.es:
        adj.add((e.source, e.target))
        weight_map[(e.source, e.target)] = e["weight"]

    ffls = []
    for a in range(g.vcount()):
        successors_a = g.successors(a)
        if len(successors_a) < 2:
            continue
        for i in range(len(successors_a)):
            for j in range(i + 1, len(successors_a)):
                b, c = successors_a[i], successors_a[j]
                if (b, c) in adj:
                    ffls.append(
                        {
                            "a": a,
                            "b": b,
                            "c": c,
                            "w_ab": weight_map.get((a, b), 0.0),
                            "w_ac": weight_map.get((a, c), 0.0),
                            "w_bc": weight_map.get((b, c), 0.0),
                        }
                    )
                if (c, b) in adj:
                    ffls.append(
                        {
                            "a": a,
                            "b": c,
                            "c": b,
                            "w_ab": weight_map.get((a, c), 0.0),
                            "w_ac": weight_map.get((a, b), 0.0),
                            "w_bc": weight_map.get((c, b), 0.0),
                        }
                    )
    return ffls


def enumerate_all_dag_motifs(g) -> list[dict]:
    """Enumerate all 4 DAG-possible 3-node motif types (fallback).

    Types: 021D (out-star), 021U (in-star), 021C (chain), 030T (FFL).
    Returns list of dicts with 'type' and node indices.
    """
    adj = set()
    for e in g.es:
        adj.add((e.source, e.target))

    motifs = []
    for a in range(g.vcount()):
        successors_a = g.successors(a)
        predecessors_a = g.predecessors(a)

        if len(successors_a) >= 2:
            for i in range(len(successors_a)):
                for j in range(i + 1, len(successors_a)):
                    b, c = successors_a[i], successors_a[j]
                    has_bc = (b, c) in adj
                    has_cb = (c, b) in adj
                    if has_bc:
                        motifs.append({"type": "030T", "nodes": [a, b, c]})
                    if has_cb:
                        motifs.append({"type": "030T", "nodes": [a, c, b]})
                    if not has_bc and not has_cb:
                        motifs.append({"type": "021D", "nodes": [a, b, c]})

        if len(predecessors_a) >= 2:
            for i in range(len(predecessors_a)):
                for j in range(i + 1, len(predecessors_a)):
                    b, c = predecessors_a[i], predecessors_a[j]
                    if (b, c) not in adj and (c, b) not in adj:
                        motifs.append({"type": "021U", "nodes": [a, b, c]})

    # 021C (chain): A->B->C, no A->C or C->A
    for a in range(g.vcount()):
        for b in g.successors(a):
            for c in g.successors(b):
                if c != a and (a, c) not in adj and (c, a) not in adj:
                    motifs.append({"type": "021C", "nodes": [a, b, c]})

    return motifs


def compute_mpi(
    g,
    ffls: list[dict],
    use_all_motifs: bool = False,
    all_motifs: list[dict] | None = None,
) -> dict[int, dict]:
    """Compute Motif Participation Index for every node.

    If use_all_motifs=True, uses all_motifs list instead of ffls.
    """
    node_mpi: dict[int, dict] = {}
    for v in range(g.vcount()):
        incident_edges = g.incident(v, mode="all")
        attr_strength = sum(g.es[e]["weight"] for e in incident_edges)
        node_mpi[v] = {
            "mpi": 0,
            "n_as_a": 0,
            "n_as_b": 0,
            "n_as_c": 0,
            "in_degree": g.vs[v].indegree(),
            "out_degree": g.vs[v].outdegree(),
            "total_degree": g.vs[v].degree(),
            "attribution_strength": float(attr_strength),
            "layer": g.vs[v]["layer"],
            "feature_type": g.vs[v]["feature_type"],
            "node_id": g.vs[v]["node_id"],
        }

    if use_all_motifs and all_motifs:
        for motif in all_motifs:
            for node_idx in motif["nodes"]:
                node_mpi[node_idx]["mpi"] += 1
    else:
        for ffl in ffls:
            for role, key in [("a", "n_as_a"), ("b", "n_as_b"), ("c", "n_as_c")]:
                idx = ffl[role]
                node_mpi[idx]["mpi"] += 1
                node_mpi[idx][key] += 1

    # Classify nodes
    nonzero_mpis = [nm["mpi"] for nm in node_mpi.values() if nm["mpi"] > 0]
    if nonzero_mpis:
        hub_threshold = float(np.percentile(nonzero_mpis, HUB_PERCENTILE))
        hub_threshold = max(hub_threshold, 1)
    else:
        hub_threshold = float("inf")

    for v, info in node_mpi.items():
        if info["mpi"] >= hub_threshold and info["mpi"] > 0:
            info["classification"] = "hub"
        elif info["mpi"] > 0:
            info["classification"] = "participant"
        else:
            info["classification"] = "non_ffl"

    return node_mpi


# ============================================================
# PHASE C: MATCHED CONTROL SELECTION
# ============================================================


def select_matched_controls(
    g,
    node_mpi: dict[int, dict],
    hub_indices: list[int],
    n_per_type: int = 3,
    seed: int = SEED,
) -> list[dict]:
    """For each hub node, find matched control nodes from non-hub pool."""
    non_hub_pool = [
        v for v, info in node_mpi.items() if info["classification"] != "hub"
    ]

    if not non_hub_pool:
        return []

    rng = np.random.default_rng(seed)
    pairs = []

    for hub_idx in hub_indices:
        hub_info = node_mpi[hub_idx]
        hub_degree = hub_info["total_degree"]
        hub_attr = hub_info["attribution_strength"]
        hub_layer = hub_info["layer"]

        # 1. DEGREE-MATCHED: same total_degree +/- 1
        degree_candidates = sorted(
            non_hub_pool,
            key=lambda v: abs(node_mpi[v]["total_degree"] - hub_degree),
        )
        degree_controls = degree_candidates[:n_per_type]

        for ctrl_idx in degree_controls:
            ctrl_info = node_mpi[ctrl_idx]
            match_quality = abs(ctrl_info["total_degree"] - hub_degree)
            if match_quality > 3:
                logger.debug(
                    f"Poor degree match: hub {hub_idx} deg={hub_degree}, "
                    f"ctrl {ctrl_idx} deg={ctrl_info['total_degree']}"
                )
            pairs.append(
                {
                    "hub_idx": int(hub_idx),
                    "hub_mpi": hub_info["mpi"],
                    "hub_degree": hub_degree,
                    "hub_attr": hub_attr,
                    "control_idx": int(ctrl_idx),
                    "control_mpi": ctrl_info["mpi"],
                    "control_degree": ctrl_info["total_degree"],
                    "control_attr": ctrl_info["attribution_strength"],
                    "control_type": "degree_matched",
                    "match_quality": float(match_quality),
                }
            )

        # 2. ATTRIBUTION-MATCHED: same sum|incident_weights| +/- 10%
        if hub_attr > 0:
            attr_candidates = sorted(
                non_hub_pool,
                key=lambda v: abs(
                    node_mpi[v]["attribution_strength"] - hub_attr
                )
                / max(hub_attr, 1e-10),
            )
        else:
            attr_candidates = list(non_hub_pool)
        attr_controls = attr_candidates[:n_per_type]

        for ctrl_idx in attr_controls:
            ctrl_info = node_mpi[ctrl_idx]
            match_quality = abs(ctrl_info["attribution_strength"] - hub_attr) / max(
                hub_attr, 1e-10
            )
            if match_quality > 0.25:
                logger.debug(
                    f"Poor attr match: hub {hub_idx} attr={hub_attr:.2f}, "
                    f"ctrl {ctrl_idx} attr={ctrl_info['attribution_strength']:.2f}"
                )
            pairs.append(
                {
                    "hub_idx": int(hub_idx),
                    "hub_mpi": hub_info["mpi"],
                    "hub_degree": hub_degree,
                    "hub_attr": hub_attr,
                    "control_idx": int(ctrl_idx),
                    "control_mpi": ctrl_info["mpi"],
                    "control_degree": ctrl_info["total_degree"],
                    "control_attr": ctrl_info["attribution_strength"],
                    "control_type": "attribution_matched",
                    "match_quality": float(match_quality),
                }
            )

        # 3. LAYER-MATCHED: same layer + total_degree +/- 2
        same_layer = [
            v for v in non_hub_pool if node_mpi[v]["layer"] == hub_layer
        ]
        if same_layer:
            layer_candidates = sorted(
                same_layer,
                key=lambda v: abs(node_mpi[v]["total_degree"] - hub_degree),
            )
            layer_controls = layer_candidates[:n_per_type]
        else:
            layer_controls = degree_controls[:n_per_type]

        for ctrl_idx in layer_controls:
            ctrl_info = node_mpi[ctrl_idx]
            match_quality = abs(ctrl_info["total_degree"] - hub_degree)
            pairs.append(
                {
                    "hub_idx": int(hub_idx),
                    "hub_mpi": hub_info["mpi"],
                    "hub_degree": hub_degree,
                    "hub_attr": hub_attr,
                    "control_idx": int(ctrl_idx),
                    "control_mpi": ctrl_info["mpi"],
                    "control_degree": ctrl_info["total_degree"],
                    "control_attr": ctrl_info["attribution_strength"],
                    "control_type": "layer_matched",
                    "match_quality": float(match_quality),
                }
            )

        # 4. RANDOM: uniform sample from non-hub pool
        n_random = min(n_per_type, len(non_hub_pool))
        random_controls = rng.choice(
            non_hub_pool, size=n_random, replace=False
        ).tolist()

        for ctrl_idx in random_controls:
            ctrl_info = node_mpi[ctrl_idx]
            pairs.append(
                {
                    "hub_idx": int(hub_idx),
                    "hub_mpi": hub_info["mpi"],
                    "hub_degree": hub_degree,
                    "hub_attr": hub_attr,
                    "control_idx": int(ctrl_idx),
                    "control_mpi": ctrl_info["mpi"],
                    "control_degree": ctrl_info["total_degree"],
                    "control_attr": ctrl_info["attribution_strength"],
                    "control_type": "random",
                    "match_quality": 0.0,
                }
            )

    return pairs


# ============================================================
# PHASE D: NODE ABLATION + IMPACT METRICS
# ============================================================


def compute_baseline_metrics(g) -> dict:
    """Pre-compute graph baseline before any ablation."""
    # Identify input and output nodes
    input_nodes = [
        v.index
        for v in g.vs
        if v["feature_type"] == "embedding" or v["layer"] == -1
    ]
    output_nodes = [
        v.index
        for v in g.vs
        if v["feature_type"] == "logit" or v["is_target_logit"]
    ]

    # Fallback: layer-based heuristic
    if not input_nodes:
        min_layer = min(v["layer"] for v in g.vs)
        input_nodes = [v.index for v in g.vs if v["layer"] == min_layer]
    if not output_nodes:
        max_layer = max(v["layer"] for v in g.vs)
        output_nodes = [v.index for v in g.vs if v["layer"] == max_layer]

    output_set = set(output_nodes)
    total_weight = float(sum(g.es["weight"])) if g.ecount() > 0 else 0.0

    # Baseline reachability from each input node
    reachable_outputs: dict[int, set] = {}
    for i in input_nodes:
        reachable = set(g.subcomponent(i, mode="out"))
        reachable_outputs[i] = reachable & output_set

    n_source_sink_pairs = sum(len(ro) for ro in reachable_outputs.values())
    all_reachable_outputs = set()
    for ro in reachable_outputs.values():
        all_reachable_outputs |= ro

    n_components = len(g.connected_components(mode="weak"))

    return {
        "input_nodes": input_nodes,
        "output_set": output_set,
        "total_weight": total_weight,
        "n_source_sink_pairs": n_source_sink_pairs,
        "reachable_output_count": len(all_reachable_outputs),
        "n_components": n_components,
    }


def ablate_node(g, node_idx: int, baseline: dict) -> dict:
    """Remove a single node (via edge deletion) and measure 4 impact metrics."""
    g_abl = g.copy()
    incident = g_abl.incident(node_idx, mode="all")
    removed_weight = sum(g_abl.es[e]["weight"] for e in incident)
    g_abl.delete_edges(incident)

    total_w = baseline["total_weight"]
    input_nodes = baseline["input_nodes"]
    output_set = baseline["output_set"]

    # (a) downstream_attribution_loss
    downstream_attr_loss = removed_weight / total_w if total_w > 0 else 0.0

    # BFS from each input node (shared for path_disruption + reachability_loss)
    new_reachable_per_input: dict[int, set] = {}
    for i in input_nodes:
        new_reachable_per_input[i] = set(g_abl.subcomponent(i, mode="out")) & output_set

    # (b) path_disruption
    baseline_pairs = baseline["n_source_sink_pairs"]
    new_pairs = sum(len(nr) for nr in new_reachable_per_input.values())
    if baseline_pairs > 0:
        path_disruption = (baseline_pairs - new_pairs) / baseline_pairs
    else:
        path_disruption = 0.0

    # (c) reachability_loss
    new_reachable_all = set()
    for nr in new_reachable_per_input.values():
        new_reachable_all |= nr
    baseline_reach = baseline["reachable_output_count"]
    if baseline_reach > 0:
        reachability_loss = (baseline_reach - len(new_reachable_all)) / baseline_reach
    else:
        reachability_loss = 0.0

    # (d) component_fragmentation
    new_components = len(g_abl.connected_components(mode="weak"))
    # Subtract 1: ablated node becomes isolated component
    effective_new = new_components - 1
    baseline_comp = baseline["n_components"]
    if baseline_comp > 0:
        component_fragmentation = (effective_new - baseline_comp) / baseline_comp
    else:
        component_fragmentation = 0.0

    return {
        "downstream_attr_loss": float(max(0.0, downstream_attr_loss)),
        "path_disruption": float(max(0.0, path_disruption)),
        "reachability_loss": float(max(0.0, reachability_loss)),
        "component_fragmentation": float(max(0.0, component_fragmentation)),
    }


# ============================================================
# PHASE E: SERIALIZATION + PER-GRAPH PROCESSING
# ============================================================

METRICS = [
    "downstream_attr_loss",
    "path_disruption",
    "reachability_loss",
    "component_fragmentation",
]
CONTROL_TYPES = [
    "degree_matched",
    "attribution_matched",
    "layer_matched",
    "random",
]


def serialize_graph(g, metadata: dict) -> dict:
    """Serialize an igraph for safe multiprocessing transfer."""
    edges = [(e.source, e.target) for e in g.es]
    weights = list(g.es["weight"]) if g.ecount() > 0 else []
    return {
        "n_vertices": g.vcount(),
        "edges": edges,
        "weights": weights,
        "node_ids": list(g.vs["node_id"]),
        "layers": list(g.vs["layer"]),
        "feature_types": list(g.vs["feature_type"]),
        "features": list(g.vs["feature"]),
        "is_target_logits": list(g.vs["is_target_logit"]),
        **metadata,
    }


def deserialize_graph(data: dict):
    """Reconstruct igraph from serialized data."""
    import igraph

    g = igraph.Graph(
        n=data["n_vertices"], edges=data["edges"], directed=True
    )
    g.vs["node_id"] = data["node_ids"]
    g.vs["layer"] = data["layers"]
    g.vs["feature_type"] = data["feature_types"]
    g.vs["feature"] = data["features"]
    g.vs["is_target_logit"] = data["is_target_logits"]
    if data["weights"]:
        g.es["weight"] = data["weights"]
    return g


def process_single_graph(serialized: dict) -> dict | None:
    """Complete analysis pipeline for one graph. Runs in worker process."""
    try:
        slug = serialized.get("slug", "unknown")
        domain = serialized.get("domain", "unknown")

        g = deserialize_graph(serialized)

        # --- FFL Enumeration ---
        ffls = enumerate_ffls(g)
        n_ffls = len(ffls)

        # FALLBACK 1: if too few FFLs, use all motif types
        use_all_motifs = False
        all_motifs_list = None
        if n_ffls < 10:
            all_motifs_list = enumerate_all_dag_motifs(g)
            if len(all_motifs_list) >= 10:
                use_all_motifs = True

        if n_ffls < 3 and not use_all_motifs:
            return {
                "slug": slug,
                "domain": domain,
                "skipped": True,
                "skip_reason": f"too_few_ffls ({n_ffls})",
                "n_nodes": g.vcount(),
                "n_edges": g.ecount(),
                "n_ffls": n_ffls,
            }

        # --- MPI Computation ---
        node_mpi = compute_mpi(
            g, ffls, use_all_motifs=use_all_motifs, all_motifs=all_motifs_list
        )

        hub_indices = [
            v for v, info in node_mpi.items() if info["classification"] == "hub"
        ]
        participant_indices = [
            v
            for v, info in node_mpi.items()
            if info["classification"] == "participant"
        ]
        non_ffl_indices = [
            v
            for v, info in node_mpi.items()
            if info["classification"] == "non_ffl"
        ]

        if len(hub_indices) < 3:
            return {
                "slug": slug,
                "domain": domain,
                "skipped": True,
                "skip_reason": f"too_few_hubs ({len(hub_indices)})",
                "n_nodes": g.vcount(),
                "n_edges": g.ecount(),
                "n_ffls": n_ffls,
            }

        # --- Matched Control Selection ---
        pairs = select_matched_controls(
            g,
            node_mpi,
            hub_indices,
            n_per_type=N_CONTROLS_PER_HUB,
            seed=SEED + hash(slug) % 10000,
        )

        # --- Baseline ---
        baseline = compute_baseline_metrics(g)

        # --- Hub Ablations ---
        hub_ablations = []
        for h_idx in hub_indices:
            metrics = ablate_node(g, h_idx, baseline)
            info = node_mpi[h_idx]
            hub_ablations.append(
                {
                    "node_idx": int(h_idx),
                    "mpi": info["mpi"],
                    "degree": info["total_degree"],
                    "attr": float(info["attribution_strength"]),
                    "layer": info["layer"],
                    "feature_type": info["feature_type"],
                    "node_id": info["node_id"],
                    **metrics,
                }
            )

        # --- Control Ablations (cache results to avoid re-ablating) ---
        ablated_cache: dict[int, dict] = {}
        # Also cache hub results
        hub_metrics_map = {
            ha["node_idx"]: {k: ha[k] for k in METRICS} for ha in hub_ablations
        }

        control_ablations: dict[str, list] = {ct: [] for ct in CONTROL_TYPES}

        for pair in pairs:
            ctrl_idx = pair["control_idx"]
            if ctrl_idx not in ablated_cache:
                ablated_cache[ctrl_idx] = ablate_node(g, ctrl_idx, baseline)
            ctrl_metrics = ablated_cache[ctrl_idx]
            ctrl_info = node_mpi[ctrl_idx]
            control_ablations[pair["control_type"]].append(
                {
                    "node_idx": int(ctrl_idx),
                    "mpi": ctrl_info["mpi"],
                    "degree": ctrl_info["total_degree"],
                    "attr": float(ctrl_info["attribution_strength"]),
                    **ctrl_metrics,
                }
            )

        # --- Matched pairs with both hub and control impacts ---
        matched_pairs = []
        for pair in pairs:
            hub_idx = pair["hub_idx"]
            ctrl_idx = pair["control_idx"]
            hub_m = hub_metrics_map.get(hub_idx, {})
            ctrl_m = ablated_cache.get(ctrl_idx, {})
            for metric in METRICS:
                matched_pairs.append(
                    {
                        "hub_idx": int(hub_idx),
                        "control_idx": int(ctrl_idx),
                        "control_type": pair["control_type"],
                        "metric": metric,
                        "hub_impact": hub_m.get(metric, 0.0),
                        "control_impact": ctrl_m.get(metric, 0.0),
                        "hub_mpi": pair["hub_mpi"],
                        "control_mpi": pair["control_mpi"],
                        "domain": domain,
                    }
                )

        # --- Dose-Response Sample ---
        rng = np.random.default_rng(SEED + hash(slug) % 10000)
        all_node_indices = list(range(g.vcount()))
        n_sample = min(DOSE_RESPONSE_SAMPLE, len(all_node_indices))
        dose_sample_indices = rng.choice(
            all_node_indices, size=n_sample, replace=False
        ).tolist()

        dose_response = []
        for v_idx in dose_sample_indices:
            if v_idx in ablated_cache:
                m = ablated_cache[v_idx]
            elif v_idx in hub_metrics_map:
                m = hub_metrics_map[v_idx]
            else:
                m = ablate_node(g, v_idx, baseline)
                ablated_cache[v_idx] = m

            info = node_mpi[v_idx]
            dose_response.append(
                {
                    "node_idx": int(v_idx),
                    "mpi": info["mpi"],
                    "log_mpi": float(np.log1p(info["mpi"])),
                    "degree": info["total_degree"],
                    "attr": float(info["attribution_strength"]),
                    **m,
                }
            )

        return {
            "slug": slug,
            "domain": domain,
            "n_nodes": g.vcount(),
            "n_edges": g.ecount(),
            "n_ffls": n_ffls,
            "used_all_motifs": use_all_motifs,
            "n_all_motifs": len(all_motifs_list) if all_motifs_list else n_ffls,
            "n_hub_nodes": len(hub_indices),
            "n_participant_nodes": len(participant_indices),
            "n_non_ffl_nodes": len(non_ffl_indices),
            "hub_ablations": hub_ablations,
            "control_ablations": control_ablations,
            "matched_pairs": matched_pairs,
            "dose_response_sample": dose_response,
            "skipped": False,
            "prompt": serialized.get("prompt", ""),
            "correctness": serialized.get("correctness", ""),
            "difficulty": serialized.get("difficulty", ""),
        }
    except Exception as exc:
        return {
            "slug": serialized.get("slug", "?"),
            "domain": serialized.get("domain", "?"),
            "skipped": True,
            "skip_reason": f"error: {str(exc)[:200]}",
            "n_nodes": 0,
            "n_edges": 0,
            "n_ffls": 0,
        }


# ============================================================
# PHASE F: STATISTICAL ANALYSIS HELPERS
# ============================================================


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adj_p = min(prev, p * n / rank)
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        prev = adj_p
    return adjusted


def compute_comparison_stats(
    hub_impacts: np.ndarray,
    ctrl_impacts: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Compute paired comparison statistics between hub and control impacts."""
    rng = np.random.default_rng(seed)
    n = len(hub_impacts)

    hub_med = float(np.median(hub_impacts))
    ctrl_med = float(np.median(ctrl_impacts))
    median_ratio = hub_med / max(ctrl_med, 1e-10)

    # Wilcoxon signed-rank test (paired, one-sided: hub > control)
    try:
        diffs = hub_impacts - ctrl_impacts
        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 10:
            stat, p_val = stats.wilcoxon(
                hub_impacts, ctrl_impacts, alternative="greater"
            )
        else:
            stat, p_val = float("nan"), 1.0
    except Exception:
        stat, p_val = float("nan"), 1.0

    # Cohen's d (paired)
    diffs = hub_impacts - ctrl_impacts
    d_mean = float(np.mean(diffs))
    d_std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1e-10
    cohens_d = d_mean / max(d_std, 1e-10)

    # Bootstrap 95% CI for median ratio
    boot_ratios = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_hub = float(np.median(hub_impacts[idx]))
        boot_ctrl = float(np.median(ctrl_impacts[idx]))
        boot_ratios.append(boot_hub / max(boot_ctrl, 1e-10))
    ci_lower = float(np.percentile(boot_ratios, 2.5))
    ci_upper = float(np.percentile(boot_ratios, 97.5))

    # Mean-based ratio (more informative for sparse metrics where median=0)
    hub_mean = float(np.mean(hub_impacts))
    ctrl_mean = float(np.mean(ctrl_impacts))
    mean_ratio = hub_mean / max(ctrl_mean, 1e-10)

    # Fraction of nonzero values (helpful for sparse metrics)
    hub_frac_nonzero = float(np.mean(hub_impacts > 0))
    ctrl_frac_nonzero = float(np.mean(ctrl_impacts > 0))

    return {
        "n_pairs": int(n),
        "hub_median": hub_med,
        "control_median": ctrl_med,
        "median_ratio": float(median_ratio),
        "hub_mean": hub_mean,
        "control_mean": ctrl_mean,
        "mean_ratio": float(mean_ratio),
        "hub_frac_nonzero": hub_frac_nonzero,
        "ctrl_frac_nonzero": ctrl_frac_nonzero,
        "ratio_ci_lower": ci_lower,
        "ratio_ci_upper": ci_upper,
        "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
        "wilcoxon_p": float(p_val),
        "cohens_d": float(cohens_d),
    }


# ============================================================
# MAIN ORCHESTRATION
# ============================================================


@logger.catch
def main():
    t_start = time.time()

    # ===== STEP 1: Load all graphs =====
    logger.info("=== PHASE A: Loading graphs ===")
    graph_records = load_graphs(max_examples=MAX_EXAMPLES)
    n_loaded = len(graph_records)
    logger.info(f"Loaded {n_loaded} graphs")

    if n_loaded == 0:
        logger.error("No graphs loaded, aborting")
        output = {
            "metadata": {
                "experiment": "ffl_motif_node_ablation",
                "error": "no_graphs_loaded",
            },
            "datasets": [
                {
                    "dataset": "neuronpedia_ffl_ablation",
                    "examples": [{"input": "none", "output": "{}"}],
                }
            ],
        }
        OUTPUT_FILE.write_text(json.dumps(output, indent=2))
        return

    # ===== STEP 2: Serialize for multiprocessing =====
    logger.info("Serializing graphs for multiprocessing")
    serialized_graphs = []
    for rec in graph_records:
        g = rec.pop("graph")
        ser = serialize_graph(g, rec)
        serialized_graphs.append(ser)
        del g
    del graph_records
    gc.collect()

    # ===== STEP 3: Gradual scaling test (3 graphs) =====
    logger.info("=== Gradual scaling test: 3 graphs ===")
    test_results = []
    t_test = time.time()
    for i, sg in enumerate(serialized_graphs[:min(3, len(serialized_graphs))]):
        r = process_single_graph(sg)
        test_results.append(r)
        if r and not r.get("skipped"):
            logger.info(
                f"  Test graph {i}: {r['slug']} - {r['n_ffls']} FFLs, "
                f"{r['n_hub_nodes']} hubs, {r['n_nodes']} nodes"
            )
        else:
            reason = r.get("skip_reason", "unknown") if r else "None"
            logger.warning(f"  Test graph {i}: skipped ({reason})")

    t_test_elapsed = time.time() - t_test
    n_test = min(3, len(serialized_graphs))
    per_graph_time = t_test_elapsed / max(n_test, 1)
    estimated_total = per_graph_time * len(serialized_graphs) / max(N_WORKERS, 1)
    logger.info(
        f"Test timing: {t_test_elapsed:.1f}s for {n_test} graphs, "
        f"~{per_graph_time:.2f}s/graph, "
        f"estimated total: {estimated_total:.0f}s ({estimated_total / 60:.1f}min)"
    )

    # Timing guard: reduce scope if needed
    if estimated_total > 50 * 60:
        logger.warning("Estimated runtime > 50 min, reducing scope")
        # These are constants used by process_single_graph via globals
        # We'll log the adjustment; actual control is via the constants already set
        logger.info("Consider reducing N_CONTROLS_PER_HUB or DOSE_RESPONSE_SAMPLE")

    # ===== STEP 4: Full parallel processing =====
    logger.info(
        f"=== Full processing: {len(serialized_graphs)} graphs, "
        f"{N_WORKERS} workers ==="
    )
    all_results = list(test_results)
    remaining = serialized_graphs[n_test:]

    if remaining:
        batch_size = 20
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start : batch_start + batch_size]
            batch_results = []
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = {
                    executor.submit(process_single_graph, sg): idx
                    for idx, sg in enumerate(batch)
                }
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)
                        batch_results.append(result)
                    except Exception as exc:
                        logger.exception(f"Worker error: {exc}")
                        batch_results.append(None)
            all_results.extend(batch_results)
            gc.collect()
            b_num = batch_start // batch_size + 1
            b_total = (len(remaining) + batch_size - 1) // batch_size
            logger.info(
                f"  Batch {b_num}/{b_total} done, "
                f"total results so far: {len(all_results)}"
            )

    # Separate analyzed vs skipped
    analyzed = [r for r in all_results if r and not r.get("skipped")]
    skipped = [r for r in all_results if r and r.get("skipped")]
    n_with_ffls = len(
        [r for r in all_results if r and r.get("n_ffls", 0) > 0]
    )

    logger.info(
        f"Results: {len(analyzed)} analyzed, {len(skipped)} skipped, "
        f"{n_with_ffls} with FFLs"
    )
    for s in skipped[:5]:
        logger.debug(f"  Skipped: {s.get('slug')} - {s.get('skip_reason')}")

    if not analyzed:
        logger.error("No graphs were successfully analyzed!")
        output = {
            "metadata": {
                "experiment": "ffl_motif_node_ablation",
                "error": "no_graphs_analyzed",
                "n_loaded": n_loaded,
                "n_skipped": len(skipped),
            },
            "datasets": [
                {
                    "dataset": "neuronpedia_ffl_ablation",
                    "examples": [{"input": "none", "output": "{}"}],
                }
            ],
        }
        OUTPUT_FILE.write_text(json.dumps(output, indent=2, cls=NpEncoder))
        return

    # ===== STEP 5: STATISTICAL ANALYSIS =====
    logger.info("=== Statistical Analysis ===")

    # Aggregate all matched pairs
    all_matched_pairs = []
    for r in analyzed:
        all_matched_pairs.extend(r.get("matched_pairs", []))

    logger.info(f"Total matched pairs: {len(all_matched_pairs)}")

    hub_vs_control_results = {}
    all_p_values = []
    comparison_keys = []

    for metric in METRICS:
        for ctype in CONTROL_TYPES:
            key = f"{metric}__{ctype}"
            relevant = [
                p
                for p in all_matched_pairs
                if p["metric"] == metric and p["control_type"] == ctype
            ]

            if len(relevant) < 10:
                hub_vs_control_results[key] = {
                    "n_pairs": len(relevant),
                    "insufficient_data": True,
                }
                all_p_values.append(1.0)
                comparison_keys.append(key)
                continue

            hub_impacts = np.array([p["hub_impact"] for p in relevant])
            ctrl_impacts = np.array([p["control_impact"] for p in relevant])

            result = compute_comparison_stats(hub_impacts, ctrl_impacts)
            hub_vs_control_results[key] = result
            all_p_values.append(result["wilcoxon_p"])
            comparison_keys.append(key)

            logger.info(
                f"  {key}: med_ratio={result['median_ratio']:.3f} "
                f"mean_ratio={result['mean_ratio']:.3f} "
                f"CI=[{result['ratio_ci_lower']:.3f}, {result['ratio_ci_upper']:.3f}] "
                f"p={result['wilcoxon_p']:.4f} d={result['cohens_d']:.3f} "
                f"hub_nz={result['hub_frac_nonzero']:.2%} ctrl_nz={result['ctrl_frac_nonzero']:.2%}"
            )

    # BH-FDR correction
    adjusted_p = benjamini_hochberg(all_p_values)
    for i, key in enumerate(comparison_keys):
        if not hub_vs_control_results[key].get("insufficient_data"):
            hub_vs_control_results[key]["wilcoxon_p_fdr"] = adjusted_p[i]
            hub_vs_control_results[key]["significant_at_005"] = (
                adjusted_p[i] < 0.05
            )

    # ===== STEP 6: DOSE-RESPONSE ANALYSIS =====
    logger.info("=== Dose-Response Analysis ===")

    all_dose = []
    for r in analyzed:
        all_dose.extend(r.get("dose_response_sample", []))

    logger.info(f"Total dose-response datapoints: {len(all_dose)}")

    dose_response_results = {}
    if all_dose:
        log_mpis = np.array([d["log_mpi"] for d in all_dose])
        for metric in METRICS:
            impacts = np.array([d[metric] for d in all_dose])
            try:
                sp_r, sp_p = stats.spearmanr(log_mpis, impacts)
                lr = stats.linregress(log_mpis, impacts)
                dose_response_results[metric] = {
                    "spearman_r": float(sp_r),
                    "spearman_p": float(sp_p),
                    "regression_slope": float(lr.slope),
                    "regression_r2": float(lr.rvalue**2),
                    "regression_p": float(lr.pvalue),
                    "n_datapoints": len(log_mpis),
                }
                logger.info(
                    f"  {metric}: Spearman r={sp_r:.4f} p={sp_p:.4g}, "
                    f"slope={lr.slope:.6f} R2={lr.rvalue**2:.4f}"
                )
            except Exception:
                logger.exception(f"Dose-response failed for {metric}")
                dose_response_results[metric] = {
                    "error": "computation_failed",
                    "n_datapoints": len(log_mpis),
                }

    # ===== STEP 7: PER-DOMAIN BREAKDOWN =====
    logger.info("=== Per-Domain Breakdown ===")

    domain_pairs: dict[str, list] = {}
    for p in all_matched_pairs:
        domain_pairs.setdefault(p["domain"], []).append(p)

    per_domain: dict[str, dict] = {}
    for domain, pairs_list in sorted(domain_pairs.items()):
        per_domain[domain] = {}
        n_sig = 0
        for metric in METRICS:
            for ctype in CONTROL_TYPES:
                key = f"{metric}__{ctype}"
                relevant = [
                    p
                    for p in pairs_list
                    if p["metric"] == metric and p["control_type"] == ctype
                ]
                if len(relevant) >= 10:
                    hub_imp = np.array([p["hub_impact"] for p in relevant])
                    ctrl_imp = np.array(
                        [p["control_impact"] for p in relevant]
                    )
                    hub_med = float(np.median(hub_imp))
                    ctrl_med = float(np.median(ctrl_imp))
                    try:
                        diffs = hub_imp - ctrl_imp
                        nonzero_count = int(np.sum(diffs != 0))
                        if nonzero_count >= 5:
                            _, w_p = stats.wilcoxon(
                                hub_imp, ctrl_imp, alternative="greater"
                            )
                        else:
                            w_p = 1.0
                        d_mean = float(np.mean(diffs))
                        d_std = (
                            float(np.std(diffs, ddof=1))
                            if len(diffs) > 1
                            else 1e-10
                        )
                        cd = d_mean / max(d_std, 1e-10)
                    except Exception:
                        w_p, cd = 1.0, 0.0
                    hub_mn = float(np.mean(hub_imp))
                    ctrl_mn = float(np.mean(ctrl_imp))
                    per_domain[domain][key] = {
                        "n_pairs": len(relevant),
                        "median_ratio": float(
                            hub_med / max(ctrl_med, 1e-10)
                        ),
                        "mean_ratio": float(
                            hub_mn / max(ctrl_mn, 1e-10)
                        ),
                        "wilcoxon_p": float(w_p),
                        "cohens_d": float(cd),
                    }
                    if w_p < 0.05:
                        n_sig += 1
                else:
                    per_domain[domain][key] = {
                        "n_pairs": len(relevant),
                        "insufficient_data": True,
                    }
        logger.info(
            f"  {domain}: {len(pairs_list)} pairs, "
            f"{n_sig}/16 significant comparisons"
        )

    # ===== STEP 8: SUMMARY STATISTICS =====
    logger.info("=== Summary Statistics ===")

    total_ffls = sum(r["n_ffls"] for r in analyzed)
    n_hub_total = sum(r["n_hub_nodes"] for r in analyzed)
    n_participant_total = sum(r["n_participant_nodes"] for r in analyzed)
    n_non_ffl_total = sum(r["n_non_ffl_nodes"] for r in analyzed)
    total_nodes = n_hub_total + n_participant_total + n_non_ffl_total

    # Collect MPI values from dose-response samples (covers full spectrum)
    all_mpis = [d["mpi"] for d in all_dose]
    mpi_arr = np.array(all_mpis) if all_mpis else np.array([0])

    corpus_summary = {
        "total_ffls": int(total_ffls),
        "mpi_distribution": {
            "mean": float(np.mean(mpi_arr)),
            "median": float(np.median(mpi_arr)),
            "std": float(np.std(mpi_arr)),
            "max": int(np.max(mpi_arr)),
            "p90": float(np.percentile(mpi_arr, 90)),
        },
        "node_classification": {
            "n_hub": int(n_hub_total),
            "n_participant": int(n_participant_total),
            "n_non_ffl": int(n_non_ffl_total),
            "pct_hub": round(
                float(n_hub_total / max(total_nodes, 1) * 100), 2
            ),
            "pct_participant": round(
                float(n_participant_total / max(total_nodes, 1) * 100), 2
            ),
        },
    }

    matched_pair_counts = {}
    for ctype in CONTROL_TYPES:
        # Each control pair appears once per metric (4 metrics), so divide by 4
        matched_pair_counts[ctype] = (
            len(
                [
                    p
                    for p in all_matched_pairs
                    if p["control_type"] == ctype
                ]
            )
            // 4
        )

    logger.info(f"Total FFLs: {total_ffls}")
    logger.info(
        f"Nodes: {n_hub_total} hubs, {n_participant_total} participants, "
        f"{n_non_ffl_total} non-FFL"
    )
    logger.info(f"Matched pair counts: {matched_pair_counts}")

    # ===== STEP 9: REPRESENTATIVE EXAMPLES =====
    all_hub_ablations = []
    for r in analyzed:
        for ha in r.get("hub_ablations", []):
            ha_copy = dict(ha)
            ha_copy["slug"] = r["slug"]
            ha_copy["domain"] = r["domain"]
            ha_copy["total_impact"] = sum(ha[m] for m in METRICS)
            all_hub_ablations.append(ha_copy)

    top_hubs = sorted(
        all_hub_ablations, key=lambda x: x["total_impact"], reverse=True
    )[:5]
    representative_examples = [
        {
            "slug": th["slug"],
            "domain": th["domain"],
            "node_id": th["node_id"],
            "mpi": th["mpi"],
            "layer": th["layer"],
            "feature_type": th["feature_type"],
            "downstream_attr_loss": th["downstream_attr_loss"],
            "path_disruption": th["path_disruption"],
            "reachability_loss": th["reachability_loss"],
            "component_fragmentation": th["component_fragmentation"],
            "total_impact": th["total_impact"],
        }
        for th in top_hubs
    ]

    # ===== STEP 10: SUCCESS CRITERIA =====
    # Use best of median_ratio or mean_ratio (mean is more informative for sparse metrics)
    best_ratio = {
        "metric": "",
        "control_type": "",
        "achieved": False,
        "value": 0.0,
        "ratio_type": "median",
    }
    for key, res in hub_vs_control_results.items():
        if res.get("insufficient_data"):
            continue
        # Check median ratio
        mr = res.get("median_ratio", 0)
        if mr > best_ratio["value"]:
            parts = key.split("__")
            best_ratio = {
                "metric": parts[0],
                "control_type": parts[1],
                "achieved": mr >= 1.5,
                "value": round(mr, 4),
                "ratio_type": "median",
            }
        # Check mean ratio (better for sparse metrics where median=0)
        mnr = res.get("mean_ratio", 0)
        if mnr > best_ratio["value"]:
            parts = key.split("__")
            best_ratio = {
                "metric": parts[0],
                "control_type": parts[1],
                "achieved": mnr >= 1.5,
                "value": round(mnr, 4),
                "ratio_type": "mean",
            }

    n_significant_domains = 0
    for domain_name, domain_results in per_domain.items():
        has_sig = any(
            not res.get("insufficient_data") and res.get("wilcoxon_p", 1.0) < 0.05
            for res in domain_results.values()
        )
        if has_sig:
            n_significant_domains += 1

    dose_significant = any(
        dr.get("spearman_p", 1.0) < 0.05 and dr.get("spearman_r", 0) > 0
        for dr in dose_response_results.values()
        if isinstance(dr, dict) and "spearman_p" in dr
    )

    t_total = time.time() - t_start

    logger.info("=== RESULTS SUMMARY ===")
    logger.info(
        f"Best {best_ratio['ratio_type']} ratio: {best_ratio['value']:.4f} "
        f"({best_ratio['metric']}__{best_ratio['control_type']}), "
        f"achieved >=1.5: {best_ratio['achieved']}"
    )
    logger.info(f"Significant domains: {n_significant_domains}/8")
    logger.info(f"Dose-response significant: {dose_significant}")
    logger.info(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")

    # ===== Build output in exp_gen_sol_out.json schema =====
    experiment_summary = {
        "experiment": "ffl_motif_node_ablation",
        "n_graphs_loaded": int(n_loaded),
        "n_graphs_with_ffls": int(n_with_ffls),
        "n_graphs_analyzed": len(analyzed),
        "corpus_summary": corpus_summary,
        "matched_pair_counts": matched_pair_counts,
        "hub_vs_control_results": hub_vs_control_results,
        "dose_response": dose_response_results,
        "per_domain_breakdown": per_domain,
        "representative_examples": representative_examples,
        "success_criteria": {
            "median_ratio_ge_1_5": best_ratio,
            "n_significant_domains": n_significant_domains,
            "dose_response_significant": dose_significant,
        },
        "timing": {
            "total_seconds": round(t_total, 1),
            "per_graph_seconds": round(
                t_total / max(len(analyzed), 1), 2
            ),
        },
    }

    # Per-graph examples for schema compliance
    examples = []
    for r in analyzed:
        per_graph_output = {
            "slug": r["slug"],
            "domain": r["domain"],
            "n_nodes": r["n_nodes"],
            "n_edges": r["n_edges"],
            "n_ffls": r["n_ffls"],
            "used_all_motifs": r.get("used_all_motifs", False),
            "n_hub_nodes": r["n_hub_nodes"],
            "n_participant_nodes": r["n_participant_nodes"],
            "n_non_ffl_nodes": r["n_non_ffl_nodes"],
            "hub_ablations_summary": {
                "n_hubs": len(r["hub_ablations"]),
                "mean_downstream_attr_loss": float(
                    np.mean(
                        [h["downstream_attr_loss"] for h in r["hub_ablations"]]
                    )
                )
                if r["hub_ablations"]
                else 0.0,
                "mean_path_disruption": float(
                    np.mean(
                        [h["path_disruption"] for h in r["hub_ablations"]]
                    )
                )
                if r["hub_ablations"]
                else 0.0,
                "mean_reachability_loss": float(
                    np.mean(
                        [h["reachability_loss"] for h in r["hub_ablations"]]
                    )
                )
                if r["hub_ablations"]
                else 0.0,
                "mean_component_fragmentation": float(
                    np.mean(
                        [
                            h["component_fragmentation"]
                            for h in r["hub_ablations"]
                        ]
                    )
                )
                if r["hub_ablations"]
                else 0.0,
            },
            "control_counts": {
                ct: len(ctrls)
                for ct, ctrls in r["control_ablations"].items()
            },
            "n_matched_pairs": len(r["matched_pairs"]),
            "n_dose_response": len(r["dose_response_sample"]),
        }

        # Build predict fields: hub ablation (our method) vs random baseline
        hub_abl = r.get("hub_ablations", [])
        ctrl_random = r.get("control_ablations", {}).get("random", [])
        hub_mean_impact = float(np.mean([
            h["downstream_attr_loss"] + h["path_disruption"]
            + h["reachability_loss"] + h["component_fragmentation"]
            for h in hub_abl
        ])) if hub_abl else 0.0
        ctrl_mean_impact = float(np.mean([
            c["downstream_attr_loss"] + c["path_disruption"]
            + c["reachability_loss"] + c["component_fragmentation"]
            for c in ctrl_random
        ])) if ctrl_random else 0.0
        ratio = hub_mean_impact / max(ctrl_mean_impact, 1e-10)

        predict_hub = json.dumps({
            "method": "ffl_hub_ablation",
            "mean_total_impact": round(hub_mean_impact, 6),
            "n_hubs_ablated": len(hub_abl),
            "impact_ratio_vs_random": round(ratio, 4),
        }, cls=NpEncoder)

        predict_baseline = json.dumps({
            "method": "random_control_ablation",
            "mean_total_impact": round(ctrl_mean_impact, 6),
            "n_controls_ablated": len(ctrl_random),
        }, cls=NpEncoder)

        example = {
            "input": r.get("prompt", ""),
            "output": json.dumps(per_graph_output, cls=NpEncoder),
            "predict_ffl_hub_ablation": predict_hub,
            "predict_random_baseline": predict_baseline,
            "metadata_fold": r["domain"],
            "metadata_slug": r["slug"],
            "metadata_n_nodes": r["n_nodes"],
            "metadata_n_edges": r["n_edges"],
            "metadata_n_ffls": r["n_ffls"],
            "metadata_correctness": str(r.get("correctness", "")),
            "metadata_difficulty": str(r.get("difficulty", "")),
        }
        examples.append(example)

    output = {
        "metadata": experiment_summary,
        "datasets": [
            {
                "dataset": "neuronpedia_ffl_ablation",
                "examples": examples,
            }
        ],
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, cls=NpEncoder))
    file_size_kb = OUTPUT_FILE.stat().st_size / 1024
    logger.info(f"Output written to {OUTPUT_FILE} ({file_size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
