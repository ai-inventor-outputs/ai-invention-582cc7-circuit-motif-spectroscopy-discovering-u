#!/usr/bin/env python3
"""Functional characterization of 4 universally overrepresented 4-node motif
types (IDs 77, 80, 82, 83) from LLM attribution graphs.

Tests whether their universality is FFL-derivative or independent, analyzes
layer ordering, semantic roles, and cross-domain consistency. Includes a
random connected 4-node subgraph baseline for comparison.

Dependencies:
  - data_id5_it3: 200 graphs in 12 split files
  - data_id4_it2: 6574 feature explanations
  - research_id2_it1: Methodology reference
"""

import gc
import json
import math
import os
import random
import re
import resource
import signal
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from itertools import combinations
from pathlib import Path

import igraph
import numpy as np
from loguru import logger
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

# ============================================================
# LOGGING
# ============================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "run.log", rotation="30 MB", level="DEBUG")

# ============================================================
# HARDWARE DETECTION
# ============================================================

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

# ============================================================
# MEMORY LIMITS
# ============================================================
RAM_BUDGET_GB = min(TOTAL_RAM_GB * 0.75, 22.0)
RAM_BUDGET_BYTES = int(RAM_BUDGET_GB * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET_GB:.1f}GB")

# ============================================================
# CONSTANTS & PATHS
# ============================================================
WORKSPACE = Path(__file__).parent.resolve()
ITER_BASE = WORKSPACE.parents[2]
DATA_DIR = ITER_BASE / "iter_3" / "gen_art" / "data_id5_it3__opus" / "data_out"
DATA_FILES = [f"full_data_out_{i}.json" for i in range(1, 13)]
EXPL_DIR = ITER_BASE / "iter_2" / "gen_art" / "data_id4_it2__opus"
EXPL_FILE = EXPL_DIR / "full_data_out.json"
OUTPUT_FILE = WORKSPACE / "method_out.json"

PRUNE_PERCENTILE = 99
MIN_NODES = 30
MAX_NODES = 700
TARGET_MOTIF_IDS = [77, 80, 82, 83]
MAX_INSTANCES_PER_TYPE = 500
RANDOM_SEED = 42
N_SEMANTIC_SAMPLES = 100
N_RANDOM_BASELINE = 200
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "9999"))
VF2_TIMEOUT_SEC = 60

# ============================================================
# UTILITIES
# ============================================================

def cantor_decode(z: int) -> tuple[int, int]:
    """Decode Cantor-paired feature integer -> (layer_num, feature_index)."""
    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    t = (w * w + w) // 2
    feat_index = z - t
    layer_num = w - feat_index
    return (int(layer_num), int(feat_index))


def parse_layer(layer_str: str) -> int:
    """Parse layer string to integer. 'E' -> -1, numeric -> int."""
    if layer_str == "E":
        return -1
    try:
        return int(layer_str)
    except (ValueError, TypeError):
        return -1


def safe_round(val, digits=4):
    """Safely round a value, handling None/NaN."""
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None


def compute_cramers_v(table: np.ndarray) -> float:
    """Compute Cramer's V from a contingency table."""
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    col_sums = table.sum(axis=0)
    nonzero = col_sums > 0
    table_f = table[:, nonzero]
    if table_f.shape[1] < 2:
        return 0.0
    try:
        chi2_val, _, _, _ = chi2_contingency(table_f)
    except ValueError:
        return 0.0
    n = np.sum(table_f)
    min_dim = min(table_f.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return math.sqrt(chi2_val / (n * min_dim))


# ============================================================
# STEP 1: Load Feature Explanation Lookup Table
# ============================================================

def load_explanations() -> dict[str, dict]:
    """Load feature explanations into a lookup dict keyed by 'layer_featureindex'."""
    logger.info(f"Loading explanations from {EXPL_FILE}")
    data = json.loads(EXPL_FILE.read_text())
    examples = data["datasets"][0]["examples"]
    lookup: dict[str, dict] = {}
    for ex in examples:
        key = ex["input"]
        try:
            output = json.loads(ex["output"])
        except json.JSONDecodeError:
            continue
        lookup[key] = {
            "explanation": output.get("explanation", ""),
            "source_domains": output.get("source_domains", []),
            "frac_nonzero": output.get("frac_nonzero"),
            "max_activation": output.get("max_activation"),
        }
    logger.info(f"Loaded {len(lookup)} feature explanations")
    return lookup


# ============================================================
# STEP 2: Load and Prune All 200 Graphs
# ============================================================

def load_all_graphs() -> list[dict]:
    """Load all graphs from 12 split files, prune at 99th percentile."""
    all_graphs: list[dict] = []
    examples_loaded = 0

    for data_file in DATA_FILES:
        if examples_loaded >= MAX_EXAMPLES:
            break
        fpath = DATA_DIR / data_file
        if not fpath.exists():
            logger.warning(f"Data file not found: {fpath}")
            continue
        logger.info(f"Loading {fpath.name}...")
        raw = json.loads(fpath.read_text())
        examples = raw["datasets"][0]["examples"]

        for ex in examples:
            if examples_loaded >= MAX_EXAMPLES:
                break
            try:
                graph_json = json.loads(ex["output"])
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON in example {ex.get('metadata_slug', '?')}")
                continue

            nodes = graph_json["nodes"]
            links = graph_json["links"]

            node_id_to_idx: dict[str, int] = {}
            for i, n in enumerate(nodes):
                node_id_to_idx[n["node_id"]] = i

            node_layers = [parse_layer(n.get("layer", "0")) for n in nodes]
            feature_types = [n.get("feature_type", "") for n in nodes]
            features = [n.get("feature", None) for n in nodes]
            node_ids = [n["node_id"] for n in nodes]

            all_abs_weights = [abs(link.get("weight", 0.0)) for link in links]
            threshold = float(np.percentile(all_abs_weights, PRUNE_PERCENTILE)) if all_abs_weights else 0

            signed_weight_dict: dict[tuple[str, str], float] = {}
            edges: list[tuple[int, int]] = []
            abs_weights: list[float] = []
            signed_weights: list[float] = []

            for link in links:
                raw_w = link.get("weight", 0.0)
                if abs(raw_w) >= threshold:
                    src_id = link["source"]
                    tgt_id = link["target"]
                    src = node_id_to_idx.get(src_id)
                    tgt = node_id_to_idx.get(tgt_id)
                    if src is not None and tgt is not None and src != tgt:
                        edges.append((src, tgt))
                        abs_weights.append(abs(raw_w))
                        signed_weights.append(float(raw_w))
                        signed_weight_dict[(src_id, tgt_id)] = float(raw_w)

            if len(edges) == 0:
                continue

            g = igraph.Graph(n=len(nodes), edges=edges, directed=True)
            g.vs["node_id"] = node_ids
            g.vs["layer"] = node_layers
            g.vs["feature_type"] = feature_types
            g.vs["feature"] = features
            g.es["weight"] = abs_weights
            g.es["signed_weight"] = signed_weights

            # Simplify: keep max abs weight for multi-edges
            edge_signed_before: dict[tuple[int, int], float] = {}
            for e in g.es:
                key = (e.source, e.target)
                if key not in edge_signed_before or abs(e["signed_weight"]) > abs(edge_signed_before[key]):
                    edge_signed_before[key] = e["signed_weight"]

            g.simplify(multiple=True, loops=True, combine_edges={"weight": "max"})

            # Restore signed weights
            for e in g.es:
                src_nid = g.vs[e.source]["node_id"]
                tgt_nid = g.vs[e.target]["node_id"]
                if (src_nid, tgt_nid) in signed_weight_dict:
                    e["signed_weight"] = signed_weight_dict[(src_nid, tgt_nid)]
                elif (e.source, e.target) in edge_signed_before:
                    e["signed_weight"] = edge_signed_before[(e.source, e.target)]
                else:
                    e["signed_weight"] = e["weight"]

            # Remove isolated vertices
            isolated = [v.index for v in g.vs if g.degree(v) == 0]
            if isolated:
                g.delete_vertices(isolated)

            if g.vcount() < MIN_NODES:
                continue
            if g.vcount() > MAX_NODES:
                continue
            if not g.is_dag():
                continue

            all_graphs.append({
                "graph": g,
                "domain": ex.get("metadata_fold", "unknown"),
                "prompt": ex.get("input", ""),
                "slug": ex.get("metadata_slug", ""),
                "n_nodes": g.vcount(),
                "n_edges": g.ecount(),
                "correctness": ex.get("metadata_model_correct", "unknown"),
                "difficulty": ex.get("metadata_difficulty", "unknown"),
            })
            examples_loaded += 1

        del raw
        gc.collect()

    logger.info(f"Loaded {len(all_graphs)} feasible graphs (out of {MAX_EXAMPLES} max)")
    domain_counts = Counter(gd["domain"] for gd in all_graphs)
    for d, c in sorted(domain_counts.items()):
        logger.info(f"  {d}: {c} graphs")
    return all_graphs


# ============================================================
# STEP 3: Build 4-Node and 3-Node Canonical Pattern Reference
# ============================================================

def build_canonical_references() -> tuple[dict, dict]:
    """Build canonical pattern references for 4-node targets and 3-node sub-motifs."""
    # 4-node target patterns
    target_patterns: dict[int, dict] = {}
    for mid in TARGET_MOTIF_IDS:
        pat = igraph.Graph.Isoclass(n=4, cls=mid, directed=True)
        edges = [(e.source, e.target) for e in pat.es]
        target_patterns[mid] = {
            "edge_count": pat.ecount(),
            "edges": edges,
            "adjacency": pat.get_adjacency().data,
        }
        logger.info(f"Motif {mid}: {pat.ecount()} edges, pattern = {edges}")

    # 3-node sub-motif reference
    three_node_ref: dict[int, str] = {}
    man_labels = {2: "021U", 4: "021C", 6: "021D", 7: "030T"}
    for cid in range(16):
        can = igraph.Graph.Isoclass(n=3, cls=cid, directed=True)
        if can.is_dag() and can.is_connected(mode="weak") and can.ecount() >= 2:
            three_node_ref[cid] = man_labels.get(cid, f"cls_{cid}")
    logger.info(f"3-node DAG-valid connected types: {three_node_ref}")
    return target_patterns, three_node_ref


# ============================================================
# STEP 4 (PHASE A): 4-Node Motif Instance Enumeration
# ============================================================

def _serialize_graph(gd: dict) -> dict:
    """Serialize graph dict for multiprocessing (igraph not picklable)."""
    g = gd["graph"]
    edges = [(e.source, e.target) for e in g.es]
    return {
        "n_vertices": g.vcount(),
        "edges": edges,
        "vs_node_id": list(g.vs["node_id"]),
        "vs_layer": list(g.vs["layer"]),
        "vs_feature_type": list(g.vs["feature_type"]),
        "vs_feature": list(g.vs["feature"]),
        "es_weight": list(g.es["weight"]),
        "es_signed_weight": list(g.es["signed_weight"]),
        "domain": gd["domain"],
        "slug": gd["slug"],
        "prompt": gd["prompt"],
    }


def _rebuild_graph(sd: dict) -> igraph.Graph:
    """Rebuild igraph from serialized dict."""
    g = igraph.Graph(n=sd["n_vertices"], edges=sd["edges"], directed=True)
    g.vs["node_id"] = sd["vs_node_id"]
    g.vs["layer"] = sd["vs_layer"]
    g.vs["feature_type"] = sd["vs_feature_type"]
    g.vs["feature"] = sd["vs_feature"]
    g.es["weight"] = sd["es_weight"]
    g.es["signed_weight"] = sd["es_signed_weight"]
    return g


def _enumerate_motifs_for_graph(args: tuple) -> dict:
    """Worker function: enumerate target motif instances in one graph."""
    sd, target_ids, max_inst, timeout_sec = args
    g = _rebuild_graph(sd)
    slug = sd["slug"]

    # Build adjacency set and weight lookup
    adj_set: set[tuple[int, int]] = set()
    weight_lookup: dict[tuple[int, int], dict] = {}
    for e in g.es:
        adj_set.add((e.source, e.target))
        weight_lookup[(e.source, e.target)] = {
            "abs": float(e["weight"]),
            "signed": float(e["signed_weight"]),
        }

    results: dict[int, list] = {}
    for mid in target_ids:
        pattern = igraph.Graph.Isoclass(n=4, cls=mid, directed=True)
        t0 = time.time()
        try:
            raw_matches = g.get_subisomorphisms_vf2(pattern)
        except Exception as exc:
            logger.warning(f"VF2 failed for motif {mid} on {slug}: {exc}")
            results[mid] = []
            continue

        elapsed = time.time() - t0
        if elapsed > timeout_sec:
            logger.warning(f"VF2 slow for motif {mid} on {slug}: {elapsed:.1f}s")

        # Deduplicate by frozenset of vertex IDs
        seen: set[frozenset] = set()
        unique_instances: list[list[int]] = []
        for match in raw_matches:
            key = frozenset(match)
            if key not in seen:
                seen.add(key)
                unique_instances.append(match)

        # Cap at max_inst
        if len(unique_instances) > max_inst:
            rng = random.Random(42)
            rng.shuffle(unique_instances)
            unique_instances = unique_instances[:max_inst]

        # Extract instance details
        instances: list[dict] = []
        for match in unique_instances:
            layers = [g.vs[v]["layer"] for v in match]
            features = [g.vs[v]["feature"] for v in match]
            ftypes = [g.vs[v]["feature_type"] for v in match]
            node_ids_m = [g.vs[v]["node_id"] for v in match]

            # Get edge weights between the 4 nodes
            edge_weights: dict[str, dict] = {}
            for i_idx in range(4):
                for j_idx in range(4):
                    if i_idx != j_idx:
                        vi, vj = match[i_idx], match[j_idx]
                        if (vi, vj) in weight_lookup:
                            edge_weights[f"{i_idx}->{j_idx}"] = weight_lookup[(vi, vj)]

            instances.append({
                "vertex_ids": match,
                "layers": layers,
                "features": features,
                "feature_types": ftypes,
                "node_ids": node_ids_m,
                "edge_weights": edge_weights,
            })

        results[mid] = instances
        logger.debug(f"Motif {mid} on {slug}: {len(instances)} instances ({elapsed:.2f}s)")

    return {"slug": slug, "domain": sd["domain"], "results": results}


def enumerate_all_motifs(
    graph_dicts: list[dict],
    target_ids: list[int],
    max_inst: int = MAX_INSTANCES_PER_TYPE,
) -> dict[int, list]:
    """Enumerate target 4-node motif instances across all graphs using multiprocessing."""
    serialized = [_serialize_graph(gd) for gd in graph_dicts]

    # Prepare args
    args_list = [
        (sd, target_ids, max_inst, VF2_TIMEOUT_SEC)
        for sd in serialized
    ]

    all_results: dict[int, list] = {mid: [] for mid in target_ids}
    n_workers = max(1, NUM_CPUS)
    batch_size = max(4, n_workers)

    logger.info(f"Enumerating motifs across {len(graph_dicts)} graphs with {n_workers} workers")

    # Process in batches
    for batch_start in range(0, len(args_list), batch_size):
        batch = args_list[batch_start:batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(args_list))
        logger.info(f"Processing batch {batch_start+1}-{batch_end}/{len(args_list)}")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_enumerate_motifs_for_graph, arg): arg[0]["slug"]
                       for arg in batch}
            for future in as_completed(futures):
                slug = futures[future]
                try:
                    result = future.result(timeout=300)
                    for mid in target_ids:
                        for inst in result["results"].get(mid, []):
                            inst["domain"] = result["domain"]
                            inst["slug"] = result["slug"]
                            all_results[mid].append(inst)
                except TimeoutError:
                    logger.warning(f"Timeout for graph {slug}")
                except Exception:
                    logger.exception(f"Error processing graph {slug}")

        gc.collect()

    for mid in target_ids:
        logger.info(f"Motif {mid}: {len(all_results[mid])} total instances")

    return all_results


# ============================================================
# STEP 5 (PHASE B): 3->4 Hierarchical Embedding Analysis
# ============================================================

def classify_triplet(edges_in_triplet: list[tuple[int, int]], nodes_in_triplet: set[int]) -> str:
    """Classify a 3-node induced subgraph by its edge pattern."""
    n_edges = len(edges_in_triplet)
    if n_edges < 2:
        return "disconnected"
    if n_edges == 2:
        targets = [e[1] for e in edges_in_triplet]
        sources = [e[0] for e in edges_in_triplet]
        if targets[0] == targets[1]:
            return "021U"  # in-star
        elif sources[0] == sources[1]:
            return "021D"  # out-star
        else:
            return "021C"  # chain
    if n_edges == 3:
        return "030T"  # FFL - only 3-edge connected triad possible in DAG
    return "disconnected"


def analyze_3to4_embedding(
    all_instances: dict[int, list],
    graph_dicts: list[dict],
) -> dict:
    """Analyze 3-node sub-motif composition of each 4-node motif type."""
    # Build adjacency sets for all graphs
    adj_by_slug: dict[str, set] = {}
    for gd in graph_dicts:
        g = gd["graph"]
        adj_set: set[tuple[int, int]] = set()
        for e in g.es:
            adj_set.add((e.source, e.target))
        adj_by_slug[gd["slug"]] = adj_set

    results: dict[int, dict] = {}

    for mid in TARGET_MOTIF_IDS:
        instances = all_instances.get(mid, [])
        if not instances:
            results[mid] = {
                "n_instances": 0,
                "ffl_containment_frac": 0.0,
                "mean_ffl_count_per_instance": 0.0,
                "sub_motif_composition": {},
                "independence_verdict": "no_data",
            }
            continue

        ffl_count_per_inst: list[int] = []
        all_sub_types: list[str] = []

        for inst in instances:
            vids = inst["vertex_ids"]
            slug = inst["slug"]
            adj = adj_by_slug.get(slug, set())

            ffl_in_this = 0
            triplet_combos = list(combinations(range(4), 3))

            for combo in triplet_combos:
                nodes_3 = [vids[i] for i in combo]
                nodes_set = set(nodes_3)
                edges_in = [(nodes_3[i], nodes_3[j])
                            for i in range(3) for j in range(3)
                            if i != j and (nodes_3[i], nodes_3[j]) in adj]
                sub_type = classify_triplet(edges_in, nodes_set)
                all_sub_types.append(sub_type)
                if sub_type == "030T":
                    ffl_in_this += 1

            ffl_count_per_inst.append(ffl_in_this)

        ffl_containment = sum(1 for c in ffl_count_per_inst if c >= 1) / len(ffl_count_per_inst)
        mean_ffl = float(np.mean(ffl_count_per_inst))

        # Sub-motif composition histogram
        type_counter = Counter(all_sub_types)
        total_triplets = sum(type_counter.values())
        composition = {k: round(v / total_triplets, 4) for k, v in type_counter.items()}

        # Independence verdict
        if ffl_containment > 0.90:
            verdict = "FFL-derivative"
        elif ffl_containment < 0.50:
            verdict = "genuinely_independent"
        else:
            verdict = "partially_embedded"

        results[mid] = {
            "n_instances": len(instances),
            "ffl_containment_frac": round(ffl_containment, 4),
            "mean_ffl_count_per_instance": round(mean_ffl, 4),
            "sub_motif_composition": composition,
            "independence_verdict": verdict,
        }
        logger.info(f"Motif {mid}: FFL containment={ffl_containment:.3f}, "
                     f"mean_ffl={mean_ffl:.2f}, verdict={verdict}")

    return results


# ============================================================
# STEP 6 (PHASE C): Layer Ordering and Span Analysis
# ============================================================

def sample_random_connected_4node(g: igraph.Graph, n_samples: int, rng: random.Random) -> list[dict]:
    """Sample random connected 4-node subgraphs for baseline comparison."""
    adj_set: set[tuple[int, int]] = set()
    for e in g.es:
        adj_set.add((e.source, e.target))

    samples: list[dict] = []
    attempts = 0
    max_attempts = n_samples * 50
    n = g.vcount()

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        start = rng.randint(0, n - 1)
        # BFS to get connected neighbors
        neighbors = set()
        queue = [start]
        visited = {start}
        while queue and len(visited) < 4:
            curr = queue.pop(0)
            for nb in g.neighbors(curr, mode="all"):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    if len(visited) == 4:
                        break

        if len(visited) < 4:
            continue

        vids = list(visited)[:4]
        # Verify weakly connected
        sub = g.induced_subgraph(vids)
        if not sub.is_connected(mode="weak"):
            continue

        layers = [g.vs[v]["layer"] for v in vids]
        valid_layers = [l for l in layers if l >= 0]

        # Check strict ordering of all edges in induced subgraph
        strict_ordered = True
        for e in sub.es:
            src_layer = sub.vs[e.source]["layer"]
            tgt_layer = sub.vs[e.target]["layer"]
            if src_layer >= tgt_layer:
                strict_ordered = False
                break

        span = (max(valid_layers) - min(valid_layers)) if len(valid_layers) >= 2 else 0

        samples.append({
            "vertex_ids": vids,
            "layers": layers,
            "span": span,
            "strict_ordered": strict_ordered,
        })

    return samples


def analyze_layer_ordering(
    all_instances: dict[int, list],
    graph_dicts: list[dict],
) -> dict:
    """Analyze layer ordering and span for each 4-node motif type + random baseline."""
    # Build graph lookup by slug
    graph_by_slug: dict[str, igraph.Graph] = {}
    for gd in graph_dicts:
        graph_by_slug[gd["slug"]] = gd["graph"]

    rng = random.Random(RANDOM_SEED)
    results: dict[int, dict] = {}

    # Random baseline: sample from each graph
    logger.info("Sampling random 4-node baselines...")
    all_baseline_spans: list[float] = []
    all_baseline_strict: list[bool] = []
    for gd in graph_dicts:
        samples = sample_random_connected_4node(gd["graph"], N_RANDOM_BASELINE, rng)
        for s in samples:
            all_baseline_spans.append(s["span"])
            all_baseline_strict.append(s["strict_ordered"])

    baseline_mean_span = float(np.mean(all_baseline_spans)) if all_baseline_spans else 0.0
    baseline_std_span = float(np.std(all_baseline_spans)) if all_baseline_spans else 0.0
    baseline_strict_frac = sum(all_baseline_strict) / len(all_baseline_strict) if all_baseline_strict else 0.0

    logger.info(f"Baseline: mean_span={baseline_mean_span:.2f}, strict_frac={baseline_strict_frac:.3f}")

    for mid in TARGET_MOTIF_IDS:
        instances = all_instances.get(mid, [])
        if not instances:
            results[mid] = {"n_instances": 0}
            continue

        strict_order_count = 0
        spans: list[float] = []
        position_layers: dict[int, list[int]] = {i: [] for i in range(4)}

        for inst in instances:
            layers = inst["layers"]
            vids = inst["vertex_ids"]
            slug = inst["slug"]
            g = graph_by_slug.get(slug)
            if g is None:
                continue

            valid_layers = [l for l in layers if l >= 0]
            if len(valid_layers) >= 2:
                spans.append(max(valid_layers) - min(valid_layers))

            # Strict ordering: every directed edge goes from lower to higher layer
            strict = True
            for i_idx in range(4):
                for j_idx in range(4):
                    if i_idx != j_idx:
                        key = f"{i_idx}->{j_idx}"
                        if key in inst["edge_weights"]:
                            src_layer = layers[i_idx]
                            tgt_layer = layers[j_idx]
                            if src_layer >= tgt_layer:
                                strict = False
                                break
                if not strict:
                    break
            if strict:
                strict_order_count += 1

            for pos_idx in range(4):
                position_layers[pos_idx].append(layers[pos_idx])

        n = len(instances)
        strict_frac = strict_order_count / n if n > 0 else 0.0
        mean_span = float(np.mean(spans)) if spans else 0.0
        std_span = float(np.std(spans)) if spans else 0.0

        # Mann-Whitney U test: motif spans vs baseline spans
        mw_u, mw_p = 0.0, 1.0
        if spans and all_baseline_spans:
            try:
                mw_u, mw_p = mannwhitneyu(spans, all_baseline_spans, alternative="two-sided")
                mw_u = float(mw_u)
                mw_p = float(mw_p)
            except ValueError:
                pass

        # Position layer distributions
        pos_layer_dist: dict[int, dict] = {}
        for pos_idx in range(4):
            c = Counter(position_layers[pos_idx])
            pos_layer_dist[pos_idx] = {str(k): v for k, v in sorted(c.items())}

        results[mid] = {
            "n_instances": n,
            "strict_order_frac": round(strict_frac, 4),
            "mean_layer_span": round(mean_span, 3),
            "std_layer_span": round(std_span, 3),
            "random_baseline_mean_span": round(baseline_mean_span, 3),
            "random_baseline_std_span": round(baseline_std_span, 3),
            "random_baseline_strict_frac": round(baseline_strict_frac, 4),
            "mann_whitney_U": safe_round(mw_u, 2),
            "mann_whitney_p": safe_round(mw_p, 6),
            "position_layer_distributions": pos_layer_dist,
            "comparison_to_ffl": "same" if abs(strict_frac - 1.0) < 0.05 else "different",
        }
        logger.info(f"Motif {mid}: strict_order={strict_frac:.3f}, mean_span={mean_span:.2f}, "
                     f"MW p={mw_p:.4g}")

    return results


# ============================================================
# STEP 7 (PHASE D): Semantic Role Mapping
# ============================================================

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "input_encoding": [
        "token", "word", "letter", "character", "the word",
        "mentions of", "beginning of", "end of", "prefix",
        "strings", "text containing", "input",
    ],
    "intermediate_concept": [
        "concept", "meaning", "related to", "associated with",
        "references to", "instances of", "topics", "context",
        "country", "capital", "language", "number", "math",
        "opposite", "similar", "sentiment", "emotion",
    ],
    "output_generation": [
        "predict", "output", "logit", "next token", "completion",
        "answer", "response", "generation", "following",
    ],
    "syntactic": [
        "grammar", "syntax", "punctuation", "structure", "formatting",
        "comma", "period", "parenthes", "bracket", "newline",
        "whitespace", "indentation",
    ],
}


def categorize_explanation(explanation_text: str | None) -> str:
    """Classify a feature explanation into a semantic category."""
    if not explanation_text or not explanation_text.strip():
        return "no_explanation"
    text_lower = explanation_text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "other"


def analyze_semantic_roles(
    all_instances: dict[int, list],
    graph_dicts: list[dict],
    expl_lookup: dict[str, dict],
) -> dict:
    """Map semantic roles at each motif position using Neuronpedia explanations."""
    graph_by_slug: dict[str, igraph.Graph] = {}
    for gd in graph_dicts:
        graph_by_slug[gd["slug"]] = gd["graph"]

    rng = random.Random(RANDOM_SEED)
    results: dict[int, dict] = {}

    for mid in TARGET_MOTIF_IDS:
        instances = all_instances.get(mid, [])
        if not instances:
            results[mid] = {"n_instances": 0, "explanation_match_rate": 0.0}
            continue

        # Stratified sample by domain
        by_domain: dict[str, list] = defaultdict(list)
        for inst in instances:
            by_domain[inst["domain"]].append(inst)

        sampled: list[dict] = []
        n_per_domain = max(1, N_SEMANTIC_SAMPLES // max(len(by_domain), 1))
        for domain, dom_insts in by_domain.items():
            n_take = min(n_per_domain, len(dom_insts))
            sampled.extend(rng.sample(dom_insts, n_take))

        # Cap at N_SEMANTIC_SAMPLES
        if len(sampled) > N_SEMANTIC_SAMPLES:
            sampled = rng.sample(sampled, N_SEMANTIC_SAMPLES)

        # Analyze positions
        match_count = 0
        total_count = 0
        position_categories: dict[int, list[str]] = {i: [] for i in range(4)}

        for inst in sampled:
            slug = inst["slug"]
            g = graph_by_slug.get(slug)
            if g is None:
                continue

            for pos_idx in range(4):
                total_count += 1
                vid = inst["vertex_ids"][pos_idx]
                feat = g.vs[vid]["feature"]
                ftype = g.vs[vid]["feature_type"]

                category = "no_explanation"
                if ftype == "cross layer transcoder" and feat is not None and feat >= 0:
                    try:
                        l, f = cantor_decode(feat)
                        key = f"{l}_{f}"
                        if key in expl_lookup:
                            explanation = expl_lookup[key]["explanation"]
                            category = categorize_explanation(explanation)
                            match_count += 1
                    except (ValueError, OverflowError):
                        pass
                elif ftype == "embedding":
                    category = "input_encoding"
                    match_count += 1
                elif ftype == "logit":
                    category = "output_generation"
                    match_count += 1
                elif ftype == "mlp reconstruction error":
                    category = "intermediate_concept"
                    match_count += 1

                position_categories[pos_idx].append(category)

        match_rate = match_count / total_count if total_count > 0 else 0.0

        # Per-position role distribution
        all_cats = sorted(set(c for cats in position_categories.values() for c in cats))
        per_pos_dist: dict[int, dict] = {}
        chi2_per_pos: dict[int, dict] = {}

        for pos_idx in range(4):
            c = Counter(position_categories[pos_idx])
            total_pos = sum(c.values())
            per_pos_dist[pos_idx] = {cat: round(c.get(cat, 0) / total_pos, 4) if total_pos > 0 else 0.0
                                      for cat in all_cats}

        # Overall contingency table: positions x categories
        table: list[list[int]] = []
        for pos_idx in range(4):
            c = Counter(position_categories[pos_idx])
            table.append([c.get(cat, 0) for cat in all_cats])
        table_arr = np.array(table)

        # Filter zero columns
        col_sums = table_arr.sum(axis=0)
        nonzero = col_sums > 0
        table_f = table_arr[:, nonzero]
        cats_f = [cat for cat, nz in zip(all_cats, nonzero) if nz]

        chi2_val, p_val, dof = 0.0, 1.0, 0
        overall_v = 0.0
        if table_f.shape[1] >= 2:
            try:
                chi2_val, p_val, dof, _ = chi2_contingency(table_f)
                overall_v = compute_cramers_v(table_f)
            except ValueError:
                pass

        # Per-position chi-squared (position vs uniform)
        for pos_idx in range(4):
            c = Counter(position_categories[pos_idx])
            observed = np.array([c.get(cat, 0) for cat in cats_f])
            if len(observed) >= 2 and observed.sum() > 0:
                # Compare to uniform
                expected = np.full_like(observed, observed.sum() / len(observed), dtype=float)
                if all(expected > 0):
                    from scipy.stats import chisquare
                    try:
                        chi2_pos, p_pos = chisquare(observed, f_exp=expected)
                        chi2_per_pos[pos_idx] = {"chi2": safe_round(chi2_pos), "p": safe_round(p_pos, 6)}
                    except ValueError:
                        chi2_per_pos[pos_idx] = {"chi2": 0.0, "p": 1.0}
                else:
                    chi2_per_pos[pos_idx] = {"chi2": 0.0, "p": 1.0}
            else:
                chi2_per_pos[pos_idx] = {"chi2": 0.0, "p": 1.0}

        results[mid] = {
            "n_sampled": len(sampled),
            "explanation_match_rate": round(match_rate, 4),
            "per_position_role_distribution": {str(k): v for k, v in per_pos_dist.items()},
            "chi_squared_per_position": {str(k): v for k, v in chi2_per_pos.items()},
            "overall_chi2": safe_round(chi2_val),
            "overall_p": safe_round(p_val, 6),
            "overall_cramers_v": round(overall_v, 4),
            "categories_used": cats_f,
        }
        logger.info(f"Motif {mid}: match_rate={match_rate:.3f}, V={overall_v:.3f}")

    return results


# ============================================================
# STEP 8 (PHASE E): Cross-Domain Consistency
# ============================================================

def analyze_cross_domain(
    all_instances: dict[int, list],
    graph_dicts: list[dict],
    expl_lookup: dict[str, dict],
) -> dict:
    """Analyze cross-domain consistency for each 4-node motif type."""
    graph_by_slug: dict[str, igraph.Graph] = {}
    domain_graph_sizes: dict[str, list[int]] = defaultdict(list)
    for gd in graph_dicts:
        graph_by_slug[gd["slug"]] = gd["graph"]
        domain_graph_sizes[gd["domain"]].append(gd["n_nodes"])

    results: dict[int, dict] = {}

    for mid in TARGET_MOTIF_IDS:
        instances = all_instances.get(mid, [])
        if not instances:
            results[mid] = {"n_instances": 0}
            continue

        # Group by domain
        by_domain: dict[str, list] = defaultdict(list)
        for inst in instances:
            by_domain[inst["domain"]].append(inst)

        per_domain_stats: dict[str, dict] = {}
        domain_spans: dict[str, list[float]] = {}
        domain_semantic_counts: dict[str, Counter] = {}

        for domain, dom_insts in sorted(by_domain.items()):
            # Normalized count
            mean_graph_size = float(np.mean(domain_graph_sizes.get(domain, [100])))
            normalized_count = len(dom_insts) / mean_graph_size if mean_graph_size > 0 else 0.0

            # Layer spans
            spans_d: list[float] = []
            for inst in dom_insts:
                valid_layers = [l for l in inst["layers"] if l >= 0]
                if len(valid_layers) >= 2:
                    spans_d.append(max(valid_layers) - min(valid_layers))
            domain_spans[domain] = spans_d

            # Semantic categories (use feature_type as proxy for speed)
            sem_counter: Counter = Counter()
            for inst in dom_insts[:50]:  # Sample for speed
                g = graph_by_slug.get(inst["slug"])
                if g is None:
                    continue
                for pos_idx in range(4):
                    vid = inst["vertex_ids"][pos_idx]
                    ftype = g.vs[vid]["feature_type"]
                    feat = g.vs[vid]["feature"]

                    cat = "no_explanation"
                    if ftype == "cross layer transcoder" and feat is not None and feat >= 0:
                        try:
                            l, f = cantor_decode(feat)
                            key = f"{l}_{f}"
                            if key in expl_lookup:
                                cat = categorize_explanation(expl_lookup[key]["explanation"])
                        except (ValueError, OverflowError):
                            pass
                    elif ftype == "embedding":
                        cat = "input_encoding"
                    elif ftype == "logit":
                        cat = "output_generation"
                    elif ftype == "mlp reconstruction error":
                        cat = "intermediate_concept"
                    sem_counter[cat] += 1
            domain_semantic_counts[domain] = sem_counter

            per_domain_stats[domain] = {
                "instance_count": len(dom_insts),
                "normalized_count": round(normalized_count, 4),
                "mean_layer_span": round(float(np.mean(spans_d)), 3) if spans_d else 0.0,
                "std_layer_span": round(float(np.std(spans_d)), 3) if spans_d else 0.0,
            }

        # Cross-domain Cramer's V for semantic role distribution
        all_cats = sorted(set(c for counter in domain_semantic_counts.values() for c in counter))
        domains_order = sorted(domain_semantic_counts.keys())

        if len(domains_order) >= 2 and len(all_cats) >= 2:
            cross_table = np.array([
                [domain_semantic_counts[d].get(c, 0) for c in all_cats]
                for d in domains_order
            ])
            semantic_v = compute_cramers_v(cross_table)
        else:
            semantic_v = 0.0

        # Kruskal-Wallis test: does layer span differ across domains?
        span_groups = [domain_spans[d] for d in domains_order if len(domain_spans.get(d, [])) >= 2]
        kruskal_h, kruskal_p = 0.0, 1.0
        if len(span_groups) >= 2:
            try:
                kruskal_h, kruskal_p = kruskal(*span_groups)
                kruskal_h = float(kruskal_h)
                kruskal_p = float(kruskal_p)
            except ValueError:
                pass

        results[mid] = {
            "per_domain": per_domain_stats,
            "semantic_cramers_v": round(semantic_v, 4),
            "kruskal_h": safe_round(kruskal_h, 4),
            "kruskal_p": safe_round(kruskal_p, 6),
            "comparison_to_ffl_v": "stronger" if semantic_v > 0.13 else "weaker_or_equal",
            "n_domains": len(domains_order),
        }
        logger.info(f"Motif {mid}: semantic_V={semantic_v:.3f}, kruskal_p={kruskal_p:.4g}")

    return results


# ============================================================
# STEP 9 (PHASE F): Motif Family Taxonomy
# ============================================================

def build_taxonomy(
    all_instances: dict[int, list],
    embedding_results: dict,
    graph_dicts: list[dict],
    target_patterns: dict,
) -> dict:
    """Build motif family taxonomy: clustering, ranking, edge weight profiles."""
    # A. Cluster by sub-motif composition vectors
    sub_types = ["021U", "021C", "021D", "030T", "disconnected"]
    vectors: dict[int, list[float]] = {}
    for mid in TARGET_MOTIF_IDS:
        comp = embedding_results.get(mid, {}).get("sub_motif_composition", {})
        vectors[mid] = [comp.get(st, 0.0) for st in sub_types]

    # Cosine similarity matrix
    cos_sim: dict[str, float] = {}
    for i, m1 in enumerate(TARGET_MOTIF_IDS):
        for m2 in TARGET_MOTIF_IDS[i+1:]:
            v1 = np.array(vectors.get(m1, [0]*5))
            v2 = np.array(vectors.get(m2, [0]*5))
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                sim = float(np.dot(v1, v2) / (norm1 * norm2))
            else:
                sim = 0.0
            cos_sim[f"{m1}-{m2}"] = round(sim, 4)

    # B. Rank by FFL-embeddedness
    ffl_fracs = {mid: embedding_results.get(mid, {}).get("ffl_containment_frac", 0.0)
                 for mid in TARGET_MOTIF_IDS}
    ranked = sorted(TARGET_MOTIF_IDS, key=lambda m: ffl_fracs.get(m, 0.0), reverse=True)

    # C. Edge weight profiles per type
    graph_by_slug: dict[str, igraph.Graph] = {}
    for gd in graph_dicts:
        graph_by_slug[gd["slug"]] = gd["graph"]

    edge_profiles: dict[int, dict] = {}
    for mid in TARGET_MOTIF_IDS:
        instances = all_instances.get(mid, [])
        if not instances:
            edge_profiles[mid] = {}
            continue

        # Collect edge weights by canonical edge position
        canonical_edges = target_patterns[mid]["edges"]
        edge_weights_by_pos: dict[str, list[float]] = {f"{s}->{t}": [] for s, t in canonical_edges}

        for inst in instances:
            for s, t in canonical_edges:
                key = f"{s}->{t}"
                if key in inst["edge_weights"]:
                    edge_weights_by_pos[key].append(inst["edge_weights"][key]["signed"])

        profiles: dict[str, dict] = {}
        for edge_key, weights in edge_weights_by_pos.items():
            if weights:
                profiles[edge_key] = {
                    "mean_abs": round(float(np.mean(np.abs(weights))), 4),
                    "mean_signed": round(float(np.mean(weights)), 4),
                    "frac_positive": round(float(np.mean([w > 0 for w in weights])), 4),
                    "std": round(float(np.std(weights)), 4),
                    "n_samples": len(weights),
                }
        edge_profiles[mid] = profiles

    # D. Candidate functional interpretations
    candidate_functions: dict[int, str] = {}
    for mid in TARGET_MOTIF_IDS:
        n_edges = target_patterns[mid]["edge_count"]
        ffl_frac = ffl_fracs.get(mid, 0.0)
        verdict = embedding_results.get(mid, {}).get("independence_verdict", "unknown")

        if n_edges >= 6:
            func = f"Type {mid} ({n_edges} edges, near-complete DAG): exhaustive information cascade with {verdict} FFL relationship (containment={ffl_frac:.0%})"
        elif n_edges >= 5:
            func = f"Type {mid} ({n_edges} edges): verified parallel computation with {verdict} FFL relationship (containment={ffl_frac:.0%})"
        elif n_edges >= 4:
            func = f"Type {mid} ({n_edges} edges): structured information relay with {verdict} FFL relationship (containment={ffl_frac:.0%})"
        else:
            func = f"Type {mid} ({n_edges} edges): minimal connected motif with {verdict} FFL relationship (containment={ffl_frac:.0%})"
        candidate_functions[mid] = func

    # Overall verdict
    all_ffl_fracs = [ffl_fracs.get(mid, 0.0) for mid in TARGET_MOTIF_IDS]
    mean_ffl = float(np.mean(all_ffl_fracs)) if all_ffl_fracs else 0.0
    if mean_ffl > 0.90:
        overall_verdict = "FFL-derivative family"
    elif mean_ffl < 0.50:
        overall_verdict = "genuinely novel structures"
    else:
        overall_verdict = "mixed independence"

    return {
        "sub_motif_similarity_matrix": cos_sim,
        "ffl_embeddedness_ranking": [{"motif_id": m, "ffl_frac": round(ffl_fracs.get(m, 0.0), 4)} for m in ranked],
        "edge_weight_profiles": {str(k): v for k, v in edge_profiles.items()},
        "candidate_functions": {str(k): v for k, v in candidate_functions.items()},
        "overall_verdict": overall_verdict,
        "mean_ffl_containment": round(mean_ffl, 4),
    }


# ============================================================
# STEP 10: ASSEMBLE OUTPUT
# ============================================================

def compute_per_graph_results(
    all_instances: dict[int, list],
    graph_dicts: list[dict],
) -> dict[str, dict]:
    """Compute per-graph motif instance counts and layer span stats for output examples."""
    # Group instances by slug
    inst_by_slug: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for mid in TARGET_MOTIF_IDS:
        for inst in all_instances.get(mid, []):
            inst_by_slug[inst["slug"]][mid].append(inst)

    # Build adjacency sets for embedding analysis
    adj_by_slug: dict[str, set] = {}
    for gd in graph_dicts:
        g = gd["graph"]
        adj_set: set[tuple[int, int]] = set()
        for e in g.es:
            adj_set.add((e.source, e.target))
        adj_by_slug[gd["slug"]] = adj_set

    rng = random.Random(RANDOM_SEED)
    per_graph: dict[str, dict] = {}

    for gd in graph_dicts:
        slug = gd["slug"]
        g = gd["graph"]
        graph_insts = inst_by_slug.get(slug, {})

        motif_counts: dict[str, int] = {}
        motif_ffl_fracs: dict[str, float] = {}
        motif_spans: dict[str, float] = {}

        for mid in TARGET_MOTIF_IDS:
            insts = graph_insts.get(mid, [])
            motif_counts[str(mid)] = len(insts)

            # FFL containment per graph
            ffl_count = 0
            spans_g: list[float] = []
            adj = adj_by_slug.get(slug, set())
            for inst in insts:
                vids = inst["vertex_ids"]
                has_ffl = False
                for combo in combinations(range(4), 3):
                    nodes_3 = [vids[i] for i in combo]
                    edges_in = [(nodes_3[i], nodes_3[j])
                                for i in range(3) for j in range(3)
                                if i != j and (nodes_3[i], nodes_3[j]) in adj]
                    if len(edges_in) == 3:
                        has_ffl = True
                        break
                if has_ffl:
                    ffl_count += 1
                valid_layers = [l for l in inst["layers"] if l >= 0]
                if len(valid_layers) >= 2:
                    spans_g.append(max(valid_layers) - min(valid_layers))

            motif_ffl_fracs[str(mid)] = round(ffl_count / len(insts), 4) if insts else 0.0
            motif_spans[str(mid)] = round(float(np.mean(spans_g)), 3) if spans_g else 0.0

        # Random baseline: sample connected 4-node subgraphs
        baseline_samples = sample_random_connected_4node(g, min(100, N_RANDOM_BASELINE), rng)
        baseline_span = round(float(np.mean([s["span"] for s in baseline_samples])), 3) if baseline_samples else 0.0
        baseline_strict = round(sum(s["strict_ordered"] for s in baseline_samples) / len(baseline_samples), 4) if baseline_samples else 0.0

        per_graph[slug] = {
            "motif_counts": motif_counts,
            "motif_ffl_fracs": motif_ffl_fracs,
            "motif_mean_spans": motif_spans,
            "baseline_mean_span": baseline_span,
            "baseline_strict_frac": baseline_strict,
            "n_nodes": gd["n_nodes"],
            "n_edges": gd["n_edges"],
            "domain": gd["domain"],
        }

    return per_graph


def assemble_output(
    metadata: dict,
    instance_counts: dict,
    embedding_results: dict,
    layer_results: dict,
    semantic_results: dict,
    cross_domain_results: dict,
    taxonomy: dict,
    graph_dicts: list[dict],
    per_graph_results: dict[str, dict],
) -> dict:
    """Assemble the final method_out.json output conforming to exp_gen_sol_out schema."""
    examples: list[dict] = []

    # --- Per-graph examples (one per graph = 179+ examples) ---
    for gd in graph_dicts:
        slug = gd["slug"]
        pgr = per_graph_results.get(slug, {})

        # Ground truth: the graph metadata
        ground_truth = {
            "slug": slug,
            "domain": gd["domain"],
            "n_nodes": gd["n_nodes"],
            "n_edges": gd["n_edges"],
            "prompt": gd["prompt"],
        }

        # Our method prediction: motif characterization for this graph
        our_method = {
            "motif_counts": pgr.get("motif_counts", {}),
            "motif_ffl_containment": pgr.get("motif_ffl_fracs", {}),
            "motif_mean_layer_spans": pgr.get("motif_mean_spans", {}),
            "total_4node_motifs": sum(pgr.get("motif_counts", {}).values()),
            "all_ffl_derivative": all(v >= 1.0 for v in pgr.get("motif_ffl_fracs", {}).values()
                                      if pgr.get("motif_counts", {}).get(
                                          list(pgr.get("motif_ffl_fracs", {}).keys())[0] if pgr.get("motif_ffl_fracs", {}) else "0", 0) > 0),
        }

        # Baseline prediction: random connected 4-node subgraph stats
        baseline = {
            "random_baseline_mean_span": pgr.get("baseline_mean_span", 0.0),
            "random_baseline_strict_frac": pgr.get("baseline_strict_frac", 0.0),
            "motif_counts": {str(mid): 0 for mid in TARGET_MOTIF_IDS},
            "interpretation": "random_connected_subgraphs_no_motif_structure",
        }

        examples.append({
            "input": f"graph:{slug} domain:{gd['domain']} prompt:{gd['prompt'][:100]}",
            "output": json.dumps(ground_truth, default=str),
            "metadata_fold": gd["domain"],
            "metadata_slug": slug,
            "metadata_n_nodes": gd["n_nodes"],
            "predict_motif_characterization": json.dumps(our_method, default=str),
            "predict_random_baseline": json.dumps(baseline, default=str),
        })

    # --- Per-motif-type summary examples ---
    for mid in TARGET_MOTIF_IDS:
        type_results = {
            "motif_id": mid,
            "phase_a_instances": instance_counts.get(mid, {}),
            "phase_b_embedding": embedding_results.get(mid, {}),
            "phase_c_layer_ordering": layer_results.get(mid, {}),
            "phase_d_semantic_roles": semantic_results.get(mid, {}),
            "phase_e_cross_domain": cross_domain_results.get(mid, {}),
        }
        our_summary = {
            "verdict": embedding_results.get(mid, {}).get("independence_verdict", "unknown"),
            "ffl_containment": embedding_results.get(mid, {}).get("ffl_containment_frac", 0.0),
            "strict_order_frac": layer_results.get(mid, {}).get("strict_order_frac", 0.0),
            "mean_layer_span": layer_results.get(mid, {}).get("mean_layer_span", 0.0),
            "semantic_cramers_v": semantic_results.get(mid, {}).get("overall_cramers_v", 0.0),
        }
        baseline_summary = {
            "verdict": "no_motif_structure",
            "ffl_containment": 0.0,
            "strict_order_frac": layer_results.get(mid, {}).get("random_baseline_strict_frac", 0.0),
            "mean_layer_span": layer_results.get(mid, {}).get("random_baseline_mean_span", 0.0),
            "semantic_cramers_v": 0.0,
        }
        examples.append({
            "input": f"motif_type_{mid}_summary across {len(graph_dicts)} graphs",
            "output": json.dumps(type_results, default=str),
            "metadata_fold": "summary",
            "predict_motif_characterization": json.dumps(our_summary, default=str),
            "predict_random_baseline": json.dumps(baseline_summary, default=str),
        })

    # --- Taxonomy summary ---
    examples.append({
        "input": "taxonomy_summary: motif family classification and functional interpretations",
        "output": json.dumps(taxonomy, default=str),
        "metadata_fold": "summary",
        "predict_motif_characterization": json.dumps({
            "overall_verdict": taxonomy.get("overall_verdict", "unknown"),
            "mean_ffl_containment": taxonomy.get("mean_ffl_containment", 0.0),
        }, default=str),
        "predict_random_baseline": json.dumps({
            "overall_verdict": "no_structure",
            "mean_ffl_containment": 0.0,
        }, default=str),
    })

    output = {
        "metadata": metadata,
        "datasets": [{
            "dataset": "4node_motif_characterization",
            "examples": examples,
        }],
    }
    return output


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("4-Node Motif Functional Characterization Experiment")
    logger.info("=" * 60)

    # ---- Step 1: Load explanations ----
    expl_lookup = load_explanations()

    # ---- Step 2: Load and prune graphs ----
    graph_dicts = load_all_graphs()
    if not graph_dicts:
        logger.error("No graphs loaded, aborting")
        return

    domain_counts = Counter(gd["domain"] for gd in graph_dicts)
    domains = sorted(domain_counts.keys())

    # ---- Step 3: Build canonical references ----
    target_patterns, three_node_ref = build_canonical_references()

    # ---- Step 4 (Phase A): Enumerate motif instances ----
    all_instances = enumerate_all_motifs(graph_dicts, TARGET_MOTIF_IDS)

    # Instance count summary
    instance_counts: dict[int, dict] = {}
    for mid in TARGET_MOTIF_IDS:
        insts = all_instances[mid]
        per_domain: dict[str, int] = Counter(inst["domain"] for inst in insts)
        instance_counts[mid] = {
            "total_instances": len(insts),
            "per_domain_counts": dict(per_domain),
        }

    # ---- Step 5 (Phase B): 3->4 embedding analysis ----
    logger.info("Phase B: Analyzing 3->4 embedding...")
    embedding_results = analyze_3to4_embedding(all_instances, graph_dicts)

    # ---- Step 6 (Phase C): Layer ordering ----
    logger.info("Phase C: Analyzing layer ordering...")
    layer_results = analyze_layer_ordering(all_instances, graph_dicts)

    # ---- Step 7 (Phase D): Semantic roles ----
    logger.info("Phase D: Analyzing semantic roles...")
    semantic_results = analyze_semantic_roles(all_instances, graph_dicts, expl_lookup)

    # ---- Step 8 (Phase E): Cross-domain consistency ----
    logger.info("Phase E: Analyzing cross-domain consistency...")
    cross_domain_results = analyze_cross_domain(all_instances, graph_dicts, expl_lookup)

    # ---- Step 9 (Phase F): Taxonomy ----
    logger.info("Phase F: Building taxonomy...")
    taxonomy = build_taxonomy(all_instances, embedding_results, graph_dicts, target_patterns)

    # ---- Step 9.5: Per-graph results for output examples ----
    logger.info("Computing per-graph results for output examples...")
    per_graph_results = compute_per_graph_results(all_instances, graph_dicts)
    logger.info(f"Computed per-graph results for {len(per_graph_results)} graphs")

    # ---- Step 10: Assemble output ----
    elapsed = time.time() - t_start
    metadata = {
        "experiment": "4node_motif_functional_characterization",
        "n_graphs_loaded": len(graph_dicts),
        "prune_percentile": PRUNE_PERCENTILE,
        "max_nodes": MAX_NODES,
        "target_motif_ids": TARGET_MOTIF_IDS,
        "runtime_seconds": round(elapsed, 1),
        "domains": domains,
        "domain_counts": dict(domain_counts),
        "4node_canonical_patterns": {str(k): {
            "edge_count": v["edge_count"],
            "edges": v["edges"],
        } for k, v in target_patterns.items()},
        "3node_reference": three_node_ref,
        "n_explanations": len(expl_lookup),
    }

    output = assemble_output(
        metadata=metadata,
        instance_counts=instance_counts,
        embedding_results=embedding_results,
        layer_results=layer_results,
        semantic_results=semantic_results,
        cross_domain_results=cross_domain_results,
        taxonomy=taxonomy,
        graph_dicts=graph_dicts,
        per_graph_results=per_graph_results,
    )

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output written to {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")
    logger.info(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
