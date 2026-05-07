#!/usr/bin/env python3
"""FFL Functional Characterization: Enumerate feed-forward loop instances in
attribution graphs, map semantic roles, and compare with random-triad baseline.

Reads 34 pruned attribution graphs (dependency 1) and 6,574 feature explanations
(dependency 2), enumerates all 030T (FFL) motif instances, annotates each node
position (A=regulator, B=intermediary, C=target) with Neuronpedia explanations,
and runs layer analysis, keyword-based semantic categorization, chi-squared tests,
cross-domain Cramer's V, edge-weight sign analysis (coherent vs incoherent FFLs),
plus a random-triad baseline for comparison.
"""

import gc
import glob
import json
import math
import os
import random
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import igraph
import numpy as np
from loguru import logger
from scipy.stats import chi2_contingency

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
# Budget: graphs ~170MB, explanations ~50MB, FFL dicts ~500MB headroom
RAM_BUDGET_GB = min(TOTAL_RAM_GB * 0.75, 20.0)
RAM_BUDGET_BYTES = int(RAM_BUDGET_GB * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET_GB:.1f}GB")

# ============================================================
# CONSTANTS & PATHS
# ============================================================
WORKSPACE = Path(__file__).parent.resolve()
DATA_DIR = Path(__file__).parent.resolve().parents[2] / "iter_1" / "gen_art" / "data_id4_it1__opus" / "data_out"
DATA_FILES = ["full_data_out_1.json", "full_data_out_2.json", "full_data_out_3.json"]
EXPL_DIR = Path(__file__).parent.resolve().parents[2] / "iter_2" / "gen_art" / "data_id4_it2__opus"
EXPL_FILE = EXPL_DIR / "full_data_out.json"
OUTPUT_FILE = WORKSPACE / "method_out.json"
PRUNE_PERCENTILE = 75
MIN_NODES = 30
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "9999"))
RANDOM_SEED = 42
N_BASELINE_SAMPLES = 5000  # random triads per graph for baseline

# ============================================================
# PHASE 0: Cantor Decode Utility
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


# ============================================================
# PHASE 1: Load Feature Explanation Lookup Table
# ============================================================

def load_explanations() -> dict[str, dict]:
    """Load feature explanations into a lookup dict keyed by 'layer_featureindex'."""
    logger.info(f"Loading explanations from {EXPL_FILE}")
    data = json.loads(EXPL_FILE.read_text())
    examples = data["datasets"][0]["examples"]
    lookup: dict[str, dict] = {}
    for ex in examples:
        key = ex["input"]  # format: "{layer}_{feature_index}"
        try:
            output = json.loads(ex["output"])
        except json.JSONDecodeError:
            continue
        lookup[key] = {
            "explanation": output.get("explanation", ""),
            "source_domains": output.get("source_domains", []),
            "frac_nonzero": output.get("frac_nonzero"),
            "max_activation": output.get("max_activation"),
            "top_positive_logits": output.get("top_positive_logits", []),
            "top_negative_logits": output.get("top_negative_logits", []),
        }
    logger.info(f"Loaded {len(lookup)} feature explanations")
    return lookup


# ============================================================
# PHASE 2: Load and Build Graphs (with SIGNED weights preserved)
# ============================================================

def load_all_graphs() -> list[dict]:
    """Load all 34 graphs, prune at 75th percentile, keep signed weights."""
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

            # Build node_id -> index mapping
            node_id_to_idx: dict[str, int] = {}
            for i, n in enumerate(nodes):
                node_id_to_idx[n["node_id"]] = i

            node_layers = [parse_layer(n.get("layer", "0")) for n in nodes]
            feature_types = [n.get("feature_type", "") for n in nodes]
            features = [n.get("feature", None) for n in nodes]

            # Build signed weight dict BEFORE igraph construction
            # key: (source_node_id, target_node_id) -> signed_weight
            signed_weight_dict: dict[tuple[str, str], float] = {}

            # Collect ALL edges with BOTH signed and abs weights
            all_abs_weights = [abs(link.get("weight", 0.0)) for link in links]
            threshold = float(np.percentile(all_abs_weights, PRUNE_PERCENTILE)) if all_abs_weights else 0

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
                        # Store by node_id pair for lookup after simplify
                        signed_weight_dict[(src_id, tgt_id)] = float(raw_w)

            if len(edges) == 0:
                continue

            # Build igraph
            g = igraph.Graph(n=len(nodes), edges=edges, directed=True)
            g.vs["node_id"] = [n["node_id"] for n in nodes]
            g.vs["layer"] = node_layers
            g.vs["feature_type"] = feature_types
            g.vs["feature"] = features
            g.es["weight"] = abs_weights
            g.es["signed_weight"] = signed_weights

            # Simplify: remove multi-edges (keep max abs weight) and self-loops
            # BEFORE simplify, store the signed weights separately keyed by (src_idx, tgt_idx)
            edge_signed_before: dict[tuple[int, int], float] = {}
            for e in g.es:
                key = (e.source, e.target)
                # Keep the one with max abs weight
                if key not in edge_signed_before or abs(e["signed_weight"]) > abs(edge_signed_before[key]):
                    edge_signed_before[key] = e["signed_weight"]

            g.simplify(multiple=True, loops=True, combine_edges={"weight": "max"})

            # Restore signed weights after simplify using node_id lookup
            for e in g.es:
                src_nid = g.vs[e.source]["node_id"]
                tgt_nid = g.vs[e.target]["node_id"]
                if (src_nid, tgt_nid) in signed_weight_dict:
                    e["signed_weight"] = signed_weight_dict[(src_nid, tgt_nid)]
                elif (e.source, e.target) in edge_signed_before:
                    e["signed_weight"] = edge_signed_before[(e.source, e.target)]
                else:
                    e["signed_weight"] = e["weight"]  # fallback: positive abs

            # Remove isolated vertices
            isolated = [v.index for v in g.vs if g.degree(v) == 0]
            if isolated:
                g.delete_vertices(isolated)

            if g.vcount() < MIN_NODES:
                logger.debug(f"Skipping graph {ex.get('metadata_slug', '?')}: only {g.vcount()} nodes after pruning")
                continue

            if not g.is_dag():
                logger.debug(f"Skipping graph {ex.get('metadata_slug', '?')}: not a DAG")
                continue

            all_graphs.append({
                "graph": g,
                "domain": ex.get("metadata_fold", "unknown"),
                "prompt": ex.get("input", ""),
                "slug": ex.get("metadata_slug", ""),
                "n_nodes": g.vcount(),
                "n_edges": g.ecount(),
            })
            examples_loaded += 1

        del raw
        gc.collect()

    logger.info(f"Loaded {len(all_graphs)} graphs (target was {MAX_EXAMPLES})")
    return all_graphs


# ============================================================
# PHASE 3: FFL Instance Enumeration (Core Algorithm)
# ============================================================

def enumerate_ffls(g: igraph.Graph) -> list[dict]:
    """Enumerate all FFL instances in a directed graph.

    FFL (030T motif): A->B, A->C, B->C
    For each node A, check all pairs of successors for B->C edge.
    Uses adjacency set for O(1) edge lookup.
    """
    # Build adjacency set for O(1) edge existence check
    adj_set: set[tuple[int, int]] = set()
    for e in g.es:
        adj_set.add((e.source, e.target))

    # Build edge weight lookup: (src, tgt) -> (abs_weight, signed_weight)
    weight_lookup: dict[tuple[int, int], dict] = {}
    for e in g.es:
        weight_lookup[(e.source, e.target)] = {
            "abs": float(e["weight"]),
            "signed": float(e["signed_weight"]),
        }

    ffls: list[dict] = []
    for a in range(g.vcount()):
        successors_a = g.successors(a)
        if len(successors_a) < 2:
            continue
        # For each pair (b, c) of successors
        for i, b in enumerate(successors_a):
            for c in successors_a[i + 1:]:
                # Check B->C
                if (b, c) in adj_set:
                    ffls.append({
                        "a": a, "b": b, "c": c,
                        "w_ab": weight_lookup[(a, b)],
                        "w_ac": weight_lookup[(a, c)],
                        "w_bc": weight_lookup[(b, c)],
                    })
                # Check C->B (reversed intermediary/target)
                if (c, b) in adj_set:
                    ffls.append({
                        "a": a, "b": c, "c": b,
                        "w_ab": weight_lookup[(a, c)],
                        "w_ac": weight_lookup[(a, b)],
                        "w_bc": weight_lookup[(c, b)],
                    })
    return ffls


# ============================================================
# PHASE 4: Feature Explanation Lookup for FFL Nodes
# ============================================================

def lookup_ffl_explanations(
    ffls: list[dict],
    g: igraph.Graph,
    expl_lookup: dict[str, dict],
) -> tuple[list[dict], float]:
    """Annotate each FFL instance with feature explanations for A, B, C."""
    annotated: list[dict] = []
    match_count = 0
    total_count = 0

    for ffl in ffls:
        record: dict = {}
        # Copy weight info
        record["w_ab"] = ffl["w_ab"]
        record["w_ac"] = ffl["w_ac"]
        record["w_bc"] = ffl["w_bc"]

        for role, idx in [("a", ffl["a"]), ("b", ffl["b"]), ("c", ffl["c"])]:
            total_count += 1
            feat = g.vs[idx]["feature"]
            ftype = g.vs[idx]["feature_type"]
            layer = g.vs[idx]["layer"]
            node_id = g.vs[idx]["node_id"]

            expl = None
            expl_key = None
            if ftype == "cross layer transcoder" and feat is not None and feat >= 0:
                try:
                    l, f = cantor_decode(feat)
                    expl_key = f"{l}_{f}"
                    if expl_key in expl_lookup:
                        expl = expl_lookup[expl_key]["explanation"]
                        match_count += 1
                except (ValueError, OverflowError):
                    pass

            record[f"{role}_idx"] = idx
            record[f"{role}_node_id"] = node_id
            record[f"{role}_layer"] = layer
            record[f"{role}_feature_type"] = ftype
            record[f"{role}_explanation"] = expl
            record[f"{role}_expl_key"] = expl_key

        annotated.append(record)

    match_rate = match_count / total_count if total_count > 0 else 0
    return annotated, match_rate


# ============================================================
# PHASE 5: Layer Position Analysis
# ============================================================

def analyze_layer_positions(annotated_ffls: list[dict]) -> dict:
    """Compute layer ordering statistics for FFL instances."""
    if not annotated_ffls:
        return {"total_ffls": 0}

    strict_order = 0
    a_earliest = 0
    c_latest = 0
    same_layer_any = 0
    total = len(annotated_ffls)

    gaps_ab: list[float] = []
    gaps_ac: list[float] = []
    gaps_bc: list[float] = []

    for ffl in annotated_ffls:
        la, lb, lc = ffl["a_layer"], ffl["b_layer"], ffl["c_layer"]
        gaps_ab.append(lb - la)
        gaps_ac.append(lc - la)
        gaps_bc.append(lc - lb)

        if la < lb < lc:
            strict_order += 1
        if la < lb and la < lc:
            a_earliest += 1
        if lc > la and lc > lb:
            c_latest += 1
        if la == lb or la == lc or lb == lc:
            same_layer_any += 1

    return {
        "total_ffls": total,
        "frac_strict_order_abc": round(strict_order / total, 4),
        "frac_a_earliest": round(a_earliest / total, 4),
        "frac_c_latest": round(c_latest / total, 4),
        "frac_same_layer_any": round(same_layer_any / total, 4),
        "gap_ab": {
            "mean": round(float(np.mean(gaps_ab)), 3),
            "std": round(float(np.std(gaps_ab)), 3),
            "median": round(float(np.median(gaps_ab)), 3),
        },
        "gap_ac": {
            "mean": round(float(np.mean(gaps_ac)), 3),
            "std": round(float(np.std(gaps_ac)), 3),
            "median": round(float(np.median(gaps_ac)), 3),
        },
        "gap_bc": {
            "mean": round(float(np.mean(gaps_bc)), 3),
            "std": round(float(np.std(gaps_bc)), 3),
            "median": round(float(np.median(gaps_bc)), 3),
        },
    }


# ============================================================
# PHASE 6: Semantic Role Categorization (Keyword Matching)
# ============================================================

ROLE_PATTERNS: dict[str, list[str]] = {
    "input_encoding": [
        r"\btoken\b", r"\bword\b", r"\bletter\b", r"\bcharacter\b",
        r"\binput\b", r"\bthe word\b", r"\bthe letter\b", r"\bmentions of\b",
        r"\boccurrences of\b", r"\bthe string\b", r"\bthe text\b",
        r"\bpunctuation\b", r"\bdigit\b", r"\bnumber\b.*\binput\b",
    ],
    "concept_intermediate": [
        r"\bconcept\b", r"\bcategor\b", r"\brelat\b", r"\bsimilar\b",
        r"\bassociat\b", r"\bmeaning\b", r"\bsemantic\b", r"\babstract\b",
        r"\bcountry\b", r"\bcity\b", r"\blanguage\b", r"\btopic\b",
        r"\bcontext\b", r"\btype of\b", r"\bkind of\b",
    ],
    "output_generation": [
        r"\boutput\b", r"\bpredict\b", r"\bgenerat\b", r"\bcomplet\b",
        r"\bnext token\b", r"\bfollowing\b", r"\bresponse\b",
        r"\banswer\b", r"\bresult\b", r"\bproduc\b",
    ],
    "linguistic_structural": [
        r"\bgrammar\b", r"\bsyntax\b", r"\bposition\b", r"\bending\b",
        r"\bprefix\b", r"\bsuffix\b", r"\bplural\b", r"\btense\b",
        r"\bspacing\b", r"\bformat\b",
        r"\bsentence\b.*\bstructure\b", r"\bclause\b",
    ],
    "domain_specific": [
        r"\barithmet\b", r"\baddition\b", r"\bsubtract\b", r"\bmath\b",
        r"\bcode\b", r"\bpython\b", r"\bvariable\b", r"\bfunction\b",
        r"\bcapital\b", r"\btranslat\b", r"\brhyme\b", r"\bsentiment\b",
        r"\bopposite\b", r"\bantonym\b", r"\bsynonym\b",
    ],
}

# Pre-compile patterns for speed
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    role: [re.compile(p) for p in pats]
    for role, pats in ROLE_PATTERNS.items()
}


_role_cache: dict[str, str] = {}


def classify_role(explanation: str | None) -> str:
    """Classify a feature explanation into a functional role category (cached)."""
    if explanation is None or explanation.strip() == "":
        return "unknown"
    if explanation in _role_cache:
        return _role_cache[explanation]
    text = explanation.lower()
    scores: dict[str, int] = {}
    for role, patterns in _COMPILED_PATTERNS.items():
        scores[role] = sum(1 for p in patterns if p.search(text))
    best = max(scores, key=lambda k: scores[k])
    result = "other" if scores[best] == 0 else best
    _role_cache[explanation] = result
    return result


def semantic_role_analysis(annotated_ffls: list[dict]) -> dict:
    """Compute role distributions per position and chi-squared test."""
    if not annotated_ffls:
        return {}

    position_roles: dict[str, list[str]] = {"a": [], "b": [], "c": []}
    for ffl in annotated_ffls:
        for pos in ["a", "b", "c"]:
            role = classify_role(ffl[f"{pos}_explanation"])
            position_roles[pos].append(role)

    # Build contingency table: rows=positions, cols=roles
    all_roles = sorted(set(r for roles in position_roles.values() for r in roles))
    table: list[list[int]] = []
    for pos in ["a", "b", "c"]:
        counts = Counter(position_roles[pos])
        table.append([counts.get(r, 0) for r in all_roles])

    table_arr = np.array(table)

    # Chi-squared test (need at least 2x2 with nonzero columns)
    # Remove zero-sum columns
    col_sums = table_arr.sum(axis=0)
    nonzero_cols = col_sums > 0
    table_filtered = table_arr[:, nonzero_cols]
    roles_filtered = [r for r, nz in zip(all_roles, nonzero_cols) if nz]

    chi2, p_value, dof = 0.0, 1.0, 0
    if table_filtered.shape[1] >= 2:
        try:
            chi2, p_value, dof, _ = chi2_contingency(table_filtered)
        except ValueError:
            pass

    # Cramers V for overall position-role association
    n = table_filtered.sum()
    min_dim = min(table_filtered.shape) - 1
    cramers_v_val = 0.0
    if min_dim > 0 and n > 0:
        cramers_v_val = math.sqrt(chi2 / (n * min_dim))

    # Per-position distribution
    distributions: dict[str, dict] = {}
    for pos in ["a", "b", "c"]:
        c = Counter(position_roles[pos])
        total = sum(c.values())
        distributions[pos] = {r: round(c[r] / total, 4) for r in all_roles}

    return {
        "distributions": distributions,
        "chi2": round(float(chi2), 4),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": round(cramers_v_val, 4),
        "role_labels": all_roles,
        "contingency_table": table,
    }


# ============================================================
# PHASE 7: Cross-Domain Consistency (Cramer's V)
# ============================================================

def cramers_v(contingency_table: np.ndarray) -> float:
    """Compute Cramer's V from a contingency table."""
    try:
        chi2_val, _, _, _ = chi2_contingency(contingency_table)
    except ValueError:
        return 0.0
    n = np.sum(contingency_table)
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return math.sqrt(chi2_val / (n * min_dim))


def cross_domain_analysis(annotated_ffls_by_domain: dict[str, list[dict]]) -> dict:
    """Compute per-domain role distributions and cross-domain consistency."""
    per_domain_results: dict[str, dict] = {}
    a_role_counts_by_domain: dict[str, Counter] = {}

    for domain, ffls in annotated_ffls_by_domain.items():
        if len(ffls) < 10:
            logger.debug(f"Skipping domain {domain}: only {len(ffls)} FFLs")
            continue
        roles_analysis = semantic_role_analysis(ffls)
        per_domain_results[domain] = {
            "n_ffls": len(ffls),
            "chi2": roles_analysis.get("chi2", 0),
            "p_value": roles_analysis.get("p_value", 1),
            "cramers_v": roles_analysis.get("cramers_v", 0),
        }

        a_counts = Counter(classify_role(f["a_explanation"]) for f in ffls)
        a_role_counts_by_domain[domain] = a_counts

    # Cross-domain chi-squared: are A-role distributions same across domains?
    all_roles = sorted(set(r for c in a_role_counts_by_domain.values() for r in c))
    cross_table: list[list[int]] = []
    domain_order: list[str] = []
    for domain in sorted(a_role_counts_by_domain):
        c = a_role_counts_by_domain[domain]
        cross_table.append([c.get(r, 0) for r in all_roles])
        domain_order.append(domain)
    cross_arr = np.array(cross_table) if cross_table else np.array([[]])

    cv, chi2_val, p_val = 0.0, 0.0, 1.0
    if cross_arr.ndim == 2 and cross_arr.shape[0] >= 2 and cross_arr.shape[1] >= 2:
        # Remove zero columns
        col_sums = cross_arr.sum(axis=0)
        nonzero = col_sums > 0
        cross_filtered = cross_arr[:, nonzero]
        if cross_filtered.shape[1] >= 2:
            try:
                cv = cramers_v(cross_filtered)
                chi2_val, p_val, _, _ = chi2_contingency(cross_filtered)
            except ValueError:
                pass

    return {
        "per_domain": per_domain_results,
        "domains_analyzed": domain_order,
        "cross_domain_cramers_v": round(cv, 4),
        "cross_domain_chi2": round(float(chi2_val), 4),
        "cross_domain_p": float(p_val),
    }


# ============================================================
# PHASE 8: Edge Weight & Coherence Analysis
# ============================================================

def edge_weight_analysis(annotated_ffls: list[dict]) -> dict:
    """Analyze edge weight patterns in FFL instances."""
    if not annotated_ffls:
        return {}

    direct_strengths: list[float] = []
    indirect_strengths_mult: list[float] = []
    indirect_strengths_add: list[float] = []
    ratios: list[float] = []
    coherent_count = 0
    incoherent_count = 0
    sign_patterns: Counter = Counter()

    for ffl in annotated_ffls:
        w_ab_s = ffl["w_ab"]["signed"]
        w_ac_s = ffl["w_ac"]["signed"]
        w_bc_s = ffl["w_bc"]["signed"]

        w_ab_a = ffl["w_ab"]["abs"]
        w_ac_a = ffl["w_ac"]["abs"]
        w_bc_a = ffl["w_bc"]["abs"]

        direct_strengths.append(w_ac_a)
        indirect_mult = w_ab_a * w_bc_a
        indirect_strengths_mult.append(indirect_mult)
        indirect_strengths_add.append(w_ab_a + w_bc_a)

        if indirect_mult > 1e-12:
            ratios.append(w_ac_a / indirect_mult)

        # Sign pattern
        signs = (int(np.sign(w_ab_s)), int(np.sign(w_ac_s)), int(np.sign(w_bc_s)))
        sign_patterns[signs] += 1

        # Coherent: indirect path sign matches direct path sign
        indirect_sign = np.sign(w_ab_s) * np.sign(w_bc_s)
        if np.sign(w_ac_s) == indirect_sign:
            coherent_count += 1
        else:
            incoherent_count += 1

    total = len(annotated_ffls)
    return {
        "n_ffls": total,
        "direct_strength": {
            "mean": round(float(np.mean(direct_strengths)), 4),
            "median": round(float(np.median(direct_strengths)), 4),
            "std": round(float(np.std(direct_strengths)), 4),
        },
        "indirect_mult_strength": {
            "mean": round(float(np.mean(indirect_strengths_mult)), 4),
            "median": round(float(np.median(indirect_strengths_mult)), 4),
        },
        "indirect_add_strength": {
            "mean": round(float(np.mean(indirect_strengths_add)), 4),
            "median": round(float(np.median(indirect_strengths_add)), 4),
        },
        "direct_indirect_ratio": {
            "mean": round(float(np.mean(ratios)), 4) if ratios else None,
            "median": round(float(np.median(ratios)), 4) if ratios else None,
        },
        "coherent_fraction": round(coherent_count / total, 4),
        "incoherent_fraction": round(incoherent_count / total, 4),
        "sign_pattern_distribution": {
            str(k): v for k, v in sign_patterns.most_common(10)
        },
    }


# ============================================================
# PHASE 9: Representative Examples
# ============================================================

def select_representative_ffls(
    annotated_ffls: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """Select top-k most interpretable FFL instances (all 3 positions have explanations)."""
    # Filter to those with all 3 explanations
    fully_annotated = [
        f for f in annotated_ffls
        if f["a_explanation"] and f["b_explanation"] and f["c_explanation"]
    ]
    logger.info(f"Fully annotated FFLs (all 3 have explanations): {len(fully_annotated)}")

    if not fully_annotated:
        # Fallback: pick those with at least 2 explanations
        partially = [
            f for f in annotated_ffls
            if sum(1 for p in ["a", "b", "c"] if f[f"{p}_explanation"]) >= 2
        ]
        logger.info(f"Partially annotated FFLs (>= 2): {len(partially)}")
        fully_annotated = partially

    if not fully_annotated:
        return []

    # Score by diversity of roles across positions
    def diversity_score(ffl: dict) -> float:
        roles = [
            classify_role(ffl["a_explanation"]),
            classify_role(ffl["b_explanation"]),
            classify_role(ffl["c_explanation"]),
        ]
        unique_roles = len(set(roles) - {"unknown", "other"})
        return unique_roles + 0.1 * sum(1 for r in roles if r not in ("unknown", "other"))

    scored = sorted(fully_annotated, key=diversity_score, reverse=True)
    selected = scored[:top_k]

    # Format for output
    examples: list[dict] = []
    for ffl in selected:
        examples.append({
            "a": {
                "node_id": ffl.get("a_node_id", ""),
                "layer": ffl["a_layer"],
                "feature_type": ffl["a_feature_type"],
                "explanation": ffl["a_explanation"],
                "role": classify_role(ffl["a_explanation"]),
            },
            "b": {
                "node_id": ffl.get("b_node_id", ""),
                "layer": ffl["b_layer"],
                "feature_type": ffl["b_feature_type"],
                "explanation": ffl["b_explanation"],
                "role": classify_role(ffl["b_explanation"]),
            },
            "c": {
                "node_id": ffl.get("c_node_id", ""),
                "layer": ffl["c_layer"],
                "feature_type": ffl["c_feature_type"],
                "explanation": ffl["c_explanation"],
                "role": classify_role(ffl["c_explanation"]),
            },
            "weights": {
                "w_ab": ffl["w_ab"],
                "w_ac": ffl["w_ac"],
                "w_bc": ffl["w_bc"],
            },
            "coherent": int(np.sign(ffl["w_ac"]["signed"])) == int(
                np.sign(ffl["w_ab"]["signed"]) * np.sign(ffl["w_bc"]["signed"])
            ),
        })
    return examples


# ============================================================
# PHASE 10: BASELINE — Random Triad Comparison
# ============================================================

def sample_random_triads(
    g: igraph.Graph,
    n_samples: int,
    rng: random.Random,
) -> list[dict]:
    """Sample random connected triads (A->B, A->C exist) that are NOT FFLs."""
    adj_set: set[tuple[int, int]] = set()
    for e in g.es:
        adj_set.add((e.source, e.target))

    weight_lookup: dict[tuple[int, int], dict] = {}
    for e in g.es:
        weight_lookup[(e.source, e.target)] = {
            "abs": float(e["weight"]),
            "signed": float(e["signed_weight"]),
        }

    triads: list[dict] = []
    nodes_with_2plus_out = [v.index for v in g.vs if g.outdegree(v) >= 2]
    if not nodes_with_2plus_out:
        return []

    attempts = 0
    max_attempts = n_samples * 20

    while len(triads) < n_samples and attempts < max_attempts:
        attempts += 1
        a = rng.choice(nodes_with_2plus_out)
        successors = g.successors(a)
        if len(successors) < 2:
            continue
        b, c = rng.sample(successors, 2)

        # NOT an FFL: no B->C and no C->B edge
        if (b, c) in adj_set or (c, b) in adj_set:
            continue

        triads.append({
            "a": a, "b": b, "c": c,
            "a_layer": g.vs[a]["layer"],
            "b_layer": g.vs[b]["layer"],
            "c_layer": g.vs[c]["layer"],
            "a_feature_type": g.vs[a]["feature_type"],
            "b_feature_type": g.vs[b]["feature_type"],
            "c_feature_type": g.vs[c]["feature_type"],
        })

    return triads


def baseline_layer_analysis(triads: list[dict]) -> dict:
    """Compute layer ordering statistics for random triads (baseline)."""
    if not triads:
        return {"total_triads": 0}

    strict_order = 0
    a_earliest = 0
    c_latest = 0
    total = len(triads)

    gaps_ab: list[float] = []
    gaps_ac: list[float] = []

    for t in triads:
        la, lb, lc = t["a_layer"], t["b_layer"], t["c_layer"]
        gaps_ab.append(lb - la)
        gaps_ac.append(lc - la)

        if la < lb < lc:
            strict_order += 1
        if la < lb and la < lc:
            a_earliest += 1
        if lc > la and lc > lb:
            c_latest += 1

    return {
        "total_triads": total,
        "frac_strict_order_abc": round(strict_order / total, 4),
        "frac_a_earliest": round(a_earliest / total, 4),
        "frac_c_latest": round(c_latest / total, 4),
        "gap_ab_mean": round(float(np.mean(gaps_ab)), 3),
        "gap_ac_mean": round(float(np.mean(gaps_ac)), 3),
    }


def baseline_semantic_analysis(
    triads: list[dict],
    g: igraph.Graph,
    expl_lookup: dict[str, dict],
) -> dict:
    """Compute semantic role distributions for random triads (baseline)."""
    if not triads:
        return {}

    position_roles: dict[str, list[str]] = {"a": [], "b": [], "c": []}
    for t in triads:
        for pos in ["a", "b", "c"]:
            idx = t[pos]
            feat = g.vs[idx]["feature"]
            ftype = g.vs[idx]["feature_type"]
            expl = None
            if ftype == "cross layer transcoder" and feat is not None and feat >= 0:
                try:
                    l, f_idx = cantor_decode(feat)
                    key = f"{l}_{f_idx}"
                    if key in expl_lookup:
                        expl = expl_lookup[key]["explanation"]
                except (ValueError, OverflowError):
                    pass
            position_roles[pos].append(classify_role(expl))

    all_roles = sorted(set(r for roles in position_roles.values() for r in roles))
    table: list[list[int]] = []
    for pos in ["a", "b", "c"]:
        counts = Counter(position_roles[pos])
        table.append([counts.get(r, 0) for r in all_roles])

    table_arr = np.array(table)
    col_sums = table_arr.sum(axis=0)
    nonzero = col_sums > 0
    table_filtered = table_arr[:, nonzero]

    chi2, p_value = 0.0, 1.0
    if table_filtered.shape[1] >= 2:
        try:
            chi2, p_value, _, _ = chi2_contingency(table_filtered)
        except ValueError:
            pass

    distributions: dict[str, dict] = {}
    for pos in ["a", "b", "c"]:
        c = Counter(position_roles[pos])
        total = sum(c.values())
        distributions[pos] = {r: round(c[r] / total, 4) for r in all_roles}

    return {
        "distributions": distributions,
        "chi2": round(float(chi2), 4),
        "p_value": float(p_value),
    }


# ============================================================
# OUTPUT ASSEMBLY
# ============================================================

def build_output(
    *,
    per_graph_records: list[dict],
    per_domain_records: list[dict],
    representative_ffls: list[dict],
    layer_results: dict,
    role_results: dict,
    cross_results: dict,
    weight_results: dict,
    baseline_results: dict,
    summary: dict,
    total_ffls: int,
    total_graphs: int,
    explanation_match_rate: float,
) -> dict:
    """Assemble method_out.json conforming to exp_gen_sol_out schema.

    Each example has predict_ffl_method and predict_baseline_random_triad fields.
    Produces 34 per-graph + 8 per-domain + 10 representative + 2 aggregate = 54+ examples.
    """
    examples: list[dict] = []

    # --- 34 per-graph examples ---
    for rec in per_graph_records:
        examples.append({
            "input": rec["prompt"],
            "output": json.dumps({
                "slug": rec["slug"],
                "n_nodes": rec["n_nodes"],
                "n_edges": rec["n_edges"],
            }),
            "predict_ffl_method": json.dumps({
                "n_ffls": rec["n_ffls"],
                "explanation_match_rate": rec["match_rate"],
                "layer_strict_order_frac": rec["ffl_layer"]["frac_strict_order_abc"],
                "coherent_frac": rec["ffl_weight"].get("coherent_fraction", 0),
                "semantic_chi2": rec["ffl_semantic"].get("chi2", 0),
                "semantic_p": rec["ffl_semantic"].get("p_value", 1),
            }),
            "predict_baseline_random_triad": json.dumps({
                "n_triads": rec["n_baseline_triads"],
                "layer_strict_order_frac": rec["baseline_layer"].get("frac_strict_order_abc", 0),
                "baseline_a_earliest": rec["baseline_layer"].get("frac_a_earliest", 0),
            }),
            "metadata_fold": rec["domain"],
            "metadata_slug": rec["slug"],
            "metadata_analysis_type": "per_graph",
        })

    # --- 8 per-domain aggregate examples ---
    for drec in per_domain_records:
        examples.append({
            "input": f"domain_aggregate:{drec['domain']}",
            "output": json.dumps({
                "domain": drec["domain"],
                "n_graphs": drec["n_graphs"],
                "total_ffls": drec["total_ffls"],
            }),
            "predict_ffl_method": json.dumps({
                "mean_ffls_per_graph": drec["mean_ffls"],
                "semantic_chi2": drec["semantic"].get("chi2", 0),
                "semantic_p": drec["semantic"].get("p_value", 1),
                "cramers_v": drec["semantic"].get("cramers_v", 0),
                "coherent_frac": drec["weight"].get("coherent_fraction", 0),
            }),
            "predict_baseline_random_triad": json.dumps({
                "baseline_strict_order_frac": drec["baseline_layer"].get("frac_strict_order_abc", 0),
            }),
            "metadata_fold": drec["domain"],
            "metadata_analysis_type": "per_domain_aggregate",
        })

    # --- 10 representative FFL examples ---
    for j, ffl_ex in enumerate(representative_ffls):
        examples.append({
            "input": f"representative_ffl_{j}",
            "output": json.dumps(ffl_ex),
            "predict_ffl_method": json.dumps({
                "a_role": ffl_ex["a"]["role"],
                "b_role": ffl_ex["b"]["role"],
                "c_role": ffl_ex["c"]["role"],
                "coherent": ffl_ex["coherent"],
            }),
            "predict_baseline_random_triad": json.dumps({
                "note": "no baseline equivalent for individual FFL instances",
            }),
            "metadata_fold": "representative",
            "metadata_analysis_type": "qualitative_example",
        })

    # --- Global aggregate: layer + semantic + edge weight ---
    examples.append({
        "input": "global_aggregate_layer_semantic",
        "output": json.dumps(layer_results),
        "predict_ffl_method": json.dumps({
            "chi2": role_results.get("chi2", 0),
            "p_value": role_results.get("p_value", 1),
            "cramers_v": role_results.get("cramers_v", 0),
            "coherent_frac": weight_results.get("coherent_fraction", 0),
            "cross_domain_cramers_v": cross_results.get("cross_domain_cramers_v", 0),
        }),
        "predict_baseline_random_triad": json.dumps(baseline_results.get("comparison", {})),
        "metadata_fold": "global",
        "metadata_analysis_type": "aggregate_statistics",
    })

    # --- Summary ---
    examples.append({
        "input": "summary_statistics",
        "output": json.dumps(summary),
        "predict_ffl_method": json.dumps(summary.get("key_findings", {})),
        "predict_baseline_random_triad": json.dumps({
            "baseline_strict_ordering_frac": summary.get("key_findings", {}).get("baseline_strict_ordering_frac", 0),
        }),
        "metadata_fold": "summary",
        "metadata_analysis_type": "confirmation_signals",
    })

    return {
        "metadata": {
            "method_name": "FFL Functional Characterization",
            "description": (
                "Enumerate all feed-forward loop (FFL / 030T motif) instances from "
                "34 pruned attribution graphs, join each node's position "
                "(A=regulator, B=intermediary, C=target) to its Neuronpedia semantic "
                "explanation, and test whether FFL positions carry consistent "
                "functional roles."
            ),
            "parameters": {
                "prune_percentile": PRUNE_PERCENTILE,
                "min_nodes": MIN_NODES,
                "random_seed": RANDOM_SEED,
                "n_baseline_samples_per_graph": N_BASELINE_SAMPLES,
            },
            "total_graphs": total_graphs,
            "total_ffls": total_ffls,
            "explanation_match_rate": round(explanation_match_rate, 4),
        },
        "datasets": [
            {
                "dataset": "ffl_functional_characterization",
                "examples": examples,
            }
        ],
    }


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main() -> None:
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("FFL Functional Characterization — Starting")
    logger.info(f"Workspace: {WORKSPACE}")
    logger.info(f"MAX_EXAMPLES={MAX_EXAMPLES}")
    logger.info("=" * 60)

    # ---- Phase 1: Load explanations ----
    t0 = time.time()
    expl_lookup = load_explanations()
    logger.info(f"Phase 1 done in {time.time()-t0:.1f}s")

    # ---- Phase 2: Load graphs ----
    t0 = time.time()
    graphs = load_all_graphs()
    logger.info(f"Phase 2 done in {time.time()-t0:.1f}s — {len(graphs)} graphs loaded")
    if not graphs:
        logger.error("No graphs loaded! Aborting.")
        sys.exit(1)

    # ---- Phase 3: Test FFL enumeration on first graph ----
    t0 = time.time()
    test_ffls = enumerate_ffls(graphs[0]["graph"])
    logger.info(f"Test: {len(test_ffls)} FFLs in first graph ({graphs[0]['slug']}) in {time.time()-t0:.2f}s")
    if len(test_ffls) == 0:
        logger.warning("No FFLs found in first graph — trying lower prune threshold")

    # Verify no duplicate FFL instances
    ffl_triples = set()
    for ffl in test_ffls:
        triple = (ffl["a"], ffl["b"], ffl["c"])
        assert triple not in ffl_triples, f"Duplicate FFL: {triple}"
        ffl_triples.add(triple)
    logger.info("FFL uniqueness verified on test graph")

    # ---- Phase 4+: Enumerate FFLs for ALL graphs ----
    t0 = time.time()
    all_ffls_by_domain: dict[str, list[dict]] = defaultdict(list)
    all_ffls_flat: list[dict] = []
    per_graph_records: list[dict] = []
    total_match_count = 0
    total_node_count = 0

    rng = random.Random(RANDOM_SEED)
    all_baseline_triads: list[dict] = []

    for i, g_info in enumerate(graphs):
        gt = time.time()
        g = g_info["graph"]
        slug = g_info["slug"]
        domain = g_info["domain"]

        # Enumerate FFLs
        ffls = enumerate_ffls(g)

        # Annotate with explanations
        annotated, match_rate = lookup_ffl_explanations(ffls, g, expl_lookup)

        n_matched = int(match_rate * len(annotated) * 3)
        total_match_count += n_matched
        total_node_count += len(annotated) * 3

        # Random triad baseline for this graph
        baseline_triads = sample_random_triads(g, N_BASELINE_SAMPLES, rng)
        all_baseline_triads.extend(baseline_triads)

        # Per-graph layer analysis
        pg_layer = analyze_layer_positions(annotated) if annotated else {"total_ffls": 0, "frac_strict_order_abc": 0, "frac_a_earliest": 0, "frac_c_latest": 0, "frac_same_layer_any": 0}
        pg_semantic = semantic_role_analysis(annotated) if annotated else {}
        pg_weight = edge_weight_analysis(annotated) if annotated else {}
        pg_bl_layer = baseline_layer_analysis(baseline_triads) if baseline_triads else {"total_triads": 0, "frac_strict_order_abc": 0, "frac_a_earliest": 0}

        all_ffls_by_domain[domain].extend(annotated)
        all_ffls_flat.extend(annotated)

        per_graph_records.append({
            "slug": slug,
            "domain": domain,
            "prompt": g_info["prompt"],
            "n_nodes": g_info["n_nodes"],
            "n_edges": g_info["n_edges"],
            "n_ffls": len(ffls),
            "match_rate": round(match_rate, 4),
            "n_baseline_triads": len(baseline_triads),
            "ffl_layer": pg_layer,
            "ffl_semantic": pg_semantic,
            "ffl_weight": pg_weight,
            "baseline_layer": pg_bl_layer,
        })

        elapsed = time.time() - gt
        logger.info(
            f"  [{i+1}/{len(graphs)}] {slug}: {len(ffls)} FFLs, "
            f"match={match_rate:.2%}, baseline={len(baseline_triads)}, "
            f"in {elapsed:.1f}s"
        )

    overall_match_rate = total_match_count / total_node_count if total_node_count > 0 else 0
    logger.info(
        f"Phase 3-4 done in {time.time()-t0:.1f}s — "
        f"Total FFLs: {len(all_ffls_flat)}, "
        f"Overall match rate: {overall_match_rate:.2%}"
    )

    # ---- Phase 5: Layer analysis ----
    t0 = time.time()
    layer_results = analyze_layer_positions(all_ffls_flat)
    logger.info(f"Phase 5 (layer analysis) done in {time.time()-t0:.1f}s")
    logger.info(
        f"  strict_order={layer_results.get('frac_strict_order_abc', 0):.2%}, "
        f"a_earliest={layer_results.get('frac_a_earliest', 0):.2%}, "
        f"c_latest={layer_results.get('frac_c_latest', 0):.2%}"
    )

    # ---- Phase 6: Semantic role analysis ----
    t0 = time.time()
    role_results = semantic_role_analysis(all_ffls_flat)
    logger.info(f"Phase 6 (semantic roles) done in {time.time()-t0:.1f}s")
    if role_results:
        logger.info(
            f"  chi2={role_results.get('chi2', 0):.2f}, "
            f"p={role_results.get('p_value', 1):.4e}, "
            f"Cramer's V={role_results.get('cramers_v', 0):.4f}"
        )

    # ---- Phase 7: Cross-domain analysis ----
    t0 = time.time()
    cross_results = cross_domain_analysis(dict(all_ffls_by_domain))
    logger.info(f"Phase 7 (cross-domain) done in {time.time()-t0:.1f}s")
    logger.info(
        f"  cross_domain Cramer's V={cross_results.get('cross_domain_cramers_v', 0):.4f}, "
        f"p={cross_results.get('cross_domain_p', 1):.4e}"
    )

    # ---- Phase 8: Edge weight analysis ----
    t0 = time.time()
    weight_results = edge_weight_analysis(all_ffls_flat)
    logger.info(f"Phase 8 (edge weights) done in {time.time()-t0:.1f}s")
    if weight_results:
        logger.info(
            f"  coherent={weight_results.get('coherent_fraction', 0):.2%}, "
            f"incoherent={weight_results.get('incoherent_fraction', 0):.2%}"
        )

    # ---- Phase 9: Representative examples ----
    t0 = time.time()
    representative_examples = select_representative_ffls(all_ffls_flat, top_k=10)
    logger.info(f"Phase 9 (examples) done in {time.time()-t0:.1f}s — {len(representative_examples)} selected")

    # ---- Phase 10: Baseline analysis + per-domain records ----
    t0 = time.time()
    baseline_layer = baseline_layer_analysis(all_baseline_triads)

    baseline_results = {
        "comparison": {
            "ffl_strict_order": layer_results.get("frac_strict_order_abc", 0),
            "baseline_strict_order": baseline_layer.get("frac_strict_order_abc", 0),
            "ffl_a_earliest": layer_results.get("frac_a_earliest", 0),
            "baseline_a_earliest": baseline_layer.get("frac_a_earliest", 0),
            "ffl_semantic_chi2": role_results.get("chi2", 0),
            "ffl_semantic_p": role_results.get("p_value", 1),
        },
    }

    # Build per-domain records for output
    per_domain_records: list[dict] = []
    for domain, dom_ffls in sorted(all_ffls_by_domain.items()):
        dom_graphs = [r for r in per_graph_records if r["domain"] == domain]
        dom_bl_triads: list[dict] = []
        for r in dom_graphs:
            dom_bl_triads.extend([])  # baseline triads already aggregated in per-graph
        # Compute domain-level baseline from per-graph records
        dom_bl_layer_vals = [r["baseline_layer"] for r in dom_graphs if r["baseline_layer"].get("total_triads", 0) > 0]
        avg_bl_strict = float(np.mean([v.get("frac_strict_order_abc", 0) for v in dom_bl_layer_vals])) if dom_bl_layer_vals else 0

        per_domain_records.append({
            "domain": domain,
            "n_graphs": len(dom_graphs),
            "total_ffls": sum(r["n_ffls"] for r in dom_graphs),
            "mean_ffls": round(float(np.mean([r["n_ffls"] for r in dom_graphs])), 1) if dom_graphs else 0,
            "semantic": semantic_role_analysis(dom_ffls) if len(dom_ffls) >= 10 else {},
            "weight": edge_weight_analysis(dom_ffls) if len(dom_ffls) >= 10 else {},
            "baseline_layer": {"frac_strict_order_abc": round(avg_bl_strict, 4)},
        })

    logger.info(f"Phase 10 (baseline + domain records) done in {time.time()-t0:.1f}s")
    logger.info(
        f"  Baseline triads: {len(all_baseline_triads)}, "
        f"strict_order={baseline_layer.get('frac_strict_order_abc', 0):.2%} "
        f"(FFL: {layer_results.get('frac_strict_order_abc', 0):.2%})"
    )

    # ---- Summary ----
    summary = {
        "total_graphs": len(graphs),
        "total_ffls": len(all_ffls_flat),
        "explanation_match_rate": round(overall_match_rate, 4),
        "key_findings": {
            "ffl_count_per_graph_mean": round(float(np.mean([s["n_ffls"] for s in per_graph_records])), 1),
            "ffl_count_per_graph_median": round(float(np.median([s["n_ffls"] for s in per_graph_records])), 1),
            "layer_strict_ordering_frac": layer_results.get("frac_strict_order_abc", 0),
            "semantic_chi2": role_results.get("chi2", 0),
            "semantic_p_value": role_results.get("p_value", 1),
            "semantic_cramers_v": role_results.get("cramers_v", 0),
            "cross_domain_cramers_v": cross_results.get("cross_domain_cramers_v", 0),
            "coherent_ffl_fraction": weight_results.get("coherent_fraction", 0),
            "baseline_strict_ordering_frac": baseline_layer.get("frac_strict_order_abc", 0),
        },
        "confirmation_signals": {
            "ffls_found": len(all_ffls_flat) > 0,
            "a_tends_earliest": layer_results.get("frac_a_earliest", 0) > 0.5,
            "semantic_positions_differ": role_results.get("p_value", 1) < 0.05,
            "ffl_more_ordered_than_baseline": (
                layer_results.get("frac_strict_order_abc", 0)
                > baseline_layer.get("frac_strict_order_abc", 0)
            ),
        },
    }

    # ---- Write output ----
    t0 = time.time()
    output = build_output(
        per_graph_records=per_graph_records,
        per_domain_records=per_domain_records,
        representative_ffls=representative_examples,
        layer_results=layer_results,
        role_results=role_results,
        cross_results=cross_results,
        weight_results=weight_results,
        baseline_results=baseline_results,
        summary=summary,
        total_ffls=len(all_ffls_flat),
        total_graphs=len(graphs),
        explanation_match_rate=overall_match_rate,
    )

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Output written to {OUTPUT_FILE} ({file_size_mb:.1f}MB) in {time.time()-t0:.1f}s")

    total_elapsed = time.time() - t_start
    logger.info(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)

    # Print key results
    logger.info(f"Total FFLs: {len(all_ffls_flat)}")
    logger.info(f"Mean FFLs/graph: {summary['key_findings']['ffl_count_per_graph_mean']}")
    logger.info(f"Layer strict ordering: {summary['key_findings']['layer_strict_ordering_frac']:.2%}")
    logger.info(f"Semantic chi2={summary['key_findings']['semantic_chi2']:.2f}, p={summary['key_findings']['semantic_p_value']:.4e}")
    logger.info(f"Coherent FFLs: {summary['key_findings']['coherent_ffl_fraction']:.2%}")
    logger.info(f"Cross-domain Cramer's V: {summary['key_findings']['cross_domain_cramers_v']:.4f}")


if __name__ == "__main__":
    main()
