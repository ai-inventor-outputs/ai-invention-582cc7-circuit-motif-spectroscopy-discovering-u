#!/usr/bin/env python3
"""Statistical Fortification of Motif Z-Scores across attribution graphs.

Six-phase analysis pipeline:
  A: 3-node Z-scores with configurable null models (default 50)
  B: 4-node Z-scores on qualifying graphs (heavy pruning, <=500 nodes)
  C: Benjamini-Hochberg FDR correction across all tests
  D: Pruning sensitivity sweep (multiple thresholds)
  E: Null model convergence diagnostics
  F: Layer-preserving null model validation

Output: method_out.json following exp_gen_sol_out schema.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import os
import math
import random
import time
import gc
import resource
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
import igraph

# ═══════════════════════════════════════════════════════════════════════════
# SETUP: Logging, Hardware Detection, Resource Limits
# ═══════════════════════════════════════════════════════════════════════════

WORKSPACE = Path(__file__).parent
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")


def _detect_cpus() -> int:
    """Detect actual CPU allocation (container-aware)."""
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
    """Read RAM limit from cgroup."""
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
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.7 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))

N_WORKERS = max(1, NUM_CPUS)

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, workers={N_WORKERS}")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(
    "/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability"
    "/3_invention_loop/iter_3/gen_art/data_id5_it3__opus/data_out"
)

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "200"))
N_NULLS_A = int(os.environ.get("N_NULLS_A", "50"))
N_NULLS_B = int(os.environ.get("N_NULLS_B", "20"))
N_NULLS_D = int(os.environ.get("N_NULLS_D", "15"))
N_NULLS_F = int(os.environ.get("N_NULLS_F", "15"))

BASELINE_NULLS = 20
PRUNING_PCT_PRIMARY = 75
PRUNING_PCT_4NODE = 99
PRUNING_THRESHOLDS_D = [60, 75, 90]
MAX_VCOUNT_4NODE = 500
N_GRAPHS_D_TARGET = 48
N_GRAPHS_E = 20
SWAP_FACTOR = 10

logger.info(
    f"Config: max_ex={MAX_EXAMPLES} nulls_A={N_NULLS_A} nulls_B={N_NULLS_B} "
    f"nulls_D={N_NULLS_D} nulls_F={N_NULLS_F}"
)

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_all_graphs(data_dir: Path, max_examples: int) -> list[dict]:
    """Load graphs from dependency data files."""
    all_records: list[dict] = []

    if max_examples <= 16:
        files = [data_dir / "mini_data_out.json"]
    else:
        files = sorted(
            data_dir.glob("full_data_out_*.json"),
            key=lambda f: int(f.stem.rsplit("_", 1)[-1]),
        )

    for fpath in files:
        logger.info(f"Loading {fpath.name} ...")
        raw = json.loads(fpath.read_text())
        for ex in raw["datasets"][0]["examples"]:
            try:
                graph_json = json.loads(ex["output"])
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON in example, skipping")
                continue
            all_records.append({
                "input": ex["input"],
                "slug": ex.get("metadata_slug", ""),
                "domain": ex.get("metadata_fold", ""),
                "correct": ex.get("metadata_model_correct", "unknown"),
                "difficulty": ex.get("metadata_difficulty", "unknown"),
                "graph_json": graph_json,
                "metadata": {k: v for k, v in ex.items()
                             if k.startswith("metadata_")},
            })
            if len(all_records) >= max_examples:
                break
        if len(all_records) >= max_examples:
            break

    domains = sorted({r["domain"] for r in all_records})
    logger.info(
        f"Loaded {len(all_records)} graphs across {len(domains)} domains: "
        f"{domains}"
    )
    return all_records


# ═══════════════════════════════════════════════════════════════════════════
# ISOCLASS MAPPING
# ═══════════════════════════════════════════════════════════════════════════


def build_isoclass_mappings() -> tuple[dict[int, dict], dict[int, dict]]:
    """Build isoclass-ID to structure mappings for 3/4-node DAG motifs."""

    # ---- 3-node ----
    dag_3: dict[int, dict] = {}
    for i in range(16):
        g = igraph.Graph.Isoclass(n=3, cls=i, directed=True)
        if not (g.is_dag() and g.is_connected(mode="weak")):
            continue
        edges = g.get_edgelist()
        ne = len(edges)
        if ne == 3:
            label = "030T"
        elif ne == 2:
            od = [g.degree(v, mode="out") for v in range(3)]
            id_ = [g.degree(v, mode="in") for v in range(3)]
            if max(od) == 2:
                label = "021D"
            elif max(id_) == 2:
                label = "021U"
            else:
                label = "021C"
        else:
            label = f"unk{ne}"
        dag_3[i] = {"label": label, "edges": edges, "n_edges": ne}

    if len(dag_3) != 4:
        raise ValueError(f"Expected 4 DAG 3-node types, got {len(dag_3)}")
    logger.info(
        "3-node DAG motifs: "
        + ", ".join(f"id={k}->{v['label']}" for k, v in sorted(dag_3.items()))
    )

    # ---- 4-node ----
    dag_4: dict[int, dict] = {}
    for i in range(218):
        g = igraph.Graph.Isoclass(n=4, cls=i, directed=True)
        if not (g.is_dag() and g.is_connected(mode="weak")):
            continue
        edges = tuple(sorted(g.get_edgelist()))
        dag_4[i] = {"edges": edges, "n_edges": len(edges)}

    if len(dag_4) != 24:
        raise ValueError(f"Expected 24 DAG 4-node types, got {len(dag_4)}")
    logger.info(f"4-node DAG motifs: {len(dag_4)} types identified")

    return dag_3, dag_4


# ═══════════════════════════════════════════════════════════════════════════
# CORE GRAPH FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def _layer_int(layer_str: str) -> int:
    """Convert layer string to int ('E' -> -1)."""
    if layer_str in ("E", "e"):
        return -1
    try:
        return int(layer_str)
    except (ValueError, TypeError):
        return -2


def parse_and_prune(
    graph_json: dict,
    weight_percentile: float = 75.0,
) -> tuple[igraph.Graph | None, dict]:
    """Parse graph JSON and prune by |weight| percentile."""
    nodes = graph_json["nodes"]
    links = graph_json.get("links", graph_json.get("edges", []))

    nid_to_idx = {n["node_id"]: i for i, n in enumerate(nodes)}
    layers = [_layer_int(n.get("layer", "0")) for n in nodes]

    valid_links: list[tuple[int, int, float]] = []
    abs_w: list[float] = []
    for lk in links:
        s, t = lk["source"], lk["target"]
        if s in nid_to_idx and t in nid_to_idx:
            w = float(lk.get("weight", 0.0))
            valid_links.append((nid_to_idx[s], nid_to_idx[t], w))
            abs_w.append(abs(w))

    if not abs_w:
        return None, {"error": "no_valid_links"}

    thresh = float(np.percentile(abs_w, weight_percentile))
    kept = [(s, t) for s, t, w in valid_links if abs(w) >= thresh]

    if len(kept) < 5:
        return None, {"error": "too_few_edges"}

    g = igraph.Graph(n=len(nodes), edges=kept, directed=True)
    g.vs["layer"] = layers
    g.simplify(multiple=True, loops=True, combine_edges="max")

    iso = [v.index for v in g.vs if g.degree(v) == 0]
    if iso:
        g.delete_vertices(iso)

    if g.vcount() < 10:
        return None, {"error": f"vcount={g.vcount()}"}
    if not g.is_dag():
        return None, {"error": "not_dag"}

    return g, {"n_nodes": g.vcount(), "n_edges": g.ecount()}


def generate_dag_null(
    edges: list[tuple[int, int]],
    n_verts: int,
    topo_ranks: dict[int, int],
    n_swap_factor: int = 10,
    rng: random.Random | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Goni Method 1 DAG-constrained edge swap. Returns (new_edges, acc_rate)."""
    if rng is None:
        rng = random.Random()
    el = list(edges)
    ne = len(el)
    if ne < 2:
        return el, 0.0

    eset = set(el)
    n_att = n_swap_factor * ne
    n_acc = 0

    for _ in range(n_att):
        i1 = rng.randrange(ne)
        i2 = rng.randrange(ne)
        if i1 == i2:
            continue
        u1, v1 = el[i1]
        u2, v2 = el[i2]
        if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
            continue
        e1n = (u1, v2)
        e2n = (u2, v1)
        if e1n in eset or e2n in eset:
            continue
        if topo_ranks[u1] >= topo_ranks[v2] or topo_ranks[u2] >= topo_ranks[v1]:
            continue
        eset.discard((u1, v1))
        eset.discard((u2, v2))
        eset.add(e1n)
        eset.add(e2n)
        el[i1] = e1n
        el[i2] = e2n
        n_acc += 1

    return el, n_acc / max(n_att, 1)


def generate_layer_preserving_null(
    edges: list[tuple[int, int]],
    vlayers: list[int],
    topo_ranks: dict[int, int],
    n_swap_factor: int = 10,
    rng: random.Random | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Layer-preserving DAG null: swaps only within same (src_layer, tgt_layer)."""
    if rng is None:
        rng = random.Random()

    groups: dict[tuple[int, int], list[int]] = {}
    for i, (u, v) in enumerate(edges):
        key = (vlayers[u], vlayers[v])
        groups.setdefault(key, []).append(i)

    el = list(edges)
    eset = set(el)
    ne = len(el)
    n_att = n_swap_factor * ne
    n_acc = 0

    swap_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not swap_groups:
        return el, 0.0

    gkeys = list(swap_groups.keys())
    gwts = [len(swap_groups[k]) for k in gkeys]
    tot = sum(gwts)

    for _ in range(n_att):
        r = rng.randrange(tot)
        c = 0
        gk = gkeys[0]
        for k, w in zip(gkeys, gwts):
            c += w
            if r < c:
                gk = k
                break
        gi = swap_groups[gk]
        if len(gi) < 2:
            continue
        i1, i2 = rng.sample(gi, 2)
        u1, v1 = el[i1]
        u2, v2 = el[i2]
        if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
            continue
        e1n = (u1, v2)
        e2n = (u2, v1)
        if e1n in eset or e2n in eset:
            continue
        if topo_ranks[u1] >= topo_ranks[v2] or topo_ranks[u2] >= topo_ranks[v1]:
            continue
        eset.discard((u1, v1))
        eset.discard((u2, v2))
        eset.add(e1n)
        eset.add(e2n)
        el[i1] = e1n
        el[i2] = e2n
        n_acc += 1

    return el, n_acc / max(n_att, 1)


# ═══════════════════════════════════════════════════════════════════════════
# WORKER FUNCTIONS  (module-level for ProcessPoolExecutor pickle)
# ═══════════════════════════════════════════════════════════════════════════


def _w_phase_a(args):
    """3-node Z-scores with N null models + baseline + convergence data."""
    (idx, slug, domain, correct, el, nv, vlayers,
     n_nulls, dag_ids, seed) = args
    try:
        g = igraph.Graph(n=nv, edges=el, directed=True)
        topo = g.topological_sorting()
        tr = {v: i for i, v in enumerate(topo)}
        rc_raw = g.motifs_randesu(size=3)
        rc = [0 if (x != x) else int(x) for x in rc_raw]
        del g

        ncm: list[list[int]] = []
        acc_rates: list[float] = []
        for j in range(n_nulls):
            ne_list, ar = generate_dag_null(
                el, nv, tr, SWAP_FACTOR, random.Random(seed + j + 1)
            )
            acc_rates.append(ar)
            gn = igraph.Graph(n=nv, edges=ne_list, directed=True)
            nc_raw = gn.motifs_randesu(size=3)
            ncm.append([0 if (x != x) else int(x) for x in nc_raw])
            del gn

        na = np.array(ncm, dtype=np.float64)
        mu = na.mean(axis=0)
        sd = na.std(axis=0, ddof=1)

        mr: dict[str, dict] = {}
        bl: dict[str, dict] = {}
        conv: dict[str, dict] = {}

        bn = min(BASELINE_NULLS, n_nulls)
        ba = na[:bn]
        bmu = ba.mean(axis=0)
        bsd = ba.std(axis=0, ddof=1)

        for mid in dag_ids:
            # Full Z-score
            if sd[mid] == 0:
                z = (0.0 if rc[mid] == mu[mid]
                     else 10.0 * np.sign(rc[mid] - mu[mid]))
            else:
                z = (rc[mid] - mu[mid]) / sd[mid]
            ep = float(np.mean(na[:, mid] >= rc[mid]))
            ep = max(ep, 1.0 / (n_nulls + 1))

            mr[str(mid)] = {
                "real_count": rc[mid],
                "null_mean": round(float(mu[mid]), 4),
                "null_std": round(float(sd[mid]), 4),
                "zscore": round(float(z), 4),
                "empirical_p": round(float(ep), 6),
            }

            # Baseline Z-score (first BASELINE_NULLS nulls)
            if bsd[mid] == 0:
                bz = (0.0 if rc[mid] == bmu[mid]
                      else 10.0 * np.sign(rc[mid] - bmu[mid]))
            else:
                bz = (rc[mid] - bmu[mid]) / bsd[mid]
            bl[str(mid)] = {"zscore": round(float(bz), 4)}

            # Convergence data (per-null counts for this motif)
            conv[str(mid)] = {
                "real_count": rc[mid],
                "null_counts": na[:, mid].tolist(),
            }

        return {
            "idx": idx, "slug": slug, "domain": domain, "correct": correct,
            "n_nodes": nv, "n_edges": len(el),
            "motif_results": mr,
            "baseline_results": bl,
            "convergence_data": conv,
            "mean_acceptance_rate": round(float(np.mean(acc_rates)), 4),
            "n_nulls": n_nulls,
            "error": None,
        }
    except Exception as e:
        return {"idx": idx, "slug": slug, "domain": domain,
                "correct": correct, "error": str(e)}


def _w_phase_b(args):
    """4-node Z-scores."""
    idx, slug, domain, el, nv, n_nulls, dag_ids, seed = args
    try:
        g = igraph.Graph(n=nv, edges=el, directed=True)
        topo = g.topological_sorting()
        tr = {v: i for i, v in enumerate(topo)}
        rc_raw = g.motifs_randesu(size=4)
        rc = [0 if (x != x) else int(x) for x in rc_raw]
        del g

        ncm: list[list[int]] = []
        acc_rates: list[float] = []
        for j in range(n_nulls):
            ne_list, ar = generate_dag_null(
                el, nv, tr, SWAP_FACTOR, random.Random(seed + j + 1)
            )
            acc_rates.append(ar)
            gn = igraph.Graph(n=nv, edges=ne_list, directed=True)
            nc_raw = gn.motifs_randesu(size=4)
            ncm.append([0 if (x != x) else int(x) for x in nc_raw])
            del gn

        na = np.array(ncm, dtype=np.float64)
        mu = na.mean(axis=0)
        sd = na.std(axis=0, ddof=1)

        mr: dict[str, dict] = {}
        for mid in dag_ids:
            if sd[mid] == 0:
                z = (0.0 if rc[mid] == mu[mid]
                     else 10.0 * np.sign(rc[mid] - mu[mid]))
            else:
                z = (rc[mid] - mu[mid]) / sd[mid]
            ep = float(np.mean(na[:, mid] >= rc[mid]))
            ep = max(ep, 1.0 / (n_nulls + 1))
            mr[str(mid)] = {
                "real_count": rc[mid],
                "null_mean": round(float(mu[mid]), 4),
                "null_std": round(float(sd[mid]), 4),
                "zscore": round(float(z), 4),
                "empirical_p": round(float(ep), 6),
            }

        return {
            "idx": idx, "slug": slug, "domain": domain,
            "n_nodes": nv, "n_edges": len(el),
            "motif_results": mr,
            "mean_acceptance_rate": round(float(np.mean(acc_rates)), 4),
            "n_nulls": n_nulls,
            "error": None,
        }
    except Exception as e:
        return {"idx": idx, "slug": slug, "domain": domain, "error": str(e)}


def _w_phase_d(args):
    """Pruning sensitivity: 3-node Z-scores at one threshold."""
    idx, slug, domain, graph_json, thresh, n_nulls, dag_ids, seed = args
    try:
        g, info = parse_and_prune(graph_json, weight_percentile=thresh)
        if g is None:
            return {"idx": idx, "slug": slug, "threshold": thresh,
                    "error": info.get("error", "prune_fail")}

        el = g.get_edgelist()
        nv = g.vcount()
        topo = g.topological_sorting()
        tr = {v: i for i, v in enumerate(topo)}
        rc_raw = g.motifs_randesu(size=3)
        rc = [0 if (x != x) else int(x) for x in rc_raw]
        del g

        ncm: list[list[int]] = []
        for j in range(n_nulls):
            ne_list, _ = generate_dag_null(
                el, nv, tr, SWAP_FACTOR, random.Random(seed + j + 1)
            )
            gn = igraph.Graph(n=nv, edges=ne_list, directed=True)
            nc_raw = gn.motifs_randesu(size=3)
            ncm.append([0 if (x != x) else int(x) for x in nc_raw])
            del gn

        na = np.array(ncm, dtype=np.float64)
        mu = na.mean(axis=0)
        sd = na.std(axis=0, ddof=1)

        mr: dict[str, dict] = {}
        for mid in dag_ids:
            if sd[mid] == 0:
                z = (0.0 if rc[mid] == mu[mid]
                     else 10.0 * np.sign(rc[mid] - mu[mid]))
            else:
                z = (rc[mid] - mu[mid]) / sd[mid]
            mr[str(mid)] = {"zscore": round(float(z), 4)}

        return {"idx": idx, "slug": slug, "domain": domain,
                "threshold": thresh, "n_nodes": nv, "n_edges": len(el),
                "motif_results": mr, "error": None}
    except Exception as e:
        return {"idx": idx, "slug": slug, "threshold": thresh,
                "error": str(e)}


def _w_phase_f(args):
    """Layer-preserving 3-node Z-scores."""
    idx, slug, domain, el, nv, vlayers, n_nulls, dag_ids, seed = args
    try:
        g = igraph.Graph(n=nv, edges=el, directed=True)
        topo = g.topological_sorting()
        tr = {v: i for i, v in enumerate(topo)}
        rc_raw = g.motifs_randesu(size=3)
        rc = [0 if (x != x) else int(x) for x in rc_raw]
        del g

        ncm: list[list[int]] = []
        acc_rates: list[float] = []
        for j in range(n_nulls):
            ne_list, ar = generate_layer_preserving_null(
                el, vlayers, tr, SWAP_FACTOR, random.Random(seed + j + 1)
            )
            acc_rates.append(ar)
            gn = igraph.Graph(n=nv, edges=ne_list, directed=True)
            nc_raw = gn.motifs_randesu(size=3)
            ncm.append([0 if (x != x) else int(x) for x in nc_raw])
            del gn

        na = np.array(ncm, dtype=np.float64)
        mu = na.mean(axis=0)
        sd = na.std(axis=0, ddof=1)

        mr: dict[str, dict] = {}
        for mid in dag_ids:
            if sd[mid] == 0:
                z = (0.0 if rc[mid] == mu[mid]
                     else 10.0 * np.sign(rc[mid] - mu[mid]))
            else:
                z = (rc[mid] - mu[mid]) / sd[mid]
            ep = float(np.mean(na[:, mid] >= rc[mid]))
            ep = max(ep, 1.0 / (n_nulls + 1))
            mr[str(mid)] = {
                "zscore": round(float(z), 4),
                "empirical_p": round(float(ep), 6),
            }

        return {
            "idx": idx, "slug": slug, "domain": domain,
            "motif_results": mr,
            "mean_acceptance_rate": round(float(np.mean(acc_rates)), 4),
            "error": None,
        }
    except Exception as e:
        return {"idx": idx, "slug": slug, "domain": domain, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# PHASE RUNNERS
# ═══════════════════════════════════════════════════════════════════════════


def _run_pool(worker_fn, tasks, label, log_every_frac=0.1):
    """Generic parallel runner with progress logging."""
    t0 = time.time()
    results: list[dict] = []
    log_interval = max(1, int(len(tasks) * log_every_frac))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(worker_fn, t): i for i, t in enumerate(tasks)}
        done = 0
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                logger.exception(f"  {label} worker exception")
                results.append({"idx": futs[f], "error": str(e)})
            done += 1
            if done % log_interval == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.01)
                eta = (len(tasks) - done) / max(rate, 0.001)
                logger.info(
                    f"  {label}: {done}/{len(tasks)}  "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )

    ok = sum(1 for r in results if not r.get("error"))
    logger.info(
        f"  {label} done: {ok} ok, {len(results)-ok} err  "
        f"({time.time()-t0:.0f}s total)"
    )
    return results


def run_phase_a(pruned: list[dict], dag3: dict) -> list[dict]:
    """Phase A: 3-node Z-scores with N_NULLS_A null models per graph."""
    logger.info(f"{'='*60}")
    logger.info(f"PHASE A  3-node x {N_NULLS_A} nulls x {len(pruned)} graphs")
    logger.info(f"{'='*60}")

    ids3 = list(dag3.keys())
    tasks = [
        (pg["idx"], pg["slug"], pg["domain"], pg["correct"],
         pg["edge_list"], pg["nv"], pg["vlayers"],
         N_NULLS_A, ids3, 42 + pg["idx"] * 1000)
        for pg in pruned
    ]

    results = _run_pool(_w_phase_a, tasks, "A")
    results.sort(key=lambda r: r.get("idx", -1))
    return results


def run_phase_b(all_g: list[dict], dag4: dict) -> list[dict]:
    """Phase B: 4-node Z-scores on qualifying graphs."""
    logger.info(f"{'='*60}")
    logger.info(f"PHASE B  4-node x {N_NULLS_B} nulls")
    logger.info(f"{'='*60}")

    ids4 = list(dag4.keys())

    # First try 99th percentile
    qualifying: list[dict] = []
    for i, rec in enumerate(all_g):
        g, info = parse_and_prune(
            rec["graph_json"], weight_percentile=PRUNING_PCT_4NODE
        )
        if g is not None and g.vcount() <= MAX_VCOUNT_4NODE:
            qualifying.append({
                "idx": i, "slug": rec["slug"], "domain": rec["domain"],
                "el": g.get_edgelist(), "nv": g.vcount(),
            })

    logger.info(
        f"  B qualifying: {len(qualifying)} graphs "
        f"(vcount<={MAX_VCOUNT_4NODE} at {PRUNING_PCT_4NODE}th pct)"
    )

    # Fallback: try higher pruning if too few qualify (relative threshold)
    min_qualifying = min(80, int(len(all_g) * 0.4))
    if len(qualifying) < min_qualifying:
        for fallback_pct in [99.5, 99.9]:
            logger.info(f"  Trying {fallback_pct}th percentile fallback ...")
            qualifying = []
            for i, rec in enumerate(all_g):
                g, info = parse_and_prune(
                    rec["graph_json"], weight_percentile=fallback_pct
                )
                if g is not None and g.vcount() <= MAX_VCOUNT_4NODE:
                    qualifying.append({
                        "idx": i, "slug": rec["slug"],
                        "domain": rec["domain"],
                        "el": g.get_edgelist(), "nv": g.vcount(),
                    })
            logger.info(
                f"  B qualifying ({fallback_pct}th pct): "
                f"{len(qualifying)} graphs"
            )
            if len(qualifying) >= min_qualifying:
                break

    if not qualifying:
        logger.warning("  No qualifying graphs for 4-node analysis")
        return []

    tasks = [
        (q["idx"], q["slug"], q["domain"],
         q["el"], q["nv"], N_NULLS_B, ids4,
         100000 + q["idx"] * 1000)
        for q in qualifying
    ]

    results = _run_pool(_w_phase_b, tasks, "B", log_every_frac=0.2)
    results.sort(key=lambda r: r.get("idx", -1))
    return results


def run_phase_c(
    res_a: list[dict], res_b: list[dict],
    dag3: dict, dag4: dict,
) -> dict:
    """Phase C: BH-FDR multiple testing correction."""
    logger.info(f"{'='*60}")
    logger.info("PHASE C  BH-FDR correction")
    logger.info(f"{'='*60}")

    va = [r for r in res_a if not r.get("error")]
    vb = [r for r in res_b if not r.get("error")]

    # Collect 3-node p-values
    p3: list[float] = []
    lab3: list[tuple[int, int, str]] = []
    for r in va:
        for mk, md in r["motif_results"].items():
            p3.append(md["empirical_p"])
            lab3.append((int(mk), r["idx"], r["domain"]))

    # Collect 4-node p-values
    p4: list[float] = []
    lab4: list[tuple[int, int, str]] = []
    for r in vb:
        for mk, md in r["motif_results"].items():
            p4.append(md["empirical_p"])
            lab4.append((int(mk), r["idx"], r["domain"]))

    out: dict = {"method": "Benjamini-Hochberg", "alpha": 0.05}

    # 3-node FDR
    if p3:
        pa = np.array(p3)
        rej, padj, _, _ = multipletests(pa, alpha=0.05, method="fdr_bh")
        npre = int((pa < 0.05).sum())
        npost = int(rej.sum())

        # Per-motif universality
        mds_pre: dict[str, dict[str, list[bool]]] = {}
        mds_post: dict[str, dict[str, list[bool]]] = {}
        for i, (mid, gidx, dom) in enumerate(lab3):
            mk = str(mid)
            mds_pre.setdefault(mk, {}).setdefault(dom, []).append(
                bool(pa[i] < 0.05)
            )
            mds_post.setdefault(mk, {}).setdefault(dom, []).append(
                bool(rej[i])
            )

        univ_pre: dict[str, int] = {}
        univ_post: dict[str, int] = {}
        for mk in mds_pre:
            univ_pre[mk] = sum(
                1 for sigs in mds_pre[mk].values()
                if sum(sigs) > len(sigs) / 2
            )
            univ_post[mk] = sum(
                1 for sigs in mds_post[mk].values()
                if sum(sigs) > len(sigs) / 2
            )

        out["3node"] = {
            "n_tests": len(p3),
            "n_rejected_pre_fdr": npre,
            "n_rejected_post_fdr": npost,
            "per_motif_universality_pre_fdr": univ_pre,
            "per_motif_universality_post_fdr": univ_post,
        }
        logger.info(
            f"  3-node: {npre} -> {npost} sig (of {len(p3)} tests)"
        )

    # 4-node FDR
    if p4:
        pa = np.array(p4)
        rej, _, _, _ = multipletests(pa, alpha=0.05, method="fdr_bh")
        out["4node"] = {
            "n_tests": len(p4),
            "n_rejected_pre_fdr": int((pa < 0.05).sum()),
            "n_rejected_post_fdr": int(rej.sum()),
        }
        logger.info(
            f"  4-node: {int((pa<0.05).sum())} -> {int(rej.sum())} "
            f"sig (of {len(p4)})"
        )

    # Combined FDR
    all_p = p3 + p4
    if all_p:
        pa = np.array(all_p)
        rej, _, _, _ = multipletests(pa, alpha=0.05, method="fdr_bh")
        out["combined"] = {
            "n_tests": len(all_p),
            "n_rejected_pre_fdr": int((pa < 0.05).sum()),
            "n_rejected_post_fdr": int(rej.sum()),
        }

    logger.info("Phase C done")
    return out


def run_phase_d(all_g: list[dict], dag3: dict) -> dict:
    """Phase D: Pruning sensitivity sweep."""
    logger.info(f"{'='*60}")
    logger.info(
        f"PHASE D  Pruning sweep: {len(PRUNING_THRESHOLDS_D)} thresholds "
        f"x {N_NULLS_D} nulls"
    )
    logger.info(f"{'='*60}")

    ids3 = list(dag3.keys())

    # Stratified selection
    dg: dict[str, list[int]] = {}
    for i, r in enumerate(all_g):
        dg.setdefault(r["domain"], []).append(i)

    n_per = max(1, N_GRAPHS_D_TARGET // max(len(dg), 1))
    sel: list[int] = []
    for d in sorted(dg):
        inds = sorted(dg[d])
        step = max(1, len(inds) // n_per)
        for j in range(0, len(inds), step):
            if len(sel) < N_GRAPHS_D_TARGET:
                sel.append(inds[j])

    logger.info(f"  D selected: {len(sel)} graphs")

    tasks = []
    for gi in sel:
        r = all_g[gi]
        for th in PRUNING_THRESHOLDS_D:
            tasks.append((
                gi, r["slug"], r["domain"], r["graph_json"],
                th, N_NULLS_D, ids3, 200000 + gi * 100 + th,
            ))

    results = _run_pool(_w_phase_d, tasks, "D", log_every_frac=0.2)

    # Stability analysis
    stab = _pruning_stability(results, dag3)
    return {
        "per_result": results, "stability": stab,
        "selected": sel, "thresholds": PRUNING_THRESHOLDS_D,
    }


def _pruning_stability(results: list[dict], dag3: dict) -> dict:
    """Compute Spearman correlation of Z-scores across pruning thresholds."""
    valid = [r for r in results if not r.get("error")]
    gd: dict[int, dict[int, dict]] = {}
    for r in valid:
        gd.setdefault(r["idx"], {})[r["threshold"]] = r["motif_results"]

    thresholds = sorted({r["threshold"] for r in valid})
    stab: dict[str, dict] = {}

    for mid, info in dag3.items():
        mk = str(mid)
        zbt: dict[int, list[float]] = {t: [] for t in thresholds}
        for gi in sorted(gd):
            for t in thresholds:
                z = gd[gi].get(t, {}).get(mk, {}).get("zscore", float("nan"))
                zbt[t].append(z)

        rho_mat: list[list[float | None]] = []
        for t1 in thresholds:
            row: list[float | None] = []
            for t2 in thresholds:
                v1 = np.array(zbt[t1])
                v2 = np.array(zbt[t2])
                m = ~(np.isnan(v1) | np.isnan(v2))
                if m.sum() >= 3:
                    try:
                        rho, _ = scipy_stats.spearmanr(v1[m], v2[m])
                        row.append(
                            round(float(rho), 4)
                            if not np.isnan(rho) else None
                        )
                    except Exception:
                        row.append(None)
                else:
                    row.append(None)
            rho_mat.append(row)

        stab[mk] = {
            "label": info["label"],
            "spearman_rho_matrix": rho_mat,
            "thresholds": thresholds,
        }
    return stab


def run_phase_e(res_a: list[dict], dag3: dict) -> dict:
    """Phase E: Convergence diagnostics using Phase A stored null counts."""
    logger.info(f"{'='*60}")
    logger.info("PHASE E  Convergence diagnostics")
    logger.info(f"{'='*60}")

    valid = [
        r for r in res_a
        if not r.get("error") and "convergence_data" in r
    ]
    if not valid:
        logger.warning("  No valid Phase A results for convergence")
        return {"error": "no_data"}

    step = max(1, len(valid) // N_GRAPHS_E)
    sel = valid[::step][:N_GRAPHS_E]

    cps = [cp for cp in [5, 10, 20, 30, 40, 50] if cp <= N_NULLS_A]
    if not cps:
        cps = [N_NULLS_A]

    per_graph: list[dict] = []
    all_conv_ns: list[int] = []

    for r in sel:
        gc: dict = {"idx": r["idx"], "slug": r["slug"]}
        for mk, cd in r["convergence_data"].items():
            rc = cd["real_count"]
            nc = np.array(cd["null_counts"])

            z_cp: list[float] = []
            for cp in cps:
                sub = nc[:cp]
                m, s = float(sub.mean()), float(sub.std(ddof=1))
                if s == 0:
                    z = 0.0 if rc == m else 10.0 * np.sign(rc - m)
                else:
                    z = (rc - m) / s
                z_cp.append(round(float(z), 4))

            gc[f"motif_{mk}_z_at_checkpoints"] = z_cp

            # Relative error vs final checkpoint
            fz = z_cp[-1]
            rel_errs = [
                round(abs(z - fz) / max(abs(fz), 0.1), 4) for z in z_cp
            ]
            gc[f"motif_{mk}_rel_errors"] = rel_errs

            # First checkpoint where error < 5%
            for i, e in enumerate(rel_errs):
                if e < 0.05:
                    all_conv_ns.append(cps[i])
                    break

        per_graph.append(gc)

    summary: dict = {
        "n_graphs": len(sel),
        "checkpoints": cps,
        "n_nulls_total": N_NULLS_A,
    }
    if all_conv_ns:
        summary["median_convergence_n"] = float(np.median(all_conv_ns))
        for cp in cps:
            summary[f"frac_converged_by_{cp}"] = round(
                float(np.mean(np.array(all_conv_ns) <= cp)), 3
            )

    logger.info(
        f"  Convergence: median N="
        f"{summary.get('median_convergence_n', 'N/A')}"
    )
    logger.info("Phase E done")
    return {"summary": summary, "per_graph": per_graph}


def run_phase_f(
    pruned: list[dict], dag3: dict, res_a: list[dict],
) -> dict:
    """Phase F: Layer-preserving null model validation."""
    logger.info(f"{'='*60}")
    logger.info(
        f"PHASE F  Layer-preserving x {N_NULLS_F} nulls "
        f"x {len(pruned)} graphs"
    )
    logger.info(f"{'='*60}")

    ids3 = list(dag3.keys())
    tasks = [
        (pg["idx"], pg["slug"], pg["domain"],
         pg["edge_list"], pg["nv"], pg["vlayers"],
         N_NULLS_F, ids3, 300000 + pg["idx"] * 1000)
        for pg in pruned
    ]

    results = _run_pool(_w_phase_f, tasks, "F")
    results.sort(key=lambda r: r.get("idx", -1))

    comp = _compare_dp_lp(results, res_a, dag3)
    return {"results": results, "comparison": comp}


def _compare_dp_lp(
    res_f: list[dict], res_a: list[dict], dag3: dict,
) -> dict:
    """Compare degree-preserving (A) vs layer-preserving (F) Z-scores."""
    vf = {r["idx"]: r for r in res_f if not r.get("error")}
    va = {r["idx"]: r for r in res_a if not r.get("error")}
    common = sorted(set(vf) & set(va))

    comp: dict[str, dict] = {}
    for mid, info in dag3.items():
        mk = str(mid)
        zdp: list[float] = []
        zlp: list[float] = []
        for gi in common:
            if (mk in va[gi].get("motif_results", {})
                    and mk in vf[gi].get("motif_results", {})):
                zdp.append(va[gi]["motif_results"][mk]["zscore"])
                zlp.append(vf[gi]["motif_results"][mk]["zscore"])

        if len(zdp) >= 5:
            adp = np.array(zdp)
            alp = np.array(zlp)
            try:
                _, wp = scipy_stats.wilcoxon(adp, alp)
            except ValueError:
                wp = float("nan")
            comp[mk] = {
                "label": info["label"],
                "n_graphs": len(zdp),
                "mean_z_dp": round(float(adp.mean()), 4),
                "mean_z_lp": round(float(alp.mean()), 4),
                "mean_diff": round(float((adp - alp).mean()), 4),
                "median_diff": round(float(np.median(adp - alp)), 4),
                "wilcoxon_p": (round(float(wp), 6)
                               if not np.isnan(wp) else None),
                "frac_lp_z_gt_2": round(float((alp > 2.0).mean()), 4),
            }
    return comp


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════


def assemble_output(
    all_g: list[dict],
    res_a: list[dict], res_b: list[dict],
    res_c: dict, res_d: dict, res_e: dict, res_f: dict,
    dag3: dict, dag4: dict,
) -> dict:
    """Assemble method_out.json following exp_gen_sol_out schema."""

    a_idx = {r["idx"]: r for r in res_a}
    b_idx = {r["idx"]: r for r in res_b}
    f_results = res_f.get("results", []) if isinstance(res_f, dict) else []
    f_idx = {r["idx"]: r for r in f_results}

    va = [r for r in res_a if not r.get("error")]

    # Per-motif 3-node summary
    ms3: dict[str, dict] = {}
    for mid, info in dag3.items():
        mk = str(mid)
        zs: list[float] = []
        dz: dict[str, list[float]] = {}
        for r in va:
            if mk in r.get("motif_results", {}):
                z = r["motif_results"][mk]["zscore"]
                zs.append(z)
                dz.setdefault(r["domain"], []).append(z)
        if zs:
            pdm = {
                d: round(float(np.median(v)), 4) for d, v in dz.items()
            }
            zarr = np.array(zs)
            ms3[mk] = {
                "label": info["label"],
                "median_z": round(float(np.median(zarr)), 4),
                "mean_z": round(float(np.mean(zarr)), 4),
                "std_z": round(float(np.std(zarr)), 4),
                "frac_z_gt_2": round(float((zarr > 2.0).mean()), 4),
                "per_domain_median_z": pdm,
                "n_domains_median_gt_2": sum(
                    1 for v in pdm.values() if v > 2.0
                ),
            }

    # Build examples
    examples: list[dict] = []
    for i, rec in enumerate(all_g):
        pg: dict = {}

        # Phase A
        if i in a_idx and not a_idx[i].get("error"):
            ra = a_idx[i]
            pg["phase_a"] = {
                "n_nodes": ra["n_nodes"],
                "n_edges": ra["n_edges"],
                "n_nulls": ra["n_nulls"],
                "acceptance_rate": ra["mean_acceptance_rate"],
                "zscores": {
                    k: v["zscore"]
                    for k, v in ra["motif_results"].items()
                },
                "p_values": {
                    k: v["empirical_p"]
                    for k, v in ra["motif_results"].items()
                },
                "real_counts": {
                    k: v["real_count"]
                    for k, v in ra["motif_results"].items()
                },
            }

        # Phase B
        if i in b_idx and not b_idx[i].get("error"):
            rb = b_idx[i]
            pg["phase_b"] = {
                "n_nodes": rb["n_nodes"],
                "n_edges": rb["n_edges"],
                "zscores": {
                    k: v["zscore"]
                    for k, v in rb["motif_results"].items()
                },
            }

        # Phase F
        if i in f_idx and not f_idx[i].get("error"):
            rf = f_idx[i]
            pg["phase_f"] = {
                "zscores_lp": {
                    k: v["zscore"]
                    for k, v in rf["motif_results"].items()
                },
                "acceptance_rate": rf.get("mean_acceptance_rate", 0),
            }

        # Baseline
        bl: dict = {}
        if i in a_idx and not a_idx[i].get("error"):
            bl = {
                k: v["zscore"]
                for k, v in a_idx[i].get("baseline_results", {}).items()
            }

        examples.append({
            "input": rec["input"],
            "output": json.dumps(pg),
            "predict_our_method": json.dumps({
                "zscores": pg.get("phase_a", {}).get("zscores", {}),
                "fdr_applied": True,
                "layer_validated": "phase_f" in pg,
            }),
            "predict_baseline": json.dumps({
                "zscores": bl,
                "n_nulls": BASELINE_NULLS,
            }),
            **rec["metadata"],
        })

    # Build metadata
    n_domains = len(set(r["domain"] for r in va)) if va else 0
    meta = {
        "title": "Statistical Fortification of Motif Z-Scores",
        "summary": (
            f"Analyzed {len(va)} graphs across {n_domains} domains. "
            f"Phase A: {N_NULLS_A} null models/graph (3-node). "
            f"Phase B: {N_NULLS_B} nulls (4-node). "
            f"BH-FDR correction applied. "
            f"Pruning sensitivity and layer-preserving validation completed."
        ),
        "phase_a": {
            "n_graphs": len(va),
            "n_nulls": N_NULLS_A,
            "prune_pct": PRUNING_PCT_PRIMARY,
            "per_motif": ms3,
        },
        "phase_b": {
            "n_graphs": len([r for r in res_b if not r.get("error")]),
            "n_nulls": N_NULLS_B,
            "prune_pct": PRUNING_PCT_4NODE,
        },
        "phase_c": res_c,
        "phase_d": {
            "thresholds": PRUNING_THRESHOLDS_D,
            "stability": (
                res_d.get("stability", {})
                if isinstance(res_d, dict) else {}
            ),
        },
        "phase_e": res_e,
        "phase_f": (
            res_f.get("comparison", {})
            if isinstance(res_f, dict) else {}
        ),
    }

    return {
        "metadata": meta,
        "datasets": [{
            "dataset": "neuronpedia_attribution_graphs_v3",
            "examples": examples,
        }],
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


@logger.catch
def main():
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading data ...")
    all_g = load_all_graphs(DATA_DIR, MAX_EXAMPLES)

    # ── Build isoclass mappings ───────────────────────────────────────
    dag3, dag4 = build_isoclass_mappings()

    # ── Pre-prune at primary threshold ────────────────────────────────
    logger.info(f"Pre-pruning at {PRUNING_PCT_PRIMARY}th percentile ...")
    pruned: list[dict] = []
    for i, rec in enumerate(all_g):
        g, info = parse_and_prune(rec["graph_json"], PRUNING_PCT_PRIMARY)
        if g is None:
            logger.warning(f"  Graph {i} ({rec['slug']}): {info}")
            continue
        pruned.append({
            "idx": i,
            "slug": rec["slug"],
            "domain": rec["domain"],
            "correct": rec["correct"],
            "edge_list": g.get_edgelist(),
            "nv": g.vcount(),
            "vlayers": list(g.vs["layer"]),
        })
    logger.info(f"Pruned: {len(pruned)}/{len(all_g)} valid")

    # Log graph size statistics
    if pruned:
        nvs = [p["nv"] for p in pruned]
        nes = [len(p["edge_list"]) for p in pruned]
        logger.info(
            f"  Node counts: min={min(nvs)} med={int(np.median(nvs))} "
            f"max={max(nvs)}"
        )
        logger.info(
            f"  Edge counts: min={min(nes)} med={int(np.median(nes))} "
            f"max={max(nes)}"
        )

    # ── Phase A ───────────────────────────────────────────────────────
    res_a = run_phase_a(pruned, dag3)
    t_a = time.time() - t0
    logger.info(f"Cumulative time after A: {t_a:.0f}s ({t_a/60:.1f} min)")

    # Free graph_json memory - no longer needed for A/F
    gc.collect()

    # ── Phase B ───────────────────────────────────────────────────────
    res_b = run_phase_b(all_g, dag4)
    t_b = time.time() - t0
    logger.info(f"Cumulative time after B: {t_b:.0f}s ({t_b/60:.1f} min)")

    # ── Phase C (fast, in-process) ────────────────────────────────────
    res_c = run_phase_c(res_a, res_b, dag3, dag4)

    # ── Phase D ───────────────────────────────────────────────────────
    res_d = run_phase_d(all_g, dag3)
    t_d = time.time() - t0
    logger.info(f"Cumulative time after D: {t_d:.0f}s ({t_d/60:.1f} min)")

    # ── Phase E (fast, in-process) ────────────────────────────────────
    res_e = run_phase_e(res_a, dag3)

    # ── Phase F ───────────────────────────────────────────────────────
    res_f = run_phase_f(pruned, dag3, res_a)
    t_f = time.time() - t0
    logger.info(f"Cumulative time after F: {t_f:.0f}s ({t_f/60:.1f} min)")

    # ── Assemble output ───────────────────────────────────────────────
    logger.info("Assembling output ...")
    output = assemble_output(
        all_g, res_a, res_b, res_c, res_d, res_e, res_f, dag3, dag4,
    )

    out_path = WORKSPACE / "method_out.json"
    out_text = json.dumps(output, indent=2)
    out_path.write_text(out_text)
    sz = len(out_text) / 1e6
    logger.info(f"Output: {out_path} ({sz:.1f} MB)")

    tt = time.time() - t0
    logger.info(f"TOTAL: {tt:.0f}s ({tt/60:.1f} min)")


if __name__ == "__main__":
    main()
