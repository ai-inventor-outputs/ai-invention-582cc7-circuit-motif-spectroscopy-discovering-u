#!/usr/bin/env python3
"""Hypothesis Evidence Scorecard: Synthesizing Iterations 1-3 Experimental Results.

Comprehensive evidence evaluation of 5 hypothesis claims against 3 iteration-3 experiments:
  - exp_id1_it3: 4-node motif census on 174 graphs
  - exp_id2_it3: Failure prediction on 140 graphs
  - exp_id3_it3: FFL characterization on 34 graphs

Scores each claim on a 5-level scale with quantitative evidence, assesses
methodological validity, recommends paper narrative, produces prioritized gap list.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import os
import math
import resource
import psutil

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware Detection ───────────────────────────────────────────────────────

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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

# Memory limit — lightweight metadata extraction, 4 GB is generous
RAM_BUDGET = int(4 * 1024**3)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET / 1e9:.1f}GB > available {_avail / 1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget {RAM_BUDGET / 1e9:.1f} GB")

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
_BASE = Path("/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability"
             "/3_invention_loop/iter_3/gen_art")
EXP1_DIR = _BASE / "exp_id1_it3__opus"
EXP2_DIR = _BASE / "exp_id2_it3__opus"
EXP3_DIR = _BASE / "exp_id3_it3__opus"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_experiment(exp_dir: Path, name: str) -> dict:
    """Load full_method_out.json from an experiment directory."""
    path = exp_dir / "full_method_out.json"
    logger.info(f"Loading {name} from {path}")
    data = json.loads(path.read_text())
    n_ex = len(data.get("datasets", [{}])[0].get("examples", []))
    logger.info(f"Loaded {name}: {n_ex} examples")
    return data


def _parse_pred(ex: dict, key: str) -> dict:
    """Safely parse a JSON-string prediction field."""
    raw = ex.get(key, "{}")
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        logger.warning(f"Could not parse {key}: {raw[:120]}...")
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Per-Hypothesis Evidence Scores
# ═════════════════════════════════════════════════════════════════════════════

def score_h1_universal_overrepresentation(meta: dict) -> dict:
    """H1: Universal Motif Overrepresentation."""
    logger.info("Scoring H1: Universal Motif Overrepresentation")

    # ── 3-node results ──
    h1_3n = meta.get("hypothesis_1_universal_overrepresentation_3node", {})
    pm3 = h1_3n.get("per_motif", {})
    ffl_3n = pm3.get("7", {})
    ffl_mean_z = ffl_3n.get("mean_z", 0.0)
    ffl_n_sig = ffl_3n.get("n_domains_significant", 0)
    ffl_dom_z = ffl_3n.get("per_domain_mean_z", {})
    zvals_3n = list(ffl_dom_z.values())
    z_min_3n = min(zvals_3n) if zvals_3n else 0.0
    z_max_3n = max(zvals_3n) if zvals_3n else 0.0
    z_mean_3n = sum(zvals_3n) / len(zvals_3n) if zvals_3n else 0.0
    n_dom_z_gt2_3n = sum(1 for z in zvals_3n if z > 2)

    # ── 4-node results ──
    h1_4n = meta.get("hypothesis_1_universal_overrepresentation_4node", {})
    pm4 = h1_4n.get("per_motif", {})
    n_universal_reported = h1_4n.get("n_motifs_universal", 0)

    # Count motifs with Z > 2 in >= 6 domains
    motifs_meeting = []
    for mid, md in pm4.items():
        ns = md.get("n_domains_significant", 0)
        mz = md.get("mean_z", 0.0)
        if ns >= 6 and mz > 2:
            motifs_meeting.append({"motif_id": int(mid), "mean_z": mz,
                                   "n_domains_significant": ns})

    # The 4 universal motifs
    univ_motifs = {}
    for mid in ("77", "80", "82", "83"):
        univ_motifs[int(mid)] = pm4.get(mid, {}).get("mean_z", 0.0)

    # 3-node SP degeneracy
    mf = meta.get("methodological_finding", {})
    sp_degen = "degeneracy" in mf.get("title", "").lower()

    # Success criterion: >= 3 motifs Z > 2 in >= 6/8 domains
    criterion_met = len(motifs_meeting) >= 3

    if criterion_met and len(motifs_meeting) >= 4:
        evidence_score, numeric_score = "Strong Confirm", 0.85
    elif criterion_met:
        evidence_score, numeric_score = "Partial Confirm", 0.6
    else:
        evidence_score, numeric_score = "Inconclusive", 0.0

    caveats = []
    if sp_degen:
        caveats.append(
            "3-node SP is degenerate [-0.5,-0.5,-0.5,0.5] under degree-preserving "
            "null models; H1 3-node result (030T overrepresentation) holds via "
            "Z-scores but SP-based claims are invalid"
        )
    caveats.extend([
        "4-node census used sampling (sampled_5) not exact enumeration for most graphs",
        f"46/174 graphs skipped for 4-node analysis (too large); only 128 graphs analyzed",
        f"Only {meta.get('n_null_3_actual', 12)} null models for 3-node (low for reliable Z-score estimation)",
    ])

    return {
        "hypothesis_id": "H1",
        "claim": "Universal Motif Overrepresentation",
        "evidence_score": evidence_score,
        "numeric_score": numeric_score,
        "success_criterion_met": criterion_met,
        "confidence": "high" if criterion_met else "medium",
        "key_metrics": {
            "3node_ffl_mean_z": ffl_mean_z,
            "3node_ffl_n_domains_significant": ffl_n_sig,
            "3node_ffl_z_min": z_min_3n,
            "3node_ffl_z_max": z_max_3n,
            "3node_ffl_z_mean": z_mean_3n,
            "3node_n_domains_z_gt2": n_dom_z_gt2_3n,
            "4node_n_motifs_universal": n_universal_reported,
            "4node_motifs_meeting_criterion": len(motifs_meeting),
            "4node_universal_motif_77_z": univ_motifs.get(77, 0.0),
            "4node_universal_motif_80_z": univ_motifs.get(80, 0.0),
            "4node_universal_motif_82_z": univ_motifs.get(82, 0.0),
            "4node_universal_motif_83_z": univ_motifs.get(83, 0.0),
            "3node_sp_degenerate": sp_degen,
            "n_graphs_3node": meta.get("n_graphs_3node", 0),
            "n_graphs_4node": meta.get("n_graphs_4node", 0),
        },
        "caveats": caveats,
    }


def score_h2_capability_clustering(meta: dict) -> dict:
    """H2: Capability Clustering (Circuit Superfamilies)."""
    logger.info("Scoring H2: Capability Clustering")

    h2 = meta.get("hypothesis_2_capability_clustering", {})
    fc = h2.get("feature_set_comparison", {})

    # Extract best-NMI per feature set
    def _best_nmi(key: str) -> float:
        d = fc.get(key, {})
        return d.get("best_nmi", 0.0) if isinstance(d, dict) else 0.0

    nmi = {
        "4node_sp":              h2.get("4node_sp_best_nmi", _best_nmi("4node_sp_24d")),
        "4node_count_ratios":    h2.get("4node_count_ratios_best_nmi",
                                        _best_nmi("4node_count_ratios_24d")),
        "4node_zscores":         _best_nmi("4node_zscores_24d"),
        "3node_enriched":        h2.get("3node_enriched_best_nmi",
                                        _best_nmi("3node_enriched_12d")),
        "graph_stats":           h2.get("graph_stats_best_nmi",
                                        _best_nmi("graph_stats_16d")),
        "random":                h2.get("random_best_nmi", _best_nmi("random_24d")),
        "combined":              _best_nmi("combined_all"),
        "3node_enriched_full":   _best_nmi("3node_enriched_full_corpus"),
        "graph_stats_full":      _best_nmi("graph_stats_full_corpus"),
    }

    # Criterion checks
    #  4node_zscores: NMI=0.596 > 0.5 YES but < graph_stats 0.701 NO
    #  3node_enriched: NMI=0.742 > 0.5 YES AND > graph_stats 0.701 YES
    fournode_above_05 = nmi["4node_zscores"] > 0.5
    fournode_beats_gs = nmi["4node_zscores"] > nmi["graph_stats"]
    three_above_05 = nmi["3node_enriched"] > 0.5
    three_beats_gs = nmi["3node_enriched"] > nmi["graph_stats"]
    four_beats_bl_flag = h2.get("4node_beats_baseline", False)

    # NMI gaps
    gap_128 = nmi["3node_enriched"] - nmi["graph_stats"]
    gap_174 = nmi["3node_enriched_full"] - nmi["graph_stats_full"]
    gap_4n  = nmi["4node_zscores"] - nmi["graph_stats"]

    # Effective dimensionality
    edim = meta.get("effective_dimensionality", {})
    dim_3sp = edim.get("3node_sp", 1)
    dim_4sp = edim.get("4node_sp", 14)

    # Score: 3-node enriched beats baseline but 4-node doesn't
    if three_above_05 and three_beats_gs:
        evidence_score, numeric_score = "Partial Confirm", 0.45
    elif fournode_above_05:
        evidence_score, numeric_score = "Partial Confirm", 0.30
    else:
        evidence_score, numeric_score = "Inconclusive", 0.0

    # Success criterion technically met for 3-node enriched only
    success_met = three_above_05 and three_beats_gs

    caveats = [
        f"4-node features do NOT beat graph-stats baseline (NMI {nmi['4node_zscores']:.3f} vs {nmi['graph_stats']:.3f})",
        f"3-node enriched features (NMI={nmi['3node_enriched']:.3f}) beat baseline but use count ratios, not the degenerate SP",
        "3-node SP is constant [-0.5,-0.5,-0.5,0.5] — carries zero clustering information",
        "Clustering success may partly reflect graph-size/density differences between domains",
        f"4-node SP effective dim={dim_4sp} (95% var), richer than 3-node (dim={dim_3sp})",
    ]

    return {
        "hypothesis_id": "H2",
        "claim": "Capability Clustering (Circuit Superfamilies)",
        "evidence_score": evidence_score,
        "numeric_score": numeric_score,
        "success_criterion_met": success_met,
        "confidence": "medium",
        "key_metrics": {
            "nmi_4node_sp": nmi["4node_sp"],
            "nmi_4node_count_ratios": nmi["4node_count_ratios"],
            "nmi_4node_zscores": nmi["4node_zscores"],
            "nmi_3node_enriched": nmi["3node_enriched"],
            "nmi_graph_stats": nmi["graph_stats"],
            "nmi_random": nmi["random"],
            "nmi_combined": nmi["combined"],
            "nmi_3node_enriched_full_174": nmi["3node_enriched_full"],
            "nmi_graph_stats_full_174": nmi["graph_stats_full"],
            "nmi_gap_128_graphs": gap_128,
            "nmi_gap_174_graphs": gap_174,
            "nmi_gap_4node_vs_baseline": gap_4n,
            "effective_dim_3node_sp": dim_3sp,
            "effective_dim_4node_sp": dim_4sp,
            "4node_beats_baseline": four_beats_bl_flag,
        },
        "caveats": caveats,
    }


def score_h3_failure_prediction(exp2: dict) -> dict:
    """H3: Failure / Error Prediction."""
    logger.info("Scoring H3: Failure/Error Prediction")

    meta = exp2["metadata"]
    examples = exp2["datasets"][0]["examples"]

    # ── Classifier results ──
    classifiers: dict[str, dict] = {}
    for ex in examples:
        if ex.get("metadata_analysis_type") == "classification":
            clf = ex.get("metadata_classifier", "")
            classifiers[clf] = _parse_pred(ex, "predict_motif_classifier")

    rf_all = classifiers.get("RF_all", {})
    best_auc = rf_all.get("auc_mean", 0.0)
    best_auc_std = rf_all.get("auc_std", 0.0)

    bl_gs   = classifiers.get("BL_graph_stats_only", {})
    bl_rand = classifiers.get("BL_random", {})
    bl_maj  = classifiers.get("BL_majority", {})
    auc_gs  = bl_gs.get("auc_mean", 0.0)
    auc_rand = bl_rand.get("auc_mean", 0.0)
    auc_maj = bl_maj.get("auc_mean", 0.0)

    # ── Statistical tests ──
    stat_tests: dict[str, dict] = {}
    perm_test: dict = {}
    for ex in examples:
        if ex.get("metadata_analysis_type") == "statistical_test":
            pred = _parse_pred(ex, "predict_motif_classifier")
            comp = pred.get("comparison", "")
            if comp:
                stat_tests[comp] = pred
            if "permutation" in ex.get("input", "").lower():
                perm_test = pred

    rf_vs_gs = stat_tests.get("RF_all vs BL_graph_stats_only", {})
    boot_ci = perm_test.get("bootstrap_ci", {})
    boot_lo = boot_ci.get("ci_lower", 0.0)
    boot_hi = boot_ci.get("ci_upper", 0.0)
    perm_p  = perm_test.get("empirical_p_value", 1.0)

    # ── Ablation results ──
    ablations: dict[str, dict] = {}
    for ex in examples:
        if ex.get("metadata_analysis_type") == "ablation":
            fg = ex.get("metadata_feature_group", "")
            ablations[fg] = _parse_pred(ex, "predict_motif_classifier")

    auc_cr_only = ablations.get("count_ratios_only", {}).get("auc_mean", 0.0)
    auc_gs_abl  = ablations.get("graph_stats_only", {}).get("auc_mean", 0.0)
    auc_zs_only = ablations.get("zscore_only", {}).get("auc_mean", 0.0)
    auc_dev_only = ablations.get("deviation_only", {}).get("auc_mean", 0.0)

    # ── Domain confound analysis ──
    cdist = meta.get("class_distribution", {})
    per_domain = cdist.get("per_domain", {})
    n_pure_true, n_pure_unk, n_mixed = 0, 0, 0
    domain_details: dict[str, dict] = {}
    for dom, cnts in per_domain.items():
        nt = cnts.get("true", 0)
        nu = cnts.get("unknown", 0)
        domain_details[dom] = {"true": nt, "unknown": nu}
        if nu == 0 and nt > 0:
            n_pure_true += 1
        elif nt == 0 and nu > 0:
            n_pure_unk += 1
        else:
            n_mixed += 1

    total_domains = len(per_domain)
    confound_sev = ((n_pure_true + n_pure_unk) / total_domains
                    if total_domains else 0.0)

    n_pos   = cdist.get("unknown", 0)
    n_total = cdist.get("total", 0)
    pos_rate = n_pos / n_total if n_total else 0.0
    label_note = meta.get("NOTE", "")

    # ── Score ──
    surface_met = best_auc > 0.65 and best_auc > auc_gs
    if surface_met and confound_sev < 0.3:
        evidence_score, numeric_score = "Partial Confirm", 0.55
    elif surface_met and confound_sev < 0.6:
        evidence_score, numeric_score = "Partial Confirm", 0.35
    elif surface_met:
        evidence_score, numeric_score = "Partial Confirm", 0.25
    else:
        evidence_score, numeric_score = "Inconclusive", 0.0

    caveats = [
        f"CRITICAL: Label confound — testing true vs unknown, NOT true vs false ({label_note})",
        (f"CRITICAL: Domain confound — {n_pure_true} domains purely 'true', "
         f"{n_pure_unk} purely 'unknown' (rhyme=17/17 unknown); "
         f"domain_confound_severity={confound_sev:.2f}"),
        (f"Class imbalance: {n_total - n_pos} true vs {n_pos} unknown "
         f"({pos_rate:.1%} positive rate)"),
        (f"Ablation: graph_stats_only AUC={auc_gs_abl:.3f}, "
         f"count_ratios_only={auc_cr_only:.3f} — motif features add marginal value"),
        "No cross-validation stratified by domain reported",
        f"Only {n_pos} positive samples — very small for reliable classification",
    ]

    return {
        "hypothesis_id": "H3",
        "claim": "Failure/Error Prediction via Motif Features",
        "evidence_score": evidence_score,
        "numeric_score": numeric_score,
        "success_criterion_met": surface_met,
        "confidence": "low",
        "key_metrics": {
            "best_auc_rf_all": best_auc,
            "best_auc_std": best_auc_std,
            "bootstrap_ci_lower": boot_lo,
            "bootstrap_ci_upper": boot_hi,
            "permutation_p_value": perm_p,
            "auc_graph_stats_baseline": auc_gs,
            "auc_random_baseline": auc_rand,
            "auc_majority_baseline": auc_maj,
            "stat_test_t": rf_vs_gs.get("t_statistic", 0.0),
            "stat_test_p": rf_vs_gs.get("p_value", 1.0),
            "auc_diff_vs_graph_stats": rf_vs_gs.get("auc_diff_mean", 0.0),
            "auc_diff_ci_lower": rf_vs_gs.get("ci_95_lower", 0.0),
            "auc_diff_ci_upper": rf_vs_gs.get("ci_95_upper", 0.0),
            "ablation_count_ratios_only_auc": auc_cr_only,
            "ablation_graph_stats_only_auc": auc_gs_abl,
            "ablation_zscore_only_auc": auc_zs_only,
            "ablation_deviation_only_auc": auc_dev_only,
            "n_pure_true_domains": n_pure_true,
            "n_pure_unknown_domains": n_pure_unk,
            "n_mixed_domains": n_mixed,
            "domain_confound_severity": confound_sev,
            "positive_rate": pos_rate,
            "n_samples": n_total,
            "n_positive": n_pos,
        },
        "domain_class_distribution": domain_details,
        "caveats": caveats,
    }


def score_h4_causal_validation() -> dict:
    """H4: Causal Validation — NOT TESTED."""
    logger.info("Scoring H4: Causal Validation (NOT TESTED)")
    return {
        "hypothesis_id": "H4",
        "claim": "Causal Validation via Neuronpedia Steering",
        "evidence_score": "Inconclusive",
        "numeric_score": 0.0,
        "success_criterion_met": False,
        "confidence": "low",
        "key_metrics": {
            "experiments_conducted": 0,
            "api_available": 1,
        },
        "caveats": [
            "NOT TESTED in any experiment — no evidence either way",
            "Neuronpedia /api/steer endpoint exists in principle but no experiment attempted",
            "Causal validation would require ablating/steering specific motif nodes and measuring behavioral impact",
            "This is the most important missing piece for mechanistic interpretability claims",
        ],
    }


def score_h5_ffl_characterization(exp3: dict) -> dict:
    """H5: Functional Characterization (FFL)."""
    logger.info("Scoring H5: Functional Characterization (FFL)")

    meta = exp3["metadata"]
    examples = exp3["datasets"][0]["examples"]

    # ── Aggregate / confirmation signals ──
    agg_stats: dict = {}
    agg_baseline: dict = {}
    confirmation: dict = {}
    for ex in examples:
        at = ex.get("metadata_analysis_type", "")
        if at == "aggregate_statistics":
            agg_stats = _parse_pred(ex, "predict_ffl_method")
            agg_baseline = _parse_pred(ex, "predict_baseline_random_triad")
        elif at == "confirmation_signals":
            confirmation = _parse_pred(ex, "predict_ffl_method")

    total_ffls = meta.get("total_ffls", 0)
    total_graphs = meta.get("total_graphs", 0)
    match_rate = meta.get("explanation_match_rate", 0.0)

    layer_strict = confirmation.get("layer_strict_ordering_frac",
                                    agg_baseline.get("ffl_strict_order", 0.0))
    bl_strict = confirmation.get("baseline_strict_ordering_frac",
                                 agg_baseline.get("baseline_strict_order", 0.0))
    chi2       = agg_stats.get("chi2", confirmation.get("semantic_chi2", 0.0))
    cramers_v  = agg_stats.get("cramers_v",
                               confirmation.get("semantic_cramers_v", 0.0))
    coherent   = agg_stats.get("coherent_frac",
                               confirmation.get("coherent_ffl_fraction", 0.0))
    cross_v    = agg_stats.get("cross_domain_cramers_v",
                               confirmation.get("cross_domain_cramers_v", 0.0))

    # Per-domain coherent fractions
    pdc: dict[str, float] = {}
    for ex in examples:
        if ex.get("metadata_analysis_type") == "per_domain_aggregate":
            dom = ex.get("metadata_fold", "")
            pred = _parse_pred(ex, "predict_ffl_method")
            pdc[dom] = pred.get("coherent_frac", 0.0)

    # Layer ordering ratio
    layer_ratio = (layer_strict / bl_strict) if bl_strict > 0 else float("inf")

    # ── Scoring ──
    layer_strong   = layer_strict > 0.95 and (layer_strict - bl_strict) > 0.5
    sem_signif     = chi2 > 1000 and cramers_v > 0.1
    cross_consist  = cross_v > 0.3

    if layer_strong and sem_signif and cross_consist:
        evidence_score, numeric_score = "Strong Confirm", 0.80
    elif layer_strong and sem_signif:
        evidence_score, numeric_score = "Partial Confirm", 0.50
    elif layer_strong:
        evidence_score, numeric_score = "Partial Confirm", 0.35
    else:
        evidence_score, numeric_score = "Inconclusive", 0.0

    # Only 1 motif type (FFL) characterised; criterion requires >= 2
    success_met = False

    caveats = [
        "Only 1 motif type (FFL/030T) characterised — criterion requires >= 2 overrepresented motifs with interpretable characterizations",
        f"Cross-domain Cramer's V = {cross_v:.3f} — LOW, different domains use different semantic patterns",
        f"Explanation match rate = {match_rate:.1%} — {1 - match_rate:.1%} of features lack explanations",
        f"Only {total_graphs} graphs analysed (subset of full 174-graph corpus)",
        f"Coherent FFL fraction = {coherent:.1%} (vs 50% chance baseline) — modest effect",
        "Semantic role categories rely on auto-generated Neuronpedia explanations (may be noisy)",
    ]

    return {
        "hypothesis_id": "H5",
        "claim": "Functional Characterization of Feed-Forward Loops",
        "evidence_score": evidence_score,
        "numeric_score": numeric_score,
        "success_criterion_met": success_met,
        "confidence": "medium",
        "key_metrics": {
            "total_ffls": total_ffls,
            "total_graphs": total_graphs,
            "layer_strict_ordering_frac": layer_strict,
            "baseline_strict_ordering_frac": bl_strict,
            "layer_ordering_ratio": layer_ratio,
            "semantic_chi2": chi2,
            "semantic_cramers_v": cramers_v,
            "coherent_ffl_fraction": coherent,
            "cross_domain_cramers_v": cross_v,
            "explanation_match_rate": match_rate,
        },
        "per_domain_coherent": pdc,
        "caveats": caveats,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. Methodological Validity Assessment
# ═════════════════════════════════════════════════════════════════════════════

def assess_methodological_validity(
    exp1_meta: dict, exp2: dict, exp3: dict
) -> list[dict]:
    logger.info("Assessing methodological validity")

    cm = exp1_meta.get("4node_census_methods", {})
    n_skip = sum(1 for v in cm.values() if v == "skipped_too_large")
    n_samp = sum(1 for v in cm.values() if v.startswith("sampled"))
    n_exact = sum(1 for v in cm.values() if v.startswith("exact"))
    n_tot = len(cm)

    exp1_val = {
        "experiment": "exp_id1_it3: 4-Node Motif Census",
        "sample_size_adequate": True,
        "sample_size_rationale":
            "n=174 (3-node), n=128 (4-node) across 8 domains — adequate for large effects",
        "null_model_quality": {
            "n_null_3node": exp1_meta.get("n_null_3_actual", 0),
            "n_null_4node": exp1_meta.get("n_null_4_actual", 0),
            "assessment": (
                f"{exp1_meta.get('n_null_3_actual', 12)} nulls for 3-node is LOW "
                f"(recommend >= 100); {exp1_meta.get('n_null_4_actual', 50)} for 4-node is acceptable"
            ),
        },
        "multiple_testing_corrected": False,
        "multiple_testing_note":
            "No explicit correction for 24 motif types x 8 domains; universal criterion (>= 6/8) provides implicit correction",
        "confound_control": [
            f"{n_skip}/{n_tot} graphs skipped for 4-node (selection bias toward smaller graphs)",
            f"{n_samp}/{n_tot} graphs used approximate sampling, {n_exact} exact enumeration",
            "3-node SP degeneracy correctly identified",
        ],
        "reproducibility_seeds": True,
        "pruning_sensitivity_tested": True,
        "pruning_thresholds": exp1_meta.get("config", {}).get(
            "prune_thresholds_tested", []),
    }

    exp2_meta = exp2["metadata"]
    cdist = exp2_meta.get("class_distribution", {})
    exp2_val = {
        "experiment": "exp_id2_it3: Failure/Ambiguity Prediction",
        "sample_size_adequate": False,
        "sample_size_rationale":
            f"n={cdist.get('total', 140)} total but only {cdist.get('unknown', 24)} positive — too few",
        "null_model_quality": {
            "n_null_models": exp2_meta.get("parameters", {}).get("n_null_models", 0),
            "assessment": "50 null models for feature extraction — adequate",
        },
        "multiple_testing_corrected": False,
        "multiple_testing_note":
            "Multiple classifiers and feature sets compared without correction",
        "confound_control": [
            "CRITICAL: Domain-label confound — rhyme=17/17 unknown, antonym/code_completion/multi_hop=all true",
            "No domain-stratified CV reported",
            "Label confound: true vs unknown, NOT true vs false",
        ],
        "reproducibility_seeds": True,
        "pruning_sensitivity_tested": False,
        "pruning_thresholds": [exp2_meta.get("parameters", {}).get(
            "prune_percentile", 75)],
    }

    exp3_meta = exp3["metadata"]
    exp3_val = {
        "experiment": "exp_id3_it3: FFL Functional Characterization",
        "sample_size_adequate": False,
        "sample_size_rationale":
            f"Only {exp3_meta.get('total_graphs', 34)} graphs — small subset, may not generalise",
        "null_model_quality": {
            "baseline_samples_per_graph":
                exp3_meta.get("parameters", {}).get("n_baseline_samples_per_graph", 0),
            "assessment": "5000 random triads per graph — adequate for comparison",
        },
        "multiple_testing_corrected": False,
        "multiple_testing_note":
            "Chi-squared on large contingency tables without correction",
        "confound_control": [
            f"Explanation match rate = {exp3_meta.get('explanation_match_rate', 0):.1%} — missing data",
            "Semantic roles depend on auto-generated Neuronpedia explanations",
            "FFL enumeration on pruned graphs may bias results",
        ],
        "reproducibility_seeds": True,
        "pruning_sensitivity_tested": False,
        "pruning_thresholds": [exp3_meta.get("parameters", {}).get(
            "prune_percentile", 75)],
    }

    return [exp1_val, exp2_val, exp3_val]


# ═════════════════════════════════════════════════════════════════════════════
# 3. Paper Narrative Recommendation
# ═════════════════════════════════════════════════════════════════════════════

def generate_paper_narrative(h_scores: list[dict]) -> dict:
    logger.info("Generating paper narrative recommendation")

    by_score = sorted(h_scores, key=lambda x: x["numeric_score"], reverse=True)
    centerpiece = by_score[0]
    supporting  = [s for s in by_score[1:] if s["numeric_score"] >= 0.35]
    exploratory = [s for s in by_score if 0.1 < s["numeric_score"] < 0.35]
    drop        = [s for s in by_score if s["numeric_score"] <= 0.1]

    def _rationale(h: dict) -> str:
        hid = h["hypothesis_id"]
        if hid == "H1":
            return ("4/24 four-node motifs are universally overrepresented "
                    "(Z > 2 in >= 6/8 domains), with motif 83 achieving mean Z = 8.12. "
                    "Strongest finding; survives null-model controls across domains.")
        if hid == "H5":
            return ("FFL layer ordering is perfectly consistent (100% vs 35% baseline). "
                    "Semantic roles are statistically significant but weakly consistent across domains.")
        if hid == "H2":
            return ("3-node enriched count-ratio features achieve NMI = 0.742, beating baseline. "
                    "4-node SP underperforms baseline; 3-node SP is degenerate.")
        if hid == "H3":
            return ("AUC = 0.853 is impressive on surface but severe domain confound "
                    "(rhyme = all unknown, 3 domains = all true) means classifier may detect "
                    "domain identity, not error topology. Requires domain-stratified validation.")
        return f"{h['claim']}: {h['evidence_score']}"

    narrative = {
        "centerpiece_claim": {
            "hypothesis": centerpiece["hypothesis_id"],
            "claim": centerpiece["claim"],
            "evidence_score": centerpiece["evidence_score"],
            "rationale": _rationale(centerpiece),
        },
        "supporting_claims": [
            {"hypothesis": s["hypothesis_id"], "claim": s["claim"],
             "evidence_score": s["evidence_score"], "rationale": _rationale(s)}
            for s in supporting
        ],
        "exploratory_claims": [
            {"hypothesis": s["hypothesis_id"], "claim": s["claim"],
             "evidence_score": s["evidence_score"], "rationale": _rationale(s)}
            for s in exploratory
        ],
        "claims_to_drop_or_reframe": [
            {"hypothesis": s["hypothesis_id"], "claim": s["claim"],
             "evidence_score": s["evidence_score"],
             "rationale": ("No experiments conducted. Future work direction."
                           if s["hypothesis_id"] == "H4"
                           else f"{s['claim']}: no evidence")}
            for s in drop
        ],
        "suggested_abstract": (
            "We perform the first systematic motif analysis of neural network attribution "
            "graphs from Neuronpedia, discovering four 4-node directed motifs that are "
            "universally overrepresented across 8 capability domains (Z > 2 in >= 6/8 domains). "
            "We show that motif count-ratio features enable capability clustering (NMI = 0.742), "
            "while revealing that 3-node significance profiles are degenerate under degree-preserving "
            "null models — a methodological finding with implications for graph-theoretic analysis "
            "of neural circuits. Feed-forward loops exhibit perfect layer ordering (100% vs 35% "
            "baseline), suggesting structural regularities in how neural networks compose "
            "information across layers."
        ),
        "reframing_notes": [
            "Lead with H1 (universal overrepresentation) as main contribution",
            "Present 3-node SP degeneracy as a methodological contribution, not a failure",
            "Reframe H2: emphasise count-ratio features beating baseline; discuss 4-node SP dimensionality",
            "H3 must carry heavy caveats about domain confound — or run domain-stratified validation first",
            "H5: emphasise structural regularity (layer ordering) over semantic classification",
            "H4 is future work, not a claim",
            "Overall: 'structural motif signatures exist and carry information about neural circuit "
            "function, but disentangling this from simpler graph statistics remains an open challenge'",
        ],
    }
    return narrative


# ═════════════════════════════════════════════════════════════════════════════
# 4. Prioritised Gap List
# ═════════════════════════════════════════════════════════════════════════════

def generate_gap_list(h_scores: list[dict]) -> list[dict]:
    logger.info("Generating prioritised gap list")
    gaps = [
        {
            "gap_id": "G1_domain_stratified_h3",
            "description": "H3 failure prediction has severe domain confound — "
                           "need domain-stratified CV or within-domain analysis",
            "priority": "must_fix",
            "impact": 5.0,
            "feasibility": 5.0,
            "priority_score": 25.0,
            "artifact_type_needed": "experiment",
            "specific_action": (
                "Re-run failure prediction with leave-one-domain-out CV, or restrict "
                "to mixed domains (country_capital 14/4, arithmetic 16/2, sentiment 16/1). "
                "Compute per-domain AUC where both classes exist."
            ),
        },
        {
            "gap_id": "G2_4node_characterization",
            "description": "Only FFL (030T, 3-node) is functionally characterised — "
                           "need characterisation of the 4 universal 4-node motifs",
            "priority": "must_fix",
            "impact": 5.0,
            "feasibility": 4.0,
            "priority_score": 20.0,
            "artifact_type_needed": "experiment",
            "specific_action": (
                "Enumerate instances of 4-node motifs 77, 80, 82, 83 and characterise "
                "semantic roles (analogous to FFL characterisation). Directly addresses "
                "H5 success criterion."
            ),
        },
        {
            "gap_id": "G3_within_domain_error_signal",
            "description": "Test if motif features predict errors within domains "
                           "(not just across domains)",
            "priority": "must_fix",
            "impact": 4.0,
            "feasibility": 4.0,
            "priority_score": 16.0,
            "artifact_type_needed": "experiment",
            "specific_action": (
                "For domains with both classes (country_capital 14/4, arithmetic 16/2, "
                "sentiment 16/1), compute within-domain motif-based prediction. "
                "Isolates motif signal from domain confound."
            ),
        },
        {
            "gap_id": "G4_graph_stats_ablation",
            "description": "Rigorously test what motif features add beyond graph statistics",
            "priority": "would_strengthen",
            "impact": 4.0,
            "feasibility": 4.0,
            "priority_score": 16.0,
            "artifact_type_needed": "experiment",
            "specific_action": (
                "Run clustering (H2) with motif features residualised against graph stats "
                "(partial correlation / regression). If motif NMI drops to baseline after "
                "controlling graph stats, the motif signal is redundant."
            ),
        },
        {
            "gap_id": "G5_more_null_models_3node",
            "description": "Only 12 null models for 3-node census — Z-scores may be unreliable",
            "priority": "would_strengthen",
            "impact": 3.0,
            "feasibility": 4.0,
            "priority_score": 12.0,
            "artifact_type_needed": "experiment",
            "specific_action": (
                "Rerun 3-node census with >= 100 null models. "
                "Check if Z-scores change substantially."
            ),
        },
        {
            "gap_id": "G6_causal_validation",
            "description": "H4 causal validation not attempted — strongest theoretical gap",
            "priority": "would_strengthen",
            "impact": 5.0,
            "feasibility": 2.0,
            "priority_score": 10.0,
            "artifact_type_needed": "experiment",
            "specific_action": (
                "Use Neuronpedia /api/steer to ablate/steer nodes in overrepresented "
                "motif instances and measure behavioural change. Even a small pilot "
                "(10 graphs) would add causal evidence."
            ),
        },
        {
            "gap_id": "G7_multiple_testing_correction",
            "description": "No explicit multiple-testing correction for 24 motif types x 8 domains",
            "priority": "would_strengthen",
            "impact": 2.0,
            "feasibility": 5.0,
            "priority_score": 10.0,
            "artifact_type_needed": "evaluation",
            "specific_action": (
                "Apply Bonferroni or FDR correction to H1 Z-scores across 24 x 8 = 192 tests. "
                "Verify the 4 universal motifs survive correction."
            ),
        },
        {
            "gap_id": "G8_pruning_sensitivity",
            "description": "Most experiments use single pruning threshold — "
                           "results may be sensitive",
            "priority": "would_strengthen",
            "impact": 3.0,
            "feasibility": 3.0,
            "priority_score": 9.0,
            "artifact_type_needed": "experiment",
            "specific_action": (
                "Rerun H1 4-node overrepresentation and H5 FFL characterisation at "
                "multiple pruning thresholds (50, 75, 90, 95, 99 percentile) "
                "and check stability."
            ),
        },
        {
            "gap_id": "G9_full_corpus_ffl",
            "description": "FFL characterisation only on 34/174 graphs",
            "priority": "would_strengthen",
            "impact": 3.0,
            "feasibility": 3.0,
            "priority_score": 9.0,
            "artifact_type_needed": "experiment",
            "specific_action": (
                "Extend FFL characterisation to full 174-graph corpus. "
                "Compare per-domain patterns across larger sample."
            ),
        },
        {
            "gap_id": "G10_weighted_motif_analysis",
            "description": "Current analysis ignores edge weights (attribution scores)",
            "priority": "future_work",
            "impact": 4.0,
            "feasibility": 2.0,
            "priority_score": 8.0,
            "artifact_type_needed": "experiment",
            "specific_action": (
                "Develop weighted motif census using edge attribution scores. "
                "Test if weighted features improve clustering (H2) and prediction (H3)."
            ),
        },
    ]
    gaps.sort(key=lambda g: g["priority_score"], reverse=True)
    return gaps


# ═════════════════════════════════════════════════════════════════════════════
# 5. Cross-Hypothesis Interaction Analysis
# ═════════════════════════════════════════════════════════════════════════════

def generate_cross_hypothesis_interactions() -> dict[str, dict]:
    logger.info("Generating cross-hypothesis interaction analysis")
    return {
        "H1_H2": {
            "hypotheses": ["H1: Universal Overrepresentation",
                           "H2: Capability Clustering"],
            "interaction": (
                "3-node SP degeneracy (relevant to H2) does NOT affect H1 — H1 uses "
                "raw Z-scores not SP. However, the degeneracy means 3-node clustering "
                "success (NMI = 0.742) comes from count ratios, not motif spectra per se. "
                "The effective dimensionality of 4-node SP (14) confirms richer structural "
                "vocabulary, suggesting 4-node features have untapped potential for clustering."
            ),
        },
        "H2_H3": {
            "hypotheses": ["H2: Capability Clustering",
                           "H3: Failure Prediction"],
            "interaction": (
                "Domain confound in H3 also affects H2 — clustering success may reflect "
                "graph-size/density differences between domains rather than genuine motif "
                "structure. If domains are distinguishable by graph stats alone (NMI = 0.701), "
                "motif-based clustering may be partly explained by the same underlying "
                "graph-level differences. Domain-stratified analysis for H3 would also "
                "help validate H2."
            ),
        },
        "H1_H5": {
            "hypotheses": ["H1: Universal Overrepresentation",
                           "H5: FFL Characterization"],
            "interaction": (
                "FFL (030T) is the only overrepresented 3-node motif (H1), and FFL "
                "characterisation shows 100% layer ordering (H5) — these strongly reinforce "
                "each other. The universality of FFL overrepresentation (Z > 32 in all 8 "
                "domains) combined with perfect structural regularity suggests FFLs are "
                "fundamental computational motifs. The 4 universal 4-node motifs (H1) need "
                "similar characterisation to strengthen H5."
            ),
        },
        "H3_H5": {
            "hypotheses": ["H3: Failure Prediction",
                           "H5: FFL Characterization"],
            "interaction": (
                "If failure prediction (H3) is domain-confounded, FFL characterisation "
                "(H5) could provide an alternative within-domain error signal. Specifically: "
                "do FFLs in verified-correct graphs have different semantic role distributions "
                "than FFLs in ambiguous graphs within the same domain?"
            ),
        },
        "H1_H3": {
            "hypotheses": ["H1: Universal Overrepresentation",
                           "H3: Failure Prediction"],
            "interaction": (
                "H1 shows motifs are universally overrepresented, while H3 uses motif "
                "features to distinguish correctness. If motif profiles are primarily "
                "domain-specific (H2 clustering), H3 may detect domain differences "
                "rather than error-related motif changes. The marginal improvement "
                "(AUC 0.853 vs 0.793 graph_stats ablation) is modest and may not "
                "survive domain stratification."
            ),
        },
        "H2_H5": {
            "hypotheses": ["H2: Capability Clustering",
                           "H5: FFL Characterization"],
            "interaction": (
                "H2 shows motif count ratios cluster by domain; H5 shows FFL semantic "
                "roles differ across domains (cross-domain V = 0.126). Consistent: different "
                "domains use circuits with different motif compositions and semantic patterns. "
                "The low cross-domain consistency in H5 actually supports H2's finding that "
                "motif features carry domain-specific information."
            ),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. Overall Hypothesis Verdict
# ═════════════════════════════════════════════════════════════════════════════

def generate_overall_verdict(h_scores: list[dict]) -> dict:
    logger.info("Generating overall verdict")

    n_confirmed = sum(1 for h in h_scores if h["numeric_score"] >= 0.6)
    n_partial   = sum(1 for h in h_scores if 0.2 <= h["numeric_score"] < 0.6)
    n_incon     = sum(1 for h in h_scores if h["numeric_score"] < 0.2)
    mean_score  = sum(h["numeric_score"] for h in h_scores) / len(h_scores)

    if mean_score >= 0.6:
        status = "Largely Confirmed"
    elif mean_score >= 0.3:
        status = "Partially Confirmed"
    elif mean_score >= 0.15:
        status = "Mixed Results"
    else:
        status = "Largely Disconfirmed"

    return {
        "overall_status": status,
        "mean_numeric_score": mean_score,
        "n_strong_confirm": n_confirmed,
        "n_partial": n_partial,
        "n_inconclusive": n_incon,
        "strongest_evidence": (
            "H1 Universal Overrepresentation: 4 of 24 four-node motifs show Z > 2 "
            "in >= 6/8 domains (motif 83: mean Z = 8.12). Most robust finding with "
            "high statistical significance."
        ),
        "weakest_evidence": (
            "H4 Causal Validation: completely untested. H3 Failure Prediction: "
            "AUC = 0.853 severely undermined by domain confound (5/8 domains are "
            "pure single-class). Without domain-stratified validation H3 is uninterpretable."
        ),
        "recommended_paper_angle": (
            "Position as 'Structural Motif Signatures in Neural Network Attribution Graphs': "
            "(1) lead with universal 4-node overrepresentation (H1); "
            "(2) present 3-node SP degeneracy as methodological insight; "
            "(3) show clustering works with count-ratio features (H2); "
            "(4) present FFL layer ordering (H5) as structural regularity evidence; "
            "(5) include failure prediction (H3) as exploratory with explicit confound discussion; "
            "(6) omit causal claims (H4) — future work."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Hypothesis Evidence Scorecard: Iterations 1-3 Results")
    logger.info("=" * 60)

    # ── Load ──
    exp1 = load_experiment(EXP1_DIR, "exp_id1_it3 (4-Node Motif Census)")
    exp2 = load_experiment(EXP2_DIR, "exp_id2_it3 (Failure Prediction)")
    exp3 = load_experiment(EXP3_DIR, "exp_id3_it3 (FFL Characterization)")
    exp1_meta = exp1["metadata"]

    # ── 1. Hypothesis scores ──
    h1 = score_h1_universal_overrepresentation(exp1_meta)
    h2 = score_h2_capability_clustering(exp1_meta)
    h3 = score_h3_failure_prediction(exp2)
    h4 = score_h4_causal_validation()
    h5 = score_h5_ffl_characterization(exp3)
    h_scores = [h1, h2, h3, h4, h5]

    for h in h_scores:
        logger.info(
            f"  {h['hypothesis_id']}: {h['evidence_score']} "
            f"(score={h['numeric_score']:.2f}, met={h['success_criterion_met']}, "
            f"conf={h['confidence']})"
        )

    # ── 2. Validity ──
    validity = assess_methodological_validity(exp1_meta, exp2, exp3)

    # ── 3. Narrative ──
    narrative = generate_paper_narrative(h_scores)

    # ── 4. Gaps ──
    gaps = generate_gap_list(h_scores)

    # ── 5. Cross-hypothesis ──
    interactions = generate_cross_hypothesis_interactions()

    # ── 6. Verdict ──
    verdict = generate_overall_verdict(h_scores)
    logger.info(
        f"Overall verdict: {verdict['overall_status']} "
        f"(mean={verdict['mean_numeric_score']:.2f})"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # Build output — conform to exp_eval_sol_out.json schema
    # ═════════════════════════════════════════════════════════════════════════

    # ── metrics_agg (flat string → number) ──
    metrics_agg: dict[str, float | int] = {
        # Per-hypothesis numeric scores
        "h1_numeric_score": h1["numeric_score"],
        "h2_numeric_score": h2["numeric_score"],
        "h3_numeric_score": h3["numeric_score"],
        "h4_numeric_score": h4["numeric_score"],
        "h5_numeric_score": h5["numeric_score"],
        "mean_hypothesis_score": verdict["mean_numeric_score"],
        "n_criteria_met": sum(1 for h in h_scores if h["success_criterion_met"]),
        "n_hypotheses_tested": sum(1 for h in h_scores if h["numeric_score"] != 0),
        "n_strong_confirm": verdict["n_strong_confirm"],
        "n_partial_confirm": verdict["n_partial"],
        "n_inconclusive": verdict["n_inconclusive"],
        # H1
        "h1_4node_n_universal_motifs":
            h1["key_metrics"]["4node_n_motifs_universal"],
        "h1_3node_ffl_mean_z":
            h1["key_metrics"]["3node_ffl_mean_z"],
        "h1_4node_motif83_z":
            h1["key_metrics"]["4node_universal_motif_83_z"],
        "h1_4node_motif77_z":
            h1["key_metrics"]["4node_universal_motif_77_z"],
        "h1_4node_motif82_z":
            h1["key_metrics"]["4node_universal_motif_82_z"],
        "h1_4node_motif80_z":
            h1["key_metrics"]["4node_universal_motif_80_z"],
        # H2
        "h2_nmi_4node_zscores":
            h2["key_metrics"]["nmi_4node_zscores"],
        "h2_nmi_3node_enriched":
            h2["key_metrics"]["nmi_3node_enriched"],
        "h2_nmi_graph_stats":
            h2["key_metrics"]["nmi_graph_stats"],
        "h2_nmi_gap_128":
            h2["key_metrics"]["nmi_gap_128_graphs"],
        "h2_nmi_gap_174":
            h2["key_metrics"]["nmi_gap_174_graphs"],
        "h2_eff_dim_4node":
            h2["key_metrics"]["effective_dim_4node_sp"],
        # H3
        "h3_best_auc":
            h3["key_metrics"]["best_auc_rf_all"],
        "h3_auc_graph_stats":
            h3["key_metrics"]["auc_graph_stats_baseline"],
        "h3_auc_diff":
            h3["key_metrics"]["auc_diff_vs_graph_stats"],
        "h3_domain_confound_severity":
            h3["key_metrics"]["domain_confound_severity"],
        "h3_n_positive":
            h3["key_metrics"]["n_positive"],
        "h3_permutation_p":
            h3["key_metrics"]["permutation_p_value"],
        "h3_ablation_count_ratios_auc":
            h3["key_metrics"]["ablation_count_ratios_only_auc"],
        "h3_ablation_graph_stats_auc":
            h3["key_metrics"]["ablation_graph_stats_only_auc"],
        # H5
        "h5_layer_strict_frac":
            h5["key_metrics"]["layer_strict_ordering_frac"],
        "h5_baseline_strict_frac":
            h5["key_metrics"]["baseline_strict_ordering_frac"],
        "h5_semantic_cramers_v":
            h5["key_metrics"]["semantic_cramers_v"],
        "h5_cross_domain_v":
            h5["key_metrics"]["cross_domain_cramers_v"],
        "h5_coherent_frac":
            h5["key_metrics"]["coherent_ffl_fraction"],
        "h5_total_ffls":
            h5["key_metrics"]["total_ffls"],
        "h5_explanation_match_rate":
            h5["key_metrics"]["explanation_match_rate"],
        # Gaps
        "n_must_fix_gaps":
            sum(1 for g in gaps if g["priority"] == "must_fix"),
        "n_would_strengthen_gaps":
            sum(1 for g in gaps if g["priority"] == "would_strengthen"),
        "n_future_work_gaps":
            sum(1 for g in gaps if g["priority"] == "future_work"),
        "top_gap_priority_score":
            gaps[0]["priority_score"] if gaps else 0,
    }

    # ── Dataset 1: hypothesis evidence scores (5 examples) ──
    hypo_examples = []
    for h in h_scores:
        hypo_examples.append({
            "input": f"Hypothesis {h['hypothesis_id']}: {h['claim']}",
            "output": json.dumps({
                "evidence_score": h["evidence_score"],
                "numeric_score": h["numeric_score"],
                "success_criterion_met": h["success_criterion_met"],
                "confidence": h["confidence"],
            }),
            "predict_evidence_assessment": json.dumps({
                "evidence_score": h["evidence_score"],
                "numeric_score": h["numeric_score"],
                "key_metrics": {k: v for k, v in h["key_metrics"].items()
                                if not isinstance(v, bool)},
                "caveats": h["caveats"],
                "success_criterion_met": h["success_criterion_met"],
                "confidence": h["confidence"],
            }),
            "eval_numeric_score": h["numeric_score"],
            "eval_criterion_met": 1.0 if h["success_criterion_met"] else 0.0,
            "metadata_hypothesis_id": h["hypothesis_id"],
            "metadata_claim": h["claim"],
            "metadata_evidence_score": h["evidence_score"],
            "metadata_confidence": h["confidence"],
        })

    # ── Dataset 2: methodological validity (3 examples) ──
    validity_examples = []
    for v in validity:
        validity_examples.append({
            "input": f"Validity assessment: {v['experiment']}",
            "output": json.dumps({
                "sample_size_adequate": v["sample_size_adequate"],
                "multiple_testing_corrected": v["multiple_testing_corrected"],
            }),
            "predict_validity_assessment": json.dumps({
                k: v_ for k, v_ in v.items()
                if k not in ("reproducibility_seeds",)
            }),
            "eval_sample_adequate":
                1.0 if v["sample_size_adequate"] else 0.0,
            "eval_testing_corrected":
                1.0 if v["multiple_testing_corrected"] else 0.0,
            "metadata_experiment": v["experiment"],
        })

    # ── Dataset 3: gap analysis (10 examples) ──
    gap_examples = []
    for g in gaps:
        gap_examples.append({
            "input": f"Gap {g['gap_id']}: {g['description']}",
            "output": json.dumps({
                "priority": g["priority"],
                "impact": g["impact"],
                "feasibility": g["feasibility"],
                "priority_score": g["priority_score"],
            }),
            "predict_gap_analysis": json.dumps(g),
            "eval_impact": g["impact"],
            "eval_feasibility": g["feasibility"],
            "eval_priority_score": g["priority_score"],
            "metadata_gap_id": g["gap_id"],
            "metadata_priority": g["priority"],
            "metadata_artifact_type": g["artifact_type_needed"],
        })

    # ── Dataset 4: cross-hypothesis interactions (6 examples) ──
    inter_examples = []
    for pid, inter in interactions.items():
        inter_examples.append({
            "input": (f"Interaction {pid}: "
                      f"{' <-> '.join(inter['hypotheses'])}"),
            "output": inter["interaction"][:500],
            "predict_interaction_analysis": json.dumps(inter),
            "eval_interaction_present": 1.0,
            "metadata_pair_id": pid,
        })

    # ── Dataset 5: paper narrative + overall verdict (1 example) ──
    narrative_examples = [{
        "input": "Paper narrative recommendation and overall hypothesis verdict",
        "output": json.dumps({
            "centerpiece": narrative["centerpiece_claim"]["hypothesis"],
            "overall_status": verdict["overall_status"],
            "mean_score": verdict["mean_numeric_score"],
        }),
        "predict_narrative": json.dumps(narrative),
        "predict_verdict": json.dumps(verdict),
        "eval_mean_score": verdict["mean_numeric_score"],
        "metadata_overall_status": verdict["overall_status"],
    }]

    # ── Assemble ──
    output = {
        "metadata": {
            "evaluation_name":
                "Hypothesis Evidence Scorecard: Iterations 1-3",
            "description":
                "Comprehensive evidence evaluation of 5 hypothesis claims "
                "against 3 iteration-3 experiments",
            "experiments_evaluated": [
                "exp_id1_it3: 4-Node Motif Census on 174 graphs",
                "exp_id2_it3: Failure Prediction on 140 graphs",
                "exp_id3_it3: FFL Characterization on 34 graphs",
            ],
            "hypothesis_scores": {
                h["hypothesis_id"]: h["evidence_score"] for h in h_scores
            },
            "overall_verdict": verdict,
            "paper_narrative": narrative,
            "cross_hypothesis_interactions": interactions,
            "methodological_validity": validity,
            "gap_list": gaps,
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {"dataset": "hypothesis_evidence_scores",
             "examples": hypo_examples},
            {"dataset": "methodological_validity",
             "examples": validity_examples},
            {"dataset": "gap_analysis",
             "examples": gap_examples},
            {"dataset": "cross_hypothesis_interactions",
             "examples": inter_examples},
            {"dataset": "paper_narrative_and_verdict",
             "examples": narrative_examples},
        ],
    }

    # ── Write ──
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Written eval_out.json ({out_path.stat().st_size / 1024:.1f} KB)")

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"Overall: {verdict['overall_status']} "
                f"(mean = {verdict['mean_numeric_score']:.2f})")
    for h in h_scores:
        logger.info(f"  {h['hypothesis_id']}: {h['evidence_score']} "
                     f"({h['numeric_score']:.2f})")
    logger.info(f"Criteria met: "
                f"{sum(1 for h in h_scores if h['success_criterion_met'])}/5")
    logger.info(f"Must-fix gaps: "
                f"{sum(1 for g in gaps if g['priority'] == 'must_fix')}")
    logger.info(f"Top gap: {gaps[0]['gap_id']} "
                f"(score = {gaps[0]['priority_score']})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
