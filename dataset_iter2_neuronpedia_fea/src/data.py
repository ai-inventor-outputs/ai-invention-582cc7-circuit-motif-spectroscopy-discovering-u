#!/usr/bin/env python3
"""Build final data_out.json from Neuronpedia API checkpoint and feature registry.

Loads the checkpoint (temp/feature_checkpoint.json) and feature registry
(temp/feature_registry.json) produced by pipeline.py, assembles the
exp_sel_data_out schema-compliant dataset, and saves to data_out/.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/3_invention_loop/iter_2/gen_art/data_id4_it2__opus")
TEMP_DIR = WORKSPACE / "temp"
DATA_OUT_DIR = WORKSPACE / "data_out"
LOG_DIR = WORKSPACE / "logs"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "data.log"), rotation="30 MB", level="DEBUG")


@logger.catch
def main():
    # ── Load checkpoint and feature registry ─────────────────────────────
    logger.info("Loading checkpoint and feature registry...")
    fetched = json.loads((TEMP_DIR / "feature_checkpoint.json").read_text())
    registry = json.loads((TEMP_DIR / "feature_registry.json").read_text())

    ok_count = sum(1 for v in fetched.values() if v.get('status') == 'ok')
    logger.info(f"Checkpoint: {len(fetched)} features, {ok_count} OK")
    logger.info(f"Registry: {len(registry)} features with graph provenance")

    # ── Build records ────────────────────────────────────────────────────
    records = []
    for key_str, reg in sorted(registry.items()):
        parts = key_str.split('_')
        layer_num = int(parts[0])
        feat_index = int(parts[1])

        api = fetched.get(key_str, {'status': 'not_fetched'})

        cantor_vals = reg.get('cantor_values', [])

        output_dict = {
            'layer_num': layer_num,
            'feature_index': feat_index,
            'cantor_paired_value': cantor_vals[0] if cantor_vals else None,
            'explanation': api.get('explanation', ''),
            'all_explanations': api.get('all_explanations', []),
            'frac_nonzero': api.get('frac_nonzero'),
            'max_activation': api.get('maxActApprox'),
            'top_positive_logits': api.get('pos_str', []),
            'top_positive_values': api.get('pos_values', []),
            'top_negative_logits': api.get('neg_str', []),
            'top_negative_values': api.get('neg_values', []),
            'has_explanation': api.get('has_explanation', False),
            'api_status': api.get('status', 'not_fetched'),
            'source_graphs': sorted(reg.get('graphs', [])),
            'source_domains': sorted(reg.get('domains', [])),
            'node_count_in_graphs': len(reg.get('node_ids', [])),
        }

        records.append({
            'input': key_str,
            'output': json.dumps(output_dict),
            'metadata_fold': 'feature_explanation',
            'metadata_layer': layer_num,
            'metadata_feature_index': feat_index,
            'metadata_has_explanation': api.get('has_explanation', False),
            'metadata_n_source_domains': len(reg.get('domains', [])),
            'metadata_source_domains': ','.join(sorted(reg.get('domains', []))),
            'metadata_frac_nonzero': api.get('frac_nonzero'),
            'metadata_max_activation': api.get('maxActApprox'),
        })

    n_with_expl = sum(1 for r in records if r['metadata_has_explanation'])
    all_domains = set()
    for r in records:
        for d in r['metadata_source_domains'].split(','):
            if d:
                all_domains.add(d)

    data_out = {
        "metadata": {
            "source": "Neuronpedia API GET /api/feature/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{index}",
            "model": "gemma-2-2b",
            "description": "Feature semantic explanations for CLT nodes in attribution graphs",
            "n_features_total": len(records),
            "n_with_explanation": n_with_expl,
            "n_source_graphs": 8,
            "source_domains": sorted(all_domains),
            "collection_date": "2026-03-19",
        },
        "datasets": [{
            "dataset": "neuronpedia_feature_explanations",
            "examples": records,
        }],
    }

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(data_out, indent=2))
    logger.info(f"Saved {len(records)} records to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info(f"Total features: {len(records)}")
    logger.info(f"With explanation: {n_with_expl} ({n_with_expl/len(records)*100:.1f}%)")

    layer_counts = Counter(r['metadata_layer'] for r in records)
    for layer in sorted(layer_counts):
        logger.info(f"  Layer {layer:2d}: {layer_counts[layer]:4d}")

    domain_counts = Counter()
    for r in records:
        for d in r['metadata_source_domains'].split(','):
            if d:
                domain_counts[d] += 1
    logger.info(f"Domain counts: {dict(domain_counts)}")

    status_counts = Counter(v.get('status') for v in fetched.values())
    logger.info(f"API status: {dict(status_counts)}")


if __name__ == "__main__":
    main()
