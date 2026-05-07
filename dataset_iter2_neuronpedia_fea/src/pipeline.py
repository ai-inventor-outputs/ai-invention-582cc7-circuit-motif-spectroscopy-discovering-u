#!/usr/bin/env python3
"""Build Neuronpedia feature explanation lookup table from attribution graph nodes.

Loads 34 attribution graphs from dependency files, selects 8 representative graphs
(one per capability domain), extracts unique CLT features via Cantor decode, calls
Neuronpedia API for feature explanations, and outputs a structured dataset.
"""

import asyncio
import gc
import json
import math
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import aiohttp
from loguru import logger

# ── Workspace paths ──────────────────────────────────────────────────────────
WORKSPACE = Path("/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/3_invention_loop/iter_2/gen_art/data_id4_it2__opus")
TEMP_DIR = WORKSPACE / "temp"
DATA_OUT_DIR = WORKSPACE / "data_out"
LOG_DIR = WORKSPACE / "logs"

TEMP_DIR.mkdir(exist_ok=True)
DATA_OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "pipeline.log"), rotation="30 MB", level="DEBUG")

# ── Memory safety ────────────────────────────────────────────────────────────
# Container limit: 29 GB. Use up to 20 GB for safety.
RAM_BUDGET = 20 * 1024**3  # 20 GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU

# ── API config ───────────────────────────────────────────────────────────────
API_KEY = os.environ.get("NEURONPEDIA_API_KEY", "sk-np-DQRQw4Us2QtJgy0kq9nZOz39qVIJ0kpy7d8ymN1Ica80")
BASE_URL = "https://www.neuronpedia.org/api/feature/gemma-2-2b"
CHECKPOINT_FILE = TEMP_DIR / "feature_checkpoint.json"

# ── Dependency file paths ────────────────────────────────────────────────────
DEP_DIR = Path("/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/3_invention_loop/iter_1/gen_art/data_id4_it1__opus/data_out")
SOURCE_FILES = [
    DEP_DIR / "full_data_out_1.json",
    DEP_DIR / "full_data_out_2.json",
    DEP_DIR / "full_data_out_3.json",
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load graphs from dependency files (one at a time for memory safety)
# ══════════════════════════════════════════════════════════════════════════════

def load_all_graphs() -> list[dict]:
    """Load all graphs from 3 dependency files, keeping only metadata + nodes."""
    all_graphs = []

    for fpath in SOURCE_FILES:
        logger.info(f"Loading {fpath.name} ({fpath.stat().st_size / 1e6:.1f} MB)...")
        t0 = time.time()

        with open(fpath, 'r') as f:
            data = json.load(f)

        examples = data['datasets'][0]['examples']
        logger.info(f"  Found {len(examples)} graphs in {fpath.name}")

        for example in examples:
            try:
                graph_data = json.loads(example['output'])
                all_graphs.append({
                    'input': example['input'],
                    'fold': example['metadata_fold'],
                    'n_nodes': example['metadata_n_nodes'],
                    'slug': example['metadata_slug'],
                    'row_index': example['metadata_row_index'],
                    'nodes': graph_data['nodes'],
                    # Do NOT store links — saves massive memory
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"  Skipping graph {example.get('metadata_slug', '?')}: {e}")

        del data, examples
        gc.collect()
        logger.info(f"  Done in {time.time() - t0:.1f}s, total graphs: {len(all_graphs)}")

    return all_graphs


def print_graph_summary(all_graphs: list[dict]) -> None:
    """Print summary statistics about loaded graphs."""
    folds = Counter(g['fold'] for g in all_graphs)
    logger.info(f"Total graphs loaded: {len(all_graphs)}")
    logger.info(f"Folds ({len(folds)}): {dict(folds)}")
    for fold in sorted(folds):
        nodes = [g['n_nodes'] for g in all_graphs if g['fold'] == fold]
        logger.info(f"  {fold}: {folds[fold]} graphs, nodes range [{min(nodes)}, {max(nodes)}], median={median(nodes):.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Select representative graphs (one per domain, median-sized)
# ══════════════════════════════════════════════════════════════════════════════

def select_representative_graphs(all_graphs: list[dict]) -> dict[str, dict]:
    """Select one graph per fold with median node count."""
    selected = {}
    folds = sorted(set(g['fold'] for g in all_graphs))

    for fold in folds:
        fold_graphs = [g for g in all_graphs if g['fold'] == fold]
        med = median(g['n_nodes'] for g in fold_graphs)
        best = min(fold_graphs, key=lambda g: abs(g['n_nodes'] - med))
        selected[fold] = best
        logger.info(f"  Selected {fold}: slug={best['slug']}, nodes={best['n_nodes']} (median={med:.0f})")

    # If fewer than 8 folds, add extra graphs from larger folds
    if len(selected) < 8:
        logger.warning(f"Only {len(selected)} folds found. Adding extras from larger folds.")
        remaining = sorted(folds, key=lambda f: -len([g for g in all_graphs if g['fold'] == f]))
        for fold in remaining:
            if len(selected) >= 8:
                break
            fold_graphs = [g for g in all_graphs if g['fold'] == fold and g['slug'] != selected[fold]['slug']]
            if fold_graphs:
                extra = fold_graphs[0]
                extra_key = f"{fold}_extra"
                selected[extra_key] = extra
                logger.info(f"  Added extra from {fold}: slug={extra['slug']}, nodes={extra['n_nodes']}")

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Extract and deduplicate CLT features via Cantor decode
# ══════════════════════════════════════════════════════════════════════════════

def cantor_decode(z: int) -> tuple[int, int]:
    """Decode Cantor-paired integer to (layer_num, feature_index).

    Verified: 902 → (0, 41), 4752 → (0, 96)
    """
    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    t = (w * w + w) // 2
    feat_index = z - t
    layer_num = w - feat_index
    return (int(layer_num), int(feat_index))


def extract_features(selected_graphs: dict[str, dict]) -> dict[tuple[int, int], dict]:
    """Extract unique CLT features from selected graphs."""
    feature_registry: dict[tuple[int, int], dict] = {}
    invalid_count = 0
    total_clt_nodes = 0

    for fold, graph in selected_graphs.items():
        fold_features = 0
        for node in graph['nodes']:
            if node.get('feature_type') != 'cross layer transcoder':
                continue
            if node.get('feature') is None:
                continue

            total_clt_nodes += 1
            try:
                layer_num, feat_index = cantor_decode(node['feature'])
            except (ValueError, OverflowError):
                invalid_count += 1
                continue

            # Validate ranges
            if not (0 <= layer_num <= 25) or not (0 <= feat_index <= 16383):
                invalid_count += 1
                logger.debug(f"  Invalid decode: feature={node['feature']} -> layer={layer_num}, idx={feat_index}")
                continue

            key = (layer_num, feat_index)
            if key not in feature_registry:
                feature_registry[key] = {
                    'graphs': set(),
                    'domains': set(),
                    'node_ids': set(),
                    'cantor_values': set(),
                }

            feature_registry[key]['graphs'].add(graph['slug'])
            feature_registry[key]['domains'].add(fold)
            feature_registry[key]['node_ids'].add(node['node_id'])
            feature_registry[key]['cantor_values'].add(node['feature'])
            fold_features += 1

        logger.info(f"  {fold}: {fold_features} CLT nodes extracted")

    logger.info(f"Total CLT nodes: {total_clt_nodes}, unique features: {len(feature_registry)}, invalid: {invalid_count}")

    # Layer distribution
    layer_counts = Counter(k[0] for k in feature_registry)
    logger.info("Layer distribution:")
    for layer in sorted(layer_counts):
        logger.info(f"  Layer {layer:2d}: {layer_counts[layer]:4d} features")

    return feature_registry


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: API retrieval with rate limiting and checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    """Load existing checkpoint data."""
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            logger.info(f"Loaded checkpoint: {len(data)} features already fetched")
            return data
        except json.JSONDecodeError:
            logger.warning("Corrupt checkpoint, starting fresh")
    return {}


def save_checkpoint(fetched: dict) -> None:
    """Save checkpoint data."""
    CHECKPOINT_FILE.write_text(json.dumps(fetched))


def parse_api_response(d: dict) -> dict:
    """Parse a successful Neuronpedia API response into compact format."""
    explanations = d.get('explanations') or []
    return {
        'status': 'ok',
        'explanation': explanations[0].get('description', '') if explanations else '',
        'all_explanations': [e.get('description', '') for e in explanations],
        'frac_nonzero': d.get('frac_nonzero'),
        'maxActApprox': d.get('maxActApprox'),
        'pos_str': (d.get('pos_str') or [])[:10],
        'pos_values': (d.get('pos_values') or [])[:10],
        'neg_str': (d.get('neg_str') or [])[:10],
        'neg_values': (d.get('neg_values') or [])[:10],
        'has_explanation': bool(explanations),
        'hookName': d.get('hookName', ''),
    }


async def fetch_feature(
    session: aiohttp.ClientSession,
    layer_num: int,
    feat_index: int,
    semaphore: asyncio.Semaphore,
    rate_delay: float = 0.5,
) -> tuple[str, dict]:
    """Fetch a single feature from Neuronpedia API."""
    key = f"{layer_num}_{feat_index}"
    url = f"{BASE_URL}/{layer_num}-gemmascope-transcoder-16k/{feat_index}"
    headers = {'x-api-key': API_KEY}

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        d = await resp.json()
                        await asyncio.sleep(rate_delay)
                        return key, parse_api_response(d)
                    elif resp.status == 404:
                        await asyncio.sleep(rate_delay)
                        return key, {'status': '404', 'has_explanation': False}
                    elif resp.status == 429:
                        wait = int(resp.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited on {key}, waiting {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        await asyncio.sleep(rate_delay)
                        return key, {'status': f'error_{resp.status}'}
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.debug(f"Attempt {attempt+1} failed for {key}: {e}")
                await asyncio.sleep(2 ** attempt)

        return key, {'status': 'exception', 'error': 'max retries exceeded'}


async def mini_batch_validation(features_sample: list[tuple[int, int]]) -> tuple[dict, bool]:
    """Validate API with 20 sample features. Returns results and success flag."""
    logger.info(f"Mini-batch validation: {len(features_sample)} features...")
    fetched = {}

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(1)  # 1 concurrent for validation
        tasks = [fetch_feature(session, l, f, sem, rate_delay=1.0) for l, f in features_sample]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    ok_count = 0
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"Mini-batch exception: {r}")
            continue
        key, data = r
        fetched[key] = data
        if data.get('status') == 'ok':
            ok_count += 1
            if data.get('has_explanation'):
                logger.info(f"  {key}: explanation='{data['explanation'][:80]}...'")
            else:
                logger.info(f"  {key}: no explanation, frac_nonzero={data.get('frac_nonzero')}")

    success_rate = ok_count / len(features_sample) if features_sample else 0
    logger.info(f"Mini-batch: {ok_count}/{len(features_sample)} OK ({success_rate:.0%})")

    return fetched, success_rate >= 0.5


async def full_batch_fetch(
    features_list: list[tuple[int, int]],
    existing_fetched: dict,
) -> dict:
    """Fetch all remaining features with checkpointing."""
    fetched = dict(existing_fetched)

    # Filter out already-fetched features
    remaining = [(l, f) for l, f in features_list if f"{l}_{f}" not in fetched]
    logger.info(f"Full batch: {len(remaining)} features to fetch ({len(fetched)} already done)")

    if not remaining:
        return fetched

    # Use 2 concurrent connections with 0.5s delay = ~2-3 req/sec effective
    concurrency = 2
    rate_delay = 0.5
    batch_size = 100  # checkpoint every 100

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(concurrency)

        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            tasks = [fetch_feature(session, l, f, sem, rate_delay=rate_delay) for l, f in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            errors_in_batch = 0
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Batch exception: {r}")
                    errors_in_batch += 1
                    continue
                key, data = r
                fetched[key] = data
                if data.get('status') not in ('ok', '404'):
                    errors_in_batch += 1

            # Checkpoint
            save_checkpoint(fetched)
            ok_count = sum(1 for v in fetched.values() if v.get('status') == 'ok')
            not_found = sum(1 for v in fetched.values() if v.get('status') == '404')
            progress = batch_start + len(batch)
            logger.info(
                f"Progress: {progress}/{len(remaining)} | "
                f"OK: {ok_count}, 404: {not_found}, errors: {errors_in_batch}"
            )

            # Back off on high error rates
            if errors_in_batch > len(batch) * 0.5:
                logger.warning("High error rate, backing off 30s...")
                await asyncio.sleep(30)

    return fetched


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Build output dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_output_dataset(
    feature_registry: dict[tuple[int, int], dict],
    fetched: dict,
    selected_graphs: dict[str, dict],
) -> dict:
    """Build the final output dataset in the required schema format."""
    records = []

    for (layer_num, feat_index), reg in sorted(feature_registry.items()):
        key = f"{layer_num}_{feat_index}"
        api = fetched.get(key, {'status': 'not_fetched'})

        cantor_vals = list(reg.get('cantor_values', set()))

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
            'source_graphs': sorted(reg['graphs']),
            'source_domains': sorted(reg['domains']),
            'node_count_in_graphs': len(reg['node_ids']),
        }

        records.append({
            'input': key,
            'output': json.dumps(output_dict),
            'metadata_fold': 'feature_explanation',
            'metadata_layer': layer_num,
            'metadata_feature_index': feat_index,
            'metadata_has_explanation': api.get('has_explanation', False),
            'metadata_n_source_domains': len(reg['domains']),
            'metadata_source_domains': ','.join(sorted(reg['domains'])),
            'metadata_frac_nonzero': api.get('frac_nonzero'),
            'metadata_max_activation': api.get('maxActApprox'),
        })

    n_with_expl = sum(1 for r in records if r['metadata_has_explanation'])

    data_out = {
        "metadata": {
            "source": "Neuronpedia API GET /api/feature/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{index}",
            "model": "gemma-2-2b",
            "description": "Feature semantic explanations for CLT nodes in attribution graphs",
            "n_features_total": len(records),
            "n_with_explanation": n_with_expl,
            "n_source_graphs": len(selected_graphs),
            "source_domains": sorted(selected_graphs.keys()),
            "collection_date": "2026-03-19",
        },
        "datasets": [{
            "dataset": "neuronpedia_feature_explanations",
            "examples": records,
        }],
    }

    logger.info(f"Built dataset: {len(records)} records, {n_with_expl} with explanations ({n_with_expl/len(records)*100:.1f}%)")
    return data_out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(data_out: dict, fetched: dict) -> None:
    """Print validation summary statistics."""
    records = data_out['datasets'][0]['examples']
    n_total = len(records)

    # Explanation coverage
    n_expl = sum(1 for r in records if r['metadata_has_explanation'])
    logger.info(f"=== VALIDATION ===")
    logger.info(f"Total features: {n_total}")
    logger.info(f"With explanation: {n_expl} ({n_expl/n_total*100:.1f}%) — target >=80%")

    # Layer distribution
    layer_counts = Counter(r['metadata_layer'] for r in records)
    logger.info("Layer distribution:")
    for layer in sorted(layer_counts):
        logger.info(f"  Layer {layer:2d}: {layer_counts[layer]:4d} features")

    # Domain coverage
    domain_counts = Counter()
    multi_domain = 0
    for r in records:
        n_domains = r['metadata_n_source_domains']
        if n_domains > 1:
            multi_domain += 1
        for d in r['metadata_source_domains'].split(','):
            if d:
                domain_counts[d] += 1
    logger.info(f"Features in 2+ domains: {multi_domain} ({multi_domain/n_total*100:.1f}%)")
    logger.info(f"Domain feature counts: {dict(domain_counts)}")

    # API status breakdown
    status_counts = Counter(v.get('status', 'unknown') for v in fetched.values())
    logger.info(f"API status breakdown: {dict(status_counts)}")

    # Sparsity distribution
    frac_vals = [r['metadata_frac_nonzero'] for r in records if r['metadata_frac_nonzero'] is not None]
    if frac_vals:
        logger.info(f"Sparsity (frac_nonzero): min={min(frac_vals):.6f}, median={sorted(frac_vals)[len(frac_vals)//2]:.6f}, max={max(frac_vals):.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Split into full/mini/preview
# ══════════════════════════════════════════════════════════════════════════════

def save_dataset_splits(data_out: dict) -> None:
    """Save full/mini/preview versions of the dataset."""
    records = data_out['datasets'][0]['examples']

    # Full
    full_path = DATA_OUT_DIR / "data_out.json"
    full_path.write_text(json.dumps(data_out, indent=2))
    logger.info(f"Saved full dataset: {full_path} ({full_path.stat().st_size / 1e6:.1f} MB)")

    # Mini: stratified by layer — up to 200 records
    layers = sorted(set(r['metadata_layer'] for r in records))
    per_layer = max(1, 200 // len(layers)) if layers else 200
    mini_records = []
    for layer in layers:
        layer_records = [r for r in records if r['metadata_layer'] == layer]
        # Prefer records with explanations
        with_expl = [r for r in layer_records if r['metadata_has_explanation']]
        without_expl = [r for r in layer_records if not r['metadata_has_explanation']]
        selected = (with_expl + without_expl)[:per_layer]
        mini_records.extend(selected)

    mini_records = mini_records[:200]
    mini_out = dict(data_out)
    mini_out = {
        "metadata": {**data_out['metadata'], "split": "mini", "n_features_total": len(mini_records)},
        "datasets": [{"dataset": "neuronpedia_feature_explanations", "examples": mini_records}],
    }

    # Preview: 24 records — 3 per domain with explanations
    preview_records = []
    all_domains = set()
    for r in records:
        for d in r['metadata_source_domains'].split(','):
            if d:
                all_domains.add(d)

    for domain in sorted(all_domains):
        domain_recs = [r for r in records if domain in r['metadata_source_domains'] and r['metadata_has_explanation']]
        if not domain_recs:
            domain_recs = [r for r in records if domain in r['metadata_source_domains']]
        preview_records.extend(domain_recs[:3])

    preview_records = preview_records[:24]
    preview_out = {
        "metadata": {**data_out['metadata'], "split": "preview", "n_features_total": len(preview_records)},
        "datasets": [{"dataset": "neuronpedia_feature_explanations", "examples": preview_records}],
    }

    # Truncate long strings in preview for readability
    def truncate_strings(obj, max_len=200):
        if isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + "..."
        elif isinstance(obj, dict):
            return {k: truncate_strings(v, max_len) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [truncate_strings(item, max_len) for item in obj]
        return obj

    preview_out_truncated = truncate_strings(preview_out)

    # Save mini and preview
    mini_path = DATA_OUT_DIR / "mini_data_out.json"
    mini_path.write_text(json.dumps(mini_out, indent=2))
    logger.info(f"Saved mini dataset: {mini_path} ({mini_path.stat().st_size / 1e6:.1f} MB, {len(mini_records)} records)")

    preview_path = DATA_OUT_DIR / "preview_data_out.json"
    preview_path.write_text(json.dumps(preview_out_truncated, indent=2))
    logger.info(f"Saved preview dataset: {preview_path} ({preview_path.stat().st_size / 1e6:.1f} MB, {len(preview_records)} records)")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("Neuronpedia Feature Explanation Lookup Table Pipeline")
    logger.info("=" * 70)

    # ── STEP 1: Load graphs ──────────────────────────────────────────────
    logger.info("STEP 1: Loading attribution graphs from dependency files...")
    all_graphs = load_all_graphs()
    print_graph_summary(all_graphs)

    # ── STEP 2: Select representative graphs ─────────────────────────────
    logger.info("STEP 2: Selecting representative graphs (one per domain)...")
    selected_graphs = select_representative_graphs(all_graphs)
    logger.info(f"Selected {len(selected_graphs)} graphs")

    # ── STEP 3: Extract features ─────────────────────────────────────────
    logger.info("STEP 3: Extracting CLT features via Cantor decode...")
    feature_registry = extract_features(selected_graphs)

    # Free graph data - we only need the registry now
    del all_graphs
    gc.collect()

    # Save feature registry for debugging
    registry_serializable = {
        f"{l}_{f}": {
            'graphs': sorted(v['graphs']),
            'domains': sorted(v['domains']),
            'node_ids': sorted(v['node_ids']),
            'cantor_values': sorted(v['cantor_values']),
        }
        for (l, f), v in feature_registry.items()
    }
    (TEMP_DIR / "feature_registry.json").write_text(json.dumps(registry_serializable, indent=2))
    logger.info(f"Saved feature registry to temp/feature_registry.json")

    # ── STEP 4: API retrieval ────────────────────────────────────────────
    logger.info("STEP 4: Fetching feature explanations from Neuronpedia API...")

    # Load existing checkpoint
    fetched = load_checkpoint()

    # Phase 4a: Mini-batch validation
    all_features = sorted(feature_registry.keys())
    # Pick 20 features spread across layers
    sample_layers = sorted(set(l for l, _ in all_features))
    sample_features = []
    for layer in sample_layers[:6]:  # First 6 unique layers
        layer_feats = [(l, f) for l, f in all_features if l == layer]
        sample_features.extend(layer_feats[:4])  # Up to 4 per layer
    sample_features = sample_features[:20]

    mini_fetched, api_ok = asyncio.run(mini_batch_validation(sample_features))
    fetched.update(mini_fetched)
    save_checkpoint(fetched)

    if not api_ok:
        logger.error("Mini-batch validation failed! API may be down.")
        logger.info("Attempting to continue with available data...")

    # Phase 4b: Full batch
    logger.info(f"Starting full batch: {len(all_features)} total features...")
    fetched = asyncio.run(full_batch_fetch(all_features, fetched))
    save_checkpoint(fetched)

    elapsed = time.time() - t_start
    logger.info(f"API retrieval complete in {elapsed/60:.1f} minutes")

    # ── STEP 5: Build output dataset ─────────────────────────────────────
    logger.info("STEP 5: Building output dataset...")
    data_out = build_output_dataset(feature_registry, fetched, selected_graphs)

    # ── STEP 6: Validation ───────────────────────────────────────────────
    logger.info("STEP 6: Validating dataset...")
    validate_dataset(data_out, fetched)

    # ── STEP 7: Save splits ──────────────────────────────────────────────
    logger.info("STEP 7: Saving dataset splits...")
    save_dataset_splits(data_out)

    elapsed = time.time() - t_start
    logger.info(f"Pipeline complete in {elapsed/60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
