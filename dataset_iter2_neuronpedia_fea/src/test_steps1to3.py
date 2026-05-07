#!/usr/bin/env python3
"""Quick test: Load graphs from mini files and extract features (Steps 1-3 only)."""

import json
import math
import gc
import sys
import time
from collections import Counter
from statistics import median
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

DEP_DIR = Path("/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/3_invention_loop/iter_1/gen_art/data_id4_it1__opus/data_out")

def cantor_decode(z: int) -> tuple[int, int]:
    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    t = (w * w + w) // 2
    feat_index = z - t
    layer_num = w - feat_index
    return (int(layer_num), int(feat_index))

# Test cantor decode
assert cantor_decode(902) == (0, 41), f"Expected (0,41) got {cantor_decode(902)}"
assert cantor_decode(4752) == (0, 96), f"Expected (0,96) got {cantor_decode(4752)}"
logger.info("Cantor decode verified: 902->(0,41), 4752->(0,96)")

# Load ONE mini file to test structure
logger.info("Loading mini_data_out_1.json...")
t0 = time.time()
with open(DEP_DIR / "mini_data_out_1.json") as f:
    data = json.load(f)

examples = data['datasets'][0]['examples']
logger.info(f"Loaded {len(examples)} examples in {time.time()-t0:.1f}s")

# Inspect structure of first graph
g0 = examples[0]
logger.info(f"Example keys: {list(g0.keys())}")
logger.info(f"fold={g0['metadata_fold']}, n_nodes={g0['metadata_n_nodes']}, slug={g0['metadata_slug']}")

# Parse output JSON
graph_data = json.loads(g0['output'])
logger.info(f"Graph data keys: {list(graph_data.keys())}")
logger.info(f"Nodes count: {len(graph_data['nodes'])}")
if 'links' in graph_data:
    logger.info(f"Links count: {len(graph_data['links'])}")

# Inspect first few nodes
node0 = graph_data['nodes'][0]
logger.info(f"Node keys: {list(node0.keys())}")
logger.info(f"Node 0: {json.dumps(node0, indent=2)[:500]}")

# Count feature types
ft_counts = Counter(n.get('feature_type') for n in graph_data['nodes'])
logger.info(f"Feature types: {dict(ft_counts)}")

# Extract CLT features from all examples in mini file
all_folds = set()
feature_registry = {}
for ex in examples:
    fold = ex['metadata_fold']
    all_folds.add(fold)
    gd = json.loads(ex['output'])
    clt_count = 0
    for node in gd['nodes']:
        if node.get('feature_type') != 'cross layer transcoder':
            continue
        if node.get('feature') is None:
            continue
        clt_count += 1
        layer_num, feat_index = cantor_decode(node['feature'])
        key = (layer_num, feat_index)
        if key not in feature_registry:
            feature_registry[key] = {'domains': set(), 'graphs': set(), 'node_ids': set(), 'cantor_values': set()}
        feature_registry[key]['domains'].add(fold)
        feature_registry[key]['graphs'].add(ex['metadata_slug'])
        feature_registry[key]['node_ids'].add(node['node_id'])
        feature_registry[key]['cantor_values'].add(node['feature'])
    logger.info(f"  {fold}: {clt_count} CLT nodes, {ex['metadata_n_nodes']} total nodes")

logger.info(f"Folds seen: {sorted(all_folds)}")
logger.info(f"Unique features from mini file: {len(feature_registry)}")

# Layer distribution
layer_counts = Counter(k[0] for k in feature_registry)
for layer in sorted(layer_counts):
    logger.info(f"  Layer {layer:2d}: {layer_counts[layer]:4d} features")

# Verify some decoded values match node_id format
for node in graph_data['nodes'][:10]:
    if node.get('feature_type') == 'cross layer transcoder' and node.get('feature') is not None:
        l, f = cantor_decode(node['feature'])
        parts = node['node_id'].split('_')
        logger.info(f"  node_id={node['node_id']}, feature={node['feature']}, decoded=({l},{f}), node_id_parts={parts}")

del data, graph_data
gc.collect()
logger.info("Test complete!")
