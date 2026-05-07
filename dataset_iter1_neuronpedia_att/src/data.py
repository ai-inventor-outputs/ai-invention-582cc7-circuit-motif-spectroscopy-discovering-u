#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Convert Neuronpedia attribution graph checkpoint to exp_sel_data_out.json schema.

Reads: temp/checkpoint_v3.json (collected attribution graphs)
Writes:
  - full_data_out.json (or data_out/full_data_out_*.json if splitting needed)
  - mini_data_out.json (3 examples total)
  - preview_data_out.json (3 examples with truncated strings)

Schema: {"datasets": [{"dataset": "...", "examples": [{"input": "...", "output": "...", "metadata_*": ...}]}]}
Each record = one attribution graph = one example.
- input: the prompt text
- output: JSON string of full graph data (nodes, links, stats)
- metadata_fold: domain name (str)
- metadata_n_nodes: node count (int)
- metadata_n_edges: edge count (int)
- metadata_density: graph density (float)
- metadata_is_dag: DAG property (bool)
- metadata_slug: Neuronpedia slug (str)
- metadata_task_type: "graph_generation" (str)
- metadata_n_classes: 8 (number of domains)
- metadata_row_index: original index in checkpoint (int)
- metadata_feature_names: list of node feature types found (list)
"""

import json
import os
import sys
import glob
import copy
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).parent
CHECKPOINT = WORKSPACE / "temp" / "checkpoint_v3.json"
DATASET_NAME = "neuronpedia_attribution_graphs"
FILE_SIZE_LIMIT_MB = 90  # Split if exceeds this (under 100MB hard limit)

METADATA_BLOCK = {
    "source": "Neuronpedia API (POST /api/graph/generate)",
    "model": "gemma-2-2b",
    "description": "Attribution graphs across 8 capability domains",
    "collection_params": {
        "nodeThreshold": 0.8,
        "edgeThreshold": 0.85,
        "maxFeatureNodes": 5000,
        "maxNLogits": 10,
        "desiredLogitProb": 0.95,
    },
}


def load_checkpoint():
    """Load the checkpoint file with all collected records."""
    if not CHECKPOINT.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        sys.exit(1)
    with open(CHECKPOINT) as f:
        data = json.load(f)
    records = data.get("records", [])
    print(f"Loaded {len(records)} records from checkpoint")
    return records


def convert_record(record, idx):
    """Convert a checkpoint record to exp_sel_data_out schema example."""
    prompt = record["input"]
    output_data = record["output"]
    domain = record["metadata_fold"]

    # Collect feature types present in nodes
    feature_types = set()
    for node in output_data.get("nodes", []):
        ft = node.get("feature_type", "")
        if ft:
            feature_types.add(ft)

    # Output is JSON-encoded string of the full graph data
    output_str = json.dumps(output_data, separators=(",", ":"))

    example = {
        "input": prompt,
        "output": output_str,
        "metadata_fold": domain,
        "metadata_n_nodes": output_data["n_nodes"],
        "metadata_n_edges": output_data["n_edges"],
        "metadata_density": output_data["density"],
        "metadata_is_dag": output_data["is_dag"],
        "metadata_slug": output_data.get("slug", ""),
        "metadata_task_type": "graph_generation",
        "metadata_n_classes": 8,
        "metadata_row_index": idx,
        "metadata_feature_names": sorted(feature_types),
    }
    return example


def print_summary(records):
    """Print summary statistics of the dataset."""
    domains = Counter(r["metadata_fold"] for r in records)
    print(f"\n{'='*70}")
    print(f"Dataset Summary: {len(records)} attribution graphs across {len(domains)} domains")
    print(f"{'='*70}")
    print(f"{'Domain':<25} {'Count':>6} {'Nodes (min-max)':>18} {'Edges (min-max)':>20} {'DAG%':>6}")
    print(f"{'-'*70}")

    for domain in sorted(domains.keys()):
        domain_records = [r for r in records if r["metadata_fold"] == domain]
        n_nodes_list = [r["output"]["n_nodes"] for r in domain_records]
        n_edges_list = [r["output"]["n_edges"] for r in domain_records]
        dag_pct = sum(1 for r in domain_records if r["output"]["is_dag"]) / len(domain_records) * 100
        print(
            f"{domain:<25} {len(domain_records):>6} "
            f"{min(n_nodes_list):>6}-{max(n_nodes_list):<6} "
            f"{min(n_edges_list):>8}-{max(n_edges_list):<8} "
            f"{dag_pct:>5.0f}%"
        )
    print(f"{'='*70}\n")


def wrap_schema(examples):
    """Wrap examples list in the exp_sel_data_out schema structure."""
    return {
        "metadata": METADATA_BLOCK,
        "datasets": [
            {
                "dataset": DATASET_NAME,
                "examples": examples,
            }
        ],
    }


def truncate_strings(obj, max_len=200):
    """Recursively truncate all strings in an object to max_len chars."""
    if isinstance(obj, str):
        return obj[:max_len] + "..." if len(obj) > max_len else obj
    elif isinstance(obj, dict):
        return {k: truncate_strings(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_strings(item, max_len) for item in obj]
    return obj


def write_full_output(examples):
    """Write full_data_out.json, splitting if file exceeds size limit."""
    result = wrap_schema(examples)
    output_str = json.dumps(result, separators=(",", ":"))
    output_size_mb = len(output_str) / (1024 * 1024)
    print(f"Full output size: {output_size_mb:.1f} MB")

    full_path = WORKSPACE / "full_data_out.json"

    if output_size_mb <= FILE_SIZE_LIMIT_MB:
        # Single file - write directly
        with open(full_path, "w") as f:
            f.write(output_str)
        print(f"Written: {full_path} ({output_size_mb:.1f} MB)")
        return [full_path], False
    else:
        # Split into parts under the limit
        split_dir = WORKSPACE / "data_out"
        split_dir.mkdir(exist_ok=True)

        # Calculate records per part to stay under limit
        avg_record_size = len(output_str) / max(len(examples), 1)
        records_per_part = max(1, int(FILE_SIZE_LIMIT_MB * 1024 * 1024 / avg_record_size * 0.85))
        print(f"Splitting: ~{records_per_part} records per part (avg {avg_record_size/1024/1024:.2f} MB/record)")

        part_paths = []
        part_num = 0
        for start in range(0, len(examples), records_per_part):
            part_num += 1
            chunk = examples[start : start + records_per_part]
            part_result = wrap_schema(chunk)
            part_path = split_dir / f"full_data_out_{part_num}.json"
            part_str = json.dumps(part_result, separators=(",", ":"))
            with open(part_path, "w") as f:
                f.write(part_str)
            part_paths.append(part_path)
            print(f"  Part {part_num}: {part_path.name} ({len(chunk)} examples, {len(part_str)/1024/1024:.1f} MB)")

        # Remove oversized combined file if it exists
        if full_path.exists():
            full_path.unlink()
            print(f"Removed oversized combined file: {full_path}")

        print(f"Split into {part_num} parts in: {split_dir}/")
        return part_paths, True


def write_mini_output(examples):
    """Write mini_data_out.json with first 3 examples."""
    mini_examples = examples[:3]
    result = wrap_schema(mini_examples)
    mini_path = WORKSPACE / "mini_data_out.json"
    with open(mini_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    size_mb = mini_path.stat().st_size / (1024 * 1024)
    print(f"Written: mini_data_out.json ({len(mini_examples)} examples, {size_mb:.1f} MB)")
    return mini_path


def write_preview_output(examples):
    """Write preview_data_out.json with first 3 examples, all strings truncated to 200 chars."""
    preview_examples = truncate_strings(examples[:3], max_len=200)
    result = wrap_schema(preview_examples)
    preview_path = WORKSPACE / "preview_data_out.json"
    with open(preview_path, "w") as f:
        json.dump(result, f, indent=2)
    size_kb = preview_path.stat().st_size / 1024
    print(f"Written: preview_data_out.json ({len(preview_examples)} examples, {size_kb:.1f} KB)")
    return preview_path


def write_split_mini_preview(split_paths):
    """For each split part, generate mini and preview versions."""
    split_dir = WORKSPACE / "data_out"
    for part_path in split_paths:
        with open(part_path) as f:
            part_data = json.load(f)
        examples = part_data["datasets"][0]["examples"]

        # Mini: first 3 examples
        mini_examples = examples[:3]
        mini_result = wrap_schema(mini_examples)
        mini_name = part_path.name.replace("full_", "mini_")
        mini_path = split_dir / mini_name
        with open(mini_path, "w") as f:
            json.dump(mini_result, f, separators=(",", ":"))
        size_mb = mini_path.stat().st_size / (1024 * 1024)
        print(f"  {mini_name} ({len(mini_examples)} examples, {size_mb:.1f} MB)")

        # Preview: first 3 examples, strings truncated
        preview_examples = truncate_strings(examples[:3], max_len=200)
        preview_result = wrap_schema(preview_examples)
        preview_name = part_path.name.replace("full_", "preview_")
        preview_path = split_dir / preview_name
        with open(preview_path, "w") as f:
            json.dump(preview_result, f, indent=2)
        size_kb = preview_path.stat().st_size / 1024
        print(f"  {preview_name} ({len(preview_examples)} examples, {size_kb:.1f} KB)")


def main():
    print("=" * 70)
    print("Neuronpedia Attribution Graph -> exp_sel_data_out.json converter")
    print("=" * 70)

    # Load checkpoint
    records = load_checkpoint()
    if not records:
        print("ERROR: No records in checkpoint")
        sys.exit(1)

    # Print summary
    print_summary(records)

    # Convert each record to an example
    examples = []
    for idx, record in enumerate(records):
        example = convert_record(record, idx)
        examples.append(example)

    print(f"Converted {len(examples)} examples")

    # Domain distribution
    domains = Counter(ex["metadata_fold"] for ex in examples)
    print(f"Domain distribution: {dict(sorted(domains.items()))}")

    # --- Write all output files ---

    # 1. Full output (may split)
    print(f"\n--- Full Output ---")
    full_paths, was_split = write_full_output(examples)

    # 2. Mini output (3 examples)
    print(f"\n--- Mini Output ---")
    write_mini_output(examples)

    # 3. Preview output (3 examples, truncated strings)
    print(f"\n--- Preview Output ---")
    write_preview_output(examples)

    # 4. If split, generate mini/preview for each part
    if was_split:
        print(f"\n--- Split Part Mini/Preview ---")
        write_split_mini_preview(full_paths)

    # Final summary
    print(f"\n{'='*70}")
    print(f"Done! {len(examples)} examples across {len(domains)} domains")
    if was_split:
        print(f"Full data split into {len(full_paths)} parts in data_out/")
        print(f"Load all: for f in sorted(glob.glob('data_out/full_data_out_*.json')):")
        print(f"            data.extend(json.load(open(f))['datasets'][0]['examples'])")
    else:
        print(f"Full data: full_data_out.json")
    print(f"Mini: mini_data_out.json (3 examples)")
    print(f"Preview: preview_data_out.json (3 examples, truncated)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
