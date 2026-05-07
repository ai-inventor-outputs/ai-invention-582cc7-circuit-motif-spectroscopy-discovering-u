#!/usr/bin/env python3
"""Convert Neuronpedia attribution graph checkpoint to exp_sel_data_out.json schema (iter2).

Reads: temp/checkpoint_v4.json
Writes:
  - full_data_out.json (or data_out/full_data_out_*.json if split needed)
  - mini_data_out.json (3 examples)
  - preview_data_out.json (3 examples with truncated strings)

Schema extends iter1 with: metadata_model_correct, metadata_difficulty, metadata_expected_answer, metadata_iter
"""

import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).parent
CHECKPOINT = WORKSPACE / "temp" / "checkpoint_v4_snapshot.json"
DATASET_NAME = "neuronpedia_attribution_graphs_v2"
FILE_SIZE_LIMIT_MB = 90

METADATA_BLOCK = {
    "source": "Neuronpedia API (POST /api/graph/generate)",
    "model": "gemma-2-2b",
    "description": "Attribution graphs across 8 capability domains with correctness annotations",
    "iter": 2,
    "collection_params": {
        "nodeThreshold": 0.8,
        "edgeThreshold": 0.85,
        "maxFeatureNodes": 5000,
        "maxNLogits": 10,
        "desiredLogitProb": 0.95,
    },
    "compatible_with_iter1": True,
}


def load_checkpoint() -> list[dict]:
    """Load checkpoint file."""
    if not CHECKPOINT.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        sys.exit(1)
    data = json.loads(CHECKPOINT.read_text())
    records = data.get("records", [])
    print(f"Loaded {len(records)} records from checkpoint")
    return records


def convert_record(record: dict, idx: int) -> dict:
    """Convert checkpoint record to output schema example."""
    prompt = record["input"]
    output_data = record["output"]
    domain = record["metadata_fold"]

    # Collect feature types
    feature_types = set()
    for node in output_data.get("nodes", []):
        ft = node.get("feature_type", "")
        if ft:
            feature_types.add(ft)

    # JSON-encode output
    output_str = json.dumps(output_data, separators=(",", ":"))

    return {
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
        # New iter2 fields
        "metadata_model_correct": record.get("metadata_model_correct", "unknown"),
        "metadata_difficulty": record.get("metadata_difficulty", "medium"),
        "metadata_expected_answer": record.get("metadata_expected_answer", ""),
        "metadata_iter": 2,
    }


def wrap_schema(examples: list[dict]) -> dict:
    """Wrap examples in exp_sel_data_out schema."""
    return {
        "metadata": METADATA_BLOCK,
        "datasets": [
            {
                "dataset": DATASET_NAME,
                "examples": examples,
            }
        ],
    }


def truncate_strings(obj, max_len: int = 200):
    """Recursively truncate strings."""
    if isinstance(obj, str):
        return obj[:max_len] + "..." if len(obj) > max_len else obj
    elif isinstance(obj, dict):
        return {k: truncate_strings(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_strings(item, max_len) for item in obj]
    return obj


def write_full_output(examples: list[dict]) -> tuple[list[Path], bool]:
    """Write full output, splitting if >90MB."""
    result = wrap_schema(examples)
    output_str = json.dumps(result, separators=(",", ":"))
    output_size_mb = len(output_str) / (1024 * 1024)
    print(f"Full output size: {output_size_mb:.1f} MB")

    full_path = WORKSPACE / "full_data_out.json"

    if output_size_mb <= FILE_SIZE_LIMIT_MB:
        full_path.write_text(output_str)
        print(f"Written: {full_path} ({output_size_mb:.1f} MB)")
        return [full_path], False
    else:
        split_dir = WORKSPACE / "data_out"
        split_dir.mkdir(exist_ok=True)

        avg_record_size = len(output_str) / max(len(examples), 1)
        records_per_part = max(1, int(FILE_SIZE_LIMIT_MB * 1024 * 1024 / avg_record_size * 0.85))
        print(f"Splitting: ~{records_per_part} records per part (avg {avg_record_size/1024/1024:.2f} MB/record)")

        part_paths = []
        part_num = 0
        for start in range(0, len(examples), records_per_part):
            part_num += 1
            chunk = examples[start: start + records_per_part]
            part_result = wrap_schema(chunk)
            part_path = split_dir / f"full_data_out_{part_num}.json"
            part_str = json.dumps(part_result, separators=(",", ":"))
            part_path.write_text(part_str)
            part_paths.append(part_path)
            print(f"  Part {part_num}: {part_path.name} ({len(chunk)} examples, {len(part_str)/1024/1024:.1f} MB)")

        if full_path.exists():
            full_path.unlink()
        print(f"Split into {part_num} parts in: {split_dir}/")
        return part_paths, True


def write_mini_output(examples: list[dict]) -> Path:
    """Write mini_data_out.json with first 3 examples."""
    mini_examples = examples[:3]
    result = wrap_schema(mini_examples)
    mini_path = WORKSPACE / "mini_data_out.json"
    mini_path.write_text(json.dumps(result, separators=(",", ":")))
    size_mb = mini_path.stat().st_size / (1024 * 1024)
    print(f"Written: mini_data_out.json ({len(mini_examples)} examples, {size_mb:.1f} MB)")
    return mini_path


def write_preview_output(examples: list[dict]) -> Path:
    """Write preview_data_out.json with first 3 examples, truncated."""
    preview_examples = truncate_strings(examples[:3], max_len=200)
    result = wrap_schema(preview_examples)
    preview_path = WORKSPACE / "preview_data_out.json"
    preview_path.write_text(json.dumps(result, indent=2))
    size_kb = preview_path.stat().st_size / 1024
    print(f"Written: preview_data_out.json ({len(preview_examples)} examples, {size_kb:.1f} KB)")
    return preview_path


def write_split_mini_preview(split_paths: list[Path]) -> None:
    """For each split part, generate mini and preview versions."""
    split_dir = WORKSPACE / "data_out"
    for part_path in split_paths:
        data = json.loads(part_path.read_text())
        examples = data["datasets"][0]["examples"]

        mini_examples = examples[:3]
        mini_result = wrap_schema(mini_examples)
        mini_name = part_path.name.replace("full_", "mini_")
        mini_path = split_dir / mini_name
        mini_path.write_text(json.dumps(mini_result, separators=(",", ":")))
        size_mb = mini_path.stat().st_size / (1024 * 1024)
        print(f"  {mini_name} ({len(mini_examples)} examples, {size_mb:.1f} MB)")

        preview_examples = truncate_strings(examples[:3], max_len=200)
        preview_result = wrap_schema(preview_examples)
        preview_name = part_path.name.replace("full_", "preview_")
        preview_path = split_dir / preview_name
        preview_path.write_text(json.dumps(preview_result, indent=2))
        size_kb = preview_path.stat().st_size / 1024
        print(f"  {preview_name} ({len(preview_examples)} examples, {size_kb:.1f} KB)")


def validate_dataset(examples: list[dict]) -> bool:
    """Validate the final dataset against quality criteria."""
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    issues = []
    total = len(examples)
    print(f"Total examples: {total}")

    # Domain distribution
    domains = Counter(ex["metadata_fold"] for ex in examples)
    print(f"\nDomain distribution:")
    for d in sorted(domains):
        cnt = domains[d]
        flag = " ***" if cnt < 5 else ""
        print(f"  {d:<25} {cnt:>4}{flag}")

    # DAG check
    dag_count = sum(1 for ex in examples if ex["metadata_is_dag"])
    print(f"\nDAGs: {dag_count}/{total} ({dag_count/total*100:.1f}%)")

    # Node count check
    min_nodes = min(ex["metadata_n_nodes"] for ex in examples)
    max_nodes = max(ex["metadata_n_nodes"] for ex in examples)
    print(f"Nodes: min={min_nodes}, max={max_nodes}")
    if min_nodes < 10:
        issues.append(f"Some graphs have < 10 nodes (min={min_nodes})")

    # Metadata completeness
    null_correct = sum(1 for ex in examples if not ex.get("metadata_model_correct"))
    null_diff = sum(1 for ex in examples if not ex.get("metadata_difficulty"))
    null_answer = sum(1 for ex in examples if not ex.get("metadata_expected_answer"))
    print(f"\nMetadata completeness:")
    print(f"  model_correct: {total - null_correct}/{total} non-null")
    print(f"  difficulty: {total - null_diff}/{total} non-null")
    print(f"  expected_answer: {total - null_answer}/{total} non-null")

    # Difficulty distribution
    difficulties = Counter(ex["metadata_difficulty"] for ex in examples)
    print(f"\nDifficulty distribution: {dict(sorted(difficulties.items()))}")

    # Model correctness distribution
    correctness = Counter(ex["metadata_model_correct"] for ex in examples)
    print(f"Correctness distribution: {dict(sorted(correctness.items()))}")

    # Feature types
    all_ftypes = set()
    for ex in examples:
        for ft in ex.get("metadata_feature_names", []):
            all_ftypes.add(ft)
    print(f"Feature types across dataset: {sorted(all_ftypes)}")

    if issues:
        print(f"\nISSUES: {issues}")
    else:
        print(f"\nAll validation checks passed!")

    print("=" * 70)
    return len(issues) == 0


def main():
    print("=" * 70)
    print("Neuronpedia Attribution Graph -> exp_sel_data_out.json (iter2)")
    print("=" * 70)

    records = load_checkpoint()
    if not records:
        print("ERROR: No records in checkpoint")
        sys.exit(1)

    # Convert
    examples = [convert_record(r, idx) for idx, r in enumerate(records)]
    print(f"Converted {len(examples)} examples")

    # Validate
    validate_dataset(examples)

    # Write outputs
    print(f"\n--- Full Output ---")
    full_paths, was_split = write_full_output(examples)

    print(f"\n--- Mini Output ---")
    write_mini_output(examples)

    print(f"\n--- Preview Output ---")
    write_preview_output(examples)

    if was_split:
        print(f"\n--- Split Part Mini/Preview ---")
        write_split_mini_preview(full_paths)

    # Final
    domains = Counter(ex["metadata_fold"] for ex in examples)
    print(f"\n{'='*70}")
    print(f"Done! {len(examples)} examples across {len(domains)} domains")
    if was_split:
        print(f"Full data split into {len(full_paths)} parts in data_out/")
    else:
        print(f"Full data: full_data_out.json")
    print(f"Mini: mini_data_out.json | Preview: preview_data_out.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
