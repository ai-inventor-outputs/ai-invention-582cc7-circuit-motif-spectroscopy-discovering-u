# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "networkx==3.4.2",
#   "requests==2.32.3",
#   "loguru==0.7.3",
# ]
# ///
"""Convert Neuronpedia attribution graph checkpoint to exp_sel_data_out.json schema.

Reads: temp/checkpoint_v4.json
Writes: full_data_out.json (single dataset, all records)

Schema: exp_sel_data_out with metadata_model_correct, metadata_difficulty,
        metadata_expected_answer, metadata_iter fields.
"""

import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).parent
CHECKPOINT = WORKSPACE / "temp" / "checkpoint_v4.json"
DATASET_NAME = "neuronpedia_attribution_graphs_v2"

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
    print(f"\nDAGs: {dag_count}/{total} ({dag_count / total * 100:.1f}%)")

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
    print("Neuronpedia Attribution Graph -> exp_sel_data_out.json")
    print("=" * 70)

    records = load_checkpoint()
    if not records:
        print("ERROR: No records in checkpoint")
        sys.exit(1)

    # Convert all records
    examples = [convert_record(r, idx) for idx, r in enumerate(records)]
    print(f"Converted {len(examples)} examples")

    # Validate
    validate_dataset(examples)

    # Write single full_data_out.json
    result = wrap_schema(examples)
    out_path = WORKSPACE / "full_data_out.json"
    output_str = json.dumps(result, separators=(",", ":"))
    out_path.write_text(output_str)
    size_mb = len(output_str) / (1024 * 1024)
    print(f"\nWritten: full_data_out.json ({len(examples)} examples, {size_mb:.1f} MB)")

    domains = Counter(ex["metadata_fold"] for ex in examples)
    print(f"\n{'=' * 70}")
    print(f"Done! {len(examples)} examples across {len(domains)} domains")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
