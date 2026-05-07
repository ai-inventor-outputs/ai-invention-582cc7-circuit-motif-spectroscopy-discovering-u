# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests==2.32.5",
#   "networkx==3.6.1",
#   "loguru==0.7.3",
# ]
# ///
"""Neuronpedia Attribution Graph Dataset — data.py

Merges iter2 (140 graphs, m4-*) with iter3 checkpoint (60 graphs, m5-*)
to produce 200 unique attribution graphs across 8 domains for gemma-2-2b.

Usage:
    uv run data.py            # Full merge (loads iter1+iter2+iter3 checkpoint)
    uv run data.py --help     # Show options

Output (in data_out/):
    full_data_out_{1..N}.json  — split files, each <90MB
    mini_data_out.json         — 16 records (2 per domain)
    preview_data_out.json      — 8 records (1 per domain, truncated output)
"""

import subprocess
import shutil
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent


def main():
    # Run collect_v5.py --merge-only which handles all the heavy lifting:
    # - Loads iter1 (34 graphs) + iter2 (140 graphs) from dependency paths
    # - Deduplicates by prompt text (iter1 fully overlaps with iter2)
    # - Loads iter3 checkpoint records (60 graphs from Neuronpedia API)
    # - Verifies correctness via logit node tokens
    # - Validates all graphs are DAGs with >=10 nodes
    # - Writes split output files (<90MB each)
    # - Writes mini (16 records) and preview (8 records) versions
    collect_script = WORKSPACE / "collect_v5.py"
    if not collect_script.exists():
        print("ERROR: collect_v5.py not found in workspace", file=sys.stderr)
        sys.exit(1)

    venv_python = WORKSPACE / ".venv" / "bin" / "python3"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable

    print("Running merge (collect_v5.py --merge-only)...")
    result = subprocess.run(
        [python_cmd, str(collect_script), "--merge-only"],
        cwd=str(WORKSPACE),
        capture_output=False,
    )
    if result.returncode != 0:
        print("ERROR: collect_v5.py --merge-only failed", file=sys.stderr)
        sys.exit(1)

    # Rename mini/preview files to standard names (remove _1 suffix)
    data_out = WORKSPACE / "data_out"
    renames = [
        (data_out / "mini_data_out_1.json", data_out / "mini_data_out.json"),
        (data_out / "preview_data_out_1.json", data_out / "preview_data_out.json"),
    ]
    for src, dst in renames:
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"Renamed {src.name} -> {dst.name}")

    # Summary
    import json
    import glob as g

    splits = sorted(g.glob(str(data_out / "full_data_out_*.json")))
    total = 0
    for f in splits:
        with open(f) as fh:
            d = json.load(fh)
        n = len(d["datasets"][0]["examples"])
        total += n
    print(f"\nDone: {total} graphs in {len(splits)} split files")
    print(f"Output: {data_out}")


if __name__ == "__main__":
    main()
