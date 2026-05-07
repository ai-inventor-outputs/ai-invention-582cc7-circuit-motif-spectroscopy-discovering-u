#!/bin/bash
# Run collect_v4.py in repeated 55-minute batches with checkpoint/resume.
# Each run collects ~27 graphs (at 120s spacing). The checkpoint ensures
# no work is repeated across runs.
#
# Usage: bash run_batches.sh [max_batches]
# Default: 6 batches (~3 hours, ~160 graphs)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/python"
COLLECTOR="$SCRIPT_DIR/collect_v4.py"
MAX_TIME=3300  # 55 minutes per batch
MAX_BATCHES=${1:-6}

echo "=== Batch Runner: $MAX_BATCHES batches x ${MAX_TIME}s ==="
echo "=== Expected: ~$((MAX_BATCHES * 27)) graphs ==="

for i in $(seq 1 "$MAX_BATCHES"); do
    echo ""
    echo "=== BATCH $i / $MAX_BATCHES ($(date)) ==="
    "$VENV" "$COLLECTOR" --max-time "$MAX_TIME" 2>&1

    # Check how many records we have
    if [ -f "$SCRIPT_DIR/temp/checkpoint_v4.json" ]; then
        COUNT=$("$VENV" -c "import json; d=json.load(open('$SCRIPT_DIR/temp/checkpoint_v4.json')); print(len(d['records']))")
        DONE=$("$VENV" -c "import json; d=json.load(open('$SCRIPT_DIR/temp/checkpoint_v4.json')); print(len(d['done']))")
        echo "=== After batch $i: $COUNT records, $DONE done ==="

        # Stop if all 264 are done
        if [ "$DONE" -ge 264 ]; then
            echo "=== All prompts collected! ==="
            break
        fi
    fi
done

echo ""
echo "=== All batches complete ==="
