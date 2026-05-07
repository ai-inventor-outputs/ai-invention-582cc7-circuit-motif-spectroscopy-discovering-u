#!/usr/bin/env python3
"""Rate-limit-aware Neuronpedia attribution graph collector.

Handles 30 req/60 min API rate limit with:
- 121s spacing between requests (safely under 30/hr)
- 429 handling with 300s wait + retry
- Unique slug prefix to avoid collisions
- Round-robin across domains for balanced coverage
- Checkpoint after every single graph
"""

import hashlib
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
LOGS_DIR = WORKSPACE / "logs"
TEMP_DIR = WORKSPACE / "temp"
LOGS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "collect_v2.log"), rotation="30 MB", level="DEBUG")

# Container-aware resource limits
def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
API_URL = "https://www.neuronpedia.org/api/graph/generate"
API_KEY = "sk-np-DQRQw4Us2QtJgy0kq9nZOz39qVIJ0kpy7d8ymN1Ica80"
SLUG_PREFIX = "m2"  # Unique prefix to avoid collisions with prior runs
REQUEST_SPACING = 121  # seconds between requests (< 30/hour)
RATE_LIMIT_WAIT = 300  # seconds to wait on 429
CHECKPOINT_FILE = TEMP_DIR / "checkpoint_v2.json"

# ---------------------------------------------------------------------------
# All prompts: 8 domains x 33 each = 264
# ---------------------------------------------------------------------------
PROMPTS = {
    "country_capital": [
        "The capital of Japan is", "The capital of Brazil is",
        "The capital of Nigeria is", "The capital of Australia is",
        "The capital of France is", "The capital of Egypt is",
        "The capital of Canada is", "The capital of Thailand is",
        "The capital of Germany is", "The capital of Mexico is",
        "The capital of India is", "The capital of South Korea is",
        "The capital of Argentina is", "The capital of Kenya is",
        "The capital of Sweden is", "The capital of Turkey is",
        "The capital of Peru is", "The capital of Indonesia is",
        "The capital of Poland is", "The capital of Vietnam is",
        "The capital of South Africa is", "The capital of Spain is",
        "The capital of Colombia is", "The capital of Iran is",
        "The capital of Norway is", "The capital of Chile is",
        "The capital of Philippines is", "The capital of Morocco is",
        "The capital of Ukraine is", "The capital of New Zealand is",
        "The capital of Italy is", "The capital of Greece is",
        "The capital of Portugal is",
    ],
    "arithmetic": [
        "3 + 5 =", "15 + 28 =", "47 + 36 =", "123 + 456 =",
        "8 + 9 =", "250 + 375 =", "7 + 4 =", "12 + 19 =",
        "33 + 67 =", "89 + 11 =", "6 + 3 =", "21 + 34 =",
        "55 + 45 =", "99 + 1 =", "14 + 27 =", "38 + 62 =",
        "2 + 8 =", "16 + 84 =", "73 + 27 =", "5 + 6 =",
        "44 + 56 =", "9 + 7 =", "31 + 69 =", "18 + 22 =",
        "150 + 250 =", "4 + 9 =", "25 + 75 =", "60 + 40 =",
        "11 + 13 =", "200 + 300 =", "17 + 83 =", "42 + 58 =",
        "1 + 2 =",
    ],
    "antonym": [
        "The opposite of happy is", "The opposite of tall is",
        "The opposite of fast is", "The opposite of dark is",
        "The opposite of cold is", "The opposite of good is",
        "The opposite of old is", "The opposite of rich is",
        "The opposite of strong is", "The opposite of hard is",
        "The opposite of loud is", "The opposite of heavy is",
        "The opposite of early is", "The opposite of clean is",
        "The opposite of wet is", "The opposite of open is",
        "The opposite of long is", "The opposite of narrow is",
        "The opposite of smooth is", "The opposite of bright is",
        "The opposite of sweet is", "The opposite of sharp is",
        "The opposite of thick is", "The opposite of deep is",
        "The opposite of full is", "The opposite of safe is",
        "The opposite of cheap is", "The opposite of brave is",
        "The opposite of kind is", "The opposite of wise is",
        "The opposite of calm is", "The opposite of alive is",
        "The opposite of true is",
    ],
    "translation": [
        "The French word for cat is", "The French word for house is",
        "The French word for water is", "The French word for book is",
        "The French word for dog is", "The French word for tree is",
        "The French word for sun is", "The French word for moon is",
        "The French word for car is", "The French word for bread is",
        "The French word for fish is", "The French word for bird is",
        "The French word for flower is", "The French word for door is",
        "The French word for chair is", "The French word for table is",
        "The French word for school is", "The French word for hand is",
        "The French word for heart is", "The French word for fire is",
        "The French word for stone is", "The French word for river is",
        "The French word for star is", "The French word for cloud is",
        "The French word for rain is", "The French word for snow is",
        "The French word for king is", "The French word for night is",
        "The French word for milk is", "The French word for egg is",
        "The French word for apple is", "The French word for cheese is",
        "The French word for friend is",
    ],
    "code_completion": [
        "def add(a, b):\n    return", "def multiply(x, y):\n    return",
        "def square(n):\n    return", "def maximum(a, b):\n    return",
        "def is_even(n):\n    return", "def absolute(x):\n    return",
        "def negate(n):\n    return", "def double(x):\n    return",
        "def minimum(a, b):\n    return", "def increment(n):\n    return",
        "def subtract(a, b):\n    return", "def divide(a, b):\n    return",
        "def power(base, exp):\n    return", "def modulo(a, b):\n    return",
        "def is_positive(n):\n    return", "def is_zero(n):\n    return",
        "def to_string(x):\n    return", "def first(lst):\n    return",
        "def last(lst):\n    return", "def length(lst):\n    return",
        "def is_empty(lst):\n    return", "def reverse(s):\n    return",
        "def upper(s):\n    return", "def lower(s):\n    return",
        "def half(n):\n    return", "def cube(n):\n    return",
        "def is_odd(n):\n    return", "def sign(n):\n    return",
        "def identity(x):\n    return", "def floor_div(a, b):\n    return",
        "def average(a, b):\n    return", "def clamp(x, lo, hi):\n    return",
        "def triple(n):\n    return",
    ],
    "multi_hop_reasoning": [
        "If Alice is taller than Bob and Bob is taller than Carol, then Alice is",
        "If X is larger than Y and Y is larger than Z, then X is",
        "If the red box is heavier than the blue box and the blue box is heavier than the green box, the red box is",
        "If John is older than Mary and Mary is older than Tom, then John is",
        "If A is faster than B and B is faster than C, then A is",
        "If the cat is bigger than the dog and the dog is bigger than the mouse, the cat is",
        "If Paris is north of Rome and Rome is north of Cairo, Paris is",
        "If iron is harder than copper and copper is harder than gold, iron is",
        "If Sam runs faster than Dan and Dan runs faster than Pat, Sam runs",
        "If the oak is taller than the pine and the pine is taller than the bush, the oak is",
        "If spring comes before summer and summer comes before fall, spring comes",
        "If diamond is harder than steel and steel is harder than wood, diamond is",
        "If the river is wider than the stream and the stream is wider than the creek, the river is",
        "If Mars is farther than Venus and Venus is farther than Mercury, Mars is",
        "If the whale is larger than the shark and the shark is larger than the tuna, the whale is",
        "If gold costs more than silver and silver costs more than bronze, gold costs",
        "If Lisa scored higher than Mike and Mike scored higher than Nina, Lisa scored",
        "If the mountain is higher than the hill and the hill is higher than the valley, the mountain is",
        "If the elephant weighs more than the horse and the horse weighs more than the sheep, the elephant weighs",
        "If Monday comes before Tuesday and Tuesday comes before Wednesday, Monday comes",
        "If the sun is brighter than the moon and the moon is brighter than a star, the sun is",
        "If French is harder than Spanish and Spanish is harder than Italian, French is",
        "If the train is faster than the car and the car is faster than the bicycle, the train is",
        "If Earth is bigger than Mars and Mars is bigger than Mercury, Earth is",
        "If the CEO earns more than the manager and the manager earns more than the clerk, the CEO earns",
        "If the Pacific is deeper than the Atlantic and the Atlantic is deeper than the Arctic, the Pacific is",
        "If Einstein was smarter than Newton and Newton was smarter than Galileo, Einstein was",
        "If the cheetah is faster than the lion and the lion is faster than the bear, the cheetah is",
        "If the piano is heavier than the guitar and the guitar is heavier than the flute, the piano is",
        "If the novel is longer than the story and the story is longer than the poem, the novel is",
        "If the lake is deeper than the pond and the pond is deeper than the puddle, the lake is",
        "If January is colder than March and March is colder than May, January is",
        "If the castle is older than the church and the church is older than the school, the castle is",
    ],
    "rhyme": [
        "The cat in the hat sat on the", "A mouse in the house found a",
        "The dog on the log saw a", "The bee in the tree drank some",
        "The fish made a wish for a", "The bear on the stair ate a",
        "The fox in the box wore some", "The hen in the den found a",
        "The goat in the boat wore a", "The frog on the log wrote a",
        "The fly in the sky ate a", "The snake by the lake ate some",
        "The crow in the snow found a", "The bat in the hat found a",
        "The owl on the towel wore a", "The bug on the rug ate a",
        "The duck in the truck had some", "The ram in the jam ate some",
        "The seal with the meal had a", "The snail on the trail found a",
        "The worm in the storm found a", "The moose on the loose drank some",
        "The mole in the hole found some", "The toad on the road had a",
        "The pup in the cup drank some", "The lark in the park ate a",
        "The hare with the flair wore a", "The eel in the reel caught a",
        "The ant on the plant ate a", "The dove from above found some",
        "The pig in the wig wore a", "The yak in the shack ate a",
        "The skunk on the bunk found a",
    ],
    "sentiment": [
        "This movie was terrible. The sentiment is",
        "This movie was wonderful. The sentiment is",
        "This movie was boring. The sentiment is",
        "This movie was exciting. The sentiment is",
        "This movie was awful. The sentiment is",
        "This movie was amazing. The sentiment is",
        "This movie was dull. The sentiment is",
        "This movie was thrilling. The sentiment is",
        "This movie was disappointing. The sentiment is",
        "This movie was brilliant. The sentiment is",
        "This movie was mediocre. The sentiment is",
        "This movie was fantastic. The sentiment is",
        "This movie was horrible. The sentiment is",
        "This movie was delightful. The sentiment is",
        "This movie was tedious. The sentiment is",
        "This movie was outstanding. The sentiment is",
        "This movie was miserable. The sentiment is",
        "This movie was superb. The sentiment is",
        "This movie was forgettable. The sentiment is",
        "This movie was spectacular. The sentiment is",
        "This movie was painful. The sentiment is",
        "This movie was charming. The sentiment is",
        "This movie was frustrating. The sentiment is",
        "This movie was inspiring. The sentiment is",
        "This movie was annoying. The sentiment is",
        "This movie was excellent. The sentiment is",
        "This movie was depressing. The sentiment is",
        "This movie was heartwarming. The sentiment is",
        "This movie was dreadful. The sentiment is",
        "This movie was captivating. The sentiment is",
        "This movie was unbearable. The sentiment is",
        "This movie was enchanting. The sentiment is",
        "This movie was atrocious. The sentiment is",
    ],
}

DOMAINS = list(PROMPTS.keys())

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            logger.info(f"Loaded checkpoint: {len(data.get('records', []))} records")
            return data
        except Exception:
            logger.exception("Checkpoint load failed, starting fresh")
    return {"records": [], "done": []}


def save_checkpoint(records: list, done: list) -> None:
    CHECKPOINT_FILE.write_text(json.dumps({"records": records, "done": done}))


# ---------------------------------------------------------------------------
# API call with rate limit handling
# ---------------------------------------------------------------------------
def generate_graph(prompt: str, domain: str, idx: int) -> dict | None:
    """Call API, handle 429/503/400, download S3 graph."""
    slug = f"{SLUG_PREFIX}-{domain}-{idx:03d}-{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
    payload = {
        "prompt": prompt,
        "modelId": "gemma-2-2b",
        "slug": slug,
        "maxNLogits": 10,
        "desiredLogitProb": 0.95,
        "nodeThreshold": 0.8,
        "edgeThreshold": 0.85,
        "maxFeatureNodes": 5000,
    }
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}

    for attempt in range(8):  # More retries for 429 handling
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)

            if resp.status_code == 200:
                result = resp.json()
                s3url = result.get("s3url", "")
                if not s3url:
                    logger.warning(f"  No s3url for {slug}")
                    return None
                graph_data = _download_s3(s3url)
                if graph_data is None:
                    return None
                return {"api_response": result, "graph_data": graph_data, "slug": slug}

            elif resp.status_code == 429:
                logger.warning(f"  429 rate limit for {slug}, waiting {RATE_LIMIT_WAIT}s (attempt {attempt+1})")
                time.sleep(RATE_LIMIT_WAIT)
                continue

            elif resp.status_code == 503:
                wait = 30 * (2 ** attempt)
                wait = min(wait, 480)
                logger.warning(f"  503 GPUs busy, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)

            elif resp.status_code == 400:
                error_msg = resp.text
                if "slug" in error_msg.lower() and "exists" in error_msg.lower():
                    slug = slug + f"-r{attempt}"
                    payload["slug"] = slug
                    logger.info(f"  Slug collision, retrying as {slug}")
                    continue
                else:
                    logger.warning(f"  400 error: {error_msg[:200]}")
                    return None
            else:
                logger.warning(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(10 * (attempt + 1))

        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout (attempt {attempt+1})")
            time.sleep(30)
        except Exception:
            logger.exception(f"  Request error (attempt {attempt+1})")
            time.sleep(15)

    logger.error(f"  All retries exhausted for {slug}")
    return None


def _download_s3(s3url: str) -> dict | None:
    for attempt in range(3):
        try:
            resp = requests.get(s3url, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"  S3 HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            logger.warning(f"  S3 timeout (attempt {attempt+1})")
        except json.JSONDecodeError:
            logger.warning(f"  S3 invalid JSON")
        except Exception:
            logger.exception("  S3 error")
        time.sleep(5)
    return None


# ---------------------------------------------------------------------------
# Graph parsing
# ---------------------------------------------------------------------------
def parse_graph(prompt: str, domain: str, api_result: dict) -> dict | None:
    try:
        graph_data = api_result["graph_data"]
        api_resp = api_result["api_response"]
        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", [])
        n_nodes = len(nodes)
        n_edges = len(links)

        if n_nodes < 10:
            return None

        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node.get("node_id", ""))
        for link in links:
            G.add_edge(link.get("source", ""), link.get("target", ""))
        is_dag = nx.is_directed_acyclic_graph(G)

        metadata = graph_data.get("metadata", {})
        return {
            "input": prompt,
            "output": {
                "nodes": nodes,
                "links": links,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": round(density, 6),
                "is_dag": is_dag,
                "prompt_tokens": metadata.get("prompt_tokens", []),
                "slug": api_result.get("slug", metadata.get("slug", "")),
                "s3url": api_resp.get("s3url", ""),
                "neuronpedia_url": api_resp.get("url", ""),
            },
            "metadata_fold": domain,
        }
    except Exception:
        logger.exception(f"  Parse error for: {prompt[:50]}")
        return None


# ---------------------------------------------------------------------------
# Build round-robin work queue
# ---------------------------------------------------------------------------
def build_work_queue() -> list[tuple[str, int, str]]:
    """Round-robin: (domain, idx, prompt) cycling across domains."""
    max_prompts = max(len(v) for v in PROMPTS.values())
    queue = []
    for idx in range(max_prompts):
        for domain in DOMAINS:
            if idx < len(PROMPTS[domain]):
                queue.append((domain, idx, PROMPTS[domain][idx]))
    return queue


# ---------------------------------------------------------------------------
# Main collection
# ---------------------------------------------------------------------------
@logger.catch
def main():
    total_prompts = sum(len(v) for v in PROMPTS.values())
    logger.info(f"Prompts: {total_prompts} across {len(DOMAINS)} domains")

    ckpt = load_checkpoint()
    records = ckpt["records"]
    done_set = set(ckpt["done"])
    logger.info(f"Resuming with {len(records)} records, {len(done_set)} done")

    queue = build_work_queue()
    remaining = [(d, i, p) for d, i, p in queue if f"{d}:{i}" not in done_set]
    logger.info(f"Remaining work items: {len(remaining)}")

    success_count = len(records)
    fail_count = 0
    start_time = time.time()

    for work_idx, (domain, idx, prompt) in enumerate(remaining):
        key = f"{domain}:{idx}"
        elapsed_total = (time.time() - start_time) / 60
        logger.info(
            f"[{work_idx+1}/{len(remaining)}] {domain}[{idx}]: "
            f"{prompt[:55]}... ({elapsed_total:.0f}m elapsed, {success_count} ok)"
        )

        t0 = time.time()
        result = generate_graph(prompt, domain, idx)
        gen_time = time.time() - t0

        if result is not None:
            record = parse_graph(prompt, domain, result)
            if record is not None:
                records.append(record)
                done_set.add(key)
                success_count += 1
                logger.info(
                    f"  OK: {record['output']['n_nodes']} nodes, "
                    f"{record['output']['n_edges']} edges, "
                    f"DAG={record['output']['is_dag']}, {gen_time:.1f}s"
                )
                # Checkpoint after every graph
                save_checkpoint(records, list(done_set))
            else:
                logger.warning(f"  Graph too small, skipping")
                done_set.add(key)
                fail_count += 1
                save_checkpoint(records, list(done_set))
        else:
            fail_count += 1
            # Don't add to done_set so we can retry later

        # Rate-limit pacing: wait between requests
        # Subtract time already spent on this request
        wait_needed = max(0, REQUEST_SPACING - gen_time)
        if wait_needed > 0 and work_idx < len(remaining) - 1:
            logger.debug(f"  Pacing wait: {wait_needed:.0f}s")
            time.sleep(wait_needed)

    # --- Collection complete ---
    logger.info("=" * 60)
    logger.info(f"DONE: {success_count} records, {fail_count} failures")

    # Per-domain summary
    domain_counts = defaultdict(int)
    for r in records:
        domain_counts[r["metadata_fold"]] += 1
    for d in DOMAINS:
        logger.info(f"  {d}: {domain_counts[d]}")

    # Write outputs
    write_outputs(records)


# ---------------------------------------------------------------------------
# Validation & output
# ---------------------------------------------------------------------------
def write_outputs(records: list[dict]) -> None:
    if not records:
        logger.error("No records to write!")
        return

    # Validation summary
    logger.info("=" * 60)
    logger.info("VALIDATION")
    domain_recs = defaultdict(list)
    for r in records:
        domain_recs[r["metadata_fold"]].append(r)

    logger.info(f"Total: {len(records)} records")
    logger.info(f"{'Domain':<25} {'N':>4} {'MinN':>6} {'MaxN':>6} {'MedN':>6} {'MedE':>7} {'%DAG':>6}")
    logger.info("-" * 70)
    for d in sorted(domain_recs):
        recs = domain_recs[d]
        nodes = [r["output"]["n_nodes"] for r in recs]
        edges = [r["output"]["n_edges"] for r in recs]
        dags = [r["output"]["is_dag"] for r in recs]
        med_n = sorted(nodes)[len(nodes)//2]
        med_e = sorted(edges)[len(edges)//2]
        pct = sum(dags)/len(dags)*100
        logger.info(f"{d:<25} {len(recs):>4} {min(nodes):>6} {max(nodes):>6} {med_n:>6} {med_e:>7} {pct:>5.1f}%")

    total_dags = sum(1 for r in records if r["output"]["is_dag"])
    logger.info(f"DAG: {total_dags}/{len(records)} ({total_dags/len(records)*100:.1f}%)")

    ftypes = set()
    for r in records:
        for n in r["output"]["nodes"]:
            ft = n.get("feature_type", "")
            if ft:
                ftypes.add(ft)
    logger.info(f"Feature types: {sorted(ftypes)}")

    # Write full data_out.json
    out = WORKSPACE / "data_out.json"
    out.write_text(json.dumps(records, ensure_ascii=False))
    logger.info(f"Wrote data_out.json: {out.stat().st_size/1e6:.1f} MB")

    # Mini: 3 per domain
    mini = []
    for d in sorted(domain_recs):
        mini.extend(domain_recs[d][:3])
    (WORKSPACE / "data_out_mini.json").write_text(json.dumps(mini, ensure_ascii=False))
    logger.info(f"Wrote data_out_mini.json: {len(mini)} records")

    # Preview: 1 per domain, truncated
    preview = []
    for d in sorted(domain_recs):
        if domain_recs[d]:
            r = json.loads(json.dumps(domain_recs[d][0]))
            r["output"]["nodes"] = r["output"]["nodes"][:10]
            r["output"]["links"] = r["output"]["links"][:10]
            preview.append(r)
    (WORKSPACE / "data_out_preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    logger.info(f"Wrote data_out_preview.json: {len(preview)} records")


if __name__ == "__main__":
    main()
