#!/usr/bin/env python3
"""Collect 250+ attribution graphs from Neuronpedia API across 8 capability domains."""

import asyncio
import aiohttp
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import networkx as nx
from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware-aware resource limits (container: 29 GB RAM, 4 CPUs)
# ---------------------------------------------------------------------------
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
RAM_BUDGET = int(TOTAL_RAM_GB * 0.5 * 1e9)  # 50% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))  # 2 hour CPU time
logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f} GB, total: {TOTAL_RAM_GB:.1f} GB")

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
API_URL = "https://www.neuronpedia.org/api/graph/generate"
API_KEY = "sk-np-DQRQw4Us2QtJgy0kq9nZOz39qVIJ0kpy7d8ymN1Ica80"
BASE_DELAY = 3.0          # seconds between calls
BATCH_PAUSE_EVERY = 20    # pause after this many graphs
BATCH_PAUSE_SECS = 30     # seconds to pause
CHECKPOINT_EVERY = 10     # save checkpoint after this many graphs
CONCURRENCY = 1           # sequential: API is GPU-bound, concurrent would cause 503s

WORKSPACE = Path(__file__).parent
CHECKPOINT_FILE = WORKSPACE / "temp" / "checkpoint.json"
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)
(WORKSPACE / "temp").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Prompt definitions: 8 domains x 33 prompts each = 264 total
# ---------------------------------------------------------------------------
PROMPTS: dict[str, list[str]] = {
    "country_capital": [
        "The capital of Japan is",
        "The capital of Brazil is",
        "The capital of Nigeria is",
        "The capital of Australia is",
        "The capital of France is",
        "The capital of Egypt is",
        "The capital of Canada is",
        "The capital of Thailand is",
        "The capital of Germany is",
        "The capital of Mexico is",
        "The capital of India is",
        "The capital of South Korea is",
        "The capital of Argentina is",
        "The capital of Kenya is",
        "The capital of Sweden is",
        "The capital of Turkey is",
        "The capital of Peru is",
        "The capital of Indonesia is",
        "The capital of Poland is",
        "The capital of Vietnam is",
        "The capital of South Africa is",
        "The capital of Spain is",
        "The capital of Colombia is",
        "The capital of Iran is",
        "The capital of Norway is",
        "The capital of Chile is",
        "The capital of Philippines is",
        "The capital of Morocco is",
        "The capital of Ukraine is",
        "The capital of New Zealand is",
        "The capital of Italy is",
        "The capital of Greece is",
        "The capital of Portugal is",
    ],
    "arithmetic": [
        "3 + 5 =",
        "15 + 28 =",
        "47 + 36 =",
        "123 + 456 =",
        "8 + 9 =",
        "250 + 375 =",
        "7 + 4 =",
        "12 + 19 =",
        "33 + 67 =",
        "89 + 11 =",
        "6 + 3 =",
        "21 + 34 =",
        "55 + 45 =",
        "99 + 1 =",
        "14 + 27 =",
        "38 + 62 =",
        "2 + 8 =",
        "16 + 84 =",
        "73 + 27 =",
        "5 + 6 =",
        "44 + 56 =",
        "9 + 7 =",
        "31 + 69 =",
        "18 + 22 =",
        "150 + 250 =",
        "4 + 9 =",
        "25 + 75 =",
        "60 + 40 =",
        "11 + 13 =",
        "200 + 300 =",
        "17 + 83 =",
        "42 + 58 =",
        "1 + 2 =",
    ],
    "antonym": [
        "The opposite of happy is",
        "The opposite of tall is",
        "The opposite of fast is",
        "The opposite of dark is",
        "The opposite of cold is",
        "The opposite of good is",
        "The opposite of old is",
        "The opposite of rich is",
        "The opposite of strong is",
        "The opposite of hard is",
        "The opposite of loud is",
        "The opposite of heavy is",
        "The opposite of early is",
        "The opposite of clean is",
        "The opposite of wet is",
        "The opposite of open is",
        "The opposite of long is",
        "The opposite of narrow is",
        "The opposite of smooth is",
        "The opposite of bright is",
        "The opposite of sweet is",
        "The opposite of sharp is",
        "The opposite of thick is",
        "The opposite of deep is",
        "The opposite of full is",
        "The opposite of safe is",
        "The opposite of cheap is",
        "The opposite of brave is",
        "The opposite of kind is",
        "The opposite of wise is",
        "The opposite of calm is",
        "The opposite of alive is",
        "The opposite of true is",
    ],
    "translation": [
        "The French word for cat is",
        "The French word for house is",
        "The French word for water is",
        "The French word for book is",
        "The French word for dog is",
        "The French word for tree is",
        "The French word for sun is",
        "The French word for moon is",
        "The French word for car is",
        "The French word for bread is",
        "The French word for fish is",
        "The French word for bird is",
        "The French word for flower is",
        "The French word for door is",
        "The French word for chair is",
        "The French word for table is",
        "The French word for school is",
        "The French word for hand is",
        "The French word for heart is",
        "The French word for fire is",
        "The French word for stone is",
        "The French word for river is",
        "The French word for star is",
        "The French word for cloud is",
        "The French word for rain is",
        "The French word for snow is",
        "The French word for king is",
        "The French word for night is",
        "The French word for milk is",
        "The French word for egg is",
        "The French word for apple is",
        "The French word for cheese is",
        "The French word for friend is",
    ],
    "code_completion": [
        "def add(a, b):\n    return",
        "def multiply(x, y):\n    return",
        "def square(n):\n    return",
        "def maximum(a, b):\n    return",
        "def is_even(n):\n    return",
        "def absolute(x):\n    return",
        "def negate(n):\n    return",
        "def double(x):\n    return",
        "def minimum(a, b):\n    return",
        "def increment(n):\n    return",
        "def subtract(a, b):\n    return",
        "def divide(a, b):\n    return",
        "def power(base, exp):\n    return",
        "def modulo(a, b):\n    return",
        "def is_positive(n):\n    return",
        "def is_zero(n):\n    return",
        "def to_string(x):\n    return",
        "def first(lst):\n    return",
        "def last(lst):\n    return",
        "def length(lst):\n    return",
        "def is_empty(lst):\n    return",
        "def reverse(s):\n    return",
        "def upper(s):\n    return",
        "def lower(s):\n    return",
        "def half(n):\n    return",
        "def cube(n):\n    return",
        "def is_odd(n):\n    return",
        "def sign(n):\n    return",
        "def identity(x):\n    return",
        "def floor_div(a, b):\n    return",
        "def average(a, b):\n    return",
        "def clamp(x, lo, hi):\n    return",
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
        "The cat in the hat sat on the",
        "A mouse in the house found a",
        "The dog on the log saw a",
        "The bee in the tree drank some",
        "The fish made a wish for a",
        "The bear on the stair ate a",
        "The fox in the box wore some",
        "The hen in the den found a",
        "The goat in the boat wore a",
        "The frog on the log wrote a",
        "The fly in the sky ate a",
        "The snake by the lake ate some",
        "The crow in the snow found a",
        "The bat in the hat found a",
        "The owl on the towel wore a",
        "The bug on the rug ate a",
        "The duck in the truck had some",
        "The ram in the jam ate some",
        "The seal with the meal had a",
        "The snail on the trail found a",
        "The worm in the storm found a",
        "The moose on the loose drank some",
        "The mole in the hole found some",
        "The toad on the road had a",
        "The pup in the cup drank some",
        "The lark in the park ate a",
        "The hare with the flair wore a",
        "The eel in the reel caught a",
        "The ant on the plant ate a",
        "The dove from above found some",
        "The pig in the wig wore a",
        "The yak in the shack ate a",
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

# ---------------------------------------------------------------------------
# Graph generation + S3 download
# ---------------------------------------------------------------------------
async def generate_graph(
    session: aiohttp.ClientSession,
    prompt: str,
    domain: str,
    idx: int,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Call Neuronpedia API to generate a graph, download from S3, return parsed result."""
    slug = f"motif-{domain}-{idx:03d}-{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
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
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
    }

    async with sem:
        for attempt in range(5):
            try:
                async with session.post(
                    API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        api_result = await resp.json()
                        s3url = api_result.get("s3url", "")
                        if not s3url:
                            logger.warning(f"  No s3url in response for {slug}")
                            return None

                        # Download graph JSON from S3 (with retries)
                        graph_data = await _download_s3(session, s3url)
                        if graph_data is None:
                            return None

                        return {
                            "api_response": api_result,
                            "graph_data": graph_data,
                            "slug": slug,
                        }

                    elif resp.status == 503:
                        wait = 30 * (2 ** attempt)
                        logger.warning(f"  GPUs busy for {slug}, waiting {wait}s (attempt {attempt+1})")
                        await asyncio.sleep(wait)

                    elif resp.status == 400:
                        error_msg = await resp.text()
                        if "slug" in error_msg.lower() and "exists" in error_msg.lower():
                            slug = slug + f"-r{attempt}"
                            payload["slug"] = slug
                            logger.info(f"  Slug collision, retrying with {slug}")
                            continue
                        else:
                            logger.warning(f"  400 error for {slug}: {error_msg[:200]}")
                            return None

                    else:
                        body = await resp.text()
                        logger.warning(f"  HTTP {resp.status} for {slug}: {body[:200]}")
                        await asyncio.sleep(10 * (attempt + 1))

            except asyncio.TimeoutError:
                logger.warning(f"  Timeout for {slug}, attempt {attempt+1}")
                await asyncio.sleep(30)
            except Exception as e:
                logger.exception(f"  Error for {slug}: {e}")
                await asyncio.sleep(15)

        logger.error(f"  All retries exhausted for {slug}")
        return None


async def _download_s3(session: aiohttp.ClientSession, s3url: str) -> dict | None:
    """Download graph JSON from S3 URL with retries."""
    for attempt in range(3):
        try:
            async with session.get(s3url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"  S3 HTTP {resp.status} (attempt {attempt+1})")
        except asyncio.TimeoutError:
            logger.warning(f"  S3 timeout (attempt {attempt+1})")
        except json.JSONDecodeError:
            logger.warning(f"  S3 invalid JSON (attempt {attempt+1})")
        except Exception as e:
            logger.exception(f"  S3 error: {e}")
        await asyncio.sleep(5)
    logger.error(f"  S3 download failed after 3 retries: {s3url[:100]}")
    return None


# ---------------------------------------------------------------------------
# Graph parsing
# ---------------------------------------------------------------------------
def parse_graph(prompt: str, domain: str, api_result: dict) -> dict | None:
    """Parse API result into a standardized record."""
    try:
        graph_data = api_result["graph_data"]
        api_resp = api_result["api_response"]

        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", [])
        n_nodes = len(nodes)
        n_edges = len(links)

        if n_nodes < 10:
            logger.info(f"  Skipping graph with only {n_nodes} nodes for: {prompt[:50]}")
            return None

        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        # Verify DAG property
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node.get("node_id", ""), **{k: v for k, v in node.items() if k != "node_id"})
        for link in links:
            G.add_edge(link.get("source", ""), link.get("target", ""), weight=link.get("weight", 0.0))
        is_dag = nx.is_directed_acyclic_graph(G)

        metadata = graph_data.get("metadata", {})
        prompt_tokens = metadata.get("prompt_tokens", [])

        return {
            "input": prompt,
            "output": {
                "nodes": nodes,
                "links": links,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": round(density, 6),
                "is_dag": is_dag,
                "prompt_tokens": prompt_tokens,
                "slug": api_result.get("slug", metadata.get("slug", "")),
                "s3url": api_resp.get("s3url", ""),
                "neuronpedia_url": api_resp.get("url", ""),
            },
            "metadata_fold": domain,
        }
    except Exception:
        logger.exception(f"  Parse error for prompt: {prompt[:50]}")
        return None


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict:
    """Load checkpoint file if it exists."""
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            logger.info(f"Loaded checkpoint with {len(data.get('records', []))} records")
            return data
        except Exception:
            logger.exception("Failed to load checkpoint, starting fresh")
    return {"records": [], "completed_keys": []}


def save_checkpoint(records: list, completed_keys: list) -> None:
    """Save checkpoint to disk."""
    CHECKPOINT_FILE.write_text(json.dumps(
        {"records": records, "completed_keys": completed_keys},
        ensure_ascii=False,
    ))
    logger.debug(f"Checkpoint saved: {len(records)} records")


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------
async def collect_graphs(mode: str = "full") -> list[dict]:
    """
    Collect graphs from Neuronpedia API.
    mode: "mini" = first 3 per domain, "full" = all prompts
    """
    checkpoint = load_checkpoint() if mode == "full" else {"records": [], "completed_keys": []}
    records = checkpoint["records"]
    completed_keys = set(checkpoint["completed_keys"])

    sem = asyncio.Semaphore(CONCURRENCY)
    total_calls = 0
    total_success = 0
    total_fail = 0
    domain_stats: dict[str, dict] = {}

    connector = aiohttp.TCPConnector(limit=5, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        for domain, prompts in PROMPTS.items():
            domain_stats[domain] = {"success": 0, "fail": 0, "small": 0}
            subset = prompts[:3] if mode == "mini" else prompts

            for idx, prompt in enumerate(subset):
                key = f"{domain}:{idx}"
                if key in completed_keys:
                    logger.debug(f"  Skipping (already done): {key}")
                    domain_stats[domain]["success"] += 1
                    total_success += 1
                    continue

                logger.info(f"[{domain}] {idx+1}/{len(subset)}: {prompt[:60]}")
                t0 = time.time()
                result = await generate_graph(session, prompt, domain, idx, sem)
                elapsed = time.time() - t0

                if result is not None:
                    record = parse_graph(prompt, domain, result)
                    if record is not None:
                        records.append(record)
                        completed_keys.add(key)
                        domain_stats[domain]["success"] += 1
                        total_success += 1
                        logger.info(
                            f"  OK: {record['output']['n_nodes']} nodes, "
                            f"{record['output']['n_edges']} edges, "
                            f"DAG={record['output']['is_dag']}, "
                            f"{elapsed:.1f}s"
                        )
                    else:
                        # Graph too small — try with lower threshold
                        domain_stats[domain]["small"] += 1
                        logger.info(f"  Graph too small, retrying with lower threshold...")
                        result2 = await generate_graph_lower_threshold(
                            session, prompt, domain, idx, sem
                        )
                        if result2 is not None:
                            record2 = parse_graph(prompt, domain, result2)
                            if record2 is not None:
                                records.append(record2)
                                completed_keys.add(key)
                                domain_stats[domain]["success"] += 1
                                total_success += 1
                                logger.info(f"  OK (lower threshold): {record2['output']['n_nodes']} nodes")
                            else:
                                total_fail += 1
                                domain_stats[domain]["fail"] += 1
                        else:
                            total_fail += 1
                            domain_stats[domain]["fail"] += 1
                else:
                    total_fail += 1
                    domain_stats[domain]["fail"] += 1

                total_calls += 1

                # Rate limiting
                await asyncio.sleep(BASE_DELAY)

                # Batch pause
                if total_calls > 0 and total_calls % BATCH_PAUSE_EVERY == 0:
                    logger.info(f"  Batch pause ({BATCH_PAUSE_SECS}s) after {total_calls} calls...")
                    await asyncio.sleep(BATCH_PAUSE_SECS)

                # Checkpoint
                if mode == "full" and total_calls % CHECKPOINT_EVERY == 0:
                    save_checkpoint(records, list(completed_keys))

    # Final checkpoint
    if mode == "full":
        save_checkpoint(records, list(completed_keys))

    # Summary
    logger.info("=" * 60)
    logger.info(f"Collection complete: {total_success} success, {total_fail} fail, {total_calls} total calls")
    logger.info(f"{'Domain':<25} {'OK':>5} {'Fail':>5} {'Small':>5}")
    for domain, stats in domain_stats.items():
        logger.info(f"{domain:<25} {stats['success']:>5} {stats['fail']:>5} {stats['small']:>5}")
    logger.info("=" * 60)

    return records


async def generate_graph_lower_threshold(
    session: aiohttp.ClientSession,
    prompt: str,
    domain: str,
    idx: int,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Retry graph generation with lower nodeThreshold (0.6 instead of 0.8)."""
    slug = f"motif-{domain}-{idx:03d}-{hashlib.md5(prompt.encode()).hexdigest()[:8]}-lo"
    payload = {
        "prompt": prompt,
        "modelId": "gemma-2-2b",
        "slug": slug,
        "maxNLogits": 10,
        "desiredLogitProb": 0.95,
        "nodeThreshold": 0.6,
        "edgeThreshold": 0.85,
        "maxFeatureNodes": 5000,
    }
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}

    async with sem:
        for attempt in range(3):
            try:
                async with session.post(
                    API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        api_result = await resp.json()
                        s3url = api_result.get("s3url", "")
                        if not s3url:
                            return None
                        graph_data = await _download_s3(session, s3url)
                        if graph_data is None:
                            return None
                        return {"api_response": api_result, "graph_data": graph_data, "slug": slug}
                    elif resp.status == 503:
                        wait = 30 * (2 ** attempt)
                        await asyncio.sleep(wait)
                    elif resp.status == 400:
                        error_msg = await resp.text()
                        if "slug" in error_msg.lower() and "exists" in error_msg.lower():
                            slug = slug + f"-r{attempt}"
                            payload["slug"] = slug
                            continue
                        return None
                    else:
                        await asyncio.sleep(10 * (attempt + 1))
            except asyncio.TimeoutError:
                await asyncio.sleep(30)
            except Exception as e:
                logger.exception(f"  Lower threshold error: {e}")
                await asyncio.sleep(15)
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(records: list[dict]) -> bool:
    """Run all validation checks on collected records."""
    logger.info("=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)

    all_ok = True

    # 1. Count check
    total = len(records)
    logger.info(f"Total records: {total}")
    if total < 250:
        logger.warning(f"  WARN: fewer than 250 records ({total})")
        all_ok = False

    # 2. Per-domain counts and stats
    from collections import defaultdict
    domain_records: dict[str, list] = defaultdict(list)
    for r in records:
        domain_records[r["metadata_fold"]].append(r)

    logger.info(f"\n{'Domain':<25} {'Count':>6} {'MinN':>6} {'MaxN':>6} {'MedN':>6} "
                f"{'MinE':>6} {'MaxE':>6} {'MedE':>6} {'Density':>8} {'%DAG':>6}")
    logger.info("-" * 100)

    for domain in sorted(domain_records.keys()):
        recs = domain_records[domain]
        count = len(recs)
        nodes = [r["output"]["n_nodes"] for r in recs]
        edges = [r["output"]["n_edges"] for r in recs]
        densities = [r["output"]["density"] for r in recs]
        dags = [r["output"]["is_dag"] for r in recs]

        min_n = min(nodes)
        max_n = max(nodes)
        med_n = sorted(nodes)[len(nodes) // 2]
        min_e = min(edges)
        max_e = max(edges)
        med_e = sorted(edges)[len(edges) // 2]
        mean_d = sum(densities) / len(densities)
        pct_dag = sum(dags) / len(dags) * 100

        logger.info(f"{domain:<25} {count:>6} {min_n:>6} {max_n:>6} {med_n:>6} "
                    f"{min_e:>6} {max_e:>6} {med_e:>6} {mean_d:>8.4f} {pct_dag:>5.1f}%")

        if count < 25:
            logger.warning(f"  WARN: {domain} has fewer than 25 records ({count})")
            all_ok = False

    # 3. DAG property
    total_dags = sum(1 for r in records if r["output"]["is_dag"])
    pct_dags = total_dags / total * 100 if total > 0 else 0
    logger.info(f"\nOverall DAG percentage: {pct_dags:.1f}% ({total_dags}/{total})")
    if pct_dags < 95:
        logger.warning(f"  WARN: DAG percentage below 95%")

    # 4. Node type coverage
    feature_types = set()
    for r in records:
        for node in r["output"]["nodes"]:
            ft = node.get("feature_type", "")
            if ft:
                feature_types.add(ft)
    logger.info(f"Feature types found: {sorted(feature_types)}")

    # 5. No empty graphs
    empty = sum(1 for r in records if r["output"]["n_nodes"] < 10 or r["output"]["n_edges"] < 1)
    if empty > 0:
        logger.warning(f"  WARN: {empty} records with <10 nodes or <1 edge")
        all_ok = False

    if all_ok:
        logger.info("\nAll validation checks PASSED")
    else:
        logger.warning("\nSome validation checks FAILED (see warnings above)")

    return all_ok


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def write_outputs(records: list[dict]) -> None:
    """Write data_out.json, data_out_mini.json, data_out_preview.json."""
    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    logger.info(f"Wrote {len(records)} records to {out_path}")

    # Mini: first 3 per domain
    from collections import defaultdict
    domain_recs: dict[str, list] = defaultdict(list)
    for r in records:
        domain_recs[r["metadata_fold"]].append(r)

    mini_records = []
    for domain in sorted(domain_recs.keys()):
        mini_records.extend(domain_recs[domain][:3])

    mini_path = WORKSPACE / "data_out_mini.json"
    mini_path.write_text(json.dumps(mini_records, ensure_ascii=False, indent=2))
    logger.info(f"Wrote {len(mini_records)} records to {mini_path}")

    # Preview: first 1 per domain, truncated nodes/links
    preview_records = []
    for domain in sorted(domain_recs.keys()):
        if domain_recs[domain]:
            r = json.loads(json.dumps(domain_recs[domain][0]))  # deep copy
            r["output"]["nodes"] = r["output"]["nodes"][:10]
            r["output"]["links"] = r["output"]["links"][:10]
            preview_records.append(r)

    preview_path = WORKSPACE / "data_out_preview.json"
    preview_path.write_text(json.dumps(preview_records, ensure_ascii=False, indent=2))
    logger.info(f"Wrote {len(preview_records)} records to {preview_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@logger.catch
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect Neuronpedia attribution graphs")
    parser.add_argument("--mode", choices=["mini", "full", "test"], default="full",
                        help="mini=3/domain, full=all, test=single request")
    args = parser.parse_args()

    total_prompts = sum(len(v) for v in PROMPTS.values())
    logger.info(f"Prompts loaded: {total_prompts} across {len(PROMPTS)} domains")

    if args.mode == "test":
        # Single test request to validate API connectivity
        logger.info("Running single test request...")
        record = asyncio.run(_test_single())
        if record:
            logger.info(f"Test OK: {record['output']['n_nodes']} nodes, {record['output']['n_edges']} edges")
        else:
            logger.error("Test FAILED — check API key and connectivity")
            sys.exit(1)
        return

    records = asyncio.run(collect_graphs(mode=args.mode))
    logger.info(f"Collected {len(records)} records")

    if len(records) == 0:
        logger.error("No records collected!")
        sys.exit(1)

    validate_dataset(records)
    write_outputs(records)

    # Check file sizes
    for f in [WORKSPACE / "data_out.json", WORKSPACE / "data_out_mini.json", WORKSPACE / "data_out_preview.json"]:
        if f.exists():
            size_mb = f.stat().st_size / 1e6
            logger.info(f"  {f.name}: {size_mb:.1f} MB")


async def _test_single() -> dict | None:
    """Test a single API request."""
    sem = asyncio.Semaphore(1)
    connector = aiohttp.TCPConnector(limit=5, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        result = await generate_graph(session, "The capital of Japan is", "test", 0, sem)
        if result:
            return parse_graph("The capital of Japan is", "test", result)
    return None


if __name__ == "__main__":
    main()
