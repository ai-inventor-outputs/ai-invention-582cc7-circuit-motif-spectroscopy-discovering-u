#!/usr/bin/env python3
"""Neuronpedia attribution graph collector v5 (iter3).

Merges iter1 (34 graphs, m3-*) and iter2 (140 graphs, m4-*), deduplicates,
backfills iter1 metadata, then collects new graphs from uncollected v4 prompts
and new hard/error-inducing prompts with m5-* prefix.

Target: 250+ unique attribution graphs across 8 domains.
"""

import argparse
import gc
import glob
import hashlib
import json
import math
import os
import resource
import signal
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
DATA_OUT = WORKSPACE / "data_out"
LOGS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
DATA_OUT.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "collect_v5.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Resource limits - container has 29GB RAM, 4 CPUs
# ---------------------------------------------------------------------------
def _container_ram_gb():
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

def _detect_cpus():
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1

TOTAL_RAM_GB = _container_ram_gb() or 29.0
NUM_CPUS = _detect_cpus()
RAM_BUDGET = int(10 * 1024**3)  # 10GB - plenty for I/O-bound work
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (36000, 36000))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget {RAM_BUDGET/1e9:.1f}GB")

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
API_URL = "https://www.neuronpedia.org/api/graph/generate"
API_KEY = os.environ.get("NEURONPEDIA_API_KEY", "sk-np-DQRQw4Us2QtJgy0kq9nZOz39qVIJ0kpy7d8ymN1Ica80")
SLUG_PREFIX = "m5"
REQUEST_SPACING = 45  # Start at 45s, adaptive
CHECKPOINT_FILE = TEMP_DIR / "checkpoint_v5.json"

# Iter1/Iter2 data paths
ITER1_DIR = Path("/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/3_invention_loop/iter_1/gen_art/data_id4_it1__opus/data_out")
ITER2_DIR = Path("/ai-inventor/aii_pipeline/runs/neuronpedia-circuit-interpretability/3_invention_loop/iter_2/gen_art/data_id3_it2__opus/data_out")

# Graceful shutdown
_shutdown = False
_start_time = time.time()
_max_wall_time = 2400  # 40 min default
_current_spacing = REQUEST_SPACING
_consecutive_429 = 0


def _handle_signal(signum, frame):
    global _shutdown
    logger.warning(f"Signal {signum} received, finishing current graph then exiting...")
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def _time_remaining():
    return _max_wall_time - (time.time() - _start_time)


# ---------------------------------------------------------------------------
# v4 PROMPTS dict (264 prompts) - copied from collect_v4.py
# ---------------------------------------------------------------------------
PROMPTS_V4 = {
    "country_capital": [
        {"prompt": "The capital of Japan is", "expected_answer": "Tokyo", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Brazil is", "expected_answer": "Brasilia", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The capital of Nigeria is", "expected_answer": "Abuja", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Australia is", "expected_answer": "Canberra", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of France is", "expected_answer": "Paris", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Egypt is", "expected_answer": "Cairo", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Canada is", "expected_answer": "Ottawa", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Thailand is", "expected_answer": "Bangkok", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Germany is", "expected_answer": "Berlin", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Mexico is", "expected_answer": "Mexico City", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of India is", "expected_answer": "New Delhi", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of South Korea is", "expected_answer": "Seoul", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Argentina is", "expected_answer": "Buenos Aires", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Kenya is", "expected_answer": "Nairobi", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Sweden is", "expected_answer": "Stockholm", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Turkey is", "expected_answer": "Ankara", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The capital of Peru is", "expected_answer": "Lima", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Indonesia is", "expected_answer": "Jakarta", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Poland is", "expected_answer": "Warsaw", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Vietnam is", "expected_answer": "Hanoi", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of South Africa is", "expected_answer": "Pretoria", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Spain is", "expected_answer": "Madrid", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Colombia is", "expected_answer": "Bogota", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Iran is", "expected_answer": "Tehran", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Norway is", "expected_answer": "Oslo", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Chile is", "expected_answer": "Santiago", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Philippines is", "expected_answer": "Manila", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of Morocco is", "expected_answer": "Rabat", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Ukraine is", "expected_answer": "Kyiv", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The capital of New Zealand is", "expected_answer": "Wellington", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Italy is", "expected_answer": "Rome", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Greece is", "expected_answer": "Athens", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The capital of Portugal is", "expected_answer": "Lisbon", "difficulty": "easy", "model_correct": "true"},
    ],
    "arithmetic": [
        {"prompt": "3 + 5 =", "expected_answer": "8", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "15 + 28 =", "expected_answer": "43", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "47 + 36 =", "expected_answer": "83", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "123 + 456 =", "expected_answer": "579", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "8 + 9 =", "expected_answer": "17", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "250 + 375 =", "expected_answer": "625", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "7 + 4 =", "expected_answer": "11", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "12 + 19 =", "expected_answer": "31", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "33 + 67 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "89 + 11 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "6 + 3 =", "expected_answer": "9", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "21 + 34 =", "expected_answer": "55", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "55 + 45 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "99 + 1 =", "expected_answer": "100", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "14 + 27 =", "expected_answer": "41", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "38 + 62 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "2 + 8 =", "expected_answer": "10", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "16 + 84 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "73 + 27 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "5 + 6 =", "expected_answer": "11", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "44 + 56 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "9 + 7 =", "expected_answer": "16", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "31 + 69 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "18 + 22 =", "expected_answer": "40", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "150 + 250 =", "expected_answer": "400", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "4 + 9 =", "expected_answer": "13", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "25 + 75 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "60 + 40 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "11 + 13 =", "expected_answer": "24", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "200 + 300 =", "expected_answer": "500", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "17 + 83 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "42 + 58 =", "expected_answer": "100", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "1 + 2 =", "expected_answer": "3", "difficulty": "easy", "model_correct": "true"},
    ],
    "antonym": [
        {"prompt": "The opposite of happy is", "expected_answer": "sad", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of tall is", "expected_answer": "short", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of fast is", "expected_answer": "slow", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of dark is", "expected_answer": "light", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of cold is", "expected_answer": "hot", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of good is", "expected_answer": "bad", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of old is", "expected_answer": "young", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of rich is", "expected_answer": "poor", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of strong is", "expected_answer": "weak", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of hard is", "expected_answer": "soft", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of loud is", "expected_answer": "quiet", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of heavy is", "expected_answer": "light", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of early is", "expected_answer": "late", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of clean is", "expected_answer": "dirty", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of wet is", "expected_answer": "dry", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of open is", "expected_answer": "closed", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of long is", "expected_answer": "short", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of narrow is", "expected_answer": "wide", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of smooth is", "expected_answer": "rough", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of bright is", "expected_answer": "dim", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of sweet is", "expected_answer": "bitter", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of sharp is", "expected_answer": "dull", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of thick is", "expected_answer": "thin", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of deep is", "expected_answer": "shallow", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of full is", "expected_answer": "empty", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of safe is", "expected_answer": "dangerous", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of cheap is", "expected_answer": "expensive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of brave is", "expected_answer": "cowardly", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of kind is", "expected_answer": "cruel", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of wise is", "expected_answer": "foolish", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of calm is", "expected_answer": "agitated", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The opposite of alive is", "expected_answer": "dead", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The opposite of true is", "expected_answer": "false", "difficulty": "easy", "model_correct": "true"},
    ],
    "translation": [
        {"prompt": "The French word for cat is", "expected_answer": "chat", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for house is", "expected_answer": "maison", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for water is", "expected_answer": "eau", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for book is", "expected_answer": "livre", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for dog is", "expected_answer": "chien", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for tree is", "expected_answer": "arbre", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for sun is", "expected_answer": "soleil", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for moon is", "expected_answer": "lune", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for car is", "expected_answer": "voiture", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for bread is", "expected_answer": "pain", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for fish is", "expected_answer": "poisson", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for bird is", "expected_answer": "oiseau", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for flower is", "expected_answer": "fleur", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for door is", "expected_answer": "porte", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for chair is", "expected_answer": "chaise", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for table is", "expected_answer": "table", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for school is", "expected_answer": "ecole", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for hand is", "expected_answer": "main", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for heart is", "expected_answer": "coeur", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for fire is", "expected_answer": "feu", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for stone is", "expected_answer": "pierre", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for river is", "expected_answer": "riviere", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The French word for star is", "expected_answer": "etoile", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for cloud is", "expected_answer": "nuage", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for rain is", "expected_answer": "pluie", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for snow is", "expected_answer": "neige", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for king is", "expected_answer": "roi", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for night is", "expected_answer": "nuit", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for milk is", "expected_answer": "lait", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for egg is", "expected_answer": "oeuf", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "The French word for apple is", "expected_answer": "pomme", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for cheese is", "expected_answer": "fromage", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "The French word for friend is", "expected_answer": "ami", "difficulty": "easy", "model_correct": "true"},
    ],
    "code_completion": [
        {"prompt": "def add(a, b):\n    return", "expected_answer": "a + b", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def multiply(x, y):\n    return", "expected_answer": "x * y", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def square(n):\n    return", "expected_answer": "n * n", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def maximum(a, b):\n    return", "expected_answer": "max(a, b)", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def is_even(n):\n    return", "expected_answer": "n % 2 == 0", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def absolute(x):\n    return", "expected_answer": "abs(x)", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def negate(n):\n    return", "expected_answer": "-n", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def double(x):\n    return", "expected_answer": "x * 2", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def minimum(a, b):\n    return", "expected_answer": "min(a, b)", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def increment(n):\n    return", "expected_answer": "n + 1", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def subtract(a, b):\n    return", "expected_answer": "a - b", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def divide(a, b):\n    return", "expected_answer": "a / b", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def power(base, exp):\n    return", "expected_answer": "base ** exp", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "def modulo(a, b):\n    return", "expected_answer": "a % b", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "def is_positive(n):\n    return", "expected_answer": "n > 0", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def is_zero(n):\n    return", "expected_answer": "n == 0", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def to_string(x):\n    return", "expected_answer": "str(x)", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def first(lst):\n    return", "expected_answer": "lst[0]", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def last(lst):\n    return", "expected_answer": "lst[-1]", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def length(lst):\n    return", "expected_answer": "len(lst)", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def is_empty(lst):\n    return", "expected_answer": "len(lst) == 0", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def reverse(s):\n    return", "expected_answer": "s[::-1]", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "def upper(s):\n    return", "expected_answer": "s.upper()", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def lower(s):\n    return", "expected_answer": "s.lower()", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def half(n):\n    return", "expected_answer": "n / 2", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def cube(n):\n    return", "expected_answer": "n ** 3", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "def is_odd(n):\n    return", "expected_answer": "n % 2 != 0", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def sign(n):\n    return", "expected_answer": "1 if n > 0 else (-1 if n < 0 else 0)", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "def identity(x):\n    return", "expected_answer": "x", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "def floor_div(a, b):\n    return", "expected_answer": "a // b", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "def average(a, b):\n    return", "expected_answer": "(a + b) / 2", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "def clamp(x, lo, hi):\n    return", "expected_answer": "max(lo, min(x, hi))", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "def triple(n):\n    return", "expected_answer": "n * 3", "difficulty": "easy", "model_correct": "true"},
    ],
    "multi_hop_reasoning": [
        {"prompt": "If Alice is taller than Bob and Bob is taller than Carol, then Alice is", "expected_answer": "taller than Carol", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If X is larger than Y and Y is larger than Z, then X is", "expected_answer": "larger than Z", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the red box is heavier than the blue box and the blue box is heavier than the green box, the red box is", "expected_answer": "heavier than the green box", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If John is older than Mary and Mary is older than Tom, then John is", "expected_answer": "older than Tom", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If A is faster than B and B is faster than C, then A is", "expected_answer": "faster than C", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the cat is bigger than the dog and the dog is bigger than the mouse, the cat is", "expected_answer": "bigger than the mouse", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If Paris is north of Rome and Rome is north of Cairo, Paris is", "expected_answer": "north of Cairo", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "If iron is harder than copper and copper is harder than gold, iron is", "expected_answer": "harder than gold", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "If Sam runs faster than Dan and Dan runs faster than Pat, Sam runs", "expected_answer": "faster than Pat", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the oak is taller than the pine and the pine is taller than the bush, the oak is", "expected_answer": "taller than the bush", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If spring comes before summer and summer comes before fall, spring comes", "expected_answer": "before fall", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If diamond is harder than steel and steel is harder than wood, diamond is", "expected_answer": "harder than wood", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "If the river is wider than the stream and the stream is wider than the creek, the river is", "expected_answer": "wider than the creek", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If Mars is farther than Venus and Venus is farther than Mercury, Mars is", "expected_answer": "farther than Mercury", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "If the whale is larger than the shark and the shark is larger than the tuna, the whale is", "expected_answer": "larger than the tuna", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If gold costs more than silver and silver costs more than bronze, gold costs", "expected_answer": "more than bronze", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If Lisa scored higher than Mike and Mike scored higher than Nina, Lisa scored", "expected_answer": "higher than Nina", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the mountain is higher than the hill and the hill is higher than the valley, the mountain is", "expected_answer": "higher than the valley", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the elephant weighs more than the horse and the horse weighs more than the sheep, the elephant weighs", "expected_answer": "more than the sheep", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If Monday comes before Tuesday and Tuesday comes before Wednesday, Monday comes", "expected_answer": "before Wednesday", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the sun is brighter than the moon and the moon is brighter than a star, the sun is", "expected_answer": "brighter than a star", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If French is harder than Spanish and Spanish is harder than Italian, French is", "expected_answer": "harder than Italian", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "If the train is faster than the car and the car is faster than the bicycle, the train is", "expected_answer": "faster than the bicycle", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If Earth is bigger than Mars and Mars is bigger than Mercury, Earth is", "expected_answer": "bigger than Mercury", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the CEO earns more than the manager and the manager earns more than the clerk, the CEO earns", "expected_answer": "more than the clerk", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the Pacific is deeper than the Atlantic and the Atlantic is deeper than the Arctic, the Pacific is", "expected_answer": "deeper than the Arctic", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "If Einstein was smarter than Newton and Newton was smarter than Galileo, Einstein was", "expected_answer": "smarter than Galileo", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "If the cheetah is faster than the lion and the lion is faster than the bear, the cheetah is", "expected_answer": "faster than the bear", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the piano is heavier than the guitar and the guitar is heavier than the flute, the piano is", "expected_answer": "heavier than the flute", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the novel is longer than the story and the story is longer than the poem, the novel is", "expected_answer": "longer than the poem", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the lake is deeper than the pond and the pond is deeper than the puddle, the lake is", "expected_answer": "deeper than the puddle", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If January is colder than March and March is colder than May, January is", "expected_answer": "colder than May", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "If the castle is older than the church and the church is older than the school, the castle is", "expected_answer": "older than the school", "difficulty": "medium", "model_correct": "true"},
    ],
    "rhyme": [
        {"prompt": "The cat in the hat sat on the", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "A mouse in the house found a", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The dog on the log saw a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The bee in the tree drank some", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The fish made a wish for a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The bear on the stair ate a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The fox in the box wore some", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The hen in the den found a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The goat in the boat wore a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The frog on the log wrote a", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The fly in the sky ate a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The snake by the lake ate some", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The crow in the snow found a", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The bat in the hat found a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The owl on the towel wore a", "expected_answer": "multiple_valid", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The bug on the rug ate a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The duck in the truck had some", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The ram in the jam ate some", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The seal with the meal had a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The snail on the trail found a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The worm in the storm found a", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The moose on the loose drank some", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The mole in the hole found some", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The toad on the road had a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The pup in the cup drank some", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The lark in the park ate a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The hare with the flair wore a", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The eel in the reel caught a", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The ant on the plant ate a", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
        {"prompt": "The dove from above found some", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The pig in the wig wore a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The yak in the shack ate a", "expected_answer": "multiple_valid", "difficulty": "easy", "model_correct": "unknown"},
        {"prompt": "The skunk on the bunk found a", "expected_answer": "multiple_valid", "difficulty": "medium", "model_correct": "unknown"},
    ],
    "sentiment": [
        {"prompt": "This movie was terrible. The sentiment is", "expected_answer": "negative", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was wonderful. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was boring. The sentiment is", "expected_answer": "negative", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was exciting. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was awful. The sentiment is", "expected_answer": "negative", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was amazing. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was dull. The sentiment is", "expected_answer": "negative", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was thrilling. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was disappointing. The sentiment is", "expected_answer": "negative", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was brilliant. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was mediocre. The sentiment is", "expected_answer": "negative", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "This movie was fantastic. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was horrible. The sentiment is", "expected_answer": "negative", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was delightful. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was tedious. The sentiment is", "expected_answer": "negative", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "This movie was outstanding. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was miserable. The sentiment is", "expected_answer": "negative", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was superb. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was forgettable. The sentiment is", "expected_answer": "negative", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "This movie was spectacular. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was painful. The sentiment is", "expected_answer": "negative", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was charming. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was frustrating. The sentiment is", "expected_answer": "negative", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "This movie was inspiring. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was annoying. The sentiment is", "expected_answer": "negative", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "This movie was excellent. The sentiment is", "expected_answer": "positive", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was depressing. The sentiment is", "expected_answer": "negative", "difficulty": "easy", "model_correct": "true"},
        {"prompt": "This movie was heartwarming. The sentiment is", "expected_answer": "positive", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "This movie was dreadful. The sentiment is", "expected_answer": "negative", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "This movie was captivating. The sentiment is", "expected_answer": "positive", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "This movie was unbearable. The sentiment is", "expected_answer": "negative", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "This movie was enchanting. The sentiment is", "expected_answer": "positive", "difficulty": "medium", "model_correct": "true"},
        {"prompt": "This movie was atrocious. The sentiment is", "expected_answer": "negative", "difficulty": "medium", "model_correct": "true"},
    ],
}

# Build prompt -> metadata lookup
_V4_PROMPT_LOOKUP = {}
for _domain, _entries in PROMPTS_V4.items():
    for _entry in _entries:
        _V4_PROMPT_LOOKUP[_entry["prompt"]] = {
            "domain": _domain,
            "expected_answer": _entry["expected_answer"],
            "difficulty": _entry["difficulty"],
            "model_correct": _entry["model_correct"],
        }

# ---------------------------------------------------------------------------
# New hard prompts for iter3
# ---------------------------------------------------------------------------
NEW_HARD_PROMPTS = {
    "country_capital": [
        {"prompt": "The capital of Myanmar is", "expected_answer": "Naypyidaw", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Ivory Coast is", "expected_answer": "Yamoussoukro", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Tanzania is", "expected_answer": "Dodoma", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Sri Lanka is", "expected_answer": "Sri Jayawardenepura Kotte", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Kazakhstan is", "expected_answer": "Astana", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Belize is", "expected_answer": "Belmopan", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The capital of Palau is", "expected_answer": "Ngerulmud", "difficulty": "hard", "model_correct": "unknown"},
    ],
    "arithmetic": [
        {"prompt": "847 + 293 =", "expected_answer": "1140", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "698 + 457 =", "expected_answer": "1155", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "999 + 1 =", "expected_answer": "1000", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "376 + 285 =", "expected_answer": "661", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "1234 + 5678 =", "expected_answer": "6912", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "509 + 491 =", "expected_answer": "1000", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "777 + 888 =", "expected_answer": "1665", "difficulty": "hard", "model_correct": "unknown"},
    ],
    "antonym": [
        {"prompt": "The opposite of garrulous is", "expected_answer": "taciturn", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The opposite of ephemeral is", "expected_answer": "eternal", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The opposite of loquacious is", "expected_answer": "taciturn", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The opposite of parsimonious is", "expected_answer": "generous", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The opposite of sanguine is", "expected_answer": "pessimistic", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The opposite of obsequious is", "expected_answer": "domineering", "difficulty": "hard", "model_correct": "unknown"},
    ],
    "translation": [
        {"prompt": "The German word for butterfly is", "expected_answer": "Schmetterling", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The Spanish word for library is", "expected_answer": "biblioteca", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The Japanese word for thank you is", "expected_answer": "arigatou", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The Italian word for window is", "expected_answer": "finestra", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The Portuguese word for cheese is", "expected_answer": "queijo", "difficulty": "hard", "model_correct": "unknown"},
    ],
    "code_completion": [
        {"prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return", "expected_answer": "fibonacci(n-1) + fibonacci(n-2)", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "def is_palindrome(s):\n    return", "expected_answer": "s == s[::-1]", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "def factorial(n):\n    if n == 0:\n        return 1\n    return", "expected_answer": "n * factorial(n-1)", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return", "expected_answer": "a", "difficulty": "hard", "model_correct": "unknown"},
    ],
    "multi_hop_reasoning": [
        {"prompt": "If A is NOT taller than B and B is NOT taller than C, then A is", "expected_answer": "not taller than C", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "If X > Y and Y > Z and Z > W, then X is", "expected_answer": "greater than W", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "If the red box is heavier than blue and blue is heavier than red, then", "expected_answer": "contradiction", "difficulty": "hard", "model_correct": "unknown"},
    ],
    "rhyme": [
        {"prompt": "A bright light in the night gave a", "expected_answer": "multiple_valid", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "The knight in the moonlight took flight to the", "expected_answer": "multiple_valid", "difficulty": "hard", "model_correct": "unknown"},
    ],
    "sentiment": [
        {"prompt": "This movie was okay. The sentiment is", "expected_answer": "neutral", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "This movie was interesting but flawed. The sentiment is", "expected_answer": "mixed", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "This movie was not bad. The sentiment is", "expected_answer": "positive", "difficulty": "hard", "model_correct": "unknown"},
        {"prompt": "This movie had its moments. The sentiment is", "expected_answer": "mixed", "difficulty": "hard", "model_correct": "unknown"},
    ],
}

DOMAINS = list(PROMPTS_V4.keys())


# ---------------------------------------------------------------------------
# Load existing data
# ---------------------------------------------------------------------------
def load_iter_data(data_dir: Path, file_pattern: str = "full_data_out_*.json") -> list[dict]:
    """Load examples from split JSON files in data_dir."""
    examples = []
    files = sorted(data_dir.glob(file_pattern))
    for f in files:
        logger.info(f"  Loading {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        try:
            data = json.loads(f.read_text())
            if "datasets" in data:
                for ds in data["datasets"]:
                    examples.extend(ds.get("examples", []))
            elif "examples" in data:
                examples.extend(data["examples"])
        except Exception:
            logger.exception(f"  Failed to load {f.name}")
        gc.collect()
    return examples


def normalize_record(record: dict, provenance: str, iter_num: int) -> dict:
    """Ensure record has all required fields, fill missing with defaults."""
    # Ensure output is a JSON string
    output = record.get("output", "")
    if isinstance(output, dict):
        output = json.dumps(output)

    # Parse output to get graph stats
    try:
        graph = json.loads(output) if isinstance(output, str) else output
        nodes = graph.get("nodes", [])
        links = graph.get("links", [])
        n_nodes = len(nodes)
        n_edges = len(links)
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
        feature_types = sorted(set(n.get("feature_type", "") for n in nodes if n.get("feature_type")))

        # Build DAG for validation
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node.get("node_id", ""))
        for link in links:
            G.add_edge(link.get("source", ""), link.get("target", ""))
        is_dag = nx.is_directed_acyclic_graph(G)
    except Exception:
        n_nodes = record.get("metadata_n_nodes", 0)
        n_edges = record.get("metadata_n_edges", 0)
        density = record.get("metadata_density", 0.0)
        is_dag = record.get("metadata_is_dag", True)
        feature_types = record.get("metadata_feature_names", [])

    prompt = record.get("input", "")
    domain = record.get("metadata_fold", "")

    # Look up v4 metadata for backfilling
    v4_meta = _V4_PROMPT_LOOKUP.get(prompt, {})

    return {
        "input": prompt,
        "output": output if isinstance(output, str) else json.dumps(output),
        "metadata_fold": domain or v4_meta.get("domain", "unknown"),
        "metadata_n_nodes": n_nodes,
        "metadata_n_edges": n_edges,
        "metadata_density": round(density, 6),
        "metadata_is_dag": is_dag,
        "metadata_model_correct": record.get("metadata_model_correct", v4_meta.get("model_correct", "unknown")),
        "metadata_difficulty": record.get("metadata_difficulty", v4_meta.get("difficulty", "medium")),
        "metadata_expected_answer": record.get("metadata_expected_answer", v4_meta.get("expected_answer", "unknown")),
        "metadata_iter": iter_num,
        "metadata_provenance": provenance,
        "metadata_slug": record.get("metadata_slug", ""),
        "metadata_task_type": "graph_generation",
        "metadata_n_classes": 8,
        "metadata_row_index": 0,  # Will be reassigned later
        "metadata_feature_names": feature_types if feature_types else record.get("metadata_feature_names", []),
    }


# ---------------------------------------------------------------------------
# Correctness verification via logit nodes
# ---------------------------------------------------------------------------
def verify_correctness(record: dict) -> str:
    """Check if model output matches expected answer by inspecting logit nodes."""
    expected = record.get("metadata_expected_answer", "")
    if not expected or expected == "unknown" or expected == "multiple_valid":
        return record.get("metadata_model_correct", "unknown")

    try:
        graph = json.loads(record["output"]) if isinstance(record["output"], str) else record["output"]
        nodes = graph.get("nodes", [])

        # Find logit nodes
        logit_nodes = [n for n in nodes if n.get("feature_type") == "logit" or n.get("is_target_logit")]
        if not logit_nodes:
            return record.get("metadata_model_correct", "unknown")

        # Extract top logit token from clerp or token field
        top_logit = None
        max_prob = -1
        for node in logit_nodes:
            prob = node.get("token_prob", 0.0)
            clerp = node.get("clerp", "")
            if prob > max_prob:
                max_prob = prob
                if clerp.startswith("logit: "):
                    top_logit = clerp[7:].strip()
                elif clerp:
                    top_logit = clerp.strip()

        if top_logit is None:
            return record.get("metadata_model_correct", "unknown")

        # Compare (case-insensitive, strip whitespace)
        top_clean = top_logit.lower().strip()
        exp_clean = expected.lower().strip()

        if exp_clean in top_clean or top_clean in exp_clean:
            return "true"
        elif top_clean:
            return "false"
        return "unknown"

    except Exception:
        return record.get("metadata_model_correct", "unknown")


# ---------------------------------------------------------------------------
# API collection
# ---------------------------------------------------------------------------
def generate_graph(prompt: str, domain: str, idx: int) -> dict | None:
    """Call Neuronpedia API, handle errors, download S3 graph."""
    global _current_spacing, _consecutive_429

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

    for attempt in range(5):
        if _time_remaining() < 180:
            logger.warning("Time running low, aborting request")
            return None

        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)

            if resp.status_code == 200:
                _consecutive_429 = 0
                if _current_spacing > 45:
                    _current_spacing = max(45, _current_spacing - 10)
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
                _consecutive_429 += 1
                body = {}
                try:
                    body = resp.json()
                except Exception:
                    pass
                remaining = body.get("remainingRequests", 0)
                if _consecutive_429 >= 3:
                    _current_spacing = 120
                elif _consecutive_429 >= 1:
                    _current_spacing = 90
                wait = max(120, _current_spacing)
                logger.warning(f"  429 (remaining={remaining}) for {slug}, spacing->{_current_spacing}s, wait {wait}s (attempt {attempt+1})")
                time.sleep(wait)
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
            logger.warning("  S3 invalid JSON")
        except Exception:
            logger.exception("  S3 error")
        time.sleep(5)
    return None


def parse_new_graph(prompt_entry: dict, domain: str, api_result: dict) -> dict | None:
    """Parse API result into a normalized record."""
    try:
        graph_data = api_result["graph_data"]
        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", [])
        n_nodes = len(nodes)
        n_edges = len(links)

        if n_nodes < 10:
            logger.warning(f"  Graph too small: {n_nodes} nodes (<10)")
            return None

        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node.get("node_id", ""))
        for link in links:
            G.add_edge(link.get("source", ""), link.get("target", ""))
        is_dag = nx.is_directed_acyclic_graph(G)

        if not is_dag:
            logger.warning(f"  Graph is NOT a DAG for {prompt_entry['prompt'][:40]}")

        feature_types = sorted(set(n.get("feature_type", "") for n in nodes if n.get("feature_type")))

        output_str = json.dumps({"nodes": nodes, "links": links})

        return {
            "input": prompt_entry["prompt"],
            "output": output_str,
            "metadata_fold": domain,
            "metadata_n_nodes": n_nodes,
            "metadata_n_edges": n_edges,
            "metadata_density": round(density, 6),
            "metadata_is_dag": is_dag,
            "metadata_model_correct": prompt_entry.get("model_correct", "unknown"),
            "metadata_difficulty": prompt_entry.get("difficulty", "medium"),
            "metadata_expected_answer": prompt_entry.get("expected_answer", "unknown"),
            "metadata_iter": 3,
            "metadata_provenance": "iter3",
            "metadata_slug": api_result.get("slug", ""),
            "metadata_task_type": "graph_generation",
            "metadata_n_classes": 8,
            "metadata_row_index": 0,
            "metadata_feature_names": feature_types,
        }
    except Exception:
        logger.exception(f"  Parse error for: {prompt_entry['prompt'][:50]}")
        return None


# ---------------------------------------------------------------------------
# Checkpoint for iter3 collection
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            logger.info(f"Loaded checkpoint: {len(data.get('records', []))} iter3 records")
            return data
        except Exception:
            logger.exception("Checkpoint load failed, starting fresh")
    return {"records": [], "done": []}


def save_checkpoint(records: list, done: list) -> None:
    CHECKPOINT_FILE.write_text(json.dumps({"records": records, "done": done}))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_split_output(records: list[dict], max_bytes: int = 90_000_000) -> list[Path]:
    """Write records into split JSON files under max_bytes each."""
    metadata = {
        "source": "Neuronpedia API (POST /api/graph/generate)",
        "model": "gemma-2-2b",
        "description": "Attribution graphs across 8 capability domains with correctness annotations (merged iter1+iter2+iter3)",
        "iter": 3,
        "collection_params": {
            "nodeThreshold": 0.8,
            "edgeThreshold": 0.85,
            "maxFeatureNodes": 5000,
            "maxNLogits": 10,
            "desiredLogitProb": 0.95,
        },
        "total_records": len(records),
        "domains": DOMAINS,
        "provenance_breakdown": {},
    }

    # Count provenance
    prov_counts = defaultdict(int)
    for r in records:
        prov_counts[r.get("metadata_provenance", "unknown")] += 1
    metadata["provenance_breakdown"] = dict(prov_counts)

    split_files = []
    current_batch = []
    current_size = 0
    file_idx = 1
    overhead = len(json.dumps({"metadata": metadata, "datasets": [{"dataset": "neuronpedia_attribution_graphs_v3", "examples": []}]}).encode())

    for record in records:
        rec_size = len(json.dumps(record).encode()) + 2  # comma + newline
        if current_size + rec_size + overhead > max_bytes and current_batch:
            out_path = DATA_OUT / f"full_data_out_{file_idx}.json"
            out_data = {
                "metadata": metadata,
                "datasets": [{
                    "dataset": "neuronpedia_attribution_graphs_v3",
                    "examples": current_batch,
                }],
            }
            out_path.write_text(json.dumps(out_data))
            split_files.append(out_path)
            logger.info(f"  Wrote {out_path.name}: {len(current_batch)} records, {out_path.stat().st_size / 1e6:.1f} MB")
            file_idx += 1
            current_batch = []
            current_size = 0

        current_batch.append(record)
        current_size += rec_size

    # Write remaining
    if current_batch:
        out_path = DATA_OUT / f"full_data_out_{file_idx}.json"
        out_data = {
            "metadata": metadata,
            "datasets": [{
                "dataset": "neuronpedia_attribution_graphs_v3",
                "examples": current_batch,
            }],
        }
        out_path.write_text(json.dumps(out_data))
        split_files.append(out_path)
        logger.info(f"  Wrote {out_path.name}: {len(current_batch)} records, {out_path.stat().st_size / 1e6:.1f} MB")

    return split_files


def write_mini_preview(records: list[dict]) -> None:
    """Write mini (16 rows, 2/domain) and preview (8 rows, 1/domain) versions."""
    metadata = {
        "source": "Neuronpedia API (POST /api/graph/generate)",
        "model": "gemma-2-2b",
        "description": "Attribution graphs (mini/preview)",
        "iter": 3,
    }

    # Mini: 2 per domain
    mini_records = []
    domain_counts = defaultdict(int)
    for r in records:
        d = r["metadata_fold"]
        if domain_counts[d] < 2:
            mini_records.append(r)
            domain_counts[d] += 1
        if len(mini_records) >= 16:
            break

    mini_data = {
        "metadata": metadata,
        "datasets": [{"dataset": "neuronpedia_attribution_graphs_v3", "examples": mini_records}],
    }
    mini_path = DATA_OUT / "mini_data_out_1.json"
    mini_path.write_text(json.dumps(mini_data))
    logger.info(f"  Wrote mini: {len(mini_records)} records, {mini_path.stat().st_size / 1e6:.1f} MB")

    # Preview: 1 per domain, truncated output
    preview_records = []
    seen_domains = set()
    for r in records:
        d = r["metadata_fold"]
        if d not in seen_domains:
            pr = dict(r)
            # Truncate output to 200 chars for preview
            if len(pr.get("output", "")) > 200:
                pr["output"] = pr["output"][:200] + "..."
            preview_records.append(pr)
            seen_domains.add(d)
        if len(preview_records) >= 8:
            break

    preview_data = {
        "metadata": metadata,
        "datasets": [{"dataset": "neuronpedia_attribution_graphs_v3", "examples": preview_records}],
    }
    preview_path = DATA_OUT / "preview_data_out_1.json"
    preview_path.write_text(json.dumps(preview_data))
    logger.info(f"  Wrote preview: {len(preview_records)} records, {preview_path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    global _max_wall_time, _start_time
    _start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-time", type=int, default=2400,
                        help="Max wall time in seconds (default 2400 = 40 min)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip API collection, just merge iter1+iter2")
    parser.add_argument("--skip-api", action="store_true",
                        help="Same as --merge-only")
    args = parser.parse_args()
    _max_wall_time = args.max_time
    merge_only = args.merge_only or args.skip_api

    logger.info("=" * 60)
    logger.info("Neuronpedia Attribution Graph Collector v5 (iter3)")
    logger.info(f"Max wall time: {_max_wall_time}s ({_max_wall_time/60:.0f} min)")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Load and merge iter1 + iter2
    # -----------------------------------------------------------------------
    logger.info("\n=== STEP 1: Loading existing data ===")

    logger.info("Loading iter1 (m3-* slugs)...")
    iter1_raw = load_iter_data(ITER1_DIR)
    logger.info(f"  iter1: {len(iter1_raw)} raw records")

    logger.info("Loading iter2 (m4-* slugs)...")
    iter2_raw = load_iter_data(ITER2_DIR)
    logger.info(f"  iter2: {len(iter2_raw)} raw records")

    # Normalize
    iter1_records = [normalize_record(r, "iter1", 1) for r in iter1_raw]
    iter2_records = [normalize_record(r, "iter2", 2) for r in iter2_raw]
    del iter1_raw, iter2_raw
    gc.collect()

    # Deduplicate by prompt text - prefer iter2 (richer metadata)
    seen_prompts = {}
    for r in iter2_records:
        seen_prompts[r["input"]] = r
    for r in iter1_records:
        if r["input"] not in seen_prompts:
            seen_prompts[r["input"]] = r

    merged = list(seen_prompts.values())
    logger.info(f"After dedup: {len(merged)} unique graphs (iter1-only: {len(merged) - len(iter2_records)}, iter2: {len(iter2_records)})")

    # -----------------------------------------------------------------------
    # Step 2: Gap analysis
    # -----------------------------------------------------------------------
    logger.info("\n=== STEP 2: Gap analysis ===")
    domain_counts = defaultdict(int)
    for r in merged:
        domain_counts[r["metadata_fold"]] += 1
    for d in DOMAINS:
        logger.info(f"  {d}: {domain_counts[d]}")
    logger.info(f"  TOTAL: {len(merged)}")

    # Find uncollected v4 prompts
    collected_prompts = set(r["input"] for r in merged)
    uncollected_v4 = []
    for domain, entries in PROMPTS_V4.items():
        for idx, entry in enumerate(entries):
            if entry["prompt"] not in collected_prompts:
                uncollected_v4.append((domain, idx, entry))
    logger.info(f"  Uncollected v4 prompts: {len(uncollected_v4)}")

    # New hard prompts (all new)
    new_hard = []
    hard_idx_counter = defaultdict(lambda: 100)  # Start at 100 to avoid collisions
    for domain, entries in NEW_HARD_PROMPTS.items():
        for entry in entries:
            if entry["prompt"] not in collected_prompts:
                idx = hard_idx_counter[domain]
                hard_idx_counter[domain] += 1
                new_hard.append((domain, idx, entry))
    logger.info(f"  New hard prompts: {len(new_hard)}")

    total_to_collect = len(uncollected_v4) + len(new_hard)
    logger.info(f"  Total to collect: {total_to_collect}")

    # -----------------------------------------------------------------------
    # Step 3: API collection (if not merge-only)
    # -----------------------------------------------------------------------
    iter3_records = []

    if not merge_only and total_to_collect > 0:
        logger.info("\n=== STEP 3: API collection ===")

        # Load checkpoint
        ckpt = load_checkpoint()
        iter3_records = ckpt["records"]
        done_set = set(ckpt["done"])
        logger.info(f"Checkpoint: {len(iter3_records)} existing iter3 records, {len(done_set)} done")

        # Add checkpoint records to collected set
        for r in iter3_records:
            collected_prompts.add(r["input"])

        # Build round-robin work queue: uncollected v4 first, then hard
        work_queue = []
        # Interleave domains for round-robin
        max_uncollected = max((len([x for x in uncollected_v4 if x[0] == d]) for d in DOMAINS), default=0)
        for round_idx in range(max_uncollected):
            for domain in DOMAINS:
                domain_items = [x for x in uncollected_v4 if x[0] == domain]
                if round_idx < len(domain_items):
                    work_queue.append(domain_items[round_idx])

        max_hard = max((len([x for x in new_hard if x[0] == d]) for d in DOMAINS), default=0)
        for round_idx in range(max_hard):
            for domain in DOMAINS:
                domain_items = [x for x in new_hard if x[0] == domain]
                if round_idx < len(domain_items):
                    work_queue.append(domain_items[round_idx])

        # Filter out already-done items
        remaining = [(d, i, p) for d, i, p in work_queue
                      if f"{d}:{i}" not in done_set and p["prompt"] not in collected_prompts]

        logger.info(f"Work queue: {len(remaining)} items (after filtering done)")
        logger.info(f"Rate limit spacing: {_current_spacing}s (adaptive)")

        success_count = len(iter3_records)
        fail_count = 0
        collection_start = time.time()

        for work_idx, (domain, idx, prompt_entry) in enumerate(remaining):
            if _shutdown:
                logger.warning("Shutdown requested, stopping collection")
                break

            if _time_remaining() < 300:
                logger.info(f"Time limit approaching ({_time_remaining():.0f}s left), stopping")
                break

            key = f"{domain}:{idx}"
            elapsed_total = (time.time() - collection_start) / 60
            logger.info(
                f"[{work_idx+1}/{len(remaining)}] {domain}[{idx}]: "
                f"{prompt_entry['prompt'][:55]}... ({elapsed_total:.0f}m elapsed, {success_count} ok)"
            )

            t0 = time.time()
            result = generate_graph(prompt_entry["prompt"], domain, idx)
            gen_time = time.time() - t0

            if result is not None:
                record = parse_new_graph(prompt_entry, domain, result)
                if record is not None:
                    # Verify correctness
                    record["metadata_model_correct"] = verify_correctness(record)
                    iter3_records.append(record)
                    done_set.add(key)
                    collected_prompts.add(record["input"])
                    success_count += 1
                    logger.info(
                        f"  OK: {record['metadata_n_nodes']} nodes, "
                        f"{record['metadata_n_edges']} edges, "
                        f"DAG={record['metadata_is_dag']}, {gen_time:.1f}s "
                        f"[{record['metadata_difficulty']}/{record['metadata_model_correct']}]"
                    )
                    save_checkpoint(iter3_records, list(done_set))
                else:
                    logger.warning("  Graph too small or parse failed, skipping")
                    done_set.add(key)
                    fail_count += 1
                    save_checkpoint(iter3_records, list(done_set))
            else:
                fail_count += 1

            # Adaptive rate-limit pacing
            wait_needed = max(0, _current_spacing - gen_time)
            if wait_needed > 0 and work_idx < len(remaining) - 1:
                if _time_remaining() < wait_needed + 180:
                    logger.info("Not enough time for next wait+request, stopping")
                    break
                logger.debug(f"  Pacing wait: {wait_needed:.0f}s")
                time.sleep(wait_needed)

        elapsed_collect = (time.time() - collection_start) / 60
        logger.info(f"Collection done: {success_count} total iter3 records, {fail_count} failures, {elapsed_collect:.1f} min")
    elif merge_only:
        logger.info("\n=== STEP 3: Skipped (merge-only mode) ===")
        # Load any existing iter3 checkpoint records
        ckpt = load_checkpoint()
        iter3_records = ckpt.get("records", [])
        logger.info(f"Loaded {len(iter3_records)} existing iter3 checkpoint records")

    # -----------------------------------------------------------------------
    # Step 4: Verify correctness for all records
    # -----------------------------------------------------------------------
    logger.info("\n=== STEP 4: Correctness verification ===")
    correct_counts = defaultdict(int)
    verified_count = 0
    for r in merged:
        # For iter2 records that already have non-"unknown" model_correct, trust the annotation
        # Only verify records with unknown correctness
        if r.get("metadata_model_correct") == "unknown":
            new_val = verify_correctness(r)
            if new_val != "unknown":
                r["metadata_model_correct"] = new_val
                verified_count += 1
        correct_counts[r["metadata_model_correct"]] += 1
    for r in iter3_records:
        # All iter3 records get verified
        new_val = verify_correctness(r)
        if new_val != "unknown":
            r["metadata_model_correct"] = new_val
            verified_count += 1
        correct_counts[r["metadata_model_correct"]] += 1
    logger.info(f"  Correctness: {dict(correct_counts)} (verified {verified_count})")

    # -----------------------------------------------------------------------
    # Step 5: Final merge
    # -----------------------------------------------------------------------
    logger.info("\n=== STEP 5: Final merge ===")

    # Merge iter3 records into the main set
    for r in iter3_records:
        if r["input"] not in seen_prompts:
            merged.append(r)
            seen_prompts[r["input"]] = r

    # Filter: only DAGs with >= 10 nodes
    valid = [r for r in merged if r["metadata_is_dag"] and r["metadata_n_nodes"] >= 10]
    invalid = len(merged) - len(valid)
    if invalid > 0:
        logger.warning(f"  Discarded {invalid} invalid graphs (non-DAG or <10 nodes)")
    logger.info(f"  Valid graphs: {len(valid)}")

    # Sort by domain -> difficulty -> prompt
    diff_order = {"easy": 0, "medium": 1, "hard": 2}
    valid.sort(key=lambda r: (r["metadata_fold"], diff_order.get(r["metadata_difficulty"], 1), r["input"]))

    # Assign sequential row indices
    for i, r in enumerate(valid):
        r["metadata_row_index"] = i

    # -----------------------------------------------------------------------
    # Step 6: Statistics
    # -----------------------------------------------------------------------
    logger.info("\n=== STEP 6: Statistics ===")
    logger.info(f"Total valid graphs: {len(valid)}")

    domain_stats = defaultdict(lambda: {"count": 0, "nodes": [], "edges": []})
    diff_counts = defaultdict(int)
    correct_counts2 = defaultdict(int)
    prov_counts = defaultdict(int)
    for r in valid:
        d = r["metadata_fold"]
        domain_stats[d]["count"] += 1
        domain_stats[d]["nodes"].append(r["metadata_n_nodes"])
        domain_stats[d]["edges"].append(r["metadata_n_edges"])
        diff_counts[r["metadata_difficulty"]] += 1
        correct_counts2[r["metadata_model_correct"]] += 1
        prov_counts[r["metadata_provenance"]] += 1

    logger.info("Per-domain counts:")
    for d in DOMAINS:
        s = domain_stats[d]
        if s["count"] > 0:
            avg_n = sum(s["nodes"]) / len(s["nodes"])
            avg_e = sum(s["edges"]) / len(s["edges"])
            logger.info(f"  {d}: {s['count']} graphs (avg {avg_n:.0f} nodes, {avg_e:.0f} edges)")
        else:
            logger.info(f"  {d}: 0 graphs")

    logger.info(f"Difficulty: {dict(diff_counts)}")
    logger.info(f"Correctness: {dict(correct_counts2)}")
    logger.info(f"Provenance: {dict(prov_counts)}")

    # -----------------------------------------------------------------------
    # Step 7: Write output
    # -----------------------------------------------------------------------
    logger.info("\n=== STEP 7: Writing output ===")
    split_files = write_split_output(valid)
    logger.info(f"  Split into {len(split_files)} files")

    write_mini_preview(valid)

    elapsed_total = (time.time() - _start_time) / 60
    logger.info(f"\nDONE in {elapsed_total:.1f} min. Output: {len(valid)} graphs in {len(split_files)} files")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
