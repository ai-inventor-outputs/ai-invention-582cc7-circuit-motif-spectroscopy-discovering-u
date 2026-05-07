#!/usr/bin/env python3
"""Neuronpedia attribution graph collector v4 (iter2).

Extends collect_v3 with:
- Slug prefix "m4" (avoids collision with iter1 "m3" graphs)
- Annotated prompts: expected_answer, difficulty, model_correct
- --max-time flag for graceful exit within 1-hour script limit
- New metadata fields in parsed records

Rate limit: 30 requests per 60-minute sliding window.
Strategy: 120s spacing, checkpoint after every graph, round-robin domains.
"""

import argparse
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
LOGS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "collect_v4.log"), rotation="30 MB", level="DEBUG")

# Resource limits - container has 29GB RAM, 4 CPUs
# RAM: use at most 8GB (this script is I/O-bound, not memory-heavy)
_container_ram = None
for _p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
    try:
        _v = Path(_p).read_text().strip()
        if _v != "max" and int(_v) < 1_000_000_000_000:
            _container_ram = int(_v)
            break
    except (FileNotFoundError, ValueError):
        pass
RAM_BUDGET = min(8 * 1024**3, int((_container_ram or 29 * 1024**3) * 0.5))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (36000, 36000))  # 10 hours CPU time

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
API_URL = "https://www.neuronpedia.org/api/graph/generate"
API_KEY = os.environ.get("NEURONPEDIA_API_KEY", "sk-np-DQRQw4Us2QtJgy0kq9nZOz39qVIJ0kpy7d8ymN1Ica80")
SLUG_PREFIX = "m4"
REQUEST_SPACING = 120  # 30/hour max throughput
CHECKPOINT_FILE = TEMP_DIR / "checkpoint_v4.json"

# Graceful shutdown
_shutdown = False
_start_time = time.time()
_max_wall_time = 3300  # 55 min default, overridden by --max-time


def _handle_signal(signum, frame):
    global _shutdown
    logger.warning(f"Signal {signum} received, finishing current graph then exiting...")
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def _time_remaining() -> float:
    """Seconds remaining before max wall time."""
    return _max_wall_time - (time.time() - _start_time)


# ---------------------------------------------------------------------------
# Annotated prompts: 8 domains x 33 = 264
# Each entry: {"prompt": str, "expected_answer": str, "difficulty": str, "model_correct": str}
# ---------------------------------------------------------------------------
PROMPTS = {
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
# API call
# ---------------------------------------------------------------------------
def generate_graph(prompt: str, domain: str, idx: int) -> dict | None:
    """Call API, handle errors, download S3 graph."""
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
        if _time_remaining() < 180:  # Stop if < 3 min left
            logger.warning("Time running low, aborting request")
            return None

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
                body = resp.json()
                remaining = body.get("remainingRequests", 0)
                logger.warning(f"  429 (remaining={remaining}) for {slug}, waiting 120s (attempt {attempt+1})")
                time.sleep(120)
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


# ---------------------------------------------------------------------------
# Graph parsing with new metadata
# ---------------------------------------------------------------------------
def parse_graph(prompt_entry: dict, domain: str, api_result: dict) -> dict | None:
    """Parse API result into a record with full metadata including annotations."""
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
            "input": prompt_entry["prompt"],
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
            "metadata_model_correct": prompt_entry["model_correct"],
            "metadata_difficulty": prompt_entry["difficulty"],
            "metadata_expected_answer": prompt_entry["expected_answer"],
        }
    except Exception:
        logger.exception(f"  Parse error for: {prompt_entry['prompt'][:50]}")
        return None


# ---------------------------------------------------------------------------
# Build round-robin work queue
# ---------------------------------------------------------------------------
def build_work_queue() -> list[tuple[str, int, dict]]:
    """Build work queue with round-robin domain ordering."""
    max_prompts = max(len(v) for v in PROMPTS.values())
    queue = []
    for idx in range(max_prompts):
        for domain in DOMAINS:
            if idx < len(PROMPTS[domain]):
                queue.append((domain, idx, PROMPTS[domain][idx]))
    return queue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    global _max_wall_time, _start_time
    _start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-time", type=int, default=3300,
                        help="Max wall time in seconds (default 3300 = 55 min)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Only collect first prompt per domain (8 graphs)")
    args = parser.parse_args()
    _max_wall_time = args.max_time

    total_prompts = sum(len(v) for v in PROMPTS.values())
    logger.info(f"Prompts: {total_prompts} across {len(DOMAINS)} domains")
    logger.info(f"Rate limit: 30 req/60 min. Spacing: {REQUEST_SPACING}s")
    logger.info(f"Max wall time: {_max_wall_time}s ({_max_wall_time/60:.0f} min)")

    ckpt = load_checkpoint()
    records = ckpt["records"]
    done_set = set(ckpt["done"])
    logger.info(f"Resuming with {len(records)} records, {len(done_set)} done")

    queue = build_work_queue()
    remaining = [(d, i, p) for d, i, p in queue if f"{d}:{i}" not in done_set]

    if args.smoke_test:
        # Only first prompt per domain
        seen_domains = set()
        smoke_remaining = []
        for d, i, p in remaining:
            if d not in seen_domains:
                smoke_remaining.append((d, i, p))
                seen_domains.add(d)
        remaining = smoke_remaining
        logger.info(f"SMOKE TEST: {len(remaining)} graphs (1 per domain)")

    logger.info(f"Remaining work items: {len(remaining)}")

    if not remaining:
        logger.info("All work items done!")
        return

    success_count = len(records)
    fail_count = 0
    start_time = time.time()

    for work_idx, (domain, idx, prompt_entry) in enumerate(remaining):
        if _shutdown:
            logger.warning("Shutdown requested, stopping collection")
            break

        if _time_remaining() < 300:  # 5 min buffer
            logger.info(f"Time limit approaching ({_time_remaining():.0f}s left), stopping")
            break

        key = f"{domain}:{idx}"
        elapsed_total = (time.time() - start_time) / 60
        logger.info(
            f"[{work_idx+1}/{len(remaining)}] {domain}[{idx}]: "
            f"{prompt_entry['prompt'][:55]}... ({elapsed_total:.0f}m elapsed, {success_count} ok)"
        )

        t0 = time.time()
        result = generate_graph(prompt_entry["prompt"], domain, idx)
        gen_time = time.time() - t0

        if result is not None:
            record = parse_graph(prompt_entry, domain, result)
            if record is not None:
                records.append(record)
                done_set.add(key)
                success_count += 1
                logger.info(
                    f"  OK: {record['output']['n_nodes']} nodes, "
                    f"{record['output']['n_edges']} edges, "
                    f"DAG={record['output']['is_dag']}, {gen_time:.1f}s "
                    f"[{record['metadata_difficulty']}/{record['metadata_model_correct']}]"
                )
                save_checkpoint(records, list(done_set))
            else:
                logger.warning("  Graph too small or parse failed, skipping")
                done_set.add(key)
                fail_count += 1
                save_checkpoint(records, list(done_set))
        else:
            fail_count += 1

        # Rate-limit pacing
        wait_needed = max(0, REQUEST_SPACING - gen_time)
        if wait_needed > 0 and work_idx < len(remaining) - 1:
            if _time_remaining() < wait_needed + 180:
                logger.info(f"Not enough time for next wait+request, stopping")
                break
            logger.debug(f"  Pacing wait: {wait_needed:.0f}s")
            time.sleep(wait_needed)

    # --- Collection summary ---
    elapsed = (time.time() - start_time) / 60
    logger.info("=" * 60)
    logger.info(f"RUN DONE: {success_count} total records, {fail_count} failures this run, {elapsed:.1f} min")

    domain_counts = defaultdict(int)
    for r in records:
        domain_counts[r["metadata_fold"]] += 1
    for d in DOMAINS:
        logger.info(f"  {d}: {domain_counts[d]}")

    total_remaining = total_prompts - len(done_set)
    logger.info(f"Total done: {len(done_set)}/{total_prompts}, remaining: {total_remaining}")
    logger.info(f"Checkpoint saved with {len(records)} records")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
