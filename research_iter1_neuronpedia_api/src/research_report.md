# Neuronpedia API

## Summary

Complete implementation reference for the Neuronpedia API covering graph generation (POST /api/graph/generate with all 9 parameters verified), graph JSON schema (4 node types, Cantor-paired feature indices, weighted directed edges), feature explanations (GET /api/feature with confirmed layer format {N}-gemmascope-transcoder-16k), steering (POST /api/steer with 100/hr limit, gemma-2-2b support uncertain for public API but confirmed in graph UI), rate limits, batch feasibility for 250+ graphs (~42 min sequential), and the circuit-tracer local fallback. Based on 20 verified sources.

## Research Findings

## Neuronpedia API Complete Implementation Reference

### 1. Graph Generation Endpoint (POST /api/graph/generate)

**URL:** `https://www.neuronpedia.org/api/graph/generate` (POST) [1]

**Authentication:** `x-api-key` header with API key obtained from neuronpedia.org/account [1, 2]

**Request Body (JSON):**

| Parameter | Type | Required | Constraints | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| prompt | string | Yes | Max 10,000 chars, capped at 64 tokens | - | Text to analyze [1] |
| modelId | string | Yes | Pattern: `[a-zA-Z0-9_-]+` | - | Currently only `gemma-2-2b` for on-demand generation [1, 3] |
| slug | string | Yes | Pattern: `[a-z0-9_-]+` | - | Unique identifier, must be unique across all graphs [1] |
| sourceSetName | string | No | Pattern: `[a-zA-Z0-9_-]+` | `gemmascope-transcoder-16k` for gemma-2-2b | Transcoder source set [1, 11] |
| maxNLogits | number | No | Min: 5, Max: 15 | 10 | Maximum output tokens to track [1] |
| desiredLogitProb | number | No | Min: 0.6, Max: 0.99 | 0.95 | Cumulative probability mass for output tokens [1] |
| nodeThreshold | number | No | Min: 0.5, Max: 1 | 0.8 | Pruning threshold for nodes [1] |
| edgeThreshold | number | No | Min: 0.8, Max: 1 | 0.85 | Pruning threshold for edges [1] |
| maxFeatureNodes | number | No | Min: 3000, Max: 10000 | 5000 | Maximum feature nodes retained [1] |

**Response (200 OK):** `{ "message": "Graph saved to database", "s3url": "string", "url": "string", "numNodes": integer, "numLinks": integer }` [1]

**Error Responses:** 400 (validation error, duplicate slug), 503 (GPUs busy - retry later) [1]

**Two-Step Retrieval:** After POST, must GET the `s3url` to download the actual graph JSON (2-100+ MB) [6].

**Working Python Example:**
```python
import requests, json, os
headers = {'x-api-key': os.environ['NEURONPEDIA_API_KEY'], 'Content-Type': 'application/json'}
config = {"prompt": "The capital of the state containing Dallas is", "modelId": "gemma-2-2b",
          "sourceSetName": "gemmascope-transcoder-16k", "slug": "dallas-capital-001",
          "nodeThreshold": 0.8, "edgeThreshold": 0.85, "maxFeatureNodes": 5000}
response = requests.post("https://www.neuronpedia.org/api/graph/generate", headers=headers, json=config, timeout=60)
s3_url = response.json()['s3url']
graph_data = requests.get(s3_url, timeout=120).json()
```
[6]

### 2. Graph JSON Schema

The complete schema from the graph validator [4]:

**Root Structure:** `{ "metadata": {}, "qParams": {}, "nodes": [], "links": [] }` — all four fields required.

**Node Object (required fields):** `node_id` (string, format `"{layer}_{feature}_{ctx_idx}"`), `feature` (integer or null, Cantor-paired), `layer` (string or integer, `"E"` for embeddings), `ctx_idx` (integer), `feature_type` (string enum: `"cross layer transcoder"`, `"mlp reconstruction error"`, `"embedding"`, `"logit"`), `jsNodeId` (string), `clerp` (string). Optional: `influence`, `activation` [4].

**Link Object:** `source` (string), `target` (string), `weight` (number = A_{s→t} = a_s × w_{s→t}) [4, 7].

**Node Type Mapping for Motif Analysis:**
- `"embedding"` → Source node (in-degree 0): prompt token embeddings [7]
- `"logit"` → Sink node (out-degree 0): candidate output tokens [7]
- `"cross layer transcoder"` → Intermediate: active CLT features [7]
- `"mlp reconstruction error"` → Intermediate: unexplained MLP output [7]

### 3. Cantor Pairing Decoder

When `neuronpedia_source_set` is used in metadata, the `feature` field encodes (layer_num, feat_index) via Cantor pairing [4]:

**Encode:** `feature = (layer_num + feat_index) * (layer_num + feat_index + 1) / 2 + feat_index`

**Decode:**
```python
import math
def cantor_decode(z):
    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    t = (w * w + w) // 2
    feat_index = z - t
    layer_num = w - feat_index
    return (int(layer_num), int(feat_index))
```

### 4. Feature Explanation Endpoint (GET /api/feature/{modelId}/{layer}/{index})

**URL:** `GET https://www.neuronpedia.org/api/feature/{modelId}/{layer}/{index}` [1]

**Layer format for transcoder features:** `{layer_num}-gemmascope-transcoder-16k` [8]. Example: `gemma-2-2b/20-gemmascope-transcoder-16k/100` returns a feature with explanation, activations, top logits, sparsity metrics, and source configuration (architecture: jumprelu_transcoder, 16384 features/layer) [9, 10].

### 5. Steering/Ablation Endpoint (POST /api/steer)

**Documented supported models:** `gemma-2b`, `gemma-2b-it`, `gpt2-small` [12]. **Rate limit:** 100 steers/hour [12].

**CRITICAL GAP:** `gemma-2-2b` is NOT listed in the public /api/steer docs. However, `gemma-2-2b-it` shows `inferenceEnabled: true` on the steer page [13], and the graph UI supports interventions for Gemma-2 (2B) [14]. The circuit-tracer library can perform local interventions as a fallback [16, 17].

### 6. Rate Limits and Batch Feasibility (250+ graphs)

| Endpoint | Rate Limit | Notes |
|----------|-----------|-------|
| Graph generation | GPU-queue limited, 503 when busy | ~10s per graph → 250 graphs ≈ 42 min [1] |
| Feature lookup | Not explicitly limited | Generous for GET [2] |
| Steering | 100/hour | Confirmed [12] |

**Higher tier:** Free whitelist access available via email [2, 3]. **Recommendation:** Email Neuronpedia before running batch.

### 7. Auxiliary Endpoints

- **POST /api/graph/signed-put:** Upload pre-generated graphs (2-step with save-to-db) [1]
- **GET /api/graph/{modelId}/{slug}:** Retrieve graph metadata [1]
- **GET /api/graph/list:** List user's graphs [1]
- **POST /api/search-all:** Top features for text (max 100 results) [1]
- Subgraph save/list/delete endpoints [1]

### 8. Existing Corpus and Alternatives

Over 7,000 attribution graphs exist on Neuronpedia [14]. Supported models: Gemma-2-2B, Llama-3.2-1B, Qwen3-4B, GPT-2 [14]. The `circuit-tracer` library can generate graphs locally with a GPU (>=15GB VRAM) as fallback [16]. The `neuronpedia` Python package (v1.2.0) provides high-level API access [18].

## Sources

[1] [Neuronpedia Scalar API Reference](https://www.neuronpedia.org/api-doc) — Primary API documentation with complete endpoint specs for graph generation, feature lookup, steering, search, and all auxiliary endpoints including parameter constraints and authentication.

[2] [Neuronpedia API and Exports Documentation](https://docs.neuronpedia.org/api) — High-level API docs noting higher-tier rate limits available via email whitelist and directing to api-doc for detailed specs.

[3] [Circuit Tracer + New Auto-Interp Method Blog Post](https://www.neuronpedia.org/blog/circuit-tracer) — Announcement describing on-demand graph generation for gemma-2-2b, the graph validator, upload capabilities, and ~5000 graphs at time of writing.

[4] [Neuronpedia Graph JSON Validator](https://www.neuronpedia.org/graph/validator) — Complete JSON schema for attribution graphs with all required/optional fields for metadata, nodes (feature_type enum), links, qParams. Confirms Cantor pairing for neuronpedia_source_set.

[5] [Neuronpedia SAE Features Documentation](https://docs.neuronpedia.org/features) — Feature data structure with three-component identifier system (model/source/index), autointerp explanation format, and scoring methodology.

[6] [Attribution Graph Probing Repository](https://github.com/peppinob-ol/attribution-graph-probing) — Working Python implementation of Neuronpedia API calls with code for requests, S3 download, error handling, local storage, and 2 req/sec rate limiting with exponential backoff.

[7] [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) — Authoritative technical reference for attribution graph methodology: node types, edge weight formula, CLT architecture, pruning algorithms, and validation metrics.

[8] [Neuronpedia Feature Page - Layer 25 Transcoder](https://www.neuronpedia.org/gemma-2-2b/25-gemmascope-transcoder-16k/2418) — Confirms layer format '{layer_num}-gemmascope-transcoder-16k' for transcoder features with feature page structure.

[9] [Live Feature API Response](https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-transcoder-16k/100) — Complete JSON response confirming explanations array, activations, statistics, logits, and source configuration from the feature endpoint.

[10] [Feature 100 Dashboard Page](https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-transcoder-16k/100) — Confirms architecture (jumprelu_transcoder), hook name, 16384 features/layer, 1024 context size, and np_max-act-logits autointerp method.

[11] [Neuronpedia Gemma-2-2B Graph Page](https://www.neuronpedia.org/gemma-2-2b/graph) — Graph generation UI with gemmascope-transcoder-16k source set, example graphs for various prompts, and management features.

[12] [Neuronpedia Steering Documentation](https://docs.neuronpedia.org/steering) — Steering API listing supported models (gemma-2b, gemma-2b-it, gpt2-small), 100 steers/hour rate limit, and strength parameters.

[13] [Neuronpedia Steer Page](https://www.neuronpedia.org/steer) — Live steer page showing gemma-2-2b-it with inferenceEnabled=true for gemmascope-res-16k and gemmascope-att-16k source sets.

[14] [Circuits Research Landscape: Results and Perspectives](https://www.neuronpedia.org/graph/info) — Reports 7000+ graphs generated, supported models (Gemma-2-2B, Llama-3.2-1B, Qwen3-4B, GPT-2), and confirms interventions for Gemma-2 (2B).

[15] [Steerify Experiment README](https://github.com/hijohnnylin/neuronpedia/blob/main/apps/experiments/steerify/README.md) — Internal graph server steer endpoint spec at localhost:5002/v1/steer/completion with request format details.

[16] [Circuit-Tracer GitHub Repository](https://github.com/decoderesearch/circuit-tracer) — Library docs covering CLI usage, Python API, supported models/transcoders, GPU requirements (15GB+), installation, and intervention capabilities.

[17] [Circuit Tracing Tutorial Notebook](https://github.com/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb) — Tutorial with graph generation, Intervention namedtuple with scaling_factor, supernode feature extraction, and Feature tuple (layer, pos, feature_idx).

[18] [neuronpedia PyPI Package](https://pypi.org/project/neuronpedia/) — Python library v1.2.0 (Nov 2025) with SAEFeature.get() for feature retrieval, NPVector for steering, Python 3.10-3.13 support.

[19] [Neuronpedia Gemma-2-2B Model Page](https://www.neuronpedia.org/gemma-2-2b) — Model specs (2304 dimensions, 26 layers, 16384 neurons/layer) and source sets including gemmascope-res-16k, gemmascope-res-65k, gemmascope-att-16k.

[20] [Neuronpedia Schemas README](https://github.com/hijohnnylin/neuronpedia/blob/main/schemas/README.md) — OpenAPI schema organization for inference/autointerp servers at openapi/inference-server.yaml, using OpenAPI Generator CLI for client generation.

## Follow-up Questions

- Does the public /api/steer endpoint support gemma-2-2b or gemma-2-2b-it for transcoder feature steering, or is only the graph UI intervention path available for Gemma 2 models?
- What is the exact neuronpedia_source_set string used in graph metadata for CLT (cross-layer transcoder) features vs PLT (per-layer transcoder) features, and does the Cantor pairing decode differently for each?
- Can existing Neuronpedia graphs be bulk-downloaded via the GET /api/graph/{modelId}/{slug} endpoint if slugs are enumerated, or is there an undocumented bulk export endpoint or S3 dataset?

---
*Generated by AI Inventor Pipeline*
