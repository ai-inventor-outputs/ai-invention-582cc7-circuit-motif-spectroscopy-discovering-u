# API & Methods

## Summary

Four-part research resolving critical execution blockers. Part 1: Neuronpedia public /api/steer does NOT support gemma-2-2b transcoder features; circuit-tracer local library is the definitive intervention path (15GB VRAM, unlimited rate). Part 2: Weighted motif methods survey covering Onnela intensity/coherence (requires positive weights), Saramaki clustering, Underwood/Elliott motifcluster package (pip install, product weighting handles signs), and C. elegans signed motif methodology (710 signed 3-node types). Recommends threshold binarization baseline + absolute-value intensity with sign-pattern categorization. Part 3: Complete causal validation protocol with logit-difference metric, Wilcoxon signed-rank test, 25 matched pairs per condition, BH-FDR correction, and confound mitigations for degree/layer/attribution matching. Part 4: Novelty confirmed - no Alon-style motif analysis of LLM circuits exists as of March 2026.

## Research Findings

## Part 1: Steering API Validation for gemma-2-2b

### Public API Status: Transcoder Features NOT Supported

The Neuronpedia public /api/steer endpoint explicitly lists only three supported models: gemma-2b, gemma-2b-it, and gpt2-small [1]. Critically, gemma-2-2b (Gemma 2, 2B parameters) is a DIFFERENT model from gemma-2b and is NOT listed [1, 2]. The gemma-2-2b-it model page shows three source sets with inferenceEnabled=true: axbench-reft-r1-res-16k, gemmascope-att-16k, and gemmascope-res-16k — but gemmascope-transcoder-16k (the source set used in attribution graphs) is NOT among them [3]. Even if the API were extended to gemma-2-2b-it, only residual-stream and attention SAE features would be supported, NOT transcoder features.

The graph UI on Neuronpedia does support interventions for Gemma-2 (2B) directly in the browser [4], but this uses an internal graph-server endpoint rather than the public API [2].

**Definitive answer: Transcoder features CANNOT be steered via the public /api/steer endpoint.**

### Circuit-Tracer: The Primary Intervention Path

The circuit-tracer library provides complete intervention capability [5, 6, 7]:
- `ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", dtype=torch.bfloat16)` for model loading [6]
- `model.get_activations(prompt)` for baseline logits [6, 7]
- `model.feature_intervention(prompt, [(layer, pos, feat_idx, 0.0)])` for zero-ablation [6, 7]
- Multiple simultaneous interventions supported in a single forward pass [7]
- GPU requirement: 15GB VRAM (Colab T4 sufficient) [5]
- Rate: Unlimited locally; 480 ablations feasible in ~40 minutes compute + overhead [5, 6]

---

## Part 2: Weighted Motif Analysis Literature Survey

### Onnela Intensity/Coherence (2005)
Subgraph intensity I(g) = geometric mean of edge weights; coherence Q(g) = ratio of geometric to arithmetic mean [8]. **Critical limitation:** geometric mean requires ALL weights > 0, making direct application to signed attribution graphs impossible without transformation [8].

### Saramaki Generalized Clustering (2007)
Weighted clustering coefficient C_w(u) uses cube-root products of normalized edge weights: (w_hat_uv * w_hat_uw * w_hat_vw)^(1/3) [9, 10]. Limited to triangles only; does not extend to arbitrary motif types [9].

### Underwood/Elliott Weighted Motif Adjacency Matrices (2020)
The `motifcluster` Python package (`pip install motifcluster`) constructs weighted motif adjacency matrices with three weighting schemes: unweighted, mean, and product [11, 12, 13]. Supports ANY 3-node motif on directed networks. Scales to million-node graphs [11]. Product weighting naturally propagates signed weights.

### C. elegans Signed Motif Analysis (2025)
First comprehensive signed motif analysis of a complete neuronal network: 710 unique signed 3-node motif types, 56 significantly overrepresented including positive feedforward loops (Z=3.09) and negative feedback loops (Z=5.14) [14]. Directly applicable methodology for signed attribution graphs.

### Software Gap
igraph's motifs_randesu operates on unweighted topology ONLY [15, 16]. No dedicated weighted subgraph census package exists; must combine igraph enumeration with post-hoc weight aggregation or use motifcluster WMAMs.

### Recommendation
Primary: threshold binarization at multiple |w|>tau values with igraph (simplest). Secondary: absolute-value Onnela intensity + sign-pattern categorization (richer). Tertiary: dual-network decomposition of positive/negative edges. Future: motifcluster product-weighted MAMs.

---

## Part 3: Causal Validation Protocol

### Metric Selection
Primary: logit difference (linear in residual stream, most principled per Heimersheim & Nanda [17]). Secondary: KL divergence. Tertiary: probability change Delta_P (most interpretable but non-linear [17]).

### Ablation Method
Zero ablation (scaling_factor=0) via circuit-tracer as practical baseline. Literature recommends mean ablation [17, 18], which requires pre-computing dataset-mean activations.

### Statistical Design
- Paired comparison: motif-node vs. matched control-node (matched on attribution strength +/-10%, degree +/-20%, layer +/-2)
- Test: Wilcoxon signed-rank (non-parametric, paired)
- Effect size: Cohen's d + ratio (threshold 1.5x)
- Sample: 25 pairs per motif type per domain = ~1,200 total ablations
- Multiple testing: BH-FDR at q=0.05 (primary); Bonferroni alpha/24 (sensitivity)
- Confounds: attribution-strength matching, degree matching, layer matching, cascading effects documented as feature [21]

### Feasibility
~2-3 hours on T4 GPU including overhead. Circuit-tracer is unlimited (no rate limit).

---

## Part 4: Novelty Update (March 2026)

Four systematic searches across multiple phrasings confirm: **NO published work applies Alon-style network motif analysis to LLM attribution graphs as of March 2026** [22, 23, 24, 25]. The term "motif" is used colloquially in MI (Anthropic's "suppression motifs" [23], OpenAI's "circuit motifs" [29], Nanda's glossary [28]) but without formal subgraph census, Z-scores, or superfamily classification. Closest prior work remains Zahn et al. 2024 on weight-topology of pruned MLPs [26] and Zambra et al. 2020 on MLP initialization [27]. The novelty claim remains fully valid.

## Sources

[1] [Neuronpedia Steering Documentation](https://docs.neuronpedia.org/steering) — Official steering API docs listing supported models (gemma-2b, gemma-2b-it, gpt2-small only), 100 steers/hour rate limit.

[2] [Neuronpedia Scalar API Reference](https://www.neuronpedia.org/api-doc) — Complete API documentation including /api/steer endpoint specifications and internal graph-server steer endpoint.

[3] [Neuronpedia Gemma-2-2B-IT Model Page](https://www.neuronpedia.org/gemma-2-2b-it) — Lists source sets with inferenceEnabled=true: axbench-reft-r1-res-16k, gemmascope-att-16k, gemmascope-res-16k. Transcoder-16k NOT listed.

[4] [Circuits Research Landscape: Results and Perspectives](https://www.neuronpedia.org/graph/info) — Confirms interventions work for Gemma-2 (2B) in graph UI and scripts/notebooks; 7000+ attribution graphs generated.

[5] [Circuit-Tracer GitHub Repository](https://github.com/decoderesearch/circuit-tracer) — Library docs: Gemma-2-2B support, 15GB VRAM requirement, CLI and Python API, intervention capabilities, CPU/disk offloading.

[6] [Circuit Tracing Tutorial Notebook](https://github.com/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb) — Model loading, Intervention namedtuple with scaling_factor, Feature tuple structure, and intervention code examples.

[7] [Gemma-2-2B Demo Notebook](https://github.com/safety-research/circuit-tracer/blob/main/demos/gemma_demo.ipynb) — Working intervention examples: language switching (11.1% to 27.5%), multiple simultaneous interventions, probability comparison.

[8] [Intensity and Coherence of Motifs in Weighted Complex Networks (Onnela et al. 2005)](https://arxiv.org/abs/cond-mat/0408629) — Introduces subgraph intensity (geometric mean of edge weights) and coherence (geometric/arithmetic mean ratio) for weighted networks.

[9] [Generalizations of the Clustering Coefficient to Weighted Complex Networks (Saramaki et al. 2007)](https://arxiv.org/abs/cond-mat/0608670) — Comparative study of weighted clustering coefficients; cube-root product formula with max-weight normalization.

[10] [NetworkX Clustering Coefficient Documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html) — Implementation of Onnela-Saramaki weighted clustering coefficient formula.

[11] [Motif-Based Spectral Clustering of Weighted Directed Networks (Underwood & Elliott 2020)](https://arxiv.org/abs/2004.01293) — Weighted motif adjacency matrices with mean/product weighting for any 3-node motif; million-node scalability.

[12] [motifcluster GitHub Repository](https://github.com/WGUNDERWOOD/motifcluster) — Python/R/Julia package for motif-based spectral clustering with build_motif_adjacency_matrix() API.

[13] [motifcluster Python Documentation](https://motifcluster.readthedocs.io/en/latest/) — Full API docs: motif adjacency matrix construction with unweighted/mean/product weighting, spectral embedding, clustering.

[14] [Signed Motif Analysis of C. elegans Connectome (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.01.09.632090v1.full) — First signed motif analysis of complete neuronal network: 710 signed 3-node types, 56 overrepresented, structure-preserving randomization.

[15] [igraph Graph Motifs Documentation](https://igraph.org/c/doc/igraph-Motifs.html) — Confirms motifs_randesu operates on unweighted topology only; directed 3-4 node and undirected 3-6 node motifs.

[16] [Motif Counting in Complex Networks: A Comprehensive Survey (2025)](https://arxiv.org/html/2503.19573v1) — Survey of subgraph census algorithms; all standard tools operate on unweighted topology.

[17] [How to Use and Interpret Activation Patching (Heimersheim & Nanda 2024)](https://arxiv.org/html/2404.15255v1) — Recommends logit difference over probability, warns against zero ablation, identifies backup behavior confounds.

[18] [Optimal Ablation for Interpretability (NeurIPS 2024)](https://arxiv.org/html/2409.09951v1) — Shows previous ablation methods may produce artificially high importance scores; proposes loss-minimizing replacement.

[19] [Mechanistic Interpretability as Statistical Estimation (2025)](https://www.arxiv.org/pdf/2510.00845) — Reframes MI as statistical inference; EAP-IG shows high structural variance and hyperparameter sensitivity.

[20] [Comparative Analysis of LLM Abliteration Methods (2026)](https://arxiv.org/pdf/2512.13655) — KL divergence values 0.043-0.076 for minimal-impact interventions; uses 100 prompts with standard errors.

[21] [Node Centrality Measures are a Poor Substitute for Causal Inference](https://www.nature.com/articles/s41598-019-43033-9) — Weak correlation between node centrality and causal influence except for eigenvector centrality.

[22] [Circuit Compositions (ACL 2025)](https://arxiv.org/abs/2410.01434) — Circuit modularity via node overlap and cross-task faithfulness; no subgraph motif census.

[23] [Circuit Tracing (Anthropic 2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) — Qualitative 'suppression motifs' identified but no systematic motif census or Z-score analysis.

[24] [GNN-based Motif Estimation (2025)](https://arxiv.org/html/2506.15709v3) — GNNs predict motif significance profiles; not applied to LLM circuits.

[25] [What Makes a Good Feedforward Computational Graph? (2025)](https://arxiv.org/pdf/2502.06751) — Graph topology for neural architecture design, not interpretability.

[26] [Motif Distribution in Sparse DNNs (Zahn et al. 2024)](https://arxiv.org/abs/2403.00974) — Closest prior work: Alon-style Z-scores on weight-topology of pruned MLPs, fundamentally different approach.

[27] [Emergence of Network Motifs in DNNs (Zambra et al. 2020)](https://www.mdpi.com/1099-4300/22/2/204) — Weight-topology motif emergence in MLPs under initialization schemes.

[28] [MI Glossary (Neel Nanda)](https://www.neelnanda.io/mechanistic-interpretability/glossary) — Defines 'motif' as fuzzy recurring pattern; confirms only colloquial usage in MI field.

[29] [Weight-Sparse Transformers Have Interpretable Circuits (OpenAI 2025)](https://cdn.openai.com/pdf/41df8f28-d4ef-43e9-aed2-823f9393e470/circuit-sparsity-paper.pdf) — Mentions 'circuit motifs' colloquially without formal subgraph census or analysis framework.

[30] [Leveraging Network Motifs for ANN Design (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-66533-x) — 882,000 three-node motifs for architecture engineering; not LLM interpretability.

## Follow-up Questions

- Can the motifcluster package's product-weighted motif adjacency matrices handle the specific scale of attribution graphs (500-5000 nodes, 2000-50000 edges) within reasonable compute time, and does product weighting correctly propagate negative edge weights through 3-node motif instances?
- For the causal validation protocol, should we use the Cantor-decoded (layer, feature_index) from attribution graph nodes directly as intervention targets in circuit-tracer, or does the mapping between graph node IDs and circuit-tracer Feature namedtuples require additional translation steps?
- Given that Heimersheim & Nanda recommend corrupted-prompt patching over zero ablation, is it feasible to define domain-specific corrupted prompts for each of the 8 capability domains, and would this substantially change the effect sizes compared to zero ablation?

---
*Generated by AI Inventor Pipeline*
