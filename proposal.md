


# Graph-Augmented Retrieval for Repository-Level Code Completion Context Collection
Research Proposal — Group 35

### Research Gap

Code completion (predicting and inserting source code based on context) is one of the most widely used AI-assisted features in modern IDEs. Contemporary approaches use large language models in a fill-in-the-middle (FIM) setting, where the model infills code between a given prefix and suffix. A growing body of research shows that the context provided alongside this prefix and suffix has a dramatic effect on completion quality to the point where a smaller model with better context can outperform a larger model with worse context (Wang et al., 2025; Ustalov et al., 2025).

This makes context collection (deciding what information from the broader codebase to feed the model) a core information retrieval problem. The recent ASE 2025 Challenge on Context Collection (Ustalov et al., 2025), organized by JetBrains in collaboration with Mistral AI, formalized this as a competition task and attracted multiple teams. The top-performing solutions converged on a shared strategy: extract symbols from the prefix/suffix, retrieve relevant code chunks using methods like BM25 or dense search, and pack results into the model's token budget.

These solutions all treat retrieval as a flat operation where each code chunk is retrieved and scored independently based on textual similarity to the query. But source code is not flat text. It has an explicit structural graph of dependencies: functions call other functions, classes inherit from parent classes, modules import symbols, and variables reference type definitions declared elsewhere. A function being completed may depend on a utility method that is never mentioned in the prefix/suffix but is structurally essential for the model to produce a correct completion.

**Research Question:** Does augmenting retrieval-based context collection with explicit graph-based expansion over a repository's dependency structure improve code completion quality compared to flat retrieval baselines? Over this, we also aim to investigate these sub-questions.

*   **RQ1:** How does graph-augmented retrieval compare to BM25, dense retrieval, and hybrid retrieval without graph expansion on the chrF metric?
*   **RQ2:** Which edge types in the dependency graph (imports, calls, inheritance, type references) contribute most to completion quality?
*   **RQ3:** What is the effect of expansion depth ($1$-hop vs. $2$-hop vs. $k$-hop) on the trade-off between context relevance and token budget utilization?

### Contribution Type

We propose a graph-augmented context collection strategy (RepoGraphRAG) and will conduct a systematic empirical analysis comparing it against established baselines. The primary contribution is the empirical investigation of whether structural graph information improves context quality for code completion. This framing is deliberate: even if graph expansion does not improve chrF scores, that would itself be a meaningful finding as it would suggest that the structural information extractable from static analysis is either redundant with what lexical/dense retrieval already captures, or that the token budget is too constrained to benefit from broader context. Both outcomes are insightful.

### Dataset & Evaluation

We wish to use the **ASE 2025 Challenge Dataset** (Ustalov et al., 2025), publicly available on Zenodo under CC BY 4.0. This dataset provides:

*   $102$ real-world repositories with $1,764$ completion points across Python and Kotlin
*   Multi-line FIM completion tasks derived from actual commit histories
*   Temporal separation between context and ground truth, avoiding data leakage
*   Established baselines (no context, random file, recent files, BM25) with published scores

**Primary Metric:** chrF (character F-score), the official metric of the ASE challenge and one of the most reliable indicators of code completion quality (Evtikhiev et al., 2023). We will compare against the baselines provided in the challenge (no context, random, recent files, BM25).

### Literatures

*   Ustalov et al. (2025). Challenge on Optimization of Context Collection for Code Completion. ASE 2025 Workshops. IEEE.
*   Evtikhiev et al. (2023). Out of the BLEU: How Should We Assess Quality of the Code Generation Models? JSS.
*   Wang et al. (2025). RLCoder: Reinforcement Learning for Repository-Level Code Completion. ICSE.

### Summary

We propose to investigate whether augmenting retrieval-based context collection with structural dependency graph information improves code completion quality. Our intended approach wishes to combine hybrid text retrieval with graph-based expansion over a repository's dependency structure. We want to evaluate on the publicly available ASE 2025 Challenge dataset using chrF as the primary metric, comparing against established baselines.