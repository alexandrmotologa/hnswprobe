# Metrics and diagnostic thresholds

This document outlines the metrics calculated by hnswprobe, their mathematical definitions, and recommended remediation thresholds.

## Metrics table

| Metric | Symbol | Description | Warning Threshold | Critical Threshold |
| :--- | :--- | :--- | :--- | :--- |
| Mean Recall@k | R_k | Proportion of approximate top-k items that match exact brute-force neighbors | < 0.90 | < 0.80 |
| Recall Decay | Delta_R | Percentage drop in recall compared to initial baseline or target | > 5% | > 15% |
| Tombstone Bloat Ratio | TBR | Ratio of soft-deleted vertices to total vertices | > 0.15 | > 0.25 |
| Orphan Ratio | O_r | Percentage of vertices with in-degree 0 in layer 0 | > 0.01 (1%) | > 0.05 (5%) |
| Trajectory Efficiency | eta_hop | Ratio of Euclidean distance to total distance traversed across greedy hops | < 0.65 | < 0.45 |
| Graph Diameter | D_est | Estimated maximum geodesic path length across layer 0 | > 2.5 * log(N) | > 4.0 * log(N) |
| Hub Congestion Index | G_deg | Variance or Gini coefficient of vertex out-degrees | > 0.40 | > 0.60 |

## Metric definitions

### 1. Empirical Recall@k
Given query q, let Omega_exact be the true top-k nearest neighbors and Omega_hnsw be the approximate neighbors:
Recall@k(q) = |Omega_exact(q) INTERSECT Omega_hnsw(q)| / k

Across Q queries:
Mean_Recall@k = (1 / Q) * SUM(Recall@k(q_i))

When Mean_Recall drops below 0.90, search relevance in downstream RAG applications begins to silently degrade.

### 2. Tombstone Bloat Ratio (TBR)
TBR = |V_dead| / |V_total|

In systems like Qdrant and pgvector, deleted or replaced vectors leave dead vertices to preserve graph structure. When TBR exceeds 0.15, traversals waste CPU cycles traversing inert nodes, and memory footprints expand unnecessarily.

### 3. Orphan Ratio
Orphan_Ratio = |{v in V : in_degree(v) = 0}| / |V|

An orphan vertex cannot be reached during greedy descent from above layers. High orphan ratios occur when neighboring nodes are deleted without proper link rewiring.

### 4. Trajectory Efficiency
eta_hop(q) = ||q - v_entry||_2 / SUM(||v_j - v_{j-1}||_2)

Measures whether greedy hops follow a direct path toward the query or wander through suboptimal detours caused by fragmented connections.

## Automated remediation triggers

When hnswprobe runs in daemon mode, it can execute automated remediation policies based on these metrics:

1. TBR > 0.20: Triggers index vacuum or segment optimization.
2. Recall < 0.85: Triggers background index rebuild (`REINDEX CONCURRENTLY` in pgvector or compaction in Qdrant/Milvus).
3. Alert webhook: Sends structured JSON alerts to monitoring services or Slack when critical thresholds are breached.
