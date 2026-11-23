# Architecture and system design

hnswprobe audits vector search indices to detect recall degradation, tombstone accumulation, and graph structural decay across production deployments.

## Design goals

1. Non-invasive auditing: The engine reads query samples and vectors without acquiring write locks or disrupting production traffic.
2. Memory efficiency: The daemon runs with under 20MB of resident memory, using bounded streaming pools and fixed-size heaps.
3. Multi-database support: Adapters connect to pgvector, Qdrant, Milvus, and Chroma, alongside an in-memory reference engine for reproducible offline benchmarks.
4. Actionable recommendations: The engine computes the empirical Pareto frontier between search latency and recall, suggesting optimal M, efConstruction, and efSearch values.

## Core pipeline

```
+--------------------------------------------------------------------+
|                         Target Vector Index                        |
|       (pgvector / Qdrant / Milvus / Chroma / In-Memory HNSW)       |
+--------------------------------------------------------------------+
                                   |
                      Streaming Samples & Metadata
                                   v
+--------------------------------------------------------------------+
|                   Streaming Ingestion Subsystem                    |
|   Reservoir sampling selects representative query vectors.         |
|   Memory allocation remains bounded regardless of corpus size.     |
+--------------------------------------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------+
|                Parallel Ground Truth Evaluator                     |
|   SIMD-friendly Cosine, Dot, and L2 distance routines.             |
|   Parallel worker pool solves exact top-k for sampled queries.     |
|   Computes recall@k, mean distance error, and rank variance.       |
+--------------------------------------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------+
|                  Topological Graph Profiler                        |
|   Layer-by-layer connectivity and degree distribution analysis.    |
|   Greedy traversal hop counting and trajectory efficiency.         |
|   Tombstone bloat ratio (TBR) and orphan node identification.      |
|   Double-sweep BFS diameter estimation.                            |
+--------------------------------------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------+
|                     Pareto Hyperparameter Engine                   |
|   Simulates search behavior across efSearch candidate values.      |
|   Identifies optimal re-indexing configurations.                   |
+--------------------------------------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------+
|                         Operational Layer                          |
|   1. Embedded Cockpit UI on port 8383 with 2D/3D graph canvas.     |
|   2. Prometheus metrics endpoint (/metrics).                       |
|   3. Compaction dispatcher (SQL REINDEX, REST vacuum, webhooks).   |
+--------------------------------------------------------------------+
```

## Memory management

To maintain an RSS footprint below 20MB:
- Vector buffers are reused through memory pooling.
- Brute-force nearest neighbor searches use pre-allocated binary heaps.
- Sampling operates via reservoir selection rather than bulk loading.
