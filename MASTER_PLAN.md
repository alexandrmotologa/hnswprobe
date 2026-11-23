# hnswprobe: Architecture and Engineering Master Plan
Vector search HNSW graph decay, tombstone bloat, and recall degradation profiler
Target: high-performance Go vector audit engine (< 20MB RAM) with embedded graph cockpit

## 1. Problem statement and operational scope

### 1.1 The silent decay of production vector indices
In Retrieval-Augmented Generation (RAG) and semantic search architectures, vector databases (such as pgvector, Qdrant, Milvus, and Chroma) rely on Hierarchical Navigable Small World (HNSW) graphs to provide logarithmic time Approximate Nearest Neighbor Search (ANNS). As production databases process continuous inserts, updates, and deletes, the HNSW graph topology degrades over time:

1. Unmonitored recall degradation: HNSW graphs are built greedily. Over continuous insert and delete cycles, navigation paths become suboptimal and queries settle in local minima. Real search recall@k can drop from 98% to below 70% with zero database errors or alerts.
2. Tombstone bloat and memory consumption: When vectors are updated or deleted, databases frequently soft-delete vertices (tombstones) to avoid breaking graph connectivity. These dead vertices remain in memory, increasing memory usage by 20% to 45% and slowing down greedy routing hops.
3. Graph fragmentation and disconnected islands: Successive deletions can sever bridge edges, isolating document clusters and making them completely unreachable from standard entry points.
4. Static hyperparameter mismatch: Initial index hyperparameters (M, efConstruction, efSearch) chosen for smaller initial datasets degrade in efficiency as the vector corpus grows by orders of magnitude.

### 1.2 The hnswprobe solution
hnswprobe is an autonomous vector index health auditor and graph diagnostic profiler written in Go:
- Non-invasive recall sampling: Samples production queries and vectors, runs an in-memory exact brute-force k-NN ground truth evaluation, and measures true recall@k without acquiring locks on operational databases.
- Multi-database adapters: Supports pgvector, Qdrant, Milvus, and Chroma, alongside an in-memory reference engine for synthetic degradation modeling and local benchmarks.
- Topological graph health analysis: Measures layer out-degree distribution, average traversal hop counts, entry-point reachability, graph diameter, and disconnected node ratios.
- Tombstone bloat auditor: Quantifies dead node ratios, tombstone density, and traversal overhead caused by unvacuumed nodes.
- Pareto hyperparameter optimizer: Simulates recall and search latency across different efSearch values to construct the empirical Pareto frontier and recommend optimal re-indexing parameters (M, efConstruction, efSearch).
- Automated compaction triggers: Dispatches webhooks and executes database compaction commands when recall decays below safety thresholds or tombstone bloat exceeds limits.
- Embedded web cockpit: Serves an interactive visualization console on port 8383 featuring 2D and 3D graph topologies, real-time recall decay timelines, degree histograms, and parameter trade-off sliders.

## 2. System architecture

```
+--------------------------------------------------------------------------+
|                             Vector Databases                             |
|          pgvector  |  Qdrant  |  Milvus  |  Chroma  |  Memory Engine     |
+--------------------------------------------------------------------------+
                                     |
              Sample Queries, Vector Embeddings, Index Graph Metadata
                                     v
+--------------------------------------------------------------------------+
|                             hnswprobe Engine                             |
|                                                                          |
|  +---------------------------+          +-----------------------------+  |
|  | Multi-Database Connectors |--------->| Stream Reservoir Sampler    |  |
|  | (Postgres, gRPC, REST)    |          | (Bounded Memory Allocator)  |  |
|  +---------------------------+          +-----------------------------+  |
|                                                        |                 |
|                                                        v                 |
|  +--------------------------------------------------------------------+  |
|  |                 Ground Truth Exact k-NN Evaluator                  |  |
|  |  - Vectorized Cosine, Dot Product, and L2 Distance Calculations    |  |
|  |  - Parallel Brute-Force Worker Pool with SIMD-Friendly Loops       |  |
|  |  - Exact Top-K Ranking and Discrepancy Matrix                      |  |
|  +--------------------------------------------------------------------+  |
|                                                        |                 |
|                                                        v                 |
|  +--------------------------------------------------------------------+  |
|  |                 Topological Graph Health Profiler                  |  |
|  |  - Layer-by-Layer In-Degree and Out-Degree Distribution Analysis   |  |
|  |  - Greedy Hop Trajectory and Navigation Efficiency Tracking        |  |
|  |  - Graph Diameter Estimation and Disconnected Component Detector   |  |
|  |  - Tombstone Bloat Ratio (TBR) and Orphan Vertex Quantification    |  |
|  +--------------------------------------------------------------------+  |
|                                                        |                 |
|                                                        v                 |
|  +--------------------------------------------------------------------+  |
|  |                  Pareto Hyperparameter Recommender                 |  |
|  |  - Latency vs Recall Pareto Frontier Construction                  |  |
|  |  - Parameter Recommendations for M, efConstruction, and efSearch   |  |
|  +--------------------------------------------------------------------+  |
+------------------------------------|-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                         Operational Outputs                              |
|  1. CLI Exit Codes & Machine-Readable Reports (JSON, JUnit XML)          |
|  2. Automated Compaction Dispatcher (Database REINDEX, Webhooks)         |
|  3. Prometheus Metrics Exporter (:8383/metrics)                          |
|  4. Embedded Web Cockpit Console (:8383) 2D/3D Visualizer & Controls     |
+--------------------------------------------------------------------------+
```

## 3. Mathematical and algorithmic formulations

### 3.1 Empirical search recall@k
Let q denote a query vector in dimension D. Let Omega_exact denote the true top-k nearest neighbors obtained via exhaustive brute-force scan, and let Omega_hnsw denote the top-k nearest neighbors returned by the vector index under test:

Recall@k(q) = |Omega_exact(q) INTERSECT Omega_hnsw(q)| / k

Across a sampled query set of size Q:

Mean_Recall@k = (1 / Q) * SUM(Recall@k(q_i)) for i = 1 to Q

### 3.2 Navigational hop distance and trajectory efficiency
Let h(q) denote the number of greedy graph hops required to reach the approximate nearest neighbor from graph entry point v_entry. Let v_j denote the vertex visited at step j:

Trajectory_Efficiency(q) = ||q - v_entry||_2 / SUM(||v_j - v_{j-1}||_2) for j = 1 to h(q)

When greedy traversal is direct, Trajectory_Efficiency approaches 1.0. As graph links deteriorate through unlinked updates, Trajectory_Efficiency drops toward 0, signaling path elongation and excessive hops.

### 3.3 Tombstone bloat ratio (TBR)
Let V_total represent the total vertex count allocated in graph memory, and let V_dead denote marked soft-deleted vertices retaining structural edges:

TBR = |V_dead| / |V_total|

When TBR exceeds 0.15 (15% dead nodes), traversal latency increases and memory efficiency degrades, requiring index vacuuming or compaction.

### 3.4 Orphan node ratio and component disconnection
A vertex v is classified as an orphan if its in-degree at layer 0 is zero, preventing greedy searches from discovering it from common entry points:

Orphan_Ratio = |{v in V : in_degree(v) = 0}| / |V|

For graph connectivity, let C denote the set of connected components in layer 0. An index with |C| > 1 contains isolated subgraphs that cannot reach each other through greedy edge traversal.

### 3.5 Layer 0 graph diameter estimation
The exact diameter of an HNSW base layer with 100,000 nodes is costly to compute directly. hnswprobe uses an iterative double-sweep Breadth-First Search (BFS) heuristic starting from peripheral sample nodes to bound the maximum geodesic hop distance:

Estimated_Diameter = max_depth(BFS(v_peripheral))

## 4. Subsystems and directory structure

```
B:\workgit\hnswprobe\
├── cmd/
│   └── hnswprobe/
│       └── main.go                  # CLI entry point, flag parsing, daemon launcher
├── pkg/
│   ├── connector/
│   │   ├── connector.go             # Storage engine interface definition
│   │   ├── pgvector.go              # PostgreSQL pgvector SQL client
│   │   ├── qdrant.go                # Qdrant HTTP/REST and gRPC client
│   │   ├── milvus.go                # Milvus REST and metadata client
│   │   ├── chroma.go                # Chroma API client
│   │   ├── memory.go                # In-memory HNSW reference engine for benchmarks
│   │   └── types.go                 # Vector, Query, IndexConfig, and GraphData structures
│   ├── groundtruth/
│   │   ├── distance.go              # SIMD-optimized Cosine, Dot, and L2 distance routines
│   │   ├── bruteforce.go            # Parallel exact k-NN solver with MinHeap pooling
│   │   └── evaluator.go             # Recall@k, discrepancy, and rank decay calculator
│   ├── topology/
│   │   ├── graph.go                 # Layered graph representation and adjacency lists
│   │   ├── hops.go                  # Traversal hop simulator and trajectory metrics
│   │   ├── tombstones.go            # Soft-delete density and memory bloat calculator
│   │   ├── orphans.go               # In-degree, BFS connectivity, and component analyzer
│   │   └── diameter.go              # Double-sweep BFS diameter estimation
│   ├── optimizer/
│   │   ├── pareto.go                # Latency-Recall Pareto frontier solver
│   │   └── recommender.go           # Index parameter recommender (M, efConstruction, efSearch)
│   ├── compactor/
│   │   ├── compactor.go             # Compaction executor and webhook dispatcher
│   │   └── triggers.go              # Threshold-based automated policy evaluation
│   ├── cockpit/
│   │   ├── server.go                # Embedded HTTP server and Prometheus exporter
│   │   ├── api.go                   # REST endpoints for live audit, topology, and tuning
│   │   └── web/
│   │       ├── index.html           # Embedded dashboard shell
│   │       ├── app.js               # Canvas 2D/3D topology renderer and chart controller
│   │       └── style.css            # Clean dark tech interface styling
│   └── report/
│       ├── json.go                  # JSON audit output formatter
│       └── junit.go                 # JUnit XML test report generator for CI/CD
├── test/
│   ├── groundtruth_test.go          # Verification of vector math and exact ranking
│   ├── topology_test.go             # Verification of graph metrics, orphans, and tombstones
│   ├── memory_engine_test.go        # Verification of synthetic decay and recall drop
│   ├── connectors_test.go           # Connector client integration tests
│   └── benchmark_test.go            # Speed and memory allocation benchmarks (< 20MB RSS)
├── docs/
│   ├── architecture.md              # Technical design document
│   ├── connectors.md                # Guide for pgvector, Qdrant, Milvus, and Chroma
│   └── metrics.md                   # Metric definitions and remediation thresholds
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions test, lint, and build workflow
├── go.mod
├── go.sum
├── LICENSE
├── README.md
└── MASTER_PLAN.md
```

## 5. Command-line interface specification

```bash
# Audit a pgvector database table
hnswprobe audit \
  --backend pgvector \
  --db "postgres://user:pass@localhost:5432/ragdb" \
  --table "document_embeddings" \
  --column "embedding" \
  --samples 200 \
  --k 10 \
  --min-recall 0.90 \
  --cockpit

# Audit a Qdrant collection for tombstone bloat and recall decay
hnswprobe audit \
  --backend qdrant \
  --endpoint "http://localhost:6333" \
  --collection "enterprise_knowledge" \
  --min-recall 0.92 \
  --json-report ./reports/qdrant_audit.json

# Audit a Milvus collection
hnswprobe audit \
  --backend milvus \
  --endpoint "http://localhost:19530" \
  --collection "articles" \
  --vector-field "vector"

# Audit a Chroma collection
hnswprobe audit \
  --backend chroma \
  --endpoint "http://localhost:8000" \
  --collection "notes"

# Run Pareto hyperparameter optimization sweep
hnswprobe optimize \
  --backend pgvector \
  --db "postgres://user:pass@localhost:5432/ragdb" \
  --table "document_embeddings" \
  --target-recall 0.95

# Start autonomous monitoring daemon with Prometheus exporter
hnswprobe daemon \
  --config ./config.yaml \
  --interval 5m \
  --port 8383 \
  --auto-compact

# Run standalone Web Cockpit on saved audit report or demo dataset
hnswprobe cockpit --port 8383 --demo
```

## 6. Implementation roadmap and milestones

### Milestone 1: Core vector mathematics and parallel ground truth solver
- Implement SIMD-friendly Cosine, Dot Product, and Squared Euclidean L2 distance routines.
- Implement parallel MinHeap-based brute-force k-NN solver with zero-allocation buffer pooling.
- Implement Recall@k, discrepancy calculation, and rank correlation evaluators.

### Milestone 2: Multi-database adapters and in-memory reference engine
- Define unified Connector interface for sampling, querying, and metadata retrieval.
- Implement pgvector connector with PostgreSQL SQL query integration.
- Implement Qdrant connector via REST/gRPC API.
- Implement Milvus connector via REST/v2 API.
- Implement Chroma connector via REST API.
- Implement in-memory reference HNSW engine with synthetic insertion, deletion, and degradation capabilities.

### Milestone 3: Topological graph profiler and tombstone auditor
- Build layer-by-layer graph representation with adjacency lists and vertex state tracking.
- Implement greedy traversal hop simulator to calculate navigation efficiency and trap rates.
- Implement tombstone bloat ratio (TBR) calculation and soft-delete memory footprint estimator.
- Implement orphan vertex detection and double-sweep BFS diameter calculator.

### Milestone 4: Pareto optimizer and automated compaction triggers
- Implement latency vs recall empirical simulation across efSearch candidates.
- Build parameter recommendation engine for M, efConstruction, and efSearch.
- Implement compaction dispatcher supporting database REINDEX, REST vacuum endpoints, and external webhooks.

### Milestone 5: Embedded web cockpit and Prometheus exporter
- Build lightweight HTTP server on port 8383 with `/metrics`, `/api/audit`, and `/api/compact` endpoints.
- Develop interactive frontend with 2D and 3D graph visualization using Canvas and WebGL.
- Render dynamic charts for recall decay timeline, degree distribution, and Pareto trade-off curves.

### Milestone 6: CLI suite, daemon scheduler, benchmarks, and documentation
- Assemble CLI subcommands: audit, optimize, daemon, compact, cockpit, and generate-sample.
- Run comprehensive unit, integration, and benchmark tests verifying RSS memory under 20MB.
- Write humanized documentation, guides, and GitHub Actions CI configuration.
