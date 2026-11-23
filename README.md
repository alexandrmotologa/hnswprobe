<p align="center">
  <img src="docs/images/logo.png" alt="hnswprobe Logo" width="128" height="128" />
</p>

<h1 align="center">hnswprobe</h1>

<p align="center">
  <strong>Autonomous HNSW Vector Index Decay, Tombstone Bloat, and Recall Profiler in Go</strong>
</p>

<p align="center">
  <a href="https://github.com/alexandrmotologa/hnswprobe/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen?style=flat-square" alt="CI Status" /></a>
  <a href="https://golang.org"><img src="https://img.shields.io/badge/go-1.22+-blue?style=flat-square" alt="Go Version" /></a>
  <a href="#health-metrics-reference"><img src="https://img.shields.io/badge/memory-%3C%2020MB-success?style=flat-square" alt="Memory Usage" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License" /></a>
</p>

Audit vector index recall degradation, tombstone bloat, and graph connectivity across pgvector, Qdrant, Milvus, and Chroma.

`hnswprobe` is a diagnostic profiler and audit daemon written in Go. It samples production vector queries, calculates exact brute-force ground truth in parallel without acquiring database locks, measures recall decay, and estimates graph structural health.

---

## Interactive Web Cockpit

`hnswprobe` embeds an HTTP cockpit on port 8383 with a real-time 2D and 3D isometric vector graph visualizer, query hop traversal animations, dynamic Pareto curves, and an interactive Chaos & Degradation Lab.

![hnswprobe Cockpit Animated Demo](docs/images/hnswprobe_demo.gif)

<p align="center"><em>Live HNSW graph exploration: 3D multi-layer inspection, query hop simulation, chaos injection, and instant self-healing compaction.</em></p>

### Responsive Multi-Device Interface

The cockpit adapts dynamically across desktop, tablet, and mobile viewports with strict layout integrity and zero text wrapping:

| Desktop 2D/3D Topology View | Mobile Health Sentinel |
| :---: | :---: |
| ![Desktop Cockpit](docs/images/hnswprobe_cockpit_topology.png) | <img src="docs/images/hnswprobe_mobile_cockpit.png" alt="Mobile Cockpit" width="340" /> |

---

## Key capabilities

- Non-invasive recall auditing: Evaluates real search recall@k against exact brute-force nearest neighbors without locking operational tables.
- Multi-database adapters: Native connectors for PostgreSQL (pgvector), Qdrant, Milvus, and Chroma, with an in-memory reference engine for offline testing.
- Graph topology diagnostics: Measures layer out-degree distributions, tombstone bloat ratio (TBR), orphan node counts (in-degree zero in layer 0), and navigation diameter.
- Pareto hyperparameter recommender: Sweeps efSearch values to construct the empirical latency vs recall trade-off curve and recommend optimal index settings (M, efConstruction, efSearch).
- Automated compaction triggers: Emits webhooks and triggers database compaction commands (REINDEX CONCURRENTLY, segment optimization) when health thresholds are breached.
- Embedded web cockpit: Web interface on port 8383 with 2D/3D graph topology exploration, recall decay timelines, live tuning sliders, and an interactive Chaos & Degradation Lab.
- Adaptive efSearch controller: Exposes dynamic self-healing configuration endpoints (`/api/config/active-ef`) to compensate for degradation during high-tombstone churn without downtime.
- Quantization distortion profiler: Evaluates angular distortion, relative L2 error, and recall penalties introduced by Scalar Quantization (SQ8).
- Dataset benchmark generator: Generates self-contained synthetic vector benchmarks with precalculated ground truth top-k for CI pipelines.
- Turnkey Grafana dashboard: Pre-configured dashboard template for Prometheus alerting and long-term decay trends.
- Low memory footprint: Runs under 20MB resident memory (typically 3 to 5MB during audits).
- CI/CD integration: Emits machine-readable JSON summaries and JUnit XML test reports with non-zero exit codes on threshold violations.

## Installation

### From binary release

Download the pre-compiled binary for your architecture from GitHub Releases:

```bash
# Verify installation
hnswprobe version
```

### From source

Requires Go 1.22 or later:

```bash
git clone https://github.com/alexandrmotologa/hnswprobe.git
cd hnswprobe
go build -o bin/hnswprobe ./cmd/hnswprobe
```

## Quick start

### 1. Audit a vector database

Audit an index and verify whether recall satisfies your threshold:

```bash
# Audit PostgreSQL pgvector
hnswprobe audit \
  --backend pgvector \
  --db "postgres://user:pass@localhost:5432/ragdb?sslmode=disable" \
  --table "document_embeddings" \
  --column "embedding" \
  --samples 100 \
  --k 10 \
  --min-recall 0.90 \
  --max-tombstones 0.15

# Audit Qdrant collection
hnswprobe audit \
  --backend qdrant \
  --endpoint "http://localhost:6333" \
  --collection "production_knowledge" \
  --samples 100 \
  --k 10

# Audit Milvus collection
hnswprobe audit \
  --backend milvus \
  --endpoint "http://localhost:19530" \
  --collection "enterprise_docs" \
  --vector-field "vector"

# Audit Chroma collection
hnswprobe audit \
  --backend chroma \
  --endpoint "http://localhost:8000" \
  --collection "chat_memory"
```

### 2. Run hyperparameter optimization sweep

Find the optimal efSearch and M parameters for your target recall:

```bash
hnswprobe optimize \
  --backend pgvector \
  --db "postgres://user:pass@localhost:5432/ragdb" \
  --table "document_embeddings" \
  --target-recall 0.95
```

Output:
```
--- Pareto Hyperparameter Sweep Results ---
efSearch   Mean Recall     Latency (ms)    QPS          Optimal   
------------------------------------------------------------------
8          0.8240          0.45            2222.2                 
16         0.8910          0.62            1612.9                 
32         0.9480          0.91            1098.9       YES (Pareto)
64         0.9820          1.45            689.7        YES (Pareto)
128        0.9910          2.60            384.6        YES (Pareto)
------------------------------------------------------------------
Optimal efSearch: 32 (Target Recall: 95.00%)
```

### 3. Launch Web Cockpit

Start the visual cockpit with an interactive 2D/3D graph canvas:

```bash
# Launch with synthetic demo graph
hnswprobe cockpit --port 8383 --demo

# Open your browser at http://localhost:8383
```

### 4. Run autonomous audit daemon

Run `hnswprobe` as a background sentinel with automated compaction and Prometheus metrics:

```bash
hnswprobe daemon \
  --backend qdrant \
  --endpoint "http://localhost:6333" \
  --collection "production_knowledge" \
  --interval 5m \
  --port 8383 \
  --auto-compact \
  --webhook "https://alerts.internal.net/hooks/vector-health"
```

Metrics are exposed at `http://localhost:8383/metrics`:
```
hnswprobe_recall_ratio{backend="qdrant",index="production_knowledge"} 0.9480
hnswprobe_tombstone_bloat_ratio{backend="qdrant",index="production_knowledge"} 0.0820
hnswprobe_orphan_ratio{backend="qdrant",index="production_knowledge"} 0.0000
hnswprobe_mean_hop_count{backend="qdrant",index="production_knowledge"} 5.20
hnswprobe_estimated_diameter{backend="qdrant",index="production_knowledge"} 4
hnswprobe_memory_bytes{backend="qdrant",index="production_knowledge"} 4194304
```

### 5. Profile scalar quantization distortion

Audit precision loss and memory savings before enabling quantization:

```bash
hnswprobe quantize --dimension 64 --samples 500 --k 10
```

### 6. Generate synthetic benchmark corpus

Generate reproducible vector corpora with pre-calculated exact ground truth:

```bash
hnswprobe generate-sample \
  --output ./data/corpus_benchmark.json \
  --vectors 1000 \
  --dimension 64 \
  --clusters 6 \
  --tombstones 15 \
  --k 10
```

## Health metrics reference

| Metric | Symbol | Description | Healthy Range | Action on Breach |
| :--- | :--- | :--- | :--- | :--- |
| Mean Recall@k | R_k | Match ratio vs exact brute-force ground truth | >= 0.90 | Increase efSearch or re-index |
| Tombstone Bloat Ratio | TBR | Soft-deleted nodes over total allocated nodes | < 0.15 | Trigger index compaction / vacuum |
| Orphan Ratio | O_r | Percentage of nodes with 0 in-degree in layer 0 | < 0.01 | Rebuild index to repair bridges |
| Trajectory Efficiency | eta | Geodesic vs actual traversed distance | >= 0.60 | Re-index with higher M |
| Graph Diameter | D_est | Estimated maximum shortest path in layer 0 | <= 4 * log(N) | Structural health indicator |

## Architecture

Detailed design notes and mathematical formulations are available in the [documentation directory](docs/):
- [Architecture & Design](docs/architecture.md)
- [Database Connectors Guide](docs/connectors.md)
- [Metrics & Remediation Thresholds](docs/metrics.md)
- [Grafana Dashboard Setup](docs/grafana.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
