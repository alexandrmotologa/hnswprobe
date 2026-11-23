# Database connectors guide

hnswprobe connects to multiple vector database engines to sample queries, read vector embeddings, and retrieve index graph statistics.

## Supported backends

### 1. pgvector (PostgreSQL)

pgvector stores vector embeddings in standard PostgreSQL tables and builds HNSW indexes over them.

```bash
hnswprobe audit \
  --backend pgvector \
  --db "postgres://user:password@localhost:5432/vectordb?sslmode=disable" \
  --table "document_embeddings" \
  --column "embedding" \
  --samples 200 \
  --k 10
```

Operations performed:
- Extracts sample queries and vectors via non-blocking indexed queries.
- Reads index size and page count from PostgreSQL catalog tables `pg_class` and `pg_am`.
- Runs queries with `SET LOCAL hnsw.ef_search = ...` to profile the Pareto frontier.
- Issues `REINDEX INDEX CONCURRENTLY` when automatic compaction is triggered.

### 2. Qdrant

Qdrant provides vector search over REST and gRPC interfaces with telemetry on segment structures.

```bash
hnswprobe audit \
  --backend qdrant \
  --endpoint "http://localhost:6333" \
  --collection "production_knowledge" \
  --samples 200 \
  --k 10
```

Operations performed:
- Inspects collection cluster telemetry and segment tombstone counts.
- Samples points and vectors using the scroll API.
- Compares default search results against brute-force exact matches.
- Calls the segment vacuum and optimization endpoint to reclaim tombstoned memory.

### 3. Milvus

Milvus manages scalable vector indexes with dedicated query nodes.

```bash
hnswprobe audit \
  --backend milvus \
  --endpoint "http://localhost:19530" \
  --collection "enterprise_docs" \
  --vector-field "vector" \
  --samples 200 \
  --k 10
```

Operations performed:
- Queries collection schema, entity counts, and index parameters.
- Pulls sample vectors and executes ANN queries with varying search parameters.
- Triggers compaction via the Milvus collection compaction API.

### 4. Chroma

Chroma is an embedded and server-based vector store commonly used in local RAG pipelines.

```bash
hnswprobe audit \
  --backend chroma \
  --endpoint "http://localhost:8000" \
  --collection "chat_memory" \
  --samples 150 \
  --k 10
```

Operations performed:
- Fetches collection metadata, embeddings, and document records.
- Measures recall drop after batch deletions.

### 5. In-Memory Reference Engine

A pure Go HNSW implementation included directly in hnswprobe for testing, local benchmarks, and offline simulations.

```bash
hnswprobe audit \
  --backend memory \
  --demo \
  --cockpit
```

Operations performed:
- Generates synthetic vector clusters with tunable dimensionalities.
- Simulates realistic deletion patterns to reproduce tombstone bloat and path disruption.
- Exposes complete layer-by-layer graph structures for 2D/3D visualization in the Cockpit.
