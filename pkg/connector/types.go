package connector

import "time"

// VectorRecord represents a vector stored in the database.
type VectorRecord struct {
	ID        string            `json:"id"`
	Values    []float32         `json:"values"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	IsDeleted bool              `json:"is_deleted"`
}

// QueryRequest represents a vector search query.
type QueryRequest struct {
	ID       string    `json:"id"`
	Values   []float32 `json:"values"`
	TopK     int       `json:"top_k"`
	EfSearch int       `json:"ef_search,omitempty"`
}

// SearchResult represents an item returned by vector search.
type SearchResult struct {
	ID       string  `json:"id"`
	Distance float32 `json:"distance"`
	Score    float32 `json:"score,omitempty"`
}

// IndexStats contains telemetry regarding the vector index.
type IndexStats struct {
	Backend           string    `json:"backend"`
	IndexName         string    `json:"index_name"`
	Dimension         int       `json:"dimension"`
	TotalVectors      int64     `json:"total_vectors"`
	ActiveVectors     int64     `json:"active_vectors"`
	TombstoneVectors  int64     `json:"tombstone_vectors"`
	TombstoneRatio    float64   `json:"tombstone_ratio"`
	MemorySizeBytes   int64     `json:"memory_size_bytes"`
	DiskSizeBytes     int64     `json:"disk_size_bytes"`
	M                 int       `json:"m"`
	EfConstruction    int       `json:"ef_construction"`
	EfSearch          int       `json:"ef_search"`
	Metric            string    `json:"metric"`
	LastCompactionAt  time.Time `json:"last_compaction_at,omitempty"`
}

// GraphNode represents a vertex in the HNSW layer graph.
type GraphNode struct {
	ID        string   `json:"id"`
	Layer     int      `json:"layer"`
	Neighbors []string `json:"neighbors"`
	IsTomb    bool     `json:"is_tomb"`
}

// GraphLayerExport contains the adjacency list for a specific HNSW layer.
type GraphLayerExport struct {
	Layer int                  `json:"layer"`
	Nodes map[string]GraphNode `json:"nodes"`
}

// GraphTopologyExport contains complete or sampled multi-layer HNSW graph topology.
type GraphTopologyExport struct {
	MaxLayer   int                         `json:"max_layer"`
	EntryPoint string                      `json:"entry_point"`
	Layers     map[int]*GraphLayerExport   `json:"layers"`
	TotalNodes int                         `json:"total_nodes"`
	TotalEdges int                         `json:"total_edges"`
}
