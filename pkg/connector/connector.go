package connector

import (
	"context"
	"fmt"
	"time"
)

// BackendType represents a supported vector database.
type BackendType string

const (
	BackendPgvector BackendType = "pgvector"
	BackendQdrant   BackendType = "qdrant"
	BackendMilvus   BackendType = "milvus"
	BackendChroma   BackendType = "chroma"
	BackendMemory   BackendType = "memory"
)

// Connector defines operations required to inspect, sample, and audit vector indexes.
type Connector interface {
	// Ping validates connection to the target database.
	Ping(ctx context.Context) error

	// GetStats retrieves index telemetry, size, vector counts, and tombstone estimates.
	GetStats(ctx context.Context) (*IndexStats, error)

	// SampleVectors extracts a representative subset of vectors from the corpus.
	SampleVectors(ctx context.Context, count int) ([]VectorRecord, error)

	// SampleQueries extracts or generates representative query vectors.
	SampleQueries(ctx context.Context, count int) ([]QueryRequest, error)

	// SearchANN executes an approximate nearest neighbor search via the database index.
	SearchANN(ctx context.Context, req QueryRequest) ([]SearchResult, time.Duration, error)

	// TriggerCompaction triggers index rebuilding, vacuuming, or segment consolidation.
	TriggerCompaction(ctx context.Context) error

	// ExportTopology exports sampled graph adjacency lists for topology analysis.
	ExportTopology(ctx context.Context, maxNodes int) (*GraphTopologyExport, error)

	// Close releases any database resources.
	Close() error
}

// Config specifies connection parameters for database adapters.
type Config struct {
	Backend        BackendType   `json:"backend"`
	Endpoint       string        `json:"endpoint"`
	Database       string        `json:"database"`
	Table          string        `json:"table"`
	Column         string        `json:"column"`
	Collection     string        `json:"collection"`
	VectorField    string        `json:"vector_field"`
	APIKey         string        `json:"api_key"`
	Timeout        time.Duration `json:"timeout"`
	Dimension      int           `json:"dimension"`
	Metric         string        `json:"metric"`
	M              int           `json:"m"`
	EfConstruction int           `json:"ef_construction"`
	EfSearch       int           `json:"ef_search"`
}

// NewConnector creates a database connector according to the configuration.
func NewConnector(cfg Config) (Connector, error) {
	if cfg.Timeout == 0 {
		cfg.Timeout = 10 * time.Second
	}

	switch cfg.Backend {
	case BackendPgvector:
		return NewPgvectorConnector(cfg)
	case BackendQdrant:
		return NewQdrantConnector(cfg)
	case BackendMilvus:
		return NewMilvusConnector(cfg)
	case BackendChroma:
		return NewChromaConnector(cfg)
	case BackendMemory:
		return NewMemoryConnector(cfg)
	default:
		return nil, fmt.Errorf("unsupported vector database backend: %s", cfg.Backend)
	}
}
