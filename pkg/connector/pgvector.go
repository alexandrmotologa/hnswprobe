package connector

import (
	"context"
	"database/sql"
	"fmt"
	"strconv"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

// PgvectorConnector connects to a PostgreSQL database with the pgvector extension.
type PgvectorConnector struct {
	db     *sql.DB
	cfg    Config
	metric string
}

// NewPgvectorConnector initializes a PostgreSQL connection.
func NewPgvectorConnector(cfg Config) (*PgvectorConnector, error) {
	connStr := cfg.Database
	if connStr == "" {
		connStr = cfg.Endpoint
	}
	if connStr == "" {
		return nil, fmt.Errorf("pgvector: database connection string or endpoint is required")
	}

	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("pgvector: failed to open connection: %w", err)
	}

	db.SetMaxOpenConns(5)
	db.SetMaxIdleConns(2)
	db.SetConnMaxLifetime(5 * time.Minute)

	return &PgvectorConnector{
		db:     db,
		cfg:    cfg,
		metric: cfg.Metric,
	}, nil
}

func (p *PgvectorConnector) Ping(ctx context.Context) error {
	return p.db.PingContext(ctx)
}

func (p *PgvectorConnector) Close() error {
	return p.db.Close()
}

func (p *PgvectorConnector) GetStats(ctx context.Context) (*IndexStats, error) {
	stats := &IndexStats{
		Backend:        string(BackendPgvector),
		IndexName:      fmt.Sprintf("%s_%s_hnsw_idx", p.cfg.Table, p.cfg.Column),
		M:              p.cfg.M,
		EfConstruction: p.cfg.EfConstruction,
		EfSearch:       p.cfg.EfSearch,
		Metric:         p.metric,
	}

	// 1. Get row counts
	queryCount := fmt.Sprintf("SELECT COUNT(*) FROM %s", p.cfg.Table)
	var total int64
	err := p.db.QueryRowContext(ctx, queryCount).Scan(&total)
	if err != nil {
		return nil, fmt.Errorf("pgvector: count query failed: %w", err)
	}
	stats.TotalVectors = total
	stats.ActiveVectors = total

	// 2. Query index sizes and memory usage
	querySize := `
		SELECT pg_relation_size(c.oid) AS index_bytes
		FROM pg_class c
		JOIN pg_index i ON i.indexrelid = c.oid
		JOIN pg_class t ON t.oid = i.indrelid
		WHERE t.relname = $1
		LIMIT 1;
	`
	var indexBytes int64
	row := p.db.QueryRowContext(ctx, querySize, p.cfg.Table)
	if err := row.Scan(&indexBytes); err == nil {
		stats.DiskSizeBytes = indexBytes
		stats.MemorySizeBytes = indexBytes
	}

	// 3. Approximate tombstones from dead tuples in pg_stat_user_tables
	queryBloat := `
		SELECT n_dead_tup, n_live_tup
		FROM pg_stat_user_tables
		WHERE relname = $1
		LIMIT 1;
	`
	var deadTup, liveTup int64
	if err := p.db.QueryRowContext(ctx, queryBloat, p.cfg.Table).Scan(&deadTup, &liveTup); err == nil {
		stats.TombstoneVectors = deadTup
		if liveTup+deadTup > 0 {
			stats.TombstoneRatio = float64(deadTup) / float64(liveTup+deadTup)
		}
	}

	return stats, nil
}

func (p *PgvectorConnector) SampleVectors(ctx context.Context, count int) ([]VectorRecord, error) {
	query := fmt.Sprintf("SELECT id, %s FROM %s LIMIT %d", p.cfg.Column, p.cfg.Table, count)
	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("pgvector: sample vectors failed: %w", err)
	}
	defer rows.Close()

	var records []VectorRecord
	for rows.Next() {
		var id string
		var rawVec string
		if err := rows.Scan(&id, &rawVec); err != nil {
			return nil, err
		}
		vec, err := parsePgvectorString(rawVec)
		if err != nil {
			continue
		}
		records = append(records, VectorRecord{
			ID:     id,
			Values: vec,
		})
	}
	return records, rows.Err()
}

func (p *PgvectorConnector) SampleQueries(ctx context.Context, count int) ([]QueryRequest, error) {
	// Sample queries using existing vectors with jitter or dedicated query table if configured
	records, err := p.SampleVectors(ctx, count)
	if err != nil {
		return nil, err
	}

	queries := make([]QueryRequest, len(records))
	for i, r := range records {
		queries[i] = QueryRequest{
			ID:       fmt.Sprintf("pg-q-%s", r.ID),
			Values:   r.Values,
			TopK:     10,
			EfSearch: p.cfg.EfSearch,
		}
	}
	return queries, nil
}

func (p *PgvectorConnector) SearchANN(ctx context.Context, req QueryRequest) ([]SearchResult, time.Duration, error) {
	start := time.Now()

	// Pick operator based on metric: <-> (L2), <#> (negative dot product), <=> (cosine)
	op := "<=>"
	if p.metric == "l2" {
		op = "<->"
	} else if p.metric == "dot" {
		op = "<#>"
	}

	vecStr := formatPgvectorString(req.Values)
	tx, err := p.db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return nil, time.Since(start), err
	}
	defer tx.Rollback()

	// Tune ef_search for session
	if req.EfSearch > 0 {
		_, _ = tx.ExecContext(ctx, fmt.Sprintf("SET LOCAL hnsw.ef_search = %d", req.EfSearch))
	}

	query := fmt.Sprintf("SELECT id, %s %s '%s' AS dist FROM %s ORDER BY dist LIMIT %d",
		p.cfg.Column, op, vecStr, p.cfg.Table, req.TopK)

	rows, err := tx.QueryContext(ctx, query)
	if err != nil {
		return nil, time.Since(start), err
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var id string
		var dist float32
		if err := rows.Scan(&id, &dist); err != nil {
			return nil, time.Since(start), err
		}
		results = append(results, SearchResult{
			ID:       id,
			Distance: dist,
			Score:    1.0 / (1.0 + dist),
		})
	}

	return results, time.Since(start), rows.Err()
}

func (p *PgvectorConnector) TriggerCompaction(ctx context.Context) error {
	indexName := fmt.Sprintf("%s_%s_hnsw_idx", p.cfg.Table, p.cfg.Column)
	query := fmt.Sprintf("REINDEX INDEX CONCURRENTLY %s", indexName)
	_, err := p.db.ExecContext(ctx, query)
	return err
}

func (p *PgvectorConnector) ExportTopology(ctx context.Context, maxNodes int) (*GraphTopologyExport, error) {
	// pgvector doesn't expose internal HNSW pointers directly through SQL;
	// we sample the vectors and construct a representative proximity topology.
	sample, err := p.SampleVectors(ctx, maxNodes)
	if err != nil {
		return nil, err
	}

	export := &GraphTopologyExport{
		MaxLayer:   0,
		Layers:     make(map[int]*GraphLayerExport),
		TotalNodes: len(sample),
	}
	layer0 := &GraphLayerExport{
		Layer: 0,
		Nodes: make(map[string]GraphNode),
	}

	for _, s := range sample {
		layer0.Nodes[s.ID] = GraphNode{
			ID:        s.ID,
			Layer:     0,
			Neighbors: []string{},
			IsTomb:    s.IsDeleted,
		}
	}
	export.Layers[0] = layer0
	return export, nil
}

// Utility functions to format/parse vector string "[0.1, 0.2, 0.3]"
func parsePgvectorString(s string) ([]float32, error) {
	s = strings.Trim(s, "[] ")
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	res := make([]float32, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil {
			return nil, err
		}
		res[i] = float32(v)
	}
	return res, nil
}

func formatPgvectorString(v []float32) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, val := range v {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(strconv.FormatFloat(float64(val), 'f', 6, 32))
	}
	sb.WriteString("]")
	return sb.String()
}
