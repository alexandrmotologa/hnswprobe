package connector

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ChromaConnector interacts with Chroma via its HTTP API.
type ChromaConnector struct {
	client       *http.Client
	endpoint     string
	collection   string
	collectionID string
	apiKey       string
	cfg          Config
}

// NewChromaConnector creates a connector for Chroma.
func NewChromaConnector(cfg Config) (*ChromaConnector, error) {
	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = "http://localhost:8000"
	}
	endpoint = strings.TrimRight(endpoint, "/")

	return &ChromaConnector{
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
		endpoint:   endpoint,
		collection: cfg.Collection,
		apiKey:     cfg.APIKey,
		cfg:        cfg,
	}, nil
}

func (c *ChromaConnector) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(data)
	}

	url := fmt.Sprintf("%s%s", c.endpoint, path)
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("chroma api error (%d): %s", resp.StatusCode, string(respBytes))
	}

	return respBytes, nil
}

func (c *ChromaConnector) Ping(ctx context.Context) error {
	_, err := c.doRequest(ctx, http.MethodGet, "/api/v1/heartbeat", nil)
	return err
}

func (c *ChromaConnector) Close() error {
	return nil
}

func (c *ChromaConnector) ensureCollectionID(ctx context.Context) error {
	if c.collectionID != "" {
		return nil
	}

	respBytes, err := c.doRequest(ctx, http.MethodGet, fmt.Sprintf("/api/v1/collections/%s", c.collection), nil)
	if err != nil {
		return err
	}

	var parsed struct {
		ID       string         `json:"id"`
		Name     string         `json:"name"`
		Metadata map[string]any `json:"metadata"`
	}
	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return err
	}
	c.collectionID = parsed.ID
	return nil
}

func (c *ChromaConnector) GetStats(ctx context.Context) (*IndexStats, error) {
	if err := c.ensureCollectionID(ctx); err != nil {
		return nil, err
	}

	countBytes, err := c.doRequest(ctx, http.MethodGet, fmt.Sprintf("/api/v1/collections/%s/count", c.collectionID), nil)
	if err != nil {
		return nil, err
	}

	var count int64
	_ = json.Unmarshal(countBytes, &count)

	return &IndexStats{
		Backend:        string(BackendChroma),
		IndexName:      c.collection,
		Dimension:      c.cfg.Dimension,
		TotalVectors:   count,
		ActiveVectors:  count,
		M:              c.cfg.M,
		EfConstruction: c.cfg.EfConstruction,
		EfSearch:       c.cfg.EfSearch,
		Metric:         "cosine",
	}, nil
}

func (c *ChromaConnector) SampleVectors(ctx context.Context, count int) ([]VectorRecord, error) {
	if err := c.ensureCollectionID(ctx); err != nil {
		return nil, err
	}

	body := map[string]any{
		"limit": count,
		"include": []string{
			"embeddings",
		},
	}

	respBytes, err := c.doRequest(ctx, http.MethodPost, fmt.Sprintf("/api/v1/collections/%s/get", c.collectionID), body)
	if err != nil {
		return nil, err
	}

	var parsed struct {
		IDs        []string      `json:"ids"`
		Embeddings [][]float32   `json:"embeddings"`
	}
	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, err
	}

	records := make([]VectorRecord, len(parsed.IDs))
	for i, id := range parsed.IDs {
		var vec []float32
		if i < len(parsed.Embeddings) {
			vec = parsed.Embeddings[i]
		}
		records[i] = VectorRecord{
			ID:     id,
			Values: vec,
		}
	}

	return records, nil
}

func (c *ChromaConnector) SampleQueries(ctx context.Context, count int) ([]QueryRequest, error) {
	records, err := c.SampleVectors(ctx, count)
	if err != nil {
		return nil, err
	}

	queries := make([]QueryRequest, len(records))
	for i, r := range records {
		queries[i] = QueryRequest{
			ID:       fmt.Sprintf("chroma-q-%s", r.ID),
			Values:   r.Values,
			TopK:     10,
			EfSearch: c.cfg.EfSearch,
		}
	}
	return queries, nil
}

func (c *ChromaConnector) SearchANN(ctx context.Context, req QueryRequest) ([]SearchResult, time.Duration, error) {
	start := time.Now()
	if err := c.ensureCollectionID(ctx); err != nil {
		return nil, time.Since(start), err
	}

	body := map[string]any{
		"query_embeddings": [][]float32{req.Values},
		"n_results":        req.TopK,
	}

	respBytes, err := c.doRequest(ctx, http.MethodPost, fmt.Sprintf("/api/v1/collections/%s/query", c.collectionID), body)
	if err != nil {
		return nil, time.Since(start), err
	}

	var parsed struct {
		IDs       [][]string    `json:"ids"`
		Distances [][]float32   `json:"distances"`
	}
	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, time.Since(start), err
	}

	var results []SearchResult
	if len(parsed.IDs) > 0 && len(parsed.IDs[0]) > 0 {
		for i, id := range parsed.IDs[0] {
			var dist float32
			if len(parsed.Distances) > 0 && i < len(parsed.Distances[0]) {
				dist = parsed.Distances[0][i]
			}
			results = append(results, SearchResult{
				ID:       id,
				Distance: dist,
				Score:    1.0 / (1.0 + dist),
			})
		}
	}

	return results, time.Since(start), nil
}

func (c *ChromaConnector) TriggerCompaction(ctx context.Context) error {
	// Chroma auto-persists; trigger collection update if needed
	return nil
}

func (c *ChromaConnector) ExportTopology(ctx context.Context, maxNodes int) (*GraphTopologyExport, error) {
	sample, err := c.SampleVectors(ctx, maxNodes)
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
