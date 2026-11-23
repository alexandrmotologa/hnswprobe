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

// QdrantConnector interacts with Qdrant via its HTTP REST API.
type QdrantConnector struct {
	client     *http.Client
	endpoint   string
	collection string
	apiKey     string
	cfg        Config
}

// NewQdrantConnector creates a connector for Qdrant.
func NewQdrantConnector(cfg Config) (*QdrantConnector, error) {
	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = "http://localhost:6333"
	}
	endpoint = strings.TrimRight(endpoint, "/")

	return &QdrantConnector{
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
		endpoint:   endpoint,
		collection: cfg.Collection,
		apiKey:     cfg.APIKey,
		cfg:        cfg,
	}, nil
}

func (q *QdrantConnector) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(data)
	}

	url := fmt.Sprintf("%s%s", q.endpoint, path)
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}

	resp, err := q.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("qdrant api error (%d): %s", resp.StatusCode, string(respBytes))
	}

	return respBytes, nil
}

func (q *QdrantConnector) Ping(ctx context.Context) error {
	_, err := q.doRequest(ctx, http.MethodGet, "/healthz", nil)
	return err
}

func (q *QdrantConnector) Close() error {
	return nil
}

func (q *QdrantConnector) GetStats(ctx context.Context) (*IndexStats, error) {
	respBytes, err := q.doRequest(ctx, http.MethodGet, fmt.Sprintf("/collections/%s", q.collection), nil)
	if err != nil {
		return nil, err
	}

	var parsed struct {
		Result struct {
			Status       string `json:"status"`
			VectorsCount int64  `json:"vectors_count"`
			PointsCount  int64  `json:"points_count"`
			SegmentsCount int   `json:"segments_count"`
			Config       struct {
				Params struct {
					Vectors struct {
						Size     int    `json:"size"`
						Distance string `json:"distance"`
					} `json:"vectors"`
				} `json:"params"`
				HnswConfig struct {
					M              int `json:"m"`
					EfConstruct    int `json:"ef_construct"`
					FullScanThreshold int `json:"full_scan_threshold"`
				} `json:"hnsw_config"`
			} `json:"config"`
		} `json:"result"`
	}

	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, err
	}

	res := parsed.Result
	total := res.PointsCount
	tombstones := total - res.VectorsCount
	ratio := 0.0
	if total > 0 && tombstones > 0 {
		ratio = float64(tombstones) / float64(total)
	}

	return &IndexStats{
		Backend:          string(BackendQdrant),
		IndexName:        q.collection,
		Dimension:        res.Config.Params.Vectors.Size,
		TotalVectors:     total,
		ActiveVectors:    res.VectorsCount,
		TombstoneVectors: tombstones,
		TombstoneRatio:   ratio,
		M:                res.Config.HnswConfig.M,
		EfConstruction:   res.Config.HnswConfig.EfConstruct,
		EfSearch:         q.cfg.EfSearch,
		Metric:           res.Config.Params.Vectors.Distance,
	}, nil
}

func (q *QdrantConnector) SampleVectors(ctx context.Context, count int) ([]VectorRecord, error) {
	body := map[string]any{
		"limit":        count,
		"with_vector":  true,
		"with_payload": false,
	}

	respBytes, err := q.doRequest(ctx, http.MethodPost, fmt.Sprintf("/collections/%s/points/scroll", q.collection), body)
	if err != nil {
		return nil, err
	}

	var parsed struct {
		Result struct {
			Points []struct {
				ID     any       `json:"id"`
				Vector []float32 `json:"vector"`
			} `json:"points"`
		} `json:"result"`
	}

	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, err
	}

	records := make([]VectorRecord, len(parsed.Result.Points))
	for i, pt := range parsed.Result.Points {
		records[i] = VectorRecord{
			ID:     fmt.Sprintf("%v", pt.ID),
			Values: pt.Vector,
		}
	}

	return records, nil
}

func (q *QdrantConnector) SampleQueries(ctx context.Context, count int) ([]QueryRequest, error) {
	records, err := q.SampleVectors(ctx, count)
	if err != nil {
		return nil, err
	}

	queries := make([]QueryRequest, len(records))
	for i, r := range records {
		queries[i] = QueryRequest{
			ID:       fmt.Sprintf("qdrant-q-%s", r.ID),
			Values:   r.Values,
			TopK:     10,
			EfSearch: q.cfg.EfSearch,
		}
	}
	return queries, nil
}

func (q *QdrantConnector) SearchANN(ctx context.Context, req QueryRequest) ([]SearchResult, time.Duration, error) {
	start := time.Now()

	body := map[string]any{
		"vector": req.Values,
		"limit":  req.TopK,
		"params": map[string]any{
			"hnsw_ef": req.EfSearch,
		},
	}

	respBytes, err := q.doRequest(ctx, http.MethodPost, fmt.Sprintf("/collections/%s/points/search", q.collection), body)
	if err != nil {
		return nil, time.Since(start), err
	}

	var parsed struct {
		Result []struct {
			ID    any     `json:"id"`
			Score float32 `json:"score"`
		} `json:"result"`
	}

	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, time.Since(start), err
	}

	results := make([]SearchResult, len(parsed.Result))
	for i, res := range parsed.Result {
		results[i] = SearchResult{
			ID:       fmt.Sprintf("%v", res.ID),
			Score:    res.Score,
			Distance: 1.0 - res.Score, // Normalized score distance
		}
	}

	return results, time.Since(start), nil
}

func (q *QdrantConnector) TriggerCompaction(ctx context.Context) error {
	// Call Qdrant update collection optimization
	body := map[string]any{
		"optimizers_config": map[string]any{
			"vacuum_deleted_segments_threshold": 0.05,
		},
	}
	_, err := q.doRequest(ctx, http.MethodPatch, fmt.Sprintf("/collections/%s", q.collection), body)
	return err
}

func (q *QdrantConnector) ExportTopology(ctx context.Context, maxNodes int) (*GraphTopologyExport, error) {
	records, err := q.SampleVectors(ctx, maxNodes)
	if err != nil {
		return nil, err
	}

	export := &GraphTopologyExport{
		MaxLayer:   0,
		Layers:     make(map[int]*GraphLayerExport),
		TotalNodes: len(records),
	}
	layer0 := &GraphLayerExport{
		Layer: 0,
		Nodes: make(map[string]GraphNode),
	}

	for _, r := range records {
		layer0.Nodes[r.ID] = GraphNode{
			ID:        r.ID,
			Layer:     0,
			Neighbors: []string{},
			IsTomb:    r.IsDeleted,
		}
	}
	export.Layers[0] = layer0
	return export, nil
}
