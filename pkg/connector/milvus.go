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

// MilvusConnector interacts with Milvus via its v2 REST API.
type MilvusConnector struct {
	client      *http.Client
	endpoint    string
	collection  string
	vectorField string
	apiKey      string
	cfg         Config
}

// NewMilvusConnector initializes a Milvus v2 REST connector.
func NewMilvusConnector(cfg Config) (*MilvusConnector, error) {
	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = "http://localhost:19530"
	}
	endpoint = strings.TrimRight(endpoint, "/")

	field := cfg.VectorField
	if field == "" {
		field = "vector"
	}

	return &MilvusConnector{
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
		endpoint:    endpoint,
		collection:  cfg.Collection,
		vectorField: field,
		apiKey:      cfg.APIKey,
		cfg:         cfg,
	}, nil
}

func (m *MilvusConnector) doRequest(ctx context.Context, path string, body any) ([]byte, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(data)
	}

	url := fmt.Sprintf("%s%s", m.endpoint, path)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bodyReader)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if m.apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", m.apiKey))
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("milvus api error (%d): %s", resp.StatusCode, string(respBytes))
	}

	return respBytes, nil
}

func (m *MilvusConnector) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/v2/vectordb/collections/list", m.endpoint), nil)
	if err != nil {
		return err
	}
	resp, err := m.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	return fmt.Errorf("milvus ping returned status: %d", resp.StatusCode)
}

func (m *MilvusConnector) Close() error {
	return nil
}

func (m *MilvusConnector) GetStats(ctx context.Context) (*IndexStats, error) {
	body := map[string]any{
		"collectionName": m.collection,
	}

	respBytes, err := m.doRequest(ctx, "/v2/vectordb/collections/describe", body)
	if err != nil {
		return nil, err
	}

	var parsed struct {
		Code int `json:"code"`
		Data struct {
			CollectionName string `json:"collectionName"`
			Fields         []struct {
				Name   string         `json:"name"`
				Type   string         `json:"type"`
				Params map[string]any `json:"params"`
			} `json:"fields"`
			Indexes []struct {
				FieldName  string `json:"fieldName"`
				IndexName  string `json:"indexName"`
				MetricType string `json:"metricType"`
			} `json:"indexes"`
		} `json:"data"`
	}

	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, err
	}

	dim := m.cfg.Dimension
	metric := "COSINE"
	if len(parsed.Data.Indexes) > 0 {
		metric = parsed.Data.Indexes[0].MetricType
	}

	return &IndexStats{
		Backend:        string(BackendMilvus),
		IndexName:      m.collection,
		Dimension:      dim,
		TotalVectors:   0,
		ActiveVectors:  0,
		M:              m.cfg.M,
		EfConstruction: m.cfg.EfConstruction,
		EfSearch:       m.cfg.EfSearch,
		Metric:         metric,
	}, nil
}

func (m *MilvusConnector) SampleVectors(ctx context.Context, count int) ([]VectorRecord, error) {
	body := map[string]any{
		"collectionName": m.collection,
		"limit":          count,
		"outputFields":   []string{"id", m.vectorField},
	}

	respBytes, err := m.doRequest(ctx, "/v2/vectordb/entities/query", body)
	if err != nil {
		return nil, err
	}

	var parsed struct {
		Code int `json:"code"`
		Data []map[string]any `json:"data"`
	}

	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, err
	}

	records := make([]VectorRecord, 0, len(parsed.Data))
	for _, item := range parsed.Data {
		idVal := fmt.Sprintf("%v", item["id"])
		vecRaw, ok := item[m.vectorField].([]any)
		if !ok {
			continue
		}
		vec := make([]float32, len(vecRaw))
		for j, val := range vecRaw {
			if f, ok := val.(float64); ok {
				vec[j] = float32(f)
			}
		}
		records = append(records, VectorRecord{
			ID:     idVal,
			Values: vec,
		})
	}

	return records, nil
}

func (m *MilvusConnector) SampleQueries(ctx context.Context, count int) ([]QueryRequest, error) {
	records, err := m.SampleVectors(ctx, count)
	if err != nil {
		return nil, err
	}

	queries := make([]QueryRequest, len(records))
	for i, r := range records {
		queries[i] = QueryRequest{
			ID:       fmt.Sprintf("milvus-q-%s", r.ID),
			Values:   r.Values,
			TopK:     10,
			EfSearch: m.cfg.EfSearch,
		}
	}
	return queries, nil
}

func (m *MilvusConnector) SearchANN(ctx context.Context, req QueryRequest) ([]SearchResult, time.Duration, error) {
	start := time.Now()

	body := map[string]any{
		"collectionName": m.collection,
		"data":           [][]float32{req.Values},
		"limit":          req.TopK,
		"searchParams": map[string]any{
			"params": map[string]any{
				"ef": req.EfSearch,
			},
		},
	}

	respBytes, err := m.doRequest(ctx, "/v2/vectordb/entities/search", body)
	if err != nil {
		return nil, time.Since(start), err
	}

	var parsed struct {
		Code int `json:"code"`
		Data [][]struct {
			ID       any     `json:"id"`
			Distance float32 `json:"distance"`
		} `json:"data"`
	}

	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return nil, time.Since(start), err
	}

	var results []SearchResult
	if len(parsed.Data) > 0 {
		for _, item := range parsed.Data[0] {
			results = append(results, SearchResult{
				ID:       fmt.Sprintf("%v", item.ID),
				Distance: item.Distance,
				Score:    1.0 / (1.0 + item.Distance),
			})
		}
	}

	return results, time.Since(start), nil
}

func (m *MilvusConnector) TriggerCompaction(ctx context.Context) error {
	body := map[string]any{
		"collectionName": m.collection,
	}
	_, err := m.doRequest(ctx, "/v2/vectordb/collections/compact", body)
	return err
}

func (m *MilvusConnector) ExportTopology(ctx context.Context, maxNodes int) (*GraphTopologyExport, error) {
	sample, err := m.SampleVectors(ctx, maxNodes)
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
