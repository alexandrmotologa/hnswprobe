package test

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
	"github.com/alexandrmotologa/hnswprobe/pkg/topology"
)

func TestTopologyAnalyzerAndOrphans(t *testing.T) {
	ctx := context.Background()
	cfg := connector.Config{
		Backend:        connector.BackendMemory,
		Dimension:      4,
		M:              16,
		EfConstruction: 32,
		EfSearch:       16,
	}

	conn, err := connector.NewConnector(cfg)
	if err != nil {
		t.Fatalf("failed to create connector: %v", err)
	}
	defer conn.Close()

	memConn := conn.(*connector.MemoryConnector)

	// Insert 100 vectors
	r := rand.New(rand.NewSource(123))
	vecMap := make(map[string][]float32, 100)
	for i := 0; i < 100; i++ {
		values := make([]float32, 4)
		for j := 0; j < 4; j++ {
			values[j] = float32(r.NormFloat64())
		}
		id := fmt.Sprintf("node-%03d", i)
		vecMap[id] = values
		memConn.Insert(id, values)
	}

	// Mark 20 tombstones
	for i := 0; i < 20; i++ {
		memConn.MarkTombstone(fmt.Sprintf("node-%03d", i))
	}

	topoExport, err := conn.ExportTopology(ctx, 100)
	if err != nil {
		t.Fatalf("failed to export topology: %v", err)
	}

	queries := []connector.QueryRequest{
		{ID: "q-1", Values: []float32{0.5, 0.5, 0.0, 0.0}, TopK: 5},
		{ID: "q-2", Values: []float32{0.0, 0.0, 0.5, 0.5}, TopK: 5},
	}

	report := topology.AnalyzeTopology(topoExport, queries, vecMap, groundtruth.MetricCosine, 16)

	if report.TotalNodes != 100 {
		t.Errorf("expected 100 total nodes, got %d", report.TotalNodes)
	}
	if report.TombstoneNodes != 20 {
		t.Errorf("expected 20 tombstone nodes, got %d", report.TombstoneNodes)
	}
	if report.TombstoneBloatRatio != 0.20 {
		t.Errorf("expected TBR 0.20, got %f", report.TombstoneBloatRatio)
	}
	if report.ConnectedComponents <= 0 {
		t.Errorf("expected at least 1 connected component, got %d", report.ConnectedComponents)
	}
	if report.EstimatedDiameter <= 0 {
		t.Errorf("expected positive estimated diameter, got %d", report.EstimatedDiameter)
	}
	if report.TrajectoryEfficiency <= 0 {
		t.Errorf("expected positive trajectory efficiency, got %f", report.TrajectoryEfficiency)
	}

	// Test degree Gini calculation
	uniform := []int{10, 10, 10, 10}
	giniUniform := topology.ComputeDegreeGini(uniform)
	if giniUniform != 0.0 {
		t.Errorf("expected uniform Gini 0, got %f", giniUniform)
	}

	skewed := []int{1, 1, 1, 100}
	giniSkewed := topology.ComputeDegreeGini(skewed)
	if giniSkewed <= 0.4 {
		t.Errorf("expected high skewed Gini > 0.4, got %f", giniSkewed)
	}
}
