package test

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
)

func TestMemoryHNSWConnectorAndDecay(t *testing.T) {
	ctx := context.Background()
	cfg := connector.Config{
		Backend:        connector.BackendMemory,
		Dimension:      8,
		M:              16,
		EfConstruction: 64,
		EfSearch:       32,
		Metric:         "cosine",
	}

	conn, err := connector.NewConnector(cfg)
	if err != nil {
		t.Fatalf("failed to create memory connector: %v", err)
	}
	defer conn.Close()

	memConn, ok := conn.(*connector.MemoryConnector)
	if !ok {
		t.Fatalf("expected *connector.MemoryConnector")
	}

	// 1. Insert 200 synthetic vectors
	r := rand.New(rand.NewSource(42))
	vecs := make([]groundtruth.VectorItem, 200)
	for i := 0; i < 200; i++ {
		values := make([]float32, 8)
		for j := 0; j < 8; j++ {
			values[j] = float32(r.NormFloat64())
		}
		id := fmt.Sprintf("vec-%03d", i)
		vecs[i] = groundtruth.VectorItem{ID: id, Values: values}
		memConn.Insert(id, values)
	}

	// Check stats
	stats, err := conn.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed to get stats: %v", err)
	}
	if stats.TotalVectors != 200 {
		t.Errorf("expected 200 vectors, got %d", stats.TotalVectors)
	}
	if stats.TombstoneRatio != 0.0 {
		t.Errorf("expected 0 tombstone ratio initially, got %f", stats.TombstoneRatio)
	}

	// 2. Perform baseline recall evaluation
	queries := make([]groundtruth.QueryItem, 10)
	for i := 0; i < 10; i++ {
		qVals := make([]float32, 8)
		for j := 0; j < 8; j++ {
			qVals[j] = float32(r.NormFloat64())
		}
		queries[i] = groundtruth.QueryItem{ID: fmt.Sprintf("q-%d", i), Values: qVals}
	}

	solver := groundtruth.NewExactSolver(groundtruth.MetricCosine)
	exactMap := solver.SolveBatch(queries, vecs, 5)

	approxMap := make(map[string][]groundtruth.Neighbor)
	for _, q := range queries {
		annRes, _, err := conn.SearchANN(ctx, connector.QueryRequest{
			ID:       q.ID,
			Values:   q.Values,
			TopK:     5,
			EfSearch: 32,
		})
		if err != nil {
			t.Fatalf("search ANN failed: %v", err)
		}
		nbrs := make([]groundtruth.Neighbor, len(annRes))
		for j, res := range annRes {
			nbrs[j] = groundtruth.Neighbor{ID: res.ID, Distance: res.Distance}
		}
		approxMap[q.ID] = nbrs
	}

	baseReport := groundtruth.EvaluateRecall(exactMap, approxMap, 5)
	if baseReport.MeanRecall < 0.80 {
		t.Errorf("expected high baseline recall >= 0.80, got %f", baseReport.MeanRecall)
	}

	// 3. Mark 50 nodes as tombstones to simulate decay
	for i := 0; i < 50; i++ {
		memConn.MarkTombstone(fmt.Sprintf("vec-%03d", i))
	}

	degradedStats, err := conn.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed to get degraded stats: %v", err)
	}
	if degradedStats.TombstoneVectors != 50 {
		t.Errorf("expected 50 tombstones, got %d", degradedStats.TombstoneVectors)
	}
	if degradedStats.TombstoneRatio < 0.24 || degradedStats.TombstoneRatio > 0.26 {
		t.Errorf("expected ~0.25 tombstone ratio, got %f", degradedStats.TombstoneRatio)
	}

	// 4. Test compaction
	if err := conn.TriggerCompaction(ctx); err != nil {
		t.Fatalf("compaction failed: %v", err)
	}

	compactStats, err := conn.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed to get compact stats: %v", err)
	}
	if compactStats.TombstoneVectors != 0 {
		t.Errorf("expected 0 tombstones after compaction, got %d", compactStats.TombstoneVectors)
	}
	if compactStats.TotalVectors != 150 {
		t.Errorf("expected 150 vectors after compaction, got %d", compactStats.TotalVectors)
	}
}
