package test

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/alexandrmotologa/hnswprobe/pkg/compactor"
	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
	"github.com/alexandrmotologa/hnswprobe/pkg/optimizer"
	"github.com/alexandrmotologa/hnswprobe/pkg/topology"
)

func TestParetoSweepAndPolicyEvaluation(t *testing.T) {
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

	r := rand.New(rand.NewSource(42))
	corpus := make([]groundtruth.VectorItem, 100)
	for i := 0; i < 100; i++ {
		vals := []float32{float32(r.NormFloat64()), float32(r.NormFloat64()), float32(r.NormFloat64()), float32(r.NormFloat64())}
		id := fmt.Sprintf("vec-%03d", i)
		corpus[i] = groundtruth.VectorItem{ID: id, Values: vals}
		memConn.Insert(id, vals)
	}

	queries := make([]connector.QueryRequest, 5)
	gtQueries := make([]groundtruth.QueryItem, 5)
	for i := 0; i < 5; i++ {
		vals := []float32{float32(r.NormFloat64()), float32(r.NormFloat64()), float32(r.NormFloat64()), float32(r.NormFloat64())}
		id := fmt.Sprintf("q-%d", i)
		queries[i] = connector.QueryRequest{ID: id, Values: vals, TopK: 5}
		gtQueries[i] = groundtruth.QueryItem{ID: id, Values: vals}
	}

	solver := groundtruth.NewExactSolver(groundtruth.MetricCosine)
	exactMap := solver.SolveBatch(gtQueries, corpus, 5)

	frontier, err := optimizer.SweepEfSearch(ctx, conn, queries, exactMap, 5, []int{8, 16, 32})
	if err != nil {
		t.Fatalf("sweep failed: %v", err)
	}

	if len(frontier.Points) != 3 {
		t.Errorf("expected 3 pareto points, got %d", len(frontier.Points))
	}
	if len(frontier.FrontierPoints) == 0 {
		t.Errorf("expected non-empty frontier points")
	}

	// Test recommendations
	stats, _ := conn.GetStats(ctx)
	evalReport := groundtruth.EvaluationReport{MeanRecall: 0.82}
	topoReport := topology.GraphReport{TombstoneBloatRatio: 0.18, OrphanRatio: 0.01}

	rec := optimizer.GenerateRecommendations(stats, evalReport, topoReport, frontier, 0.90)
	if !rec.RequiresCompaction {
		t.Errorf("expected recommendation to require compaction due to TBR 0.18")
	}
	if rec.ActionableAdvice == "" {
		t.Errorf("expected non-empty actionable advice")
	}

	// Test policy trigger
	cmpr := compactor.NewCompactor()
	policy := compactor.Policy{
		MinRecall:         0.90,
		MaxTombstoneRatio: 0.15,
		AutoCompact:       true,
	}

	event, err := compactor.EvaluatePolicy(ctx, conn, cmpr, stats, evalReport, topoReport, policy)
	if err != nil {
		t.Fatalf("policy evaluation failed: %v", err)
	}
	if !event.Breached {
		t.Errorf("expected policy breach")
	}
	if !event.CompactionExecuted {
		t.Errorf("expected compaction to be executed")
	}

	// Test AdaptiveController
	ctrl := optimizer.NewAdaptiveController(0.95, 32, 128)
	status := ctrl.GetStatus()
	if status.ActiveEfSearch != 32 {
		t.Errorf("expected initial efSearch 32, got %d", status.ActiveEfSearch)
	}

	// Degraded recall should step up efSearch
	newEf := ctrl.UpdateEvaluation(0.80, nil)
	if newEf <= 32 {
		t.Errorf("expected adaptive efSearch to increase above 32, got %d", newEf)
	}

	// Satisfied recall should relax efSearch
	relaxedEf := ctrl.UpdateEvaluation(0.98, nil)
	if relaxedEf >= newEf {
		t.Errorf("expected adaptive efSearch to relax toward base, got %d", relaxedEf)
	}
}
