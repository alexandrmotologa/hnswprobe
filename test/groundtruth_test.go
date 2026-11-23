package test

import (
	"math"
	"testing"

	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
)

func TestVectorDistances(t *testing.T) {
	a := []float32{1.0, 0.0, 0.0, 0.0}
	b := []float32{0.0, 1.0, 0.0, 0.0}
	c := []float32{1.0, 0.0, 0.0, 0.0}

	// Cosine similarity
	simAB := groundtruth.CosineSimilarity(a, b)
	if math.Abs(float64(simAB)) > 1e-6 {
		t.Errorf("expected orthogonal cosine similarity 0, got %f", simAB)
	}

	simAC := groundtruth.CosineSimilarity(a, c)
	if math.Abs(float64(simAC-1.0)) > 1e-6 {
		t.Errorf("expected identical vector cosine similarity 1.0, got %f", simAC)
	}

	// Dot product
	dotAB := groundtruth.DotProduct(a, b)
	if math.Abs(float64(dotAB)) > 1e-6 {
		t.Errorf("expected dot product 0, got %f", dotAB)
	}

	dotAC := groundtruth.DotProduct(a, c)
	if math.Abs(float64(dotAC-1.0)) > 1e-6 {
		t.Errorf("expected dot product 1.0, got %f", dotAC)
	}

	// L2 distance
	l2AB := groundtruth.SquaredL2Distance(a, b)
	if math.Abs(float64(l2AB-2.0)) > 1e-6 {
		t.Errorf("expected squared L2 distance 2.0, got %f", l2AB)
	}

	l2AC := groundtruth.SquaredL2Distance(a, c)
	if math.Abs(float64(l2AC)) > 1e-6 {
		t.Errorf("expected squared L2 distance 0.0, got %f", l2AC)
	}
}

func TestExactSolverAndEvaluator(t *testing.T) {
	corpus := []groundtruth.VectorItem{
		{ID: "vec-1", Values: []float32{1.0, 0.0, 0.0}},
		{ID: "vec-2", Values: []float32{0.9, 0.1, 0.0}},
		{ID: "vec-3", Values: []float32{0.0, 1.0, 0.0}},
		{ID: "vec-4", Values: []float32{0.0, 0.0, 1.0}},
	}

	queries := []groundtruth.QueryItem{
		{ID: "q-1", Values: []float32{1.0, 0.05, 0.0}},
	}

	solver := groundtruth.NewExactSolver(groundtruth.MetricCosine)
	exact := solver.SolveBatch(queries, corpus, 2)

	if len(exact["q-1"]) != 2 {
		t.Fatalf("expected 2 neighbors, got %d", len(exact["q-1"]))
	}

	// Top neighbors should be vec-1 or vec-2
	first := exact["q-1"][0].ID
	if first != "vec-1" && first != "vec-2" {
		t.Errorf("expected vec-1 or vec-2 as closest, got %s", first)
	}

	// Test evaluator with 100% match
	approx := exact
	report := groundtruth.EvaluateRecall(exact, approx, 2)
	if report.MeanRecall != 1.0 {
		t.Errorf("expected mean recall 1.0, got %f", report.MeanRecall)
	}

	// Test evaluator with partial match
	approxDegraded := map[string][]groundtruth.Neighbor{
		"q-1": {
			{ID: "vec-1", Distance: 0.01},
			{ID: "vec-4", Distance: 0.99}, // vec-4 instead of vec-2
		},
	}
	degradedReport := groundtruth.EvaluateRecall(exact, approxDegraded, 2)
	if degradedReport.MeanRecall != 0.5 {
		t.Errorf("expected mean recall 0.5, got %f", degradedReport.MeanRecall)
	}
}
