package test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
	"github.com/alexandrmotologa/hnswprobe/pkg/quantization"
)

func TestScalarQuantizerAndDistortion(t *testing.T) {
	sq := quantization.NewScalarQuantizer()

	// 1. Test single vector quantization
	v := []float32{-1.0, 0.0, 0.5, 1.0}
	qv := sq.Quantize("v1", v)
	if len(qv.Data) != 4 {
		t.Fatalf("expected 4 bytes, got %d", len(qv.Data))
	}
	if qv.Data[0] != 0 {
		t.Errorf("expected min value mapped to 0, got %d", qv.Data[0])
	}
	if qv.Data[3] != 255 {
		t.Errorf("expected max value mapped to 255, got %d", qv.Data[3])
	}

	dequant := sq.Dequantize(qv)
	cosSim := groundtruth.CosineSimilarity(v, dequant)
	if cosSim < 0.99 {
		t.Errorf("expected high reconstruction cosine similarity >= 0.99, got %f", cosSim)
	}

	// 2. Test corpus profiling
	r := rand.New(rand.NewSource(77))
	corpus := make([]groundtruth.VectorItem, 50)
	for i := 0; i < 50; i++ {
		vals := make([]float32, 16)
		for j := 0; j < 16; j++ {
			vals[j] = float32(r.NormFloat64())
		}
		corpus[i] = groundtruth.VectorItem{ID: fmt.Sprintf("vec-%d", i), Values: vals}
	}

	queries := make([]groundtruth.QueryItem, 5)
	for i := 0; i < 5; i++ {
		vals := make([]float32, 16)
		for j := 0; j < 16; j++ {
			vals[j] = float32(r.NormFloat64())
		}
		queries[i] = groundtruth.QueryItem{ID: fmt.Sprintf("q-%d", i), Values: vals}
	}

	report := sq.ProfileQuantization(corpus, queries, 5)
	if report.MemorySavingsPercent != 75.0 {
		t.Errorf("expected 75%% memory savings, got %f", report.MemorySavingsPercent)
	}
	if report.MeanAngularDistortion > 0.05 {
		t.Errorf("expected low angular distortion <= 0.05, got %f", report.MeanAngularDistortion)
	}
	if report.QuantizedRecallAtK < 0.80 {
		t.Errorf("expected quantized recall >= 0.80, got %f", report.QuantizedRecallAtK)
	}
}
