package test

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/alexandrmotologa/hnswprobe/pkg/cockpit"
	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
)

func BenchmarkVectorMath(b *testing.B) {
	dim := 1536
	v1 := make([]float32, dim)
	v2 := make([]float32, dim)
	r := rand.New(rand.NewSource(1337))
	for i := 0; i < dim; i++ {
		v1[i] = float32(r.NormFloat64())
		v2[i] = float32(r.NormFloat64())
	}

	b.Run("CosineSimilarity_1536d", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = groundtruth.CosineSimilarity(v1, v2)
		}
	})

	b.Run("DotProduct_1536d", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = groundtruth.DotProduct(v1, v2)
		}
	})

	b.Run("SquaredL2Distance_1536d", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = groundtruth.SquaredL2Distance(v1, v2)
		}
	})
}

func TestAuditMemoryFootprintUnder20MB(t *testing.T) {
	// Force GC before starting
	runtime.GC()
	var mBefore runtime.MemStats
	runtime.ReadMemStats(&mBefore)

	cfg := connector.Config{
		Backend:        connector.BackendMemory,
		Dimension:      64,
		M:              16,
		EfConstruction: 64,
		EfSearch:       32,
	}

	conn, err := connector.NewConnector(cfg)
	if err != nil {
		t.Fatalf("failed to create connector: %v", err)
	}
	defer conn.Close()

	memConn := conn.(*connector.MemoryConnector)

	// Populate 1000 vectors of 64 dimensions
	r := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		vals := make([]float32, 64)
		for j := 0; j < 64; j++ {
			vals[j] = float32(r.NormFloat64())
		}
		memConn.Insert(fmt.Sprintf("vec-%04d", i), vals)
	}

	// Mark 150 tombstones
	for i := 0; i < 150; i++ {
		memConn.MarkTombstone(fmt.Sprintf("vec-%04d", i*5))
	}

	server := cockpit.NewServer(conn, cockpit.Config{
		Port:         8384,
		SampleCount:  100,
		K:            10,
		TargetRecall: 0.95,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	snapshot, err := server.RunAudit(ctx)
	if err != nil {
		t.Fatalf("audit failed: %v", err)
	}

	if snapshot.Evaluation.MeanRecall <= 0 {
		t.Errorf("expected positive mean recall, got %f", snapshot.Evaluation.MeanRecall)
	}

	var mAfter runtime.MemStats
	runtime.ReadMemStats(&mAfter)

	allocMB := float64(mAfter.Alloc) / (1024.0 * 1024.0)
	t.Logf("Resident Memory Alloc: %.2f MB (threshold: 20.0 MB)", allocMB)

	if allocMB > 20.0 {
		t.Errorf("memory allocation %.2f MB exceeded 20MB limit", allocMB)
	}
}
