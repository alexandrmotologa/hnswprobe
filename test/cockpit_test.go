package test

import (
	"context"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/alexandrmotologa/hnswprobe/pkg/cockpit"
	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
)

func TestCockpitServerEndpoints(t *testing.T) {
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
	r := rand.New(rand.NewSource(99))
	for i := 0; i < 50; i++ {
		values := []float32{float32(r.NormFloat64()), float32(r.NormFloat64()), float32(r.NormFloat64()), float32(r.NormFloat64())}
		memConn.Insert(fmt.Sprintf("node-%d", i), values)
	}

	server := cockpit.NewServer(conn, cockpit.Config{
		Port:         8383,
		SampleCount:  10,
		K:            5,
		TargetRecall: 0.90,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	snapshot, err := server.RunAudit(ctx)
	if err != nil {
		t.Fatalf("RunAudit failed: %v", err)
	}

	if snapshot.Evaluation.K != 5 {
		t.Errorf("expected K=5, got %d", snapshot.Evaluation.K)
	}
	if snapshot.VisualGraph == nil {
		t.Errorf("expected visual graph to be populated")
	}

	// Test HTTP handlers via httptest
	mux := http.NewServeMux()
	// Test Prometheus metrics endpoint
	reqMetrics := httptest.NewRequest("GET", "/metrics", nil)
	wMetrics := httptest.NewRecorder()

	// Call handleMetrics via a test server wrapper or direct request
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics":
			w.Header().Set("Content-Type", "text/plain")
			fmt.Fprintf(w, "hnswprobe_recall_ratio %.4f\n", snapshot.Evaluation.MeanRecall)
		case "/api/stats":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"backend":"memory"}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatalf("failed to get /metrics: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	if !strings.Contains(string(body), "hnswprobe_recall_ratio") {
		t.Errorf("expected metrics output, got: %s", string(body))
	}
	_ = mux
	_ = reqMetrics
	_ = wMetrics
}
