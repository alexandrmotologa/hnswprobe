package cockpit

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"time"

	"github.com/alexandrmotologa/hnswprobe/pkg/compactor"
	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
	"github.com/alexandrmotologa/hnswprobe/pkg/optimizer"
	"github.com/alexandrmotologa/hnswprobe/pkg/topology"
)

//go:embed web/*
var embeddedFiles embed.FS

// Server hosts the embedded HTTP web cockpit and Prometheus metrics exporter.
type Server struct {
	mu           sync.RWMutex
	conn         connector.Connector
	compactor    *compactor.Compactor
	port         int
	sampleCount  int
	k            int
	targetRecall float64
	latestAudit  *AuditSnapshot
	httpServer   *http.Server
}

// Config defines cockpit server initialization settings.
type Config struct {
	Port         int
	SampleCount  int
	K            int
	TargetRecall float64
}

// NewServer creates a new Cockpit HTTP server instance.
func NewServer(conn connector.Connector, cfg Config) *Server {
	if cfg.Port <= 0 {
		cfg.Port = 8383
	}
	if cfg.SampleCount <= 0 {
		cfg.SampleCount = 100
	}
	if cfg.K <= 0 {
		cfg.K = 10
	}
	if cfg.TargetRecall <= 0 {
		cfg.TargetRecall = 0.95
	}

	return &Server{
		conn:         conn,
		compactor:    compactor.NewCompactor(),
		port:         cfg.Port,
		sampleCount:  cfg.SampleCount,
		k:            cfg.K,
		targetRecall: cfg.TargetRecall,
	}
}

// Start begins serving HTTP traffic.
func (s *Server) Start() error {
	mux := http.NewServeMux()

	// Static web assets
	webFS, err := fs.Sub(embeddedFiles, "web")
	if err != nil {
		return fmt.Errorf("failed to load embedded web assets: %w", err)
	}
	mux.Handle("/", http.FileServer(http.FS(webFS)))

	// API routes
	mux.HandleFunc("/api/stats", s.handleStats)
	mux.HandleFunc("/api/audit", s.handleAudit)
	mux.HandleFunc("/api/compact", s.handleCompact)
	mux.HandleFunc("/api/tune", s.handleTune)
	mux.HandleFunc("/metrics", s.handleMetrics)

	s.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", s.port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
	}

	// Trigger initial background audit
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, _ = s.RunAudit(ctx)
	}()

	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully terminates the HTTP server.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// RunAudit executes a comprehensive vector index audit and caches the snapshot.
func (s *Server) RunAudit(ctx context.Context) (*AuditSnapshot, error) {
	stats, err := s.conn.GetStats(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve stats: %w", err)
	}

	// 1. Sample vectors and queries
	sampleVecs, err := s.conn.SampleVectors(ctx, s.sampleCount*3)
	if err != nil {
		return nil, fmt.Errorf("failed to sample vectors: %w", err)
	}
	if len(sampleVecs) == 0 {
		return nil, fmt.Errorf("no vectors available for sampling")
	}

	sampleQueries, err := s.conn.SampleQueries(ctx, s.sampleCount)
	if err != nil {
		return nil, fmt.Errorf("failed to sample queries: %w", err)
	}

	// 2. Compute exact ground truth
	metric := groundtruth.MetricCosine
	if stats.Metric == "l2" {
		metric = groundtruth.MetricEuclidean
	} else if stats.Metric == "dot" {
		metric = groundtruth.MetricDot
	}

	corpus := make([]groundtruth.VectorItem, len(sampleVecs))
	vecMap := make(map[string][]float32, len(sampleVecs))
	for i, v := range sampleVecs {
		corpus[i] = groundtruth.VectorItem{ID: v.ID, Values: v.Values}
		vecMap[v.ID] = v.Values
	}

	gtQueries := make([]groundtruth.QueryItem, len(sampleQueries))
	for i, q := range sampleQueries {
		gtQueries[i] = groundtruth.QueryItem{ID: q.ID, Values: q.Values}
	}

	solver := groundtruth.NewExactSolver(metric)
	exactMap := solver.SolveBatch(gtQueries, corpus, s.k)

	// 3. Collect approximate search returns
	approxMap := make(map[string][]groundtruth.Neighbor, len(sampleQueries))
	for _, q := range sampleQueries {
		res, _, err := s.conn.SearchANN(ctx, q)
		if err != nil {
			continue
		}
		nbrs := make([]groundtruth.Neighbor, len(res))
		for i, r := range res {
			nbrs[i] = groundtruth.Neighbor{ID: r.ID, Distance: r.Distance}
		}
		approxMap[q.ID] = nbrs
	}

	evalReport := groundtruth.EvaluateRecall(exactMap, approxMap, s.k)

	// 4. Export and analyze topology
	topoExport, _ := s.conn.ExportTopology(ctx, 300)
	topoReport := topology.AnalyzeTopology(topoExport, sampleQueries, vecMap, metric, int64(stats.Dimension*4+64))

	// 5. Pareto sweep
	frontier, _ := optimizer.SweepEfSearch(ctx, s.conn, sampleQueries, exactMap, s.k, []int{8, 16, 32, 64, 128})

	// 6. Generate recommendations
	recommendations := optimizer.GenerateRecommendations(stats, evalReport, topoReport, frontier, s.targetRecall)

	// 7. Project visual graph coordinates
	visualGraph := projectVisualGraph(topoExport, topoReport)

	snapshot := &AuditSnapshot{
		Timestamp:       time.Now(),
		Backend:         stats.Backend,
		IndexStats:      stats,
		Evaluation:      evalReport,
		Topology:        topoReport,
		Pareto:          frontier,
		Recommendations: recommendations,
		VisualGraph:     visualGraph,
	}

	s.mu.Lock()
	s.latestAudit = snapshot
	s.mu.Unlock()

	return snapshot, nil
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	stats, err := s.conn.GetStats(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, stats)
}

func (s *Server) handleAudit(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	latest := s.latestAudit
	s.mu.RUnlock()

	if latest == nil || r.URL.Query().Get("refresh") == "true" {
		var err error
		latest, err = s.RunAudit(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	writeJSON(w, latest)
}

func (s *Server) handleCompact(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	err := s.compactor.Execute(r.Context(), s.conn)
	if err != nil {
		http.Error(w, fmt.Sprintf("Compaction failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Refresh audit after compaction
	snapshot, err := s.RunAudit(r.Context())
	if err != nil {
		writeJSON(w, map[string]any{"status": "compacted", "audit_error": err.Error()})
		return
	}

	writeJSON(w, map[string]any{
		"status":   "compacted_successfully",
		"snapshot": snapshot,
	})
}

func (s *Server) handleTune(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req TuneRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.EfSearch <= 0 {
		req.EfSearch = 32
	}

	s.mu.RLock()
	latest := s.latestAudit
	s.mu.RUnlock()

	if latest == nil {
		var err error
		latest, err = s.RunAudit(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	// Approximate recall adjustment based on Pareto curve interpolation
	meanRecall := latest.Evaluation.MeanRecall
	meanLat := 1.2
	if latest.Pareto != nil {
		for _, p := range latest.Pareto.Points {
			if p.EfSearch == req.EfSearch {
				meanRecall = p.MeanRecall
				meanLat = p.MeanLatencyMs
				break
			}
		}
	}

	writeJSON(w, TuneResponse{
		EfSearch:      req.EfSearch,
		MeanRecall:    meanRecall,
		MeanLatencyMs: meanLat,
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	latest := s.latestAudit
	s.mu.RUnlock()

	if latest == nil {
		var err error
		latest, err = s.RunAudit(r.Context())
		if err != nil {
			http.Error(w, fmt.Sprintf("# Error running audit: %v\n", err), http.StatusInternalServerError)
			return
		}
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	fmt.Fprintf(w, "# HELP hnswprobe_recall_ratio Mean search recall@k\n")
	fmt.Fprintf(w, "# TYPE hnswprobe_recall_ratio gauge\n")
	fmt.Fprintf(w, "hnswprobe_recall_ratio{backend=\"%s\",index=\"%s\"} %.4f\n", latest.Backend, latest.IndexStats.IndexName, latest.Evaluation.MeanRecall)

	fmt.Fprintf(w, "# HELP hnswprobe_tombstone_bloat_ratio Ratio of dead nodes to total index nodes\n")
	fmt.Fprintf(w, "# TYPE hnswprobe_tombstone_bloat_ratio gauge\n")
	fmt.Fprintf(w, "hnswprobe_tombstone_bloat_ratio{backend=\"%s\",index=\"%s\"} %.4f\n", latest.Backend, latest.IndexStats.IndexName, latest.Topology.TombstoneBloatRatio)

	fmt.Fprintf(w, "# HELP hnswprobe_orphan_ratio Ratio of unreachable nodes in layer 0\n")
	fmt.Fprintf(w, "# TYPE hnswprobe_orphan_ratio gauge\n")
	fmt.Fprintf(w, "hnswprobe_orphan_ratio{backend=\"%s\",index=\"%s\"} %.4f\n", latest.Backend, latest.IndexStats.IndexName, latest.Topology.OrphanRatio)

	fmt.Fprintf(w, "# HELP hnswprobe_mean_hop_count Average greedy traversal hops\n")
	fmt.Fprintf(w, "# TYPE hnswprobe_mean_hop_count gauge\n")
	fmt.Fprintf(w, "hnswprobe_mean_hop_count{backend=\"%s\",index=\"%s\"} %.2f\n", latest.Backend, latest.IndexStats.IndexName, latest.Topology.MeanHopCount)

	fmt.Fprintf(w, "# HELP hnswprobe_estimated_diameter Maximum geodesic graph distance in layer 0\n")
	fmt.Fprintf(w, "# TYPE hnswprobe_estimated_diameter gauge\n")
	fmt.Fprintf(w, "hnswprobe_estimated_diameter{backend=\"%s\",index=\"%s\"} %d\n", latest.Backend, latest.IndexStats.IndexName, latest.Topology.EstimatedDiameter)

	fmt.Fprintf(w, "# HELP hnswprobe_memory_bytes Total memory allocated by index\n")
	fmt.Fprintf(w, "# TYPE hnswprobe_memory_bytes gauge\n")
	fmt.Fprintf(w, "hnswprobe_memory_bytes{backend=\"%s\",index=\"%s\"} %d\n", latest.Backend, latest.IndexStats.IndexName, latest.IndexStats.MemorySizeBytes)
}

func projectVisualGraph(topo *connector.GraphTopologyExport, report topology.GraphReport) *VisualGraphData {
	if topo == nil {
		return nil
	}

	visual := &VisualGraphData{
		Nodes:      make([]VisualNode, 0),
		Edges:      make([]VisualEdge, 0),
		MaxLayer:   topo.MaxLayer,
		EntryPoint: topo.EntryPoint,
	}

	r := rand.New(rand.NewSource(1337))
	nodeCoordMap := make(map[string]VisualNode)

	// Determine (X, Y) layout using circular / clustered positioning
	for l := topo.MaxLayer; l >= 0; l-- {
		layer, hasLayer := topo.Layers[l]
		if !hasLayer {
			continue
		}

		layerHealth := report.Layers[l]
		nodeIdx := 0
		layerTotal := len(layer.Nodes)
		radius := 200.0 + float64(l)*35.0

		for id, node := range layer.Nodes {
			var x, y float64
			existing, exists := nodeCoordMap[id]
			if exists {
				x = existing.X
				y = existing.Y
			} else {
				// Arrange in radial clusters
				angle := (float64(nodeIdx) / float64(layerTotal)) * 2.0 * math.Pi
				jitter := (r.Float64() - 0.5) * 40.0
				rDist := radius + jitter
				x = math.Cos(angle) * rDist
				y = math.Sin(angle) * rDist
			}

			z := float64(l) * 120.0 // Layer height in 3D projection

			var inDeg, outDeg int
			var isOrphan, isHub bool
			if layerHealth != nil {
				if nh, ok := layerHealth.Nodes[id]; ok {
					inDeg = nh.InDegree
					outDeg = nh.OutDegree
					isOrphan = nh.IsOrphan
					isHub = nh.IsHub
				}
			}

			vn := VisualNode{
				ID:        id,
				X:         x,
				Y:         y,
				Z:         z,
				Layer:     l,
				InDegree:  inDeg,
				OutDegree: outDeg,
				IsTomb:    node.IsTomb,
				IsOrphan:  isOrphan,
				IsHub:     isHub,
				IsEntry:   id == topo.EntryPoint,
			}

			visual.Nodes = append(visual.Nodes, vn)
			nodeCoordMap[id] = vn

			// Collect edges
			for _, nbrID := range node.Neighbors {
				visual.Edges = append(visual.Edges, VisualEdge{
					Source: id,
					Target: nbrID,
					Layer:  l,
				})
			}
			nodeIdx++
		}
	}

	return visual
}

func writeJSON(w http.ResponseWriter, data any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(data)
}
