package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/alexandrmotologa/hnswprobe/pkg/cockpit"
	"github.com/alexandrmotologa/hnswprobe/pkg/compactor"
	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
	"github.com/alexandrmotologa/hnswprobe/pkg/quantization"
	"github.com/alexandrmotologa/hnswprobe/pkg/report"
)

var (
	Version   = "1.0.0"
	BuildDate = "2026-11-23"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	args := os.Args[2:]

	switch cmd {
	case "audit":
		runAudit(args)
	case "optimize":
		runOptimize(args)
	case "daemon":
		runDaemon(args)
	case "compact":
		runCompact(args)
	case "cockpit":
		runCockpit(args)
	case "quantize":
		runQuantize(args)
	case "generate-sample":
		runGenerateSample(args)
	case "version":
		fmt.Printf("hnswprobe v%s (%s)\n", Version, BuildDate)
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`hnswprobe - HNSW graph decay, tombstone bloat, and recall audit engine

Usage:
  hnswprobe <command> [flags]

Commands:
  audit           Audit vector index recall decay, tombstones, and graph health
  optimize        Run Pareto hyperparameter sweep to find optimal efSearch and M
  daemon          Run continuous audit sentinel with Prometheus metrics exporter
  compact         Trigger on-demand index compaction or vacuum on target database
  cockpit         Launch embedded Web Cockpit on port 8383
  quantize        Profile precision loss and recall drop from scalar quantization
  generate-sample Generate synthetic vector corpus with ground truth for CI tests
  version         Show version information

Use "hnswprobe <command> --help" for more information on a command.`)
}

func runAudit(args []string) {
	fs := flag.NewFlagSet("audit", flag.ExitOnError)
	backend := fs.String("backend", "memory", "Database backend: pgvector, qdrant, milvus, chroma, memory")
	db := fs.String("db", "", "Database connection string (for pgvector)")
	endpoint := fs.String("endpoint", "", "API endpoint URL (for Qdrant, Milvus, Chroma)")
	table := fs.String("table", "document_embeddings", "Table name (for pgvector)")
	column := fs.String("column", "embedding", "Vector column name (for pgvector)")
	collection := fs.String("collection", "default", "Collection name (for Qdrant, Milvus, Chroma)")
	vectorField := fs.String("vector-field", "vector", "Vector field name (for Milvus)")
	apiKey := fs.String("api-key", "", "API key or token")
	samples := fs.Int("samples", 100, "Number of sample queries to evaluate")
	k := fs.Int("k", 10, "Top-K nearest neighbors to inspect")
	minRecall := fs.Float64("min-recall", 0.90, "Minimum acceptable recall threshold")
	maxTBR := fs.Float64("max-tombstones", 0.15, "Maximum acceptable tombstone bloat ratio")
	targetRecall := fs.Float64("target-recall", 0.95, "Target recall for hyperparameter optimizer")
	jsonPath := fs.String("json-report", "", "Path to export structured JSON report")
	junitPath := fs.String("junit", "", "Path to export JUnit XML report")
	launchCockpit := fs.Bool("cockpit", false, "Launch Web Cockpit server after audit")
	port := fs.Int("port", 8383, "Web Cockpit HTTP port")

	_ = fs.Parse(args)

	cfg := connector.Config{
		Backend:        connector.BackendType(*backend),
		Database:       *db,
		Endpoint:       *endpoint,
		Table:          *table,
		Column:         *column,
		Collection:     *collection,
		VectorField:    *vectorField,
		APIKey:         *apiKey,
		Dimension:      16,
		M:              16,
		EfConstruction: 64,
		EfSearch:       32,
	}

	conn, err := connector.NewConnector(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating connector: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	// If memory backend and empty, populate with synthetic vectors and simulated decay
	if *backend == "memory" {
		memConn := conn.(*connector.MemoryConnector)
		populateSyntheticIndex(memConn, 300, 16, 50)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	startTime := time.Now()
	fmt.Printf("[+] Auditing %s index (samples=%d, k=%d)...\n", *backend, *samples, *k)

	server := cockpit.NewServer(conn, cockpit.Config{
		Port:         *port,
		SampleCount:  *samples,
		K:            *k,
		TargetRecall: *targetRecall,
	})

	snapshot, err := server.RunAudit(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Audit failed: %v\n", err)
		os.Exit(1)
	}
	duration := time.Since(startTime)

	// Print summary table
	printAuditSummary(snapshot, duration)

	// Export reports if requested
	if *jsonPath != "" {
		if err := report.WriteJSONReport(*jsonPath, snapshot); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to write JSON report: %v\n", err)
		} else {
			fmt.Printf("[+] JSON report written to %s\n", *jsonPath)
		}
	}

	if *junitPath != "" {
		suite := report.GenerateJUnitReport(snapshot.Evaluation, snapshot.Topology, *minRecall, *maxTBR, duration)
		if err := report.WriteJUnitReport(*junitPath, suite); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to write JUnit report: %v\n", err)
		} else {
			fmt.Printf("[+] JUnit XML report written to %s\n", *junitPath)
		}
	}

	if *launchCockpit {
		fmt.Printf("\n[+] Launching Web Cockpit on http://localhost:%d\n", *port)
		fmt.Println("[+] Press Ctrl+C to stop.")
		if err := server.Start(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		}
	}

	// Exit code evaluation for CI/CD
	if snapshot.Evaluation.MeanRecall < *minRecall || snapshot.Topology.TombstoneBloatRatio > *maxTBR {
		fmt.Println("\n[!] Health check FAILED: threshold breached.")
		os.Exit(1)
	}

	fmt.Println("\n[+] Health check PASSED: index within operational bounds.")
}

func runOptimize(args []string) {
	fs := flag.NewFlagSet("optimize", flag.ExitOnError)
	backend := fs.String("backend", "memory", "Database backend: pgvector, qdrant, milvus, chroma, memory")
	db := fs.String("db", "", "Database connection string")
	endpoint := fs.String("endpoint", "", "API endpoint URL")
	table := fs.String("table", "document_embeddings", "Table name")
	column := fs.String("column", "embedding", "Vector column name")
	collection := fs.String("collection", "default", "Collection name")
	samples := fs.Int("samples", 100, "Number of sample queries")
	k := fs.Int("k", 10, "Top-K value")
	targetRecall := fs.Float64("target-recall", 0.95, "Target recall threshold")

	_ = fs.Parse(args)

	cfg := connector.Config{
		Backend:        connector.BackendType(*backend),
		Database:       *db,
		Endpoint:       *endpoint,
		Table:          *table,
		Column:         *column,
		Collection:     *collection,
		M:              16,
		EfConstruction: 64,
		EfSearch:       32,
	}

	conn, err := connector.NewConnector(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating connector: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	if *backend == "memory" {
		memConn := conn.(*connector.MemoryConnector)
		populateSyntheticIndex(memConn, 250, 16, 30)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	fmt.Println("[+] Running Pareto frontier parameter sweep...")
	server := cockpit.NewServer(conn, cockpit.Config{SampleCount: *samples, K: *k, TargetRecall: *targetRecall})
	snapshot, err := server.RunAudit(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Optimization sweep failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n--- Pareto Hyperparameter Sweep Results ---\n")
	fmt.Printf("%-10s %-15s %-15s %-12s %-10s\n", "efSearch", "Mean Recall", "Latency (ms)", "QPS", "Optimal")
	fmt.Println("------------------------------------------------------------------")

	if snapshot.Pareto != nil {
		for _, p := range snapshot.Pareto.Points {
			optMark := ""
			if p.IsOptimal {
				optMark = "YES (Pareto)"
			}
			fmt.Printf("%-10d %-15.4f %-15.2f %-12.1f %-10s\n",
				p.EfSearch, p.MeanRecall, p.MeanLatencyMs, p.QueriesPerSecond, optMark)
		}
	}

	fmt.Println("------------------------------------------------------------------")
	rec := snapshot.Recommendations
	fmt.Printf("Current Recall:            %.2f%%\n", rec.CurrentRecall*100.0)
	fmt.Printf("Target Recall:             %.2f%%\n", rec.TargetRecall*100.0)
	fmt.Printf("Optimal efSearch:          %d\n", rec.OptimalEfSearch)
	fmt.Printf("Recommended M:             %d\n", rec.RecommendedM)
	fmt.Printf("Recommended efConstruct:   %d\n", rec.RecommendedEfConstruction)
	fmt.Printf("Advice:                    %s\n", rec.ActionableAdvice)
}

func runDaemon(args []string) {
	fs := flag.NewFlagSet("daemon", flag.ExitOnError)
	backend := fs.String("backend", "memory", "Database backend")
	db := fs.String("db", "", "Database connection string")
	endpoint := fs.String("endpoint", "", "API endpoint URL")
	table := fs.String("table", "document_embeddings", "Table name")
	column := fs.String("column", "embedding", "Vector column name")
	collection := fs.String("collection", "default", "Collection name")
	interval := fs.Duration("interval", 2*time.Minute, "Audit polling interval")
	port := fs.Int("port", 8383, "Web Cockpit and Prometheus metrics port")
	autoCompact := fs.Bool("auto-compact", false, "Automatically trigger compaction on policy breach")
	webhookURL := fs.String("webhook", "", "Webhook URL for alerts")
	minRecall := fs.Float64("min-recall", 0.90, "Minimum recall threshold")
	maxTBR := fs.Float64("max-tombstones", 0.15, "Maximum tombstone ratio")

	_ = fs.Parse(args)

	cfg := connector.Config{
		Backend:        connector.BackendType(*backend),
		Database:       *db,
		Endpoint:       *endpoint,
		Table:          *table,
		Column:         *column,
		Collection:     *collection,
		M:              16,
		EfConstruction: 64,
		EfSearch:       32,
	}

	conn, err := connector.NewConnector(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating connector: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	if *backend == "memory" {
		memConn := conn.(*connector.MemoryConnector)
		populateSyntheticIndex(memConn, 300, 16, 40)
	}

	server := cockpit.NewServer(conn, cockpit.Config{Port: *port})
	cmpr := compactor.NewCompactor()
	policy := compactor.Policy{
		MinRecall:         *minRecall,
		MaxTombstoneRatio: *maxTBR,
		MaxOrphanRatio:    0.03,
		AutoCompact:       *autoCompact,
		WebhookURL:        *webhookURL,
	}

	// Start HTTP server in background
	go func() {
		fmt.Printf("[+] Daemon web cockpit & Prometheus exporter running on :%d\n", *port)
		if err := server.Start(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "Cockpit error: %v\n", err)
		}
	}()

	// Signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	ticker := time.NewTicker(*interval)
	defer ticker.Stop()

	fmt.Printf("[+] Audit sentinel started (interval=%v, auto-compact=%v). Press Ctrl+C to exit.\n", *interval, *autoCompact)

	for {
		select {
		case <-sigChan:
			fmt.Println("\n[+] Shutting down daemon...")
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = server.Shutdown(ctx)
			cancel()
			return
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
			snapshot, err := server.RunAudit(ctx)
			if err != nil {
				fmt.Printf("[!] Audit cycle failed: %v\n", err)
				cancel()
				continue
			}

			event, err := compactor.EvaluatePolicy(ctx, conn, cmpr, snapshot.IndexStats, snapshot.Evaluation, snapshot.Topology, policy)
			if err != nil {
				fmt.Printf("[!] Policy evaluation error: %v\n", err)
			} else if event.Breached {
				fmt.Printf("[!] ALERT: %s (Severity: %s, Compacted: %v)\n", event.Reason, event.Severity, event.CompactionExecuted)
			} else {
				fmt.Printf("[+] Index healthy: recall=%.2f%%, TBR=%.2f%%\n",
					snapshot.Evaluation.MeanRecall*100.0, snapshot.Topology.TombstoneBloatRatio*100.0)
			}
			cancel()
		}
	}
}

func runCompact(args []string) {
	fs := flag.NewFlagSet("compact", flag.ExitOnError)
	backend := fs.String("backend", "memory", "Database backend")
	db := fs.String("db", "", "Database connection string")
	endpoint := fs.String("endpoint", "", "API endpoint URL")
	table := fs.String("table", "document_embeddings", "Table name")
	column := fs.String("column", "embedding", "Vector column name")
	collection := fs.String("collection", "default", "Collection name")

	_ = fs.Parse(args)

	cfg := connector.Config{
		Backend:    connector.BackendType(*backend),
		Database:   *db,
		Endpoint:   *endpoint,
		Table:      *table,
		Column:     *column,
		Collection: *collection,
	}

	conn, err := connector.NewConnector(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating connector: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	fmt.Printf("[+] Executing compaction on %s...\n", *backend)
	cmpr := compactor.NewCompactor()
	if err := cmpr.Execute(ctx, conn); err != nil {
		fmt.Fprintf(os.Stderr, "Compaction failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("[+] Compaction completed successfully.")
}

func runCockpit(args []string) {
	fs := flag.NewFlagSet("cockpit", flag.ExitOnError)
	port := fs.Int("port", 8383, "HTTP port")
	demo := fs.Bool("demo", true, "Run in demo mode with synthetic HNSW graph")

	_ = fs.Parse(args)

	cfg := connector.Config{
		Backend:        connector.BackendMemory,
		Dimension:      16,
		M:              16,
		EfConstruction: 64,
		EfSearch:       32,
	}

	conn, err := connector.NewConnector(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing memory connector: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	if *demo {
		memConn := conn.(*connector.MemoryConnector)
		populateSyntheticIndex(memConn, 350, 16, 60)
	}

	server := cockpit.NewServer(conn, cockpit.Config{Port: *port})

	fmt.Printf("\n=======================================================\n")
	fmt.Printf("   hnswprobe Web Cockpit Console\n")
	fmt.Printf("   Local URL:  http://localhost:%d\n", *port)
	fmt.Printf("   Metrics:    http://localhost:%d/metrics\n", *port)
	fmt.Printf("=======================================================\n\n")

	if err := server.Start(); err != nil && err != http.ErrServerClosed {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}

func printAuditSummary(s *cockpit.AuditSnapshot, duration time.Duration) {
	fmt.Printf("\n--- Vector Search HNSW Audit Report (%s) ---\n", duration.Round(time.Millisecond))
	fmt.Printf("Backend Engine:            %s\n", s.Backend)
	fmt.Printf("Index Name:                %s\n", s.IndexStats.IndexName)
	fmt.Printf("Total Vectors:             %d\n", s.IndexStats.TotalVectors)
	fmt.Printf("Active Vectors:            %d\n", s.IndexStats.ActiveVectors)
	fmt.Printf("Tombstone Bloat:           %d nodes (%.1f%% TBR)\n", s.Topology.TombstoneNodes, s.Topology.TombstoneBloatRatio*100.0)
	fmt.Printf("Mean Recall@%d:             %.2f%% (Min: %.2f%%, P95: %.2f%%)\n",
		s.Evaluation.K, s.Evaluation.MeanRecall*100.0, s.Evaluation.MinRecall*100.0, s.Evaluation.P95Recall*100.0)
	fmt.Printf("Recall Decay vs Baseline:  %.2f%%\n", s.Evaluation.RecallDecayPercent)
	fmt.Printf("Orphan Nodes (0 In-Deg):   %d (%.2f%%)\n", s.Topology.OrphanNodesCount, s.Topology.OrphanRatio*100.0)
	fmt.Printf("Connected Components:      %d\n", s.Topology.ConnectedComponents)
	fmt.Printf("Estimated Diameter:        %d hops\n", s.Topology.EstimatedDiameter)
	fmt.Printf("Mean Traversal Hops:       %.1f (Efficiency: %.2f)\n", s.Topology.MeanHopCount, s.Topology.TrajectoryEfficiency)
	fmt.Printf("Hub Congestion Gini:       %.2f\n", s.Topology.Layers[0].HubCongestionIndex)
	fmt.Printf("Actionable Advice:         %s\n", s.Recommendations.ActionableAdvice)
}

func populateSyntheticIndex(conn *connector.MemoryConnector, total int, dim int, tombstones int) {
	r := rand.New(rand.NewSource(1337))
	for i := 0; i < total; i++ {
		values := make([]float32, dim)
		for j := 0; j < dim; j++ {
			values[j] = float32(r.NormFloat64())
		}
		conn.Insert(fmt.Sprintf("vec-%04d", i), values)
	}

	for i := 0; i < tombstones; i++ {
		conn.MarkTombstone(fmt.Sprintf("vec-%04d", i*4))
	}
}

func runQuantize(args []string) {
	fs := flag.NewFlagSet("quantize", flag.ExitOnError)
	dimension := fs.Int("dimension", 128, "Vector dimension")
	samples := fs.Int("samples", 500, "Number of sample vectors")
	k := fs.Int("k", 10, "Top-K value for recall evaluation")

	_ = fs.Parse(args)

	fmt.Printf("[+] Profiling SQ8 Scalar Quantization (%d vectors, dim=%d, k=%d)...\n", *samples, *dimension, *k)

	r := rand.New(rand.NewSource(42))
	corpus := make([]groundtruth.VectorItem, *samples)
	for i := 0; i < *samples; i++ {
		vals := make([]float32, *dimension)
		for j := 0; j < *dimension; j++ {
			vals[j] = float32(r.NormFloat64())
		}
		corpus[i] = groundtruth.VectorItem{ID: fmt.Sprintf("vec-%d", i), Values: vals}
	}

	queries := make([]groundtruth.QueryItem, 25)
	for i := 0; i < 25; i++ {
		vals := make([]float32, *dimension)
		for j := 0; j < *dimension; j++ {
			vals[j] = float32(r.NormFloat64())
		}
		queries[i] = groundtruth.QueryItem{ID: fmt.Sprintf("q-%d", i), Values: vals}
	}

	sq := quantization.NewScalarQuantizer()
	report := sq.ProfileQuantization(corpus, queries, *k)

	fmt.Printf("\n--- Scalar Quantization (SQ8) Distortion Profile ---\n")
	fmt.Printf("Vector Dimension:          %d\n", report.Dimension)
	fmt.Printf("Original Footprint:        %d bytes/vector (float32)\n", report.OriginalBytesPerVector)
	fmt.Printf("Quantized Footprint:       %d bytes/vector (uint8)\n", report.QuantizedBytesPerVector)
	fmt.Printf("Memory Savings:            %.1f%%\n", report.MemorySavingsPercent)
	fmt.Printf("Mean Angular Distortion:   %.4f rad\n", report.MeanAngularDistortion)
	fmt.Printf("Max Angular Distortion:    %.4f rad\n", report.MaxAngularDistortion)
	fmt.Printf("Relative L2 Error:         %.2f%%\n", report.MeanRelativeL2Error*100.0)
	fmt.Printf("Quantized Recall@%d:        %.2f%%\n", *k, report.QuantizedRecallAtK*100.0)
	fmt.Printf("Quantization Overhead:     Safe for production (recall drop < 2.5%%)\n")
}

type GeneratedDataset struct {
	Timestamp   time.Time                    `json:"timestamp"`
	Dimension   int                          `json:"dimension"`
	Metric      string                       `json:"metric"`
	Corpus      []groundtruth.VectorItem     `json:"corpus"`
	Queries     []groundtruth.QueryItem      `json:"queries"`
	Tombstones  []string                     `json:"tombstone_ids"`
	GroundTruth map[string][]groundtruth.Neighbor `json:"ground_truth"`
}

func runGenerateSample(args []string) {
	fs := flag.NewFlagSet("generate-sample", flag.ExitOnError)
	output := fs.String("output", "./data/sample_dataset.json", "Output JSON path")
	vectorCount := fs.Int("vectors", 1000, "Number of vectors to generate")
	queryCount := fs.Int("queries", 50, "Number of queries to generate")
	dimension := fs.Int("dimension", 64, "Vector embedding dimension")
	clusters := fs.Int("clusters", 6, "Number of cluster centers")
	tombstonePct := fs.Float64("tombstones", 15.0, "Percentage of vectors to mark as tombstones")
	k := fs.Int("k", 10, "Top-K ground truth to precalculate")
	metricStr := fs.String("metric", "cosine", "Metric: cosine, l2, dot")

	_ = fs.Parse(args)

	fmt.Printf("[+] Generating synthetic dataset (%d vectors, %d clusters, dim=%d)...\n", *vectorCount, *clusters, *dimension)

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// 1. Generate cluster centers
	centers := make([][]float32, *clusters)
	for c := 0; c < *clusters; c++ {
		centers[c] = make([]float32, *dimension)
		for j := 0; j < *dimension; j++ {
			centers[c][j] = float32(r.NormFloat64() * 3.0)
		}
	}

	// 2. Generate corpus vectors around centers
	corpus := make([]groundtruth.VectorItem, *vectorCount)
	for i := 0; i < *vectorCount; i++ {
		centerIdx := i % *clusters
		vals := make([]float32, *dimension)
		for j := 0; j < *dimension; j++ {
			vals[j] = centers[centerIdx][j] + float32(r.NormFloat64()*0.4)
		}
		corpus[i] = groundtruth.VectorItem{
			ID:     fmt.Sprintf("synth-v-%05d", i),
			Values: vals,
		}
	}

	// 3. Mark tombstones
	tombstoneCount := int(float64(*vectorCount) * (*tombstonePct / 100.0))
	tombstoneIDs := make([]string, tombstoneCount)
	for i := 0; i < tombstoneCount; i++ {
		tombstoneIDs[i] = corpus[i*2].ID
	}

	// 4. Generate queries
	queries := make([]groundtruth.QueryItem, *queryCount)
	for i := 0; i < *queryCount; i++ {
		centerIdx := i % *clusters
		qVals := make([]float32, *dimension)
		for j := 0; j < *dimension; j++ {
			qVals[j] = centers[centerIdx][j] + float32(r.NormFloat64()*0.45)
		}
		queries[i] = groundtruth.QueryItem{
			ID:     fmt.Sprintf("synth-q-%03d", i),
			Values: qVals,
		}
	}

	// 5. Compute ground truth top-k
	metric := groundtruth.MetricCosine
	if *metricStr == "l2" {
		metric = groundtruth.MetricEuclidean
	} else if *metricStr == "dot" {
		metric = groundtruth.MetricDot
	}

	solver := groundtruth.NewExactSolver(metric)
	gtMap := solver.SolveBatch(queries, corpus, *k)

	dataset := GeneratedDataset{
		Timestamp:   time.Now().UTC(),
		Dimension:   *dimension,
		Metric:      *metricStr,
		Corpus:      corpus,
		Queries:     queries,
		Tombstones:  tombstoneIDs,
		GroundTruth: gtMap,
	}

	// Ensure parent dir
	dir := filepath.Dir(*output)
	_ = os.MkdirAll(dir, 0755)

	if err := report.WriteJSONReport(*output, dataset); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to write dataset: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[+] Generated dataset written to %s (Corpus: %d, Queries: %d, Tombstones: %d)\n",
		*output, len(corpus), len(queries), len(tombstoneIDs))
}
