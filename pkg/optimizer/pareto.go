package optimizer

import (
	"context"
	"sort"

	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
)

// ParetoPoint represents a single measured configuration point.
type ParetoPoint struct {
	EfSearch         int     `json:"ef_search"`
	MeanRecall       float64 `json:"mean_recall"`
	MinRecall        float64 `json:"min_recall"`
	MeanLatencyMs    float64 `json:"mean_latency_ms"`
	P95LatencyMs     float64 `json:"p95_latency_ms"`
	QueriesPerSecond float64 `json:"queries_per_second"`
	IsOptimal        bool    `json:"is_optimal"` // Belongs to empirical Pareto frontier
}

// ParetoFrontier represents the simulated trade-off curve between search latency and recall.
type ParetoFrontier struct {
	Points         []ParetoPoint `json:"points"`
	FrontierPoints []ParetoPoint `json:"frontier_points"`
	KneePoint      *ParetoPoint  `json:"knee_point,omitempty"`
}

// SweepEfSearch evaluates recall and latency across a range of efSearch candidate values.
func SweepEfSearch(
	ctx context.Context,
	conn connector.Connector,
	queries []connector.QueryRequest,
	exactMap map[string][]groundtruth.Neighbor,
	k int,
	candidates []int,
) (*ParetoFrontier, error) {
	if len(candidates) == 0 {
		candidates = []int{8, 16, 32, 64, 128, 256}
	}
	sort.Ints(candidates)

	points := make([]ParetoPoint, 0, len(candidates))

	for _, ef := range candidates {
		latencies := make([]float64, 0, len(queries))
		approxMap := make(map[string][]groundtruth.Neighbor, len(queries))

		for _, q := range queries {
			req := q
			req.TopK = k
			req.EfSearch = ef

			res, dur, err := conn.SearchANN(ctx, req)
			if err != nil {
				continue
			}

			latMs := float64(dur.Microseconds()) / 1000.0
			latencies = append(latencies, latMs)

			nbrs := make([]groundtruth.Neighbor, len(res))
			for i, r := range res {
				nbrs[i] = groundtruth.Neighbor{ID: r.ID, Distance: r.Distance}
			}
			approxMap[q.ID] = nbrs
		}

		if len(latencies) == 0 {
			continue
		}

		report := groundtruth.EvaluateRecall(exactMap, approxMap, k)

		sort.Float64s(latencies)
		var sumLat float64
		for _, l := range latencies {
			sumLat += l
		}
		meanLat := sumLat / float64(len(latencies))
		p95Lat := percentile(latencies, 0.95)

		qps := 0.0
		if meanLat > 0 {
			qps = 1000.0 / meanLat
		}

		points = append(points, ParetoPoint{
			EfSearch:         ef,
			MeanRecall:       report.MeanRecall,
			MinRecall:        report.MinRecall,
			MeanLatencyMs:    meanLat,
			P95LatencyMs:     p95Lat,
			QueriesPerSecond: qps,
		})
	}

	frontier := computeFrontier(points)
	knee := identifyKneePoint(frontier)

	return &ParetoFrontier{
		Points:         points,
		FrontierPoints: frontier,
		KneePoint:      knee,
	}, nil
}

func computeFrontier(points []ParetoPoint) []ParetoPoint {
	// Sort primarily by latency ascending, then recall descending
	sorted := make([]ParetoPoint, len(points))
	copy(sorted, points)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].MeanLatencyMs == sorted[j].MeanLatencyMs {
			return sorted[i].MeanRecall > sorted[j].MeanRecall
		}
		return sorted[i].MeanLatencyMs < sorted[j].MeanLatencyMs
	})

	frontier := make([]ParetoPoint, 0)
	var maxRecall float64 = -1.0

	for i := range sorted {
		if sorted[i].MeanRecall > maxRecall {
			maxRecall = sorted[i].MeanRecall
			sorted[i].IsOptimal = true
			frontier = append(frontier, sorted[i])
		}
	}

	return frontier
}

// identifyKneePoint finds the point on the frontier with maximum distance to the line connecting ends.
func identifyKneePoint(frontier []ParetoPoint) *ParetoPoint {
	if len(frontier) == 0 {
		return nil
	}
	if len(frontier) <= 2 {
		return &frontier[len(frontier)-1]
	}

	// Line from first to last point
	pFirst := frontier[0]
	pLast := frontier[len(frontier)-1]

	var maxDist float64 = -1.0
	var bestPoint *ParetoPoint

	// Normalized coordinates [0, 1]
	latRange := pLast.MeanLatencyMs - pFirst.MeanLatencyMs
	recRange := pLast.MeanRecall - pFirst.MeanRecall
	if latRange <= 0 {
		latRange = 1.0
	}
	if recRange <= 0 {
		recRange = 1.0
	}

	for i := 1; i < len(frontier)-1; i++ {
		p := frontier[i]
		x := (p.MeanLatencyMs - pFirst.MeanLatencyMs) / latRange
		y := (p.MeanRecall - pFirst.MeanRecall) / recRange

		// Distance from (x, y) to line connecting (0,0) and (1,1): |y - x| / sqrt(2)
		dist := (y - x)
		if dist > maxDist {
			maxDist = dist
			best := p
			bestPoint = &best
		}
	}

	if bestPoint == nil {
		bestPoint = &frontier[len(frontier)-1]
	}
	return bestPoint
}

func percentile(sortedVals []float64, p float64) float64 {
	if len(sortedVals) == 0 {
		return 0
	}
	idx := int(float64(len(sortedVals)-1) * p)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sortedVals) {
		idx = len(sortedVals) - 1
	}
	return sortedVals[idx]
}
