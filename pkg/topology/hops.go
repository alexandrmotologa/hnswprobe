package topology

import (
	"math"

	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
)

// HopMetrics details navigational hop counts and trajectory efficiency.
type HopMetrics struct {
	TotalQueries         int     `json:"total_queries"`
	MeanHops             float64 `json:"mean_hops"`
	MinHops              int     `json:"min_hops"`
	MaxHops              int     `json:"max_hops"`
	MeanTrajectoryEff    float64 `json:"mean_trajectory_efficiency"` // ||q - v_entry|| / sum(||v_j - v_{j-1}||)
	LocalMinimaTrapCount int     `json:"local_minima_trap_count"`
	TrapRate             float64 `json:"trap_rate"`
}

// TraversalStep tracks a single step during greedy routing.
type TraversalStep struct {
	NodeID   string
	Level    int
	Distance float32
}

// AnalyzeTraversalHops analyzes greedy traversal paths across sample queries.
func AnalyzeTraversalHops(
	topo *connector.GraphTopologyExport,
	queries []connector.QueryRequest,
	vectorMap map[string][]float32,
	metric groundtruth.Metric,
) HopMetrics {
	if topo == nil || len(queries) == 0 || topo.EntryPoint == "" {
		return HopMetrics{}
	}

	totalQueries := len(queries)
	var sumHops float64
	var sumEff float64
	minHops := 999999
	maxHops := 0
	trapCount := 0

	entryNode, hasEntry := vectorMap[topo.EntryPoint]
	if !hasEntry {
		return HopMetrics{}
	}

	distFunc := func(a, b []float32) float32 {
		switch metric {
		case groundtruth.MetricDot:
			return -groundtruth.DotProduct(a, b)
		case groundtruth.MetricEuclidean:
			return groundtruth.SquaredL2Distance(a, b)
		default:
			return groundtruth.CosineDistance(a, b)
		}
	}

	for _, q := range queries {
		currID := topo.EntryPoint
		currDist := distFunc(q.Values, entryNode)
		hops := 0
		var pathLength float64
		lastVec := entryNode

		// Greedy descent through layers
		for l := topo.MaxLayer; l >= 0; l-- {
			layer, hasLayer := topo.Layers[l]
			if !hasLayer {
				continue
			}

			changed := true
			for changed {
				changed = false
				node, hasNode := layer.Nodes[currID]
				if !hasNode {
					break
				}

				for _, nbrID := range node.Neighbors {
					nbrVec, hasVec := vectorMap[nbrID]
					if !hasVec {
						continue
					}
					d := distFunc(q.Values, nbrVec)
					if d < currDist {
						hops++
						stepDist := math.Sqrt(float64(groundtruth.SquaredL2Distance(lastVec, nbrVec)))
						pathLength += stepDist
						lastVec = nbrVec
						currDist = d
						currID = nbrID
						changed = true
					}
				}
			}
		}

		if hops < minHops {
			minHops = hops
		}
		if hops > maxHops {
			maxHops = hops
		}
		sumHops += float64(hops)

		// Direct Euclidean distance from entry point to query
		directDist := math.Sqrt(float64(groundtruth.SquaredL2Distance(entryNode, q.Values)))
		eff := 1.0
		if pathLength > 0 && directDist > 0 {
			eff = directDist / pathLength
			if eff > 1.0 {
				eff = 1.0
			}
		}
		sumEff += eff

		// Check if trapped: if path was short (< 2 hops) and direct distance is large
		if hops <= 1 && directDist > 0.5 {
			trapCount++
		}
	}

	if minHops == 999999 {
		minHops = 0
	}

	return HopMetrics{
		TotalQueries:         totalQueries,
		MeanHops:             sumHops / float64(totalQueries),
		MinHops:              minHops,
		MaxHops:              maxHops,
		MeanTrajectoryEff:    sumEff / float64(totalQueries),
		LocalMinimaTrapCount: trapCount,
		TrapRate:             float64(trapCount) / float64(totalQueries),
	}
}
