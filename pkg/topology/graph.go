package topology

import (
	"math"
	"sort"
)

// NodeHealth summarizes topological health metrics for an individual node.
type NodeHealth struct {
	ID         string `json:"id"`
	Layer      int    `json:"layer"`
	InDegree   int    `json:"in_degree"`
	OutDegree  int    `json:"out_degree"`
	IsOrphan   bool   `json:"is_orphan"`
	IsTomb     bool   `json:"is_tomb"`
	IsHub      bool   `json:"is_hub"`
}

// LayerTopology holds connectivity information for one level in the HNSW hierarchy.
type LayerTopology struct {
	Layer              int            `json:"layer"`
	NodeCount          int            `json:"node_count"`
	EdgeCount          int            `json:"edge_count"`
	MeanOutDegree      float64        `json:"mean_out_degree"`
	MaxOutDegree       int            `json:"max_out_degree"`
	MinOutDegree       int            `json:"min_out_degree"`
	DegreeStdDev       float64        `json:"degree_std_dev"`
	HubCongestionIndex float64        `json:"hub_congestion_index"` // Gini coefficient of out-degrees
	DegreeHistogram    map[int]int    `json:"degree_histogram"`    // degree bucket -> count
	Nodes              map[string]NodeHealth `json:"nodes,omitempty"`
}

// GraphReport contains the comprehensive topology audit results.
type GraphReport struct {
	MaxLayer            int                     `json:"max_layer"`
	TotalNodes          int                     `json:"total_nodes"`
	TotalEdges          int                     `json:"total_edges"`
	ActiveNodes         int                     `json:"active_nodes"`
	TombstoneNodes      int                     `json:"tombstone_nodes"`
	TombstoneBloatRatio float64                 `json:"tombstone_bloat_ratio"`
	OrphanNodesCount    int                     `json:"orphan_nodes_count"`
	OrphanRatio         float64                 `json:"orphan_ratio"`
	ConnectedComponents int                     `json:"connected_components"`
	EstimatedDiameter   int                     `json:"estimated_diameter"`
	Layers              map[int]*LayerTopology  `json:"layers"`
	MeanHopCount        float64                 `json:"mean_hop_count"`
	TrajectoryEfficiency float64                `json:"trajectory_efficiency"`
}

// ComputeDegreeGini calculates the Gini coefficient of degree distribution (0.0 = uniform, 1.0 = maximal hub concentration).
func ComputeDegreeGini(degrees []int) float64 {
	n := len(degrees)
	if n <= 1 {
		return 0
	}

	sorted := make([]int, n)
	copy(sorted, degrees)
	sort.Ints(sorted)

	var sum, weightedSum float64
	for i, deg := range sorted {
		val := float64(deg)
		sum += val
		weightedSum += float64(i+1) * val
	}

	if sum == 0 {
		return 0
	}

	// G = (2 * sum_{i=1}^n i*y_i) / (n * sum y_i) - (n + 1)/n
	gini := (2.0*weightedSum)/(float64(n)*sum) - float64(n+1)/float64(n)
	if gini < 0 {
		return 0
	}
	if gini > 1 {
		return 1
	}
	return gini
}

// ComputeDegreeStats analyzes the mean, variance, and degree histogram.
func ComputeDegreeStats(degrees []int) (mean float64, stdDev float64, minDeg int, maxDeg int, hist map[int]int) {
	hist = make(map[int]int)
	if len(degrees) == 0 {
		return 0, 0, 0, 0, hist
	}

	var sum float64
	minDeg = degrees[0]
	maxDeg = degrees[0]

	for _, d := range degrees {
		sum += float64(d)
		if d < minDeg {
			minDeg = d
		}
		if d > maxDeg {
			maxDeg = d
		}
		// Bucket degrees by groups of 4: 0-3, 4-7, 8-11, etc.
		bucket := (d / 4) * 4
		hist[bucket]++
	}

	mean = sum / float64(len(degrees))

	var varSum float64
	for _, d := range degrees {
		diff := float64(d) - mean
		varSum += diff * diff
	}
	stdDev = math.Sqrt(varSum / float64(len(degrees)))

	return mean, stdDev, minDeg, maxDeg, hist
}
