package cockpit

import (
	"time"

	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
	"github.com/alexandrmotologa/hnswprobe/pkg/optimizer"
	"github.com/alexandrmotologa/hnswprobe/pkg/topology"
)

// AuditSnapshot holds a complete point-in-time audit report for the web cockpit.
type AuditSnapshot struct {
	Timestamp       time.Time                     `json:"timestamp"`
	Backend         string                        `json:"backend"`
	IndexStats      *connector.IndexStats         `json:"index_stats"`
	Evaluation      groundtruth.EvaluationReport  `json:"evaluation"`
	Topology        topology.GraphReport          `json:"topology"`
	Pareto          *optimizer.ParetoFrontier     `json:"pareto,omitempty"`
	Recommendations optimizer.Recommendation      `json:"recommendations"`
	VisualGraph     *VisualGraphData              `json:"visual_graph,omitempty"`
}

// VisualNode represents a projected 2D/3D coordinate for graph visualization.
type VisualNode struct {
	ID              string  `json:"id"`
	X               float64 `json:"x"`
	Y               float64 `json:"y"`
	Z               float64 `json:"z"`
	Layer           int     `json:"layer"`
	InDegree        int     `json:"in_degree"`
	OutDegree       int     `json:"out_degree"`
	IsTomb          bool    `json:"is_tomb"`
	IsOrphan        bool    `json:"is_orphan"`
	IsHub           bool    `json:"is_hub"`
	IsEntry         bool    `json:"is_entry"`
	AscensionLevels []int   `json:"ascension_levels"` // Layers where this vertex exists (e.g. [0, 1, 2])
}

// VisualEdge represents a directed connection between two nodes in a layer.
type VisualEdge struct {
	Source    string `json:"source"`
	Target    string `json:"target"`
	Layer     int    `json:"layer"`
	IsHighway bool   `json:"is_highway"` // True for high-level navigational bridges (layer >= 1)
}

// VisualGraphData packages projected coordinates and edge lists for the cockpit renderer.
type VisualGraphData struct {
	Nodes      []VisualNode `json:"nodes"`
	Edges      []VisualEdge `json:"edges"`
	MaxLayer   int          `json:"max_layer"`
	EntryPoint string       `json:"entry_point"`
}

// TuneRequest receives user parameters for on-demand efSearch evaluation.
type TuneRequest struct {
	EfSearch int `json:"ef_search"`
}

// TuneResponse returns recall and latency for the requested efSearch value.
type TuneResponse struct {
	EfSearch      int     `json:"ef_search"`
	MeanRecall    float64 `json:"mean_recall"`
	MeanLatencyMs float64 `json:"mean_latency_ms"`
}
