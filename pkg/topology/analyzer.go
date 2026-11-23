package topology

import (
	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
)

// AnalyzeTopology produces a comprehensive topological health audit for an HNSW index.
func AnalyzeTopology(
	topo *connector.GraphTopologyExport,
	queries []connector.QueryRequest,
	vectorMap map[string][]float32,
	metric groundtruth.Metric,
	bytesPerVector int64,
) GraphReport {
	if topo == nil {
		return GraphReport{}
	}

	report := GraphReport{
		MaxLayer:   topo.MaxLayer,
		TotalNodes: topo.TotalNodes,
		TotalEdges: topo.TotalEdges,
		Layers:     make(map[int]*LayerTopology),
	}

	tombCount := 0
	activeCount := 0

	// Analyze each layer
	for l := 0; l <= topo.MaxLayer; l++ {
		layerExport, exists := topo.Layers[l]
		if !exists || len(layerExport.Nodes) == 0 {
			continue
		}

		degrees := make([]int, 0, len(layerExport.Nodes))
		layerNodesHealth := make(map[string]NodeHealth, len(layerExport.Nodes))

		for id, node := range layerExport.Nodes {
			deg := len(node.Neighbors)
			degrees = append(degrees, deg)

			if l == 0 {
				if node.IsTomb {
					tombCount++
				} else {
					activeCount++
				}
			}

			layerNodesHealth[id] = NodeHealth{
				ID:        id,
				Layer:     l,
				OutDegree: deg,
				IsTomb:    node.IsTomb,
			}
		}

		meanDeg, stdDev, minDeg, maxDeg, hist := ComputeDegreeStats(degrees)
		gini := ComputeDegreeGini(degrees)

		// Mark hub nodes (degrees > mean + 2*stdDev)
		hubThreshold := int(meanDeg + 2.0*stdDev)
		for id, nh := range layerNodesHealth {
			if nh.OutDegree > hubThreshold && hubThreshold > 0 {
				nh.IsHub = true
				layerNodesHealth[id] = nh
			}
		}

		report.Layers[l] = &LayerTopology{
			Layer:              l,
			NodeCount:          len(layerExport.Nodes),
			EdgeCount:          len(degrees),
			MeanOutDegree:      meanDeg,
			MaxOutDegree:       maxDeg,
			MinOutDegree:       minDeg,
			DegreeStdDev:       stdDev,
			HubCongestionIndex: gini,
			DegreeHistogram:    hist,
			Nodes:              layerNodesHealth,
		}
	}

	report.ActiveNodes = activeCount
	report.TombstoneNodes = tombCount
	if activeCount+tombCount > 0 {
		report.TombstoneBloatRatio = float64(tombCount) / float64(activeCount+tombCount)
	}

	// Layer 0 orphan and component analysis
	if layer0, hasL0 := topo.Layers[0]; hasL0 {
		orphanAnalysis := AnalyzeOrphansAndComponents(layer0)
		report.OrphanNodesCount = orphanAnalysis.OrphanNodesCount
		report.OrphanRatio = orphanAnalysis.OrphanRatio
		report.ConnectedComponents = orphanAnalysis.ConnectedComponents

		// Mark orphan nodes in layer 0 health map
		for _, orphanID := range orphanAnalysis.OrphanNodeIDs {
			if nodeHealth, exists := report.Layers[0].Nodes[orphanID]; exists {
				nodeHealth.IsOrphan = true
				report.Layers[0].Nodes[orphanID] = nodeHealth
			}
		}

		// Estimate diameter
		report.EstimatedDiameter = EstimateGraphDiameter(layer0, 3)
	}

	// Hop metrics and trajectory efficiency
	if len(queries) > 0 && len(vectorMap) > 0 {
		hopMetrics := AnalyzeTraversalHops(topo, queries, vectorMap, metric)
		report.MeanHopCount = hopMetrics.MeanHops
		report.TrajectoryEfficiency = hopMetrics.MeanTrajectoryEff
	}

	return report
}
