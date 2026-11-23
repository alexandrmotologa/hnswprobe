package topology

// TombstoneStats details memory bloat and traversal penalties caused by soft-deleted nodes.
type TombstoneStats struct {
	TotalVertices         int     `json:"total_vertices"`
	TombstoneVertices     int     `json:"tombstone_vertices"`
	ActiveVertices        int     `json:"active_vertices"`
	TombstoneBloatRatio   float64 `json:"tombstone_bloat_ratio"`   // TBR = dead / total
	WastedMemoryBytes     int64   `json:"wasted_memory_bytes"`
	TraversalFriction     float64 `json:"traversal_friction"`      // Estimated slowdown factor: 1.0 / (1.0 - TBR)
	CompactionRecommended bool    `json:"compaction_recommended"` // TBR > 0.15
	SeverityLevel         string  `json:"severity_level"`         // "OK", "WARNING", "CRITICAL"
}

// AnalyzeTombstones calculates tombstone density and friction factors.
func AnalyzeTombstones(totalNodes int, tombstoneNodes int, bytesPerVector int64) TombstoneStats {
	if totalNodes <= 0 {
		return TombstoneStats{SeverityLevel: "OK"}
	}

	active := totalNodes - tombstoneNodes
	if active < 0 {
		active = 0
	}

	tbr := float64(tombstoneNodes) / float64(totalNodes)
	wasted := int64(tombstoneNodes) * bytesPerVector

	friction := 1.0
	if tbr < 0.95 && tbr > 0 {
		friction = 1.0 / (1.0 - tbr)
	}

	severity := "OK"
	recommended := false
	if tbr >= 0.25 {
		severity = "CRITICAL"
		recommended = true
	} else if tbr >= 0.15 {
		severity = "WARNING"
		recommended = true
	}

	return TombstoneStats{
		TotalVertices:         totalNodes,
		TombstoneVertices:     tombstoneNodes,
		ActiveVertices:        active,
		TombstoneBloatRatio:   tbr,
		WastedMemoryBytes:     wasted,
		TraversalFriction:     friction,
		CompactionRecommended: recommended,
		SeverityLevel:         severity,
	}
}
