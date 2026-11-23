package optimizer

import (
	"fmt"

	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
	"github.com/alexandrmotologa/hnswprobe/pkg/topology"
)

// Recommendation contains tuned hyperparameters and actionable advice.
type Recommendation struct {
	TargetRecall              float64 `json:"target_recall"`
	CurrentRecall             float64 `json:"current_recall"`
	OptimalEfSearch           int     `json:"optimal_ef_search"`
	RecommendedM              int     `json:"recommended_m"`
	RecommendedEfConstruction int     `json:"recommended_ef_construction"`
	RequiresCompaction        bool    `json:"requires_compaction"`
	RequiresReindex           bool    `json:"requires_reindex"`
	ExpectedRecallGain        float64 `json:"expected_recall_gain"`
	ActionableAdvice          string  `json:"actionable_advice"`
}

// GenerateRecommendations analyzes evaluation results, topology, and Pareto curve to generate advice.
func GenerateRecommendations(
	stats *connector.IndexStats,
	evalReport groundtruth.EvaluationReport,
	topoReport topology.GraphReport,
	frontier *ParetoFrontier,
	targetRecall float64,
) Recommendation {
	if targetRecall <= 0 {
		targetRecall = 0.95
	}

	currentRecall := evalReport.MeanRecall
	optimalEf := stats.EfSearch
	if optimalEf <= 0 {
		optimalEf = 32
	}

	// 1. Find optimal efSearch from Pareto frontier
	if frontier != nil && len(frontier.FrontierPoints) > 0 {
		foundTarget := false
		for _, p := range frontier.FrontierPoints {
			if p.MeanRecall >= targetRecall {
				optimalEf = p.EfSearch
				foundTarget = true
				break
			}
		}

		if !foundTarget {
			// Pick highest recall point or knee point
			if frontier.KneePoint != nil {
				optimalEf = frontier.KneePoint.EfSearch
			} else {
				optimalEf = frontier.FrontierPoints[len(frontier.FrontierPoints)-1].EfSearch
			}
		}
	}

	// 2. Determine recommended M and efConstruction based on vector dimension and corpus size
	recM := stats.M
	if recM <= 0 {
		recM = 16
	}

	dim := stats.Dimension
	total := stats.TotalVectors

	// Scaling heuristics
	if dim >= 768 || total > 100000 {
		if recM < 32 {
			recM = 32
		}
	}
	if dim >= 1536 || total > 1000000 {
		if recM < 64 {
			recM = 64
		}
	}

	recEfConst := 2 * recM
	if recEfConst < 128 {
		recEfConst = 128
	}
	if recEfConst < stats.EfConstruction {
		recEfConst = stats.EfConstruction
	}

	// 3. Compaction and re-index triggers
	tbr := topoReport.TombstoneBloatRatio
	if tbr == 0 && stats.TombstoneRatio > 0 {
		tbr = stats.TombstoneRatio
	}

	requiresCompaction := tbr >= 0.15 || topoReport.OrphanRatio > 0.02
	requiresReindex := currentRecall < targetRecall-0.10 && tbr < 0.05 // Severe recall loss without tombstones suggests structural degeneration

	gain := 0.0
	if currentRecall < targetRecall {
		gain = (targetRecall - currentRecall) * 100.0
	}

	advice := formatAdvice(currentRecall, targetRecall, tbr, optimalEf, recM, recEfConst, requiresCompaction, requiresReindex)

	return Recommendation{
		TargetRecall:              targetRecall,
		CurrentRecall:             currentRecall,
		OptimalEfSearch:           optimalEf,
		RecommendedM:              recM,
		RecommendedEfConstruction: recEfConst,
		RequiresCompaction:        requiresCompaction,
		RequiresReindex:           requiresReindex,
		ExpectedRecallGain:        gain,
		ActionableAdvice:          advice,
	}
}

func formatAdvice(current, target, tbr float64, ef, m, efConst int, compact, reindex bool) string {
	if compact {
		return fmt.Sprintf("Tombstone bloat ratio is %.1f%% with current recall at %.1f%%. Execute index compaction to eliminate dead vertices, then set efSearch to %d for target recall of %.1f%%.",
			tbr*100.0, current*100.0, ef, target*100.0)
	}

	if reindex {
		return fmt.Sprintf("Graph connectivity has decayed (recall %.1f%% vs target %.1f%%). Re-index corpus with M=%d and efConstruction=%d, and set search-time efSearch=%d.",
			current*100.0, target*100.0, m, efConst, ef)
	}

	if current < target {
		return fmt.Sprintf("Recall is slightly below target (%.1f%% vs %.1f%%). Increase efSearch from current configuration to %d to meet target recall.",
			current*100.0, target*100.0, ef)
	}

	return fmt.Sprintf("Index health is optimal (recall %.1f%% >= target %.1f%%). Maintain efSearch=%d.",
		current*100.0, target*100.0, ef)
}
