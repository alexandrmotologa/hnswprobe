package groundtruth

import (
	"math"
	"sort"
)

// QueryEvaluation captures recall and accuracy metrics for an individual query.
type QueryEvaluation struct {
	QueryID            string    `json:"query_id"`
	RecallAtK          float64   `json:"recall_at_k"`
	ExactCount         int       `json:"exact_count"`
	MatchedCount       int       `json:"matched_count"`
	MeanDistanceError  float64   `json:"mean_distance_error"`
	ExactNeighbors     []string  `json:"exact_neighbors"`
	ApproxNeighbors    []string  `json:"approx_neighbors"`
}

// EvaluationReport aggregates recall metrics across all evaluated queries.
type EvaluationReport struct {
	K                  int               `json:"k"`
	QueryCount         int               `json:"query_count"`
	MeanRecall         float64           `json:"mean_recall"`
	MinRecall          float64           `json:"min_recall"`
	MaxRecall          float64           `json:"max_recall"`
	P50Recall          float64           `json:"p50_recall"`
	P95Recall          float64           `json:"p95_recall"`
	P99Recall          float64           `json:"p99_recall"`
	MeanDistanceError  float64           `json:"mean_distance_error"`
	RecallDecayPercent float64           `json:"recall_decay_percent"` // Compared to baseline 1.0 (or target)
	QueryEvaluations   []QueryEvaluation `json:"query_evaluations,omitempty"`
}

// EvaluateRecall compares approximate nearest neighbor outputs with ground truth.
func EvaluateRecall(exactMap map[string][]Neighbor, approxMap map[string][]Neighbor, k int) EvaluationReport {
	if k <= 0 {
		k = 10
	}

	report := EvaluationReport{
		K:          k,
		QueryCount: len(exactMap),
	}

	if len(exactMap) == 0 {
		return report
	}

	recalls := make([]float64, 0, len(exactMap))
	var sumRecall, sumDistErr float64
	minRec := 1.0
	maxRec := 0.0

	evaluations := make([]QueryEvaluation, 0, len(exactMap))

	for qID, exactNeighbors := range exactMap {
		approxNeighbors := approxMap[qID]

		// Build fast lookup set for exact top-K IDs
		limitExact := k
		if len(exactNeighbors) < limitExact {
			limitExact = len(exactNeighbors)
		}

		exactSet := make(map[string]float32, limitExact)
		exactIDs := make([]string, 0, limitExact)
		for i := 0; i < limitExact; i++ {
			exactSet[exactNeighbors[i].ID] = exactNeighbors[i].Distance
			exactIDs = append(exactIDs, exactNeighbors[i].ID)
		}

		limitApprox := k
		if len(approxNeighbors) < limitApprox {
			limitApprox = len(approxNeighbors)
		}

		matchCount := 0
		var queryDistErr float64
		approxIDs := make([]string, 0, limitApprox)

		for i := 0; i < limitApprox; i++ {
			aID := approxNeighbors[i].ID
			approxIDs = append(approxIDs, aID)
			if _, ok := exactSet[aID]; ok {
				matchCount++
			}
			// Compute relative distance error compared to corresponding rank in exact
			if i < len(exactNeighbors) {
				exactDist := math.Abs(float64(exactNeighbors[i].Distance))
				approxDist := math.Abs(float64(approxNeighbors[i].Distance))
				if exactDist > 1e-6 {
					queryDistErr += math.Abs(approxDist-exactDist) / exactDist
				}
			}
		}

		rec := 0.0
		if limitExact > 0 {
			rec = float64(matchCount) / float64(limitExact)
		}

		recalls = append(recalls, rec)
		sumRecall += rec

		if rec < minRec {
			minRec = rec
		}
		if rec > maxRec {
			maxRec = rec
		}

		meanQDistErr := 0.0
		if limitApprox > 0 {
			meanQDistErr = queryDistErr / float64(limitApprox)
			sumDistErr += meanQDistErr
		}

		evaluations = append(evaluations, QueryEvaluation{
			QueryID:           qID,
			RecallAtK:         rec,
			ExactCount:        limitExact,
			MatchedCount:      matchCount,
			MeanDistanceError: meanQDistErr,
			ExactNeighbors:    exactIDs,
			ApproxNeighbors:   approxIDs,
		})
	}

	sort.Float64s(recalls)
	n := len(recalls)

	report.MeanRecall = sumRecall / float64(n)
	report.MinRecall = minRec
	report.MaxRecall = maxRec
	report.MeanDistanceError = sumDistErr / float64(n)
	report.RecallDecayPercent = (1.0 - report.MeanRecall) * 100.0

	// Percentiles
	report.P50Recall = percentile(recalls, 0.50)
	report.P95Recall = percentile(recalls, 0.95)
	report.P99Recall = percentile(recalls, 0.99)
	report.QueryEvaluations = evaluations

	return report
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
