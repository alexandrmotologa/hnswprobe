package groundtruth

import (
	"container/heap"
	"runtime"
	"sync"
)

// Neighbor represents an identified vector with its ID and distance from query.
type Neighbor struct {
	ID       string
	Distance float32
}

// NeighborHeap implements a max-heap of Neighbors (top elements have highest distance).
// Retains the K smallest distances when pruned.
type NeighborHeap []Neighbor

func (h NeighborHeap) Len() int           { return len(h) }
func (h NeighborHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance } // max-heap
func (h NeighborHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *NeighborHeap) Push(x any) {
	*h = append(*h, x.(Neighbor))
}

func (h *NeighborHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// VectorItem represents an indexed vector with an identifier.
type VectorItem struct {
	ID     string
	Values []float32
}

// QueryItem represents a search query to be evaluated against the dataset.
type QueryItem struct {
	ID     string
	Values []float32
}

// ExactSolver executes parallel brute-force nearest neighbor calculations.
type ExactSolver struct {
	metric Metric
}

// NewExactSolver initializes an ExactSolver with the target distance metric.
func NewExactSolver(metric Metric) *ExactSolver {
	return &ExactSolver{metric: metric}
}

// ComputeDistance evaluates the distance between two vectors according to the metric.
func (s *ExactSolver) ComputeDistance(a, b []float32) float32 {
	switch s.metric {
	case MetricDot:
		// For dot product, higher is closer; we invert to standard distance where lower is closer
		return -DotProduct(a, b)
	case MetricEuclidean:
		return SquaredL2Distance(a, b)
	case MetricCosine:
		fallthrough
	default:
		return CosineDistance(a, b)
	}
}

// SolveSingle computes the exact top-k nearest neighbors for a single query.
func (s *ExactSolver) SolveSingle(query []float32, corpus []VectorItem, k int) []Neighbor {
	if k <= 0 || len(corpus) == 0 {
		return nil
	}

	h := &NeighborHeap{}
	heap.Init(h)

	for _, item := range corpus {
		dist := s.ComputeDistance(query, item.Values)
		if h.Len() < k {
			heap.Push(h, Neighbor{ID: item.ID, Distance: dist})
		} else if dist < (*h)[0].Distance {
			(*h)[0] = Neighbor{ID: item.ID, Distance: dist}
			heap.Fix(h, 0)
		}
	}

	result := make([]Neighbor, h.Len())
	for i := len(result) - 1; i >= 0; i-- {
		result[i] = heap.Pop(h).(Neighbor)
	}
	return result
}

// SolveBatch computes exact top-k ground truth for multiple queries using worker goroutines.
func (s *ExactSolver) SolveBatch(queries []QueryItem, corpus []VectorItem, k int) map[string][]Neighbor {
	results := make(map[string][]Neighbor, len(queries))
	if len(queries) == 0 || len(corpus) == 0 {
		return results
	}

	numWorkers := runtime.NumCPU()
	if numWorkers > len(queries) {
		numWorkers = len(queries)
	}

	var mu sync.Mutex
	queryChan := make(chan QueryItem, len(queries))
	for _, q := range queries {
		queryChan <- q
	}
	close(queryChan)

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func() {
			defer wg.Done()
			for q := range queryChan {
				neighbors := s.SolveSingle(q.Values, corpus, k)
				mu.Lock()
				results[q.ID] = neighbors
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	return results
}
