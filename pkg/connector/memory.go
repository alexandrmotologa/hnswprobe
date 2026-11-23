package connector

import (
	"container/heap"
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
)

// MemoryNode represents an internal vector node in the memory HNSW graph.
type MemoryNode struct {
	ID        string
	Values    []float32
	Level     int
	Neighbors map[int][]string // level -> list of neighbor IDs
	IsTomb    bool
}

// MemoryConnector implements a pure Go in-memory HNSW index for testing and benchmarking.
type MemoryConnector struct {
	mu             sync.RWMutex
	cfg            Config
	nodes          map[string]*MemoryNode
	entryPointID   string
	maxLevel       int
	m              int
	m0             int
	efConstruction int
	efSearch       int
	mL             float64
	metric         groundtruth.Metric
	tombstoneCount int64
	lastCompaction time.Time
}

// NewMemoryConnector creates an in-memory HNSW index connector.
func NewMemoryConnector(cfg Config) (*MemoryConnector, error) {
	m := cfg.M
	if m <= 0 {
		m = 16
	}
	efConst := cfg.EfConstruction
	if efConst <= 0 {
		efConst = 64
	}
	efSearch := cfg.EfSearch
	if efSearch <= 0 {
		efSearch = 32
	}

	metric := groundtruth.MetricCosine
	if cfg.Metric == "l2" {
		metric = groundtruth.MetricEuclidean
	} else if cfg.Metric == "dot" {
		metric = groundtruth.MetricDot
	}

	conn := &MemoryConnector{
		cfg:            cfg,
		nodes:          make(map[string]*MemoryNode),
		m:              m,
		m0:             2 * m,
		efConstruction: efConst,
		efSearch:       efSearch,
		mL:             1.0 / math.Log(float64(m)),
		metric:         metric,
		lastCompaction: time.Now(),
	}

	return conn, nil
}

func (c *MemoryConnector) Ping(ctx context.Context) error {
	return nil
}

func (c *MemoryConnector) Close() error {
	return nil
}

func (c *MemoryConnector) GetStats(ctx context.Context) (*IndexStats, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := int64(len(c.nodes))
	active := total - c.tombstoneCount
	ratio := 0.0
	if total > 0 {
		ratio = float64(c.tombstoneCount) / float64(total)
	}

	// Approximate memory: 4 bytes per float32 in vector + overhead
	dim := c.cfg.Dimension
	if dim == 0 && len(c.nodes) > 0 {
		for _, n := range c.nodes {
			dim = len(n.Values)
			break
		}
	}
	memBytes := total * int64(dim*4+128)

	return &IndexStats{
		Backend:          string(BackendMemory),
		IndexName:        "in_memory_hnsw_reference",
		Dimension:        dim,
		TotalVectors:     total,
		ActiveVectors:    active,
		TombstoneVectors: c.tombstoneCount,
		TombstoneRatio:   ratio,
		MemorySizeBytes:  memBytes,
		DiskSizeBytes:    memBytes,
		M:                c.m,
		EfConstruction:   c.efConstruction,
		EfSearch:         c.efSearch,
		Metric:           string(c.metric),
		LastCompactionAt: c.lastCompaction,
	}, nil
}

func (c *MemoryConnector) computeDist(a, b []float32) float32 {
	switch c.metric {
	case groundtruth.MetricDot:
		return -groundtruth.DotProduct(a, b)
	case groundtruth.MetricEuclidean:
		return groundtruth.SquaredL2Distance(a, b)
	default:
		return groundtruth.CosineDistance(a, b)
	}
}

func (c *MemoryConnector) randomLevel() int {
	r := rand.Float64()
	if r == 0 {
		r = 1e-7
	}
	lvl := int(-math.Log(r) * c.mL)
	return lvl
}

// Insert adds a vector to the HNSW graph.
func (c *MemoryConnector) Insert(id string, values []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	nodeLevel := c.randomLevel()
	newNode := &MemoryNode{
		ID:        id,
		Values:    values,
		Level:     nodeLevel,
		Neighbors: make(map[int][]string),
		IsTomb:    false,
	}
	for l := 0; l <= nodeLevel; l++ {
		newNode.Neighbors[l] = make([]string, 0)
	}

	if len(c.nodes) == 0 {
		c.entryPointID = id
		c.maxLevel = nodeLevel
		c.nodes[id] = newNode
		return
	}

	currObj := c.entryPointID
	currDist := c.computeDist(values, c.nodes[currObj].Values)

	// Phase 1: greedy zoom from maxLevel down to nodeLevel + 1
	for l := c.maxLevel; l > nodeLevel; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range c.nodes[currObj].Neighbors[l] {
				neighborNode := c.nodes[neighborID]
				d := c.computeDist(values, neighborNode.Values)
				if d < currDist {
					currDist = d
					currObj = neighborID
					changed = true
				}
			}
		}
	}

	c.nodes[id] = newNode

	// Phase 2: from min(maxLevel, nodeLevel) down to 0, search layer and connect
	startLevel := c.maxLevel
	if nodeLevel < startLevel {
		startLevel = nodeLevel
	}

	for l := startLevel; l >= 0; l-- {
		w := c.searchLayer(values, []string{currObj}, c.efConstruction, l)
		maxM := c.m
		if l == 0 {
			maxM = c.m0
		}

		selected := c.selectNeighbors(values, w, maxM)
		newNode.Neighbors[l] = selected

		// Add bidirectional edges
		for _, neighborID := range selected {
			neighborNode := c.nodes[neighborID]
			neighborNode.Neighbors[l] = append(neighborNode.Neighbors[l], id)
			if len(neighborNode.Neighbors[l]) > maxM {
				// Prune neighbor edges
				neighborNode.Neighbors[l] = c.selectNeighborsFromIDs(neighborNode.Values, neighborNode.Neighbors[l], maxM)
			}
		}

		if len(w) > 0 {
			currObj = w[0].ID
		}
	}

	if nodeLevel > c.maxLevel {
		c.maxLevel = nodeLevel
		c.entryPointID = id
	}
}

// MarkTombstone simulates soft-deleting a vector without restructuring edges.
func (c *MemoryConnector) MarkTombstone(id string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	node, exists := c.nodes[id]
	if !exists || node.IsTomb {
		return false
	}

	node.IsTomb = true
	c.tombstoneCount++
	return true
}

func (c *MemoryConnector) searchLayer(query []float32, entryPoints []string, ef int, level int) []groundtruth.Neighbor {
	visited := make(map[string]bool)
	candidates := &minDistHeap{}
	heap.Init(candidates)

	w := &groundtruth.NeighborHeap{}
	heap.Init(w)

	for _, ep := range entryPoints {
		visited[ep] = true
		d := c.computeDist(query, c.nodes[ep].Values)
		heap.Push(candidates, groundtruth.Neighbor{ID: ep, Distance: d})
		heap.Push(w, groundtruth.Neighbor{ID: ep, Distance: d})
	}

	for candidates.Len() > 0 {
		curr := heap.Pop(candidates).(groundtruth.Neighbor)
		furthestInW := (*w)[0]

		if curr.Distance > furthestInW.Distance {
			break
		}

		currNode := c.nodes[curr.ID]
		for _, neighborID := range currNode.Neighbors[level] {
			if !visited[neighborID] {
				visited[neighborID] = true
				furthestInW = (*w)[0]
				d := c.computeDist(query, c.nodes[neighborID].Values)

				if d < furthestInW.Distance || w.Len() < ef {
					heap.Push(candidates, groundtruth.Neighbor{ID: neighborID, Distance: d})
					if w.Len() < ef {
						heap.Push(w, groundtruth.Neighbor{ID: neighborID, Distance: d})
					} else {
						(*w)[0] = groundtruth.Neighbor{ID: neighborID, Distance: d}
						heap.Fix(w, 0)
					}
				}
			}
		}
	}

	results := make([]groundtruth.Neighbor, w.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(w).(groundtruth.Neighbor)
	}
	return results
}

func (c *MemoryConnector) selectNeighbors(query []float32, candidates []groundtruth.Neighbor, m int) []string {
	if len(candidates) <= m {
		res := make([]string, len(candidates))
		for i, c := range candidates {
			res[i] = c.ID
		}
		return res
	}

	res := make([]string, 0, m)
	for i := 0; i < m && i < len(candidates); i++ {
		res = append(res, candidates[i].ID)
	}
	return res
}

func (c *MemoryConnector) selectNeighborsFromIDs(origin []float32, neighborIDs []string, m int) []string {
	if len(neighborIDs) <= m {
		return neighborIDs
	}

	type item struct {
		id   string
		dist float32
	}
	items := make([]item, len(neighborIDs))
	for i, nid := range neighborIDs {
		items[i] = item{id: nid, dist: c.computeDist(origin, c.nodes[nid].Values)}
	}

	// Simple selection of m smallest
	for i := 0; i < len(items); i++ {
		for j := i + 1; j < len(items); j++ {
			if items[j].dist < items[i].dist {
				items[i], items[j] = items[j], items[i]
			}
		}
	}

	res := make([]string, m)
	for i := 0; i < m; i++ {
		res[i] = items[i].id
	}
	return res
}

type minDistHeap []groundtruth.Neighbor

func (h minDistHeap) Len() int           { return len(h) }
func (h minDistHeap) Less(i, j int) bool { return h[i].Distance < h[j].Distance } // min-heap
func (h minDistHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minDistHeap) Push(x any)        { *h = append(*h, x.(groundtruth.Neighbor)) }
func (h *minDistHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func (c *MemoryConnector) SampleVectors(ctx context.Context, count int) ([]VectorRecord, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	records := make([]VectorRecord, 0, count)
	for _, n := range c.nodes {
		records = append(records, VectorRecord{
			ID:        n.ID,
			Values:    n.Values,
			IsDeleted: n.IsTomb,
		})
		if len(records) >= count {
			break
		}
	}
	return records, nil
}

func (c *MemoryConnector) SampleQueries(ctx context.Context, count int) ([]QueryRequest, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	queries := make([]QueryRequest, 0, count)
	// Create sample queries by perturbing random active vectors slightly
	for _, n := range c.nodes {
		if n.IsTomb {
			continue
		}
		qVals := make([]float32, len(n.Values))
		copy(qVals, n.Values)
		// Add small jitter
		for j := range qVals {
			qVals[j] += float32((rand.Float64() - 0.5) * 0.05)
		}

		queries = append(queries, QueryRequest{
			ID:       fmt.Sprintf("q-%s", n.ID),
			Values:   qVals,
			TopK:     10,
			EfSearch: c.efSearch,
		})
		if len(queries) >= count {
			break
		}
	}
	return queries, nil
}

func (c *MemoryConnector) SearchANN(ctx context.Context, req QueryRequest) ([]SearchResult, time.Duration, error) {
	start := time.Now()
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.nodes) == 0 {
		return nil, time.Since(start), nil
	}

	ef := req.EfSearch
	if ef <= 0 {
		ef = c.efSearch
	}

	currObj := c.entryPointID
	currDist := c.computeDist(req.Values, c.nodes[currObj].Values)

	// Traverse from top level down to 1
	for l := c.maxLevel; l > 0; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range c.nodes[currObj].Neighbors[l] {
				neighborNode := c.nodes[neighborID]
				d := c.computeDist(req.Values, neighborNode.Values)
				if d < currDist {
					currDist = d
					currObj = neighborID
					changed = true
				}
			}
		}
	}

	// Layer 0 search
	neighbors := c.searchLayer(req.Values, []string{currObj}, ef, 0)
	results := make([]SearchResult, 0, req.TopK)

	for _, n := range neighbors {
		node := c.nodes[n.ID]
		// Skip tombstones in final return list (just as production vector DBs do)
		if node.IsTomb {
			continue
		}
		results = append(results, SearchResult{
			ID:       n.ID,
			Distance: n.Distance,
			Score:    1.0 / (1.0 + n.Distance),
		})
		if len(results) >= req.TopK {
			break
		}
	}

	return results, time.Since(start), nil
}

// TriggerCompaction removes tombstones and rebuilds the HNSW graph completely.
func (c *MemoryConnector) TriggerCompaction(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	activeNodes := make([]VectorRecord, 0)
	for _, n := range c.nodes {
		if !n.IsTomb {
			activeNodes = append(activeNodes, VectorRecord{
				ID:     n.ID,
				Values: n.Values,
			})
		}
	}

	// Reinitialize graph state
	c.nodes = make(map[string]*MemoryNode)
	c.entryPointID = ""
	c.maxLevel = 0
	c.tombstoneCount = 0
	c.lastCompaction = time.Now()

	c.mu.Unlock()
	for _, rec := range activeNodes {
		c.Insert(rec.ID, rec.Values)
	}
	c.mu.Lock()

	return nil
}

func (c *MemoryConnector) ExportTopology(ctx context.Context, maxNodes int) (*GraphTopologyExport, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	export := &GraphTopologyExport{
		MaxLayer:   c.maxLevel,
		EntryPoint: c.entryPointID,
		Layers:     make(map[int]*GraphLayerExport),
		TotalNodes: len(c.nodes),
	}

	totalEdges := 0
	for l := 0; l <= c.maxLevel; l++ {
		layerExport := &GraphLayerExport{
			Layer: l,
			Nodes: make(map[string]GraphNode),
		}

		count := 0
		for id, node := range c.nodes {
			neighbors, ok := node.Neighbors[l]
			if !ok {
				continue
			}

			// Copy neighbors
			nbrs := make([]string, len(neighbors))
			copy(nbrs, neighbors)
			totalEdges += len(nbrs)

			layerExport.Nodes[id] = GraphNode{
				ID:        id,
				Layer:     l,
				Neighbors: nbrs,
				IsTomb:    node.IsTomb,
			}

			count++
			if maxNodes > 0 && count >= maxNodes {
				break
			}
		}
		export.Layers[l] = layerExport
	}

	export.TotalEdges = totalEdges
	return export, nil
}
