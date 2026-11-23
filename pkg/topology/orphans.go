package topology

import "github.com/alexandrmotologa/hnswprobe/pkg/connector"

// OrphanAnalysis details unreachable nodes and isolated components in layer 0.
type OrphanAnalysis struct {
	TotalNodes          int      `json:"total_nodes"`
	OrphanNodesCount    int      `json:"orphan_nodes_count"`
	OrphanRatio         float64  `json:"orphan_ratio"`
	OrphanNodeIDs       []string `json:"orphan_node_ids,omitempty"`
	ConnectedComponents int      `json:"connected_components"`
	LargestComponent    int      `json:"largest_component"`
	IsolatedSubgraphs   int      `json:"isolated_subgraphs"`
}

// AnalyzeOrphansAndComponents computes in-degrees and connected components on layer 0.
func AnalyzeOrphansAndComponents(layer0 *connector.GraphLayerExport) OrphanAnalysis {
	if layer0 == nil || len(layer0.Nodes) == 0 {
		return OrphanAnalysis{}
	}

	total := len(layer0.Nodes)
	inDegrees := make(map[string]int, total)
	for id := range layer0.Nodes {
		inDegrees[id] = 0
	}

	// Calculate in-degrees
	for _, node := range layer0.Nodes {
		for _, neighborID := range node.Neighbors {
			if _, exists := inDegrees[neighborID]; exists {
				inDegrees[neighborID]++
			}
		}
	}

	orphanIDs := make([]string, 0)
	for id, inDeg := range inDegrees {
		if inDeg == 0 {
			orphanIDs = append(orphanIDs, id)
		}
	}

	orphanRatio := 0.0
	if total > 0 {
		orphanRatio = float64(len(orphanIDs)) / float64(total)
	}

	// Connected components analysis via undirected BFS on layer 0
	visited := make(map[string]bool, total)
	components := 0
	largestComp := 0

	// Build symmetric adjacency for component traversal
	adj := make(map[string][]string, total)
	for id, node := range layer0.Nodes {
		adj[id] = append(adj[id], node.Neighbors...)
		for _, nbr := range node.Neighbors {
			adj[nbr] = append(adj[nbr], id)
		}
	}

	for id := range layer0.Nodes {
		if visited[id] {
			continue
		}

		components++
		compSize := 0
		queue := []string{id}
		visited[id] = true

		for len(queue) > 0 {
			curr := queue[0]
			queue = queue[1:]
			compSize++

			for _, nbr := range adj[curr] {
				if !visited[nbr] {
					visited[nbr] = true
					queue = append(queue, nbr)
				}
			}
		}

		if compSize > largestComp {
			largestComp = compSize
		}
	}

	isolated := 0
	if components > 1 {
		isolated = components - 1
	}

	return OrphanAnalysis{
		TotalNodes:          total,
		OrphanNodesCount:    len(orphanIDs),
		OrphanRatio:         orphanRatio,
		OrphanNodeIDs:       orphanIDs,
		ConnectedComponents: components,
		LargestComponent:    largestComp,
		IsolatedSubgraphs:   isolated,
	}
}
