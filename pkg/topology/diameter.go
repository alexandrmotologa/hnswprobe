package topology

import "github.com/alexandrmotologa/hnswprobe/pkg/connector"

// EstimateGraphDiameter estimates the diameter of the layer 0 graph using double-sweep BFS.
func EstimateGraphDiameter(layer0 *connector.GraphLayerExport, sampleStarts int) int {
	if layer0 == nil || len(layer0.Nodes) <= 1 {
		return 0
	}

	if sampleStarts <= 0 {
		sampleStarts = 3
	}

	// Pick initial start nodes
	startNodes := make([]string, 0, sampleStarts)
	for id := range layer0.Nodes {
		startNodes = append(startNodes, id)
		if len(startNodes) >= sampleStarts {
			break
		}
	}

	maxDiameter := 0

	for _, startID := range startNodes {
		// Sweep 1: find farthest node from start
		farthest1, _ := bfsFarthest(layer0, startID)
		if farthest1 == "" {
			continue
		}

		// Sweep 2: find farthest node from farthest1
		_, dist2 := bfsFarthest(layer0, farthest1)
		if dist2 > maxDiameter {
			maxDiameter = dist2
		}
	}

	return maxDiameter
}

func bfsFarthest(layer0 *connector.GraphLayerExport, startID string) (string, int) {
	dist := make(map[string]int)
	queue := []string{startID}
	dist[startID] = 0

	farthestNode := startID
	maxDist := 0

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		currDist := dist[curr]

		if currDist > maxDist {
			maxDist = currDist
			farthestNode = curr
		}

		node, exists := layer0.Nodes[curr]
		if !exists {
			continue
		}

		for _, nbr := range node.Neighbors {
			if _, visited := dist[nbr]; !visited {
				dist[nbr] = currDist + 1
				queue = append(queue, nbr)
			}
		}
	}

	return farthestNode, maxDist
}
