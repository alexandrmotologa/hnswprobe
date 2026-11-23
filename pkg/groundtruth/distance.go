package groundtruth

import "math"

// Metric defines the distance computation strategy for vector comparisons.
type Metric string

const (
	MetricCosine    Metric = "cosine"
	MetricDot       Metric = "dot"
	MetricEuclidean Metric = "l2"
)

// CosineSimilarity computes cosine similarity between vectors a and b.
// Returns a value between -1.0 and 1.0.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float32
	i := 0
	length := len(a)

	// Process 4 elements per iteration to assist compiler vectorization
	for ; i+3 < length; i += 4 {
		dot += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		normA += a[i]*a[i] + a[i+1]*a[i+1] + a[i+2]*a[i+2] + a[i+3]*a[i+3]
		normB += b[i]*b[i] + b[i+1]*b[i+1] + b[i+2]*b[i+2] + b[i+3]*b[i+3]
	}

	for ; i < length; i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA <= 0 || normB <= 0 {
		return 0
	}

	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// CosineDistance computes cosine distance: 1.0 - CosineSimilarity.
func CosineDistance(a, b []float32) float32 {
	return 1.0 - CosineSimilarity(a, b)
}

// DotProduct computes scalar dot product between two float32 slices.
func DotProduct(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sum float32
	i := 0
	length := len(a)

	for ; i+3 < length; i += 4 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
	}

	for ; i < length; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// SquaredL2Distance computes squared Euclidean distance (sum of (a[i]-b[i])^2).
func SquaredL2Distance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sum float32
	i := 0
	length := len(a)

	for ; i+3 < length; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3
	}

	for ; i < length; i++ {
		d := a[i] - b[i]
		sum += d * d
	}

	return sum
}
