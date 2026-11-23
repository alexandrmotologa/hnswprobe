package quantization

import (
	"math"

	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
)

// QuantizedVector holds uint8 quantized values alongside scale and offset parameters.
type QuantizedVector struct {
	ID     string
	Data   []uint8
	MinVal float32
	MaxVal float32
	Scale  float32
}

// QuantizationReport provides metrics comparing quantized vectors with original float32 vectors.
type QuantizationReport struct {
	Dimension              int     `json:"dimension"`
	OriginalBytesPerVector int     `json:"original_bytes_per_vector"`
	QuantizedBytesPerVector int    `json:"quantized_bytes_per_vector"`
	MemorySavingsPercent   float64 `json:"memory_savings_percent"`
	MeanAngularDistortion  float64 `json:"mean_angular_distortion"` // 1.0 - CosineSimilarity(orig, dequant)
	MaxAngularDistortion   float64 `json:"max_angular_distortion"`
	MeanRelativeL2Error    float64 `json:"mean_relative_l2_error"`   // ||v - dequant|| / ||v||
	QuantizedRecallAtK     float64 `json:"quantized_recall_at_k"`     // Recall vs unquantized ground truth
}

// ScalarQuantizer quantizes float32 vector components into uint8 integers.
type ScalarQuantizer struct{}

// NewScalarQuantizer creates a scalar quantizer instance.
func NewScalarQuantizer() *ScalarQuantizer {
	return &ScalarQuantizer{}
}

// Quantize compresses a float32 vector to uint8.
func (sq *ScalarQuantizer) Quantize(id string, values []float32) QuantizedVector {
	if len(values) == 0 {
		return QuantizedVector{ID: id}
	}

	minVal := values[0]
	maxVal := values[0]
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	diff := maxVal - minVal
	if diff <= 0 {
		diff = 1.0
	}
	scale := 255.0 / diff

	data := make([]uint8, len(values))
	for i, v := range values {
		scaled := (v - minVal) * scale
		if scaled < 0 {
			scaled = 0
		} else if scaled > 255 {
			scaled = 255
		}
		data[i] = uint8(math.Round(float64(scaled)))
	}

	return QuantizedVector{
		ID:     id,
		Data:   data,
		MinVal: minVal,
		MaxVal: maxVal,
		Scale:  diff / 255.0,
	}
}

// Dequantize reconstructs the approximate float32 vector from uint8 representation.
func (sq *ScalarQuantizer) Dequantize(qv QuantizedVector) []float32 {
	res := make([]float32, len(qv.Data))
	for i, d := range qv.Data {
		res[i] = qv.MinVal + float32(d)*qv.Scale
	}
	return res
}

// ProfileQuantization evaluates distortion and recall penalty across a corpus and sample queries.
func (sq *ScalarQuantizer) ProfileQuantization(
	corpus []groundtruth.VectorItem,
	queries []groundtruth.QueryItem,
	k int,
) QuantizationReport {
	if len(corpus) == 0 {
		return QuantizationReport{}
	}

	dim := len(corpus[0].Values)
	origBytes := dim * 4
	quantBytes := dim * 1 // 1 byte per dimension in uint8

	var sumAngular, maxAngular, sumL2 float64
	dequantCorpus := make([]groundtruth.VectorItem, len(corpus))

	for i, item := range corpus {
		qv := sq.Quantize(item.ID, item.Values)
		dequant := sq.Dequantize(qv)
		dequantCorpus[i] = groundtruth.VectorItem{ID: item.ID, Values: dequant}

		// Calculate angular distortion
		cosSim := groundtruth.CosineSimilarity(item.Values, dequant)
		angularDist := 1.0 - float64(cosSim)
		if angularDist < 0 {
			angularDist = 0
		}
		sumAngular += angularDist
		if angularDist > maxAngular {
			maxAngular = angularDist
		}

		// Relative L2 error
		l2Diff := math.Sqrt(float64(groundtruth.SquaredL2Distance(item.Values, dequant)))
		l2Orig := math.Sqrt(float64(groundtruth.SquaredL2Distance(item.Values, make([]float32, dim))))
		if l2Orig > 1e-6 {
			sumL2 += l2Diff / l2Orig
		}
	}

	meanAngular := sumAngular / float64(len(corpus))
	meanL2 := sumL2 / float64(len(corpus))
	savings := (1.0 - float64(quantBytes)/float64(origBytes)) * 100.0

	// Evaluate recall impact if queries are provided
	quantRecall := 1.0
	if len(queries) > 0 {
		solver := groundtruth.NewExactSolver(groundtruth.MetricCosine)
		exactRaw := solver.SolveBatch(queries, corpus, k)
		exactQuant := solver.SolveBatch(queries, dequantCorpus, k)
		evalReport := groundtruth.EvaluateRecall(exactRaw, exactQuant, k)
		quantRecall = evalReport.MeanRecall
	}

	return QuantizationReport{
		Dimension:               dim,
		OriginalBytesPerVector:  origBytes,
		QuantizedBytesPerVector: quantBytes,
		MemorySavingsPercent:    savings,
		MeanAngularDistortion:   meanAngular,
		MaxAngularDistortion:    maxAngular,
		MeanRelativeL2Error:     meanL2,
		QuantizedRecallAtK:      quantRecall,
	}
}
