package optimizer

import (
	"sync"
	"time"
)

// AdaptiveController dynamically adjusts active efSearch to compensate for index decay.
type AdaptiveController struct {
	mu            sync.RWMutex
	targetRecall  float64
	baseEf        int
	maxEf         int
	currentEf     int
	lastRecall    float64
	decayDetected bool
	lastAdjusted  time.Time
}

// ControllerStatus reports live state of the adaptive controller.
type ControllerStatus struct {
	TargetRecall   float64   `json:"target_recall"`
	CurrentRecall  float64   `json:"current_recall"`
	ActiveEfSearch int       `json:"active_ef_search"`
	BaseEfSearch   int       `json:"base_ef_search"`
	MaxEfSearch    int       `json:"max_ef_search"`
	DecayDetected  bool      `json:"decay_detected"`
	LastAdjustedAt time.Time `json:"last_adjusted_at"`
	IsRelaxed      bool      `json:"is_relaxed"`
}

// NewAdaptiveController creates a controller with target recall and bounds.
func NewAdaptiveController(targetRecall float64, baseEf, maxEf int) *AdaptiveController {
	if targetRecall <= 0 {
		targetRecall = 0.95
	}
	if baseEf <= 0 {
		baseEf = 32
	}
	if maxEf < baseEf {
		maxEf = 256
	}

	return &AdaptiveController{
		targetRecall: targetRecall,
		baseEf:       baseEf,
		maxEf:        maxEf,
		currentEf:    baseEf,
		lastAdjusted: time.Now(),
	}
}

// UpdateEvaluation recalculates the active efSearch based on fresh evaluation metrics and Pareto frontier.
func (c *AdaptiveController) UpdateEvaluation(currentRecall float64, frontier *ParetoFrontier) int {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.lastRecall = currentRecall
	c.lastAdjusted = time.Now()

	// If recall meets target, attempt to relax toward baseEf
	if currentRecall >= c.targetRecall {
		c.decayDetected = false
		if c.currentEf > c.baseEf {
			// Step down gradually
			c.currentEf = (c.currentEf + c.baseEf) / 2
		}
		return c.currentEf
	}

	// Recall is below target; decay detected
	c.decayDetected = true

	// If Pareto frontier is available, look for first point satisfying target
	if frontier != nil && len(frontier.FrontierPoints) > 0 {
		for _, p := range frontier.FrontierPoints {
			if p.MeanRecall >= c.targetRecall {
				c.currentEf = p.EfSearch
				return c.currentEf
			}
		}
		// If no point achieves target, pick the highest recall point
		best := frontier.FrontierPoints[len(frontier.FrontierPoints)-1]
		if best.EfSearch > c.currentEf {
			c.currentEf = best.EfSearch
		}
	} else {
		// Multiplicative step up
		nextEf := int(float64(c.currentEf) * 1.5)
		if nextEf > c.maxEf {
			nextEf = c.maxEf
		}
		c.currentEf = nextEf
	}

	return c.currentEf
}

// GetStatus returns the current controller status.
func (c *AdaptiveController) GetStatus() ControllerStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return ControllerStatus{
		TargetRecall:   c.targetRecall,
		CurrentRecall:  c.lastRecall,
		ActiveEfSearch: c.currentEf,
		BaseEfSearch:   c.baseEf,
		MaxEfSearch:    c.maxEf,
		DecayDetected:  c.decayDetected,
		LastAdjustedAt: c.lastAdjusted,
		IsRelaxed:      c.currentEf == c.baseEf,
	}
}

// SetTargetRecall updates the target recall threshold.
func (c *AdaptiveController) SetTargetRecall(target float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if target > 0 && target <= 1.0 {
		c.targetRecall = target
	}
}
