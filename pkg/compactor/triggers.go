package compactor

import (
	"context"
	"fmt"
	"time"

	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
	"github.com/alexandrmotologa/hnswprobe/pkg/groundtruth"
	"github.com/alexandrmotologa/hnswprobe/pkg/topology"
)

// Policy defines threshold limits that trigger warnings and compaction actions.
type Policy struct {
	MinRecall         float64 `json:"min_recall"`          // Default 0.90
	MaxTombstoneRatio float64 `json:"max_tombstone_ratio"` // Default 0.15
	MaxOrphanRatio    float64 `json:"max_orphan_ratio"`    // Default 0.03
	AutoCompact       bool    `json:"auto_compact"`
	WebhookURL        string  `json:"webhook_url"`
}

// DefaultPolicy returns a safe production policy.
func DefaultPolicy() Policy {
	return Policy{
		MinRecall:         0.90,
		MaxTombstoneRatio: 0.15,
		MaxOrphanRatio:    0.03,
		AutoCompact:       false,
	}
}

// ActionSeverity marks the gravity of the policy breach.
type ActionSeverity string

const (
	SeverityInfo     ActionSeverity = "INFO"
	SeverityWarning  ActionSeverity = "WARNING"
	SeverityCritical ActionSeverity = "CRITICAL"
)

// TriggerEvent records an evaluation against policy rules.
type TriggerEvent struct {
	Timestamp          time.Time      `json:"timestamp"`
	Backend            string         `json:"backend"`
	IndexName          string         `json:"index_name"`
	Severity           ActionSeverity `json:"severity"`
	Breached           bool           `json:"breached"`
	Reason             string         `json:"reason"`
	CompactionExecuted bool           `json:"compaction_executed"`
	CompactionError    string         `json:"compaction_error,omitempty"`
	WebhookDispatched  bool           `json:"webhook_dispatched"`
}

// EvaluatePolicy checks index health and optionally triggers compaction and alerts.
func EvaluatePolicy(
	ctx context.Context,
	conn connector.Connector,
	compactor *Compactor,
	stats *connector.IndexStats,
	evalReport groundtruth.EvaluationReport,
	topoReport topology.GraphReport,
	policy Policy,
) (*TriggerEvent, error) {
	event := &TriggerEvent{
		Timestamp: time.Now(),
		Backend:   stats.Backend,
		IndexName: stats.IndexName,
		Severity:  SeverityInfo,
	}

	reasons := make([]string, 0)
	var severity ActionSeverity = SeverityInfo

	// Check recall
	if evalReport.MeanRecall > 0 && evalReport.MeanRecall < policy.MinRecall {
		reasons = append(reasons, fmt.Sprintf("Mean recall %.2f is below threshold %.2f", evalReport.MeanRecall, policy.MinRecall))
		if evalReport.MeanRecall < policy.MinRecall-0.10 {
			severity = SeverityCritical
		} else {
			severity = SeverityWarning
		}
	}

	// Check tombstones
	tbr := topoReport.TombstoneBloatRatio
	if tbr == 0 && stats.TombstoneRatio > 0 {
		tbr = stats.TombstoneRatio
	}
	if tbr > policy.MaxTombstoneRatio {
		reasons = append(reasons, fmt.Sprintf("Tombstone ratio %.2f exceeds limit %.2f", tbr, policy.MaxTombstoneRatio))
		if tbr > policy.MaxTombstoneRatio*1.5 {
			severity = SeverityCritical
		} else if severity != SeverityCritical {
			severity = SeverityWarning
		}
	}

	// Check orphans
	if topoReport.OrphanRatio > policy.MaxOrphanRatio {
		reasons = append(reasons, fmt.Sprintf("Orphan ratio %.2f exceeds limit %.2f", topoReport.OrphanRatio, policy.MaxOrphanRatio))
		if severity != SeverityCritical {
			severity = SeverityWarning
		}
	}

	if len(reasons) > 0 {
		event.Breached = true
		event.Severity = severity
		event.Reason = fmt.Sprintf("Policy breached: %v", reasons)

		// Execute compaction if auto-compact is active
		if policy.AutoCompact && compactor != nil {
			err := compactor.Execute(ctx, conn)
			if err != nil {
				event.CompactionError = err.Error()
			} else {
				event.CompactionExecuted = true
			}
		}

		// Dispatch webhook
		if policy.WebhookURL != "" && compactor != nil {
			_ = compactor.DispatchWebhook(ctx, policy.WebhookURL, event)
			event.WebhookDispatched = true
		}
	} else {
		event.Reason = "Index health within policy thresholds"
	}

	return event, nil
}
