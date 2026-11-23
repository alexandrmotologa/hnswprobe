package compactor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/alexandrmotologa/hnswprobe/pkg/connector"
)

// Compactor executes database compaction and notifies external systems.
type Compactor struct {
	httpClient *http.Client
}

// NewCompactor creates a compactor instance.
func NewCompactor() *Compactor {
	return &Compactor{
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

// Execute triggers the database-specific compaction or vacuum operation.
func (c *Compactor) Execute(ctx context.Context, conn connector.Connector) error {
	return conn.TriggerCompaction(ctx)
}

// DispatchWebhook sends an event payload to a webhook URL.
func (c *Compactor) DispatchWebhook(ctx context.Context, url string, payload any) error {
	if url == "" {
		return nil
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal webhook payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("failed to create webhook request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "hnswprobe-compactor/1.0")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("webhook dispatch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("webhook responded with non-2xx status: %d", resp.StatusCode)
	}

	return nil
}
