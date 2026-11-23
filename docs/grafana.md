# Grafana dashboard setup guide

hnswprobe includes a pre-configured Grafana dashboard template for monitoring HNSW graph degradation and recall decay across your vector database fleet.

## Dashboard template location

The dashboard definition is located at:
```
dashboards/hnswprobe-grafana.json
```

## Setup instructions

### 1. Configure Prometheus scrape target

Add `hnswprobe` to your `prometheus.yml` scrape configuration:

```yaml
scrape_configs:
  - job_name: "hnswprobe"
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:8383"]
        labels:
          environment: "production"
          cluster: "rag-eu-west-1"
```

Verify that metrics are accessible:
```bash
curl http://localhost:8383/metrics
```

### 2. Import into Grafana

1. Open your Grafana console (e.g. `http://localhost:3000`).
2. Navigate to **Dashboards** > **New** > **Import**.
3. Click **Upload JSON file** and select `dashboards/hnswprobe-grafana.json` (or paste the JSON content).
4. Select your Prometheus data source from the dropdown.
5. Click **Import**.

## Monitored metrics and alerts

| Metric Name | Prometheus Series | Healthy Range | Recommended Alert Condition |
| :--- | :--- | :--- | :--- |
| Search Recall@k | `hnswprobe_recall_ratio` | 0.90 to 1.00 | Alert if `< 0.85` for 5m |
| Tombstone Bloat Ratio | `hnswprobe_tombstone_bloat_ratio` | 0.00 to 0.15 | Alert if `> 0.20` for 10m |
| Orphan Ratio | `hnswprobe_orphan_ratio` | 0.00 to 0.01 | Alert if `> 0.03` for 5m |
| Mean Greedy Traversal Hops | `hnswprobe_mean_hop_count` | 2 to 8 | Alert if `> 12` |
| Index Memory (Bytes) | `hnswprobe_memory_bytes` | System dependent | Used for capacity planning |
