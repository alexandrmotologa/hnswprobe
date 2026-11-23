// hnswprobe Cockpit Frontend Application
(function () {
  'use strict';

  let currentSnapshot = null;
  let visualData = null;
  let viewMode = '2d'; // '2d' or '3d'
  let selectedLayer = 'all';

  // Canvas pan and zoom state
  const canvas = document.getElementById('graph-canvas');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('node-tooltip');

  let camera = {
    x: 0,
    y: 0,
    zoom: 1.0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
  };

  let hoveredNode = null;
  let selectedNode = null;
  let highlightHighways = true;
  let animationHopPath = [];
  let hopStep = 0;
  let isSimulatingHop = false;

  // Initialize
  function init() {
    setupCanvas();
    setupEventListeners();
    fetchAuditData();
  }

  function setupCanvas() {
    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      renderGraph();
    }
    window.addEventListener('resize', resize);
    resize();
  }

  function setupEventListeners() {
    // Mode toggles
    const btn2d = document.getElementById('view-mode-2d');
    const btn3d = document.getElementById('view-mode-3d');
    btn2d.addEventListener('click', () => {
      viewMode = '2d';
      btn2d.classList.add('active');
      btn3d.classList.remove('active');
      renderGraph();
    });
    btn3d.addEventListener('click', () => {
      viewMode = '3d';
      btn3d.classList.add('active');
      btn2d.classList.remove('active');
      renderGraph();
    });

    // Layer filter
    document.getElementById('layer-select').addEventListener('change', (e) => {
      selectedLayer = e.target.value;
      renderGraph();
    });

    // Camera reset
    document.getElementById('reset-cam-btn').addEventListener('click', () => {
      camera.x = 0;
      camera.y = 0;
      camera.zoom = 1.0;
      renderGraph();
    });

    // Simulate query hops
    document.getElementById('simulate-hop-btn').addEventListener('click', startHopSimulation);

    // Refresh audit
    document.getElementById('refresh-btn').addEventListener('click', () => fetchAuditData(true));

    // Compaction trigger
    document.getElementById('compact-btn').addEventListener('click', triggerCompaction);

    // Canvas interactions
    canvas.addEventListener('mousedown', (e) => {
      camera.isDragging = true;
      camera.dragStartX = e.clientX - camera.x;
      camera.dragStartY = e.clientY - camera.y;
    });

    window.addEventListener('mouseup', () => {
      camera.isDragging = false;
    });

    canvas.addEventListener('mousemove', (e) => {
      if (camera.isDragging) {
        camera.x = e.clientX - camera.dragStartX;
        camera.y = e.clientY - camera.dragStartY;
        renderGraph();
      } else {
        handleNodeHover(e);
      }
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      camera.zoom *= zoomFactor;
      if (camera.zoom < 0.2) camera.zoom = 0.2;
      if (camera.zoom > 5.0) camera.zoom = 5.0;
      renderGraph();
    });

    // Chaos and Highway controls
    const hwCheckbox = document.getElementById('highway-checkbox');
    if (hwCheckbox) {
      hwCheckbox.addEventListener('change', (e) => {
        highlightHighways = e.target.checked;
        renderGraph();
      });
    }

    const chaosTombBtn = document.getElementById('chaos-tombstone-btn');
    if (chaosTombBtn) {
      chaosTombBtn.addEventListener('click', () => triggerChaos('/api/chaos/tombstones', { ratio: 0.20 }));
    }

    const chaosDriftBtn = document.getElementById('chaos-drift-btn');
    if (chaosDriftBtn) {
      chaosDriftBtn.addEventListener('click', () => triggerChaos('/api/chaos/drift', { size: 35, offset: 2.0 }));
    }

    const chaosSeverBtn = document.getElementById('chaos-sever-btn');
    if (chaosSeverBtn) {
      chaosSeverBtn.addEventListener('click', () => triggerChaos('/api/chaos/sever', { ratio: 0.35 }));
    }

    // Canvas click selection
    canvas.addEventListener('click', () => {
      if (hoveredNode) {
        selectNode(hoveredNode);
      }
    });

    // Pareto slider
    const slider = document.getElementById('ef-slider');
    slider.addEventListener('input', (e) => {
      const efVal = parseInt(e.target.value, 10);
      document.getElementById('slider-ef-val').textContent = efVal;
      updateParetoPrediction(efVal);
    });
  }

  // API Calls
  function fetchAuditData(refresh = false) {
    const url = refresh ? '/api/audit?refresh=true' : '/api/audit';
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        currentSnapshot = data;
        visualData = data.visual_graph;
        updateDashboard(data);
        renderGraph();
        renderParetoChart(data.pareto);
        renderDegreeHistogram(data.topology);
      })
      .catch((err) => {
        console.error('Failed to fetch audit data:', err);
      });
  }

  function triggerChaos(endpoint, payload) {
    const modal = document.getElementById('compaction-modal');
    modal.classList.add('active');
    const titleEl = document.querySelector('.modal-title');
    const textEl = document.querySelector('.modal-text');
    if (titleEl) titleEl.textContent = 'Injecting Chaos Experiment...';
    if (textEl) textEl.textContent = 'Simulating graph disruption, drift, and evaluating new recall metrics...';

    fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then((res) => res.json())
      .then((data) => {
        modal.classList.remove('active');
        if (titleEl) titleEl.textContent = 'Graph Compaction in Progress';
        if (textEl) textEl.textContent = 'Purging tombstoned vectors, repairing broken bridges, and refreshing index metadata...';

        if (data.snapshot) {
          currentSnapshot = data.snapshot;
          visualData = data.snapshot.visual_graph;
          updateDashboard(data.snapshot);
          renderGraph();
          renderParetoChart(data.snapshot.pareto);
          renderDegreeHistogram(data.snapshot.topology);
          if (selectedNode) {
            const updated = visualData.nodes.find((n) => n.id === selectedNode.id);
            if (updated) selectNode(updated);
          }
        } else {
          fetchAuditData(true);
        }
      })
      .catch((err) => {
        modal.classList.remove('active');
        alert('Chaos injection error: ' + err);
      });
  }

  function selectNode(node) {
    selectedNode = node;
    const statusTag = document.getElementById('inspector-node-status');
    if (statusTag) statusTag.textContent = node.id;

    const details = document.getElementById('inspector-details');
    if (!details) return;

    const levels = node.ascension_levels && node.ascension_levels.length > 0 ? node.ascension_levels : [node.layer];
    const pillsHtml = levels.map((l) => `<span class="ascension-pill">L${l}</span>`).join(' ');

    const statusHtml = node.is_tomb
      ? '<span style="color:var(--crimson)">Tombstone</span>'
      : node.is_orphan
      ? '<span style="color:#ec4899">Orphan (0 In-Deg)</span>'
      : node.is_hub
      ? '<span style="color:var(--amber)">Hub Bottleneck</span>'
      : node.is_entry
      ? '<span style="color:var(--violet)">Entry Point</span>'
      : '<span style="color:var(--cyan-primary)">Active</span>';

    details.innerHTML = `
      <div class="inspector-row">
        <span class="inspector-key">Node ID:</span>
        <span class="inspector-val">${node.id}</span>
      </div>
      <div class="inspector-row">
        <span class="inspector-key">Ascension Levels:</span>
        <div class="ascension-pills">${pillsHtml}</div>
      </div>
      <div class="inspector-row">
        <span class="inspector-key">In-Degree:</span>
        <span class="inspector-val">${node.in_degree}</span>
      </div>
      <div class="inspector-row">
        <span class="inspector-key">Out-Degree:</span>
        <span class="inspector-val">${node.out_degree}</span>
      </div>
      <div class="inspector-row">
        <span class="inspector-key">Status:</span>
        <span class="inspector-val">${statusHtml}</span>
      </div>
    `;
    renderGraph();
  }

  function triggerCompaction() {
    const modal = document.getElementById('compaction-modal');
    modal.classList.add('active');

    fetch('/api/compact', { method: 'POST' })
      .then((res) => res.json())
      .then((data) => {
        modal.classList.remove('active');
        fetchAuditData(true);
      })
      .catch((err) => {
        modal.classList.remove('active');
        alert('Compaction error: ' + err);
      });
  }

  // Dashboard Updates
  function updateDashboard(data) {
    // Header
    const bBadge = document.getElementById('backend-badge');
    bBadge.textContent = `Backend: ${data.backend.toUpperCase()} | Index: ${data.index_stats?.index_name || 'N/A'}`;

    const healthBadge = document.getElementById('overall-status-badge');
    const recall = data.evaluation?.mean_recall || 0;
    const tbr = data.topology?.tombstone_bloat_ratio || 0;

    if (recall >= 0.90 && tbr < 0.15) {
      healthBadge.className = 'status-badge status-optimal';
      healthBadge.textContent = 'Health: Optimal';
    } else if (recall >= 0.80 && tbr < 0.25) {
      healthBadge.className = 'status-badge status-warning';
      healthBadge.textContent = 'Health: Warning';
    } else {
      healthBadge.className = 'status-badge status-critical';
      healthBadge.textContent = 'Health: Critical Decay';
    }

    // KPI Cards
    const recPct = (recall * 100).toFixed(1);
    document.getElementById('kpi-recall-val').textContent = `${recPct}%`;
    const recBadge = document.getElementById('kpi-recall-badge');
    if (recall >= 0.90) {
      recBadge.className = 'kpi-badge badge-green';
      recBadge.textContent = 'Optimal';
    } else if (recall >= 0.80) {
      recBadge.className = 'kpi-badge badge-yellow';
      recBadge.textContent = 'Degraded';
    } else {
      recBadge.className = 'kpi-badge badge-red';
      recBadge.textContent = 'Critical';
    }

    const tbrPct = (tbr * 100).toFixed(1);
    document.getElementById('kpi-tombstone-val').textContent = `${tbrPct}%`;
    document.getElementById('kpi-tombstone-badge').textContent = `${data.topology?.tombstone_nodes || 0} Dead`;
    document.getElementById('kpi-tombstone-sub').textContent = `${data.topology?.active_nodes || 0} active, ${data.topology?.tombstone_nodes || 0} dead nodes`;

    const orphanCount = data.topology?.orphan_nodes_count || 0;
    const orphanPct = ((data.topology?.orphan_ratio || 0) * 100).toFixed(1);
    document.getElementById('kpi-orphan-val').textContent = orphanCount;
    document.getElementById('kpi-orphan-badge').textContent = `${orphanPct}% Ratio`;

    const eff = data.topology?.trajectory_efficiency || 0;
    document.getElementById('kpi-efficiency-val').textContent = eff > 0 ? eff.toFixed(2) : '--';
    document.getElementById('kpi-hops-sub').textContent = `Avg traversal hops: ${(data.topology?.mean_hop_count || 0).toFixed(1)}`;

    document.getElementById('kpi-diameter-val').textContent = data.topology?.estimated_diameter || '--';

    // Graph panel tag
    document.getElementById('graph-stats-tag').textContent =
      `Nodes: ${data.topology?.total_nodes || 0} | Edges: ${data.topology?.total_edges || 0} | Max Layer: ${data.topology?.max_layer || 0}`;

    // Recommendations
    const recs = data.recommendations;
    if (recs) {
      document.getElementById('rec-text').textContent = recs.actionable_advice;
      document.getElementById('rec-param-ef').textContent = recs.optimal_ef_search;
      document.getElementById('rec-param-m').textContent = recs.recommended_m;
      document.getElementById('rec-param-efc').textContent = recs.recommended_ef_construction;

      const actionBadge = document.getElementById('rec-action-badge');
      if (recs.requires_compaction) {
        actionBadge.textContent = 'Vacuum Required';
        actionBadge.style.color = 'var(--amber)';
      } else if (recs.requires_reindex) {
        actionBadge.textContent = 'Reindex Required';
        actionBadge.style.color = 'var(--crimson)';
      } else {
        actionBadge.textContent = 'Optimal';
        actionBadge.style.color = 'var(--emerald)';
      }
    }
  }

  // 2D/3D Projection Helper
  function project(node, width, height) {
    const cx = width / 2 + camera.x;
    const cy = height / 2 + camera.y;

    if (viewMode === '2d') {
      return {
        x: cx + node.x * camera.zoom,
        y: cy + node.y * camera.zoom,
      };
    } else {
      // 3D Isometric Projection
      const angle = Math.PI / 6; // 30 deg
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      const isoX = (node.x - node.y) * cosA;
      const isoY = (node.x + node.y) * sinA - node.z * 0.8;

      return {
        x: cx + isoX * camera.zoom,
        y: cy + isoY * camera.zoom,
      };
    }
  }

  // Canvas Graph Renderer
  function renderGraph() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);

    if (!visualData || !visualData.nodes || visualData.nodes.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting graph topology data...', width / 2, height / 2);
      return;
    }

    // Filter nodes by layer
    const filteredNodes = visualData.nodes.filter((n) => {
      if (selectedLayer === 'all') return true;
      if (selectedLayer === '2') return n.layer >= 2;
      return n.layer === parseInt(selectedLayer, 10);
    });

    const nodePosMap = new Map();
    filteredNodes.forEach((n) => {
      nodePosMap.set(n.id, project(n, width, height));
    });

    // 1. Draw Edges
    visualData.edges.forEach((edge) => {
      if (selectedLayer !== 'all') {
        if (selectedLayer === '2' && edge.layer < 2) return;
        if (selectedLayer !== '2' && edge.layer !== parseInt(selectedLayer, 10)) return;
      }

      const p1 = nodePosMap.get(edge.source);
      const p2 = nodePosMap.get(edge.target);
      if (p1 && p2) {
        ctx.beginPath();
        if (highlightHighways && edge.is_highway) {
          ctx.lineWidth = 1.8;
          ctx.strokeStyle = 'rgba(139, 92, 246, 0.65)';
          ctx.shadowColor = '#8b5cf6';
          ctx.shadowBlur = 5;
        } else {
          ctx.lineWidth = 1;
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
          ctx.shadowBlur = 0;
        }
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    });

    // 1.5 Draw Multi-Layer Ascension Pillar for Selected Node
    if (selectedNode && viewMode === '3d' && selectedNode.ascension_levels && selectedNode.ascension_levels.length > 1) {
      const levels = selectedNode.ascension_levels;
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#c4b5fd';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      levels.forEach((l, idx) => {
        const dummyNode = { x: selectedNode.x, y: selectedNode.y, z: l * 120.0 };
        const p = project(dummyNode, width, height);
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 2. Draw Simulation Hop Path
    if (animationHopPath.length > 1) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#38bdf8';
      ctx.shadowColor = '#06b6d4';
      ctx.shadowBlur = 8;
      ctx.beginPath();

      const p0 = nodePosMap.get(animationHopPath[0].id);
      if (p0) ctx.moveTo(p0.x, p0.y);

      for (let i = 1; i <= hopStep && i < animationHopPath.length; i++) {
        const p = nodePosMap.get(animationHopPath[i].id);
        if (p) ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // 3. Draw Nodes
    filteredNodes.forEach((node) => {
      const pos = nodePosMap.get(node.id);
      if (!pos) return;

      let radius = 4 * camera.zoom;
      if (radius < 2) radius = 2;
      if (radius > 12) radius = 12;

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

      if (node.is_entry) {
        ctx.fillStyle = '#8b5cf6';
        ctx.shadowColor = '#8b5cf6';
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
        // Outer ring
        ctx.strokeStyle = '#c4b5fd';
        ctx.lineWidth = 2;
        ctx.stroke();
      } else if (node.is_tomb) {
        ctx.fillStyle = '#ef4444';
        ctx.fill();
      } else if (node.is_orphan) {
        ctx.fillStyle = '#ec4899';
        ctx.fill();
      } else if (node.is_hub) {
        ctx.fillStyle = '#f59e0b';
        ctx.fill();
      } else {
        ctx.fillStyle = '#06b6d4';
        ctx.fill();
      }
    });

    // 4. Highlight Hovered Node
    if (hoveredNode) {
      const pos = nodePosMap.get(hoveredNode.id);
      if (pos) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 8 * camera.zoom, 0, Math.PI * 2);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // 5. Highlight Selected Node
    if (selectedNode) {
      const pos = nodePosMap.get(selectedNode.id);
      if (pos) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 10 * camera.zoom, 0, Math.PI * 2);
        ctx.strokeStyle = '#c4b5fd';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
    }
  }

  function handleNodeHover(e) {
    if (!visualData || !visualData.nodes) return;

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let nearest = null;
    let minDist = 12;

    const width = rect.width;
    const height = rect.height;

    for (const node of visualData.nodes) {
      if (selectedLayer !== 'all') {
        if (selectedLayer === '2' && node.layer < 2) continue;
        if (selectedLayer !== '2' && node.layer !== parseInt(selectedLayer, 10)) continue;
      }

      const p = project(node, width, height);
      const dx = mx - p.x;
      const dy = my - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < minDist) {
        minDist = dist;
        nearest = node;
      }
    }

    if (nearest !== hoveredNode) {
      hoveredNode = nearest;
      if (hoveredNode) {
        tooltip.style.display = 'block';
        tooltip.style.left = `${mx + 14}px`;
        tooltip.style.top = `${my + 14}px`;
        tooltip.innerHTML = `
          <div><strong>ID:</strong> ${hoveredNode.id}</div>
          <div><strong>Layer:</strong> ${hoveredNode.layer}</div>
          <div><strong>Out-Degree:</strong> ${hoveredNode.out_degree}</div>
          <div><strong>Status:</strong> ${hoveredNode.is_tomb ? '<span style="color:#ef4444">Tombstone</span>' : hoveredNode.is_orphan ? '<span style="color:#ec4899">Orphan</span>' : hoveredNode.is_hub ? '<span style="color:#f59e0b">Hub Node</span>' : '<span style="color:#06b6d4">Active</span>'}</div>
        `;
      } else {
        tooltip.style.display = 'none';
      }
      renderGraph();
    }
  }

  // Hop Simulation Animation
  function startHopSimulation() {
    if (!visualData || !visualData.nodes || visualData.nodes.length < 2 || isSimulatingHop) return;

    // Pick entry point and trace hops down to nearest neighbor
    const entry = visualData.nodes.find((n) => n.is_entry) || visualData.nodes[0];
    const targets = visualData.nodes.filter((n) => !n.is_tomb && n.id !== entry.id);
    if (targets.length === 0) return;

    const target = targets[Math.floor(Math.random() * targets.length)];

    // Build synthetic greedy path
    animationHopPath = [entry];
    let curr = entry;
    for (let l = curr.layer; l >= 0; l--) {
      // Find neighbor closer to target
      const nbrEdges = visualData.edges.filter((e) => e.source === curr.id && e.layer === l);
      if (nbrEdges.length > 0) {
        const nextEdge = nbrEdges[Math.floor(Math.random() * nbrEdges.length)];
        const nextNode = visualData.nodes.find((n) => n.id === nextEdge.target);
        if (nextNode) {
          animationHopPath.push(nextNode);
          curr = nextNode;
        }
      }
    }
    animationHopPath.push(target);

    hopStep = 0;
    isSimulatingHop = true;

    function step() {
      if (hopStep < animationHopPath.length) {
        hopStep++;
        renderGraph();
        setTimeout(step, 300);
      } else {
        setTimeout(() => {
          isSimulatingHop = false;
          animationHopPath = [];
          renderGraph();
        }, 1500);
      }
    }
    step();
  }

  // Pareto Chart Canvas Renderer
  function renderParetoChart(pareto) {
    const pCanvas = document.getElementById('pareto-canvas');
    if (!pCanvas || !pareto || !pareto.points) return;

    const pCtx = pCanvas.getContext('2d');
    const w = pCanvas.width;
    const h = pCanvas.height;

    pCtx.clearRect(0, 0, w, h);

    // Padding
    const padL = 40;
    const padR = 20;
    const padT = 20;
    const padB = 30;

    const drawW = w - padL - padR;
    const drawH = h - padT - padB;

    // Axes
    pCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    pCtx.lineWidth = 1;
    pCtx.beginPath();
    pCtx.moveTo(padL, padT);
    pCtx.lineTo(padL, h - padB);
    pCtx.lineTo(w - padR, h - padB);
    pCtx.stroke();

    // Axis labels
    pCtx.fillStyle = '#64748b';
    pCtx.font = '9px monospace';
    pCtx.fillText('100%', 5, padT + 8);
    pCtx.fillText('50%', 12, padT + drawH / 2);
    pCtx.fillText('0%', 18, h - padB);
    pCtx.fillText('Latency (ms)', w / 2 - 25, h - 8);

    const points = pareto.points;
    if (points.length === 0) return;

    let maxLat = 0;
    points.forEach((p) => {
      if (p.mean_latency_ms > maxLat) maxLat = p.mean_latency_ms;
    });
    if (maxLat <= 0) maxLat = 5.0;

    function mapX(lat) {
      return padL + (lat / maxLat) * drawW;
    }
    function mapY(rec) {
      return h - padB - rec * drawH;
    }

    // Draw curve
    pCtx.beginPath();
    pCtx.strokeStyle = '#06b6d4';
    pCtx.lineWidth = 2;

    points.forEach((p, idx) => {
      const x = mapX(p.mean_latency_ms);
      const y = mapY(p.mean_recall);
      if (idx === 0) pCtx.moveTo(x, y);
      else pCtx.lineTo(x, y);
    });
    pCtx.stroke();

    // Draw points
    points.forEach((p) => {
      const x = mapX(p.mean_latency_ms);
      const y = mapY(p.mean_recall);

      pCtx.beginPath();
      pCtx.arc(x, y, 4, 0, Math.PI * 2);
      pCtx.fillStyle = p.is_optimal ? '#10b981' : '#0284c7';
      pCtx.fill();
    });

    // Knee point tag
    if (pareto.knee_point) {
      document.getElementById('pareto-knee-tag').textContent = `Knee: efSearch=${pareto.knee_point.ef_search}`;
    }
  }

  function updateParetoPrediction(efVal) {
    if (!currentSnapshot || !currentSnapshot.pareto) return;

    // Call /api/tune
    fetch('/api/tune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ef_search: efVal }),
    })
      .then((res) => res.json())
      .then((data) => {
        document.getElementById('pred-recall').textContent = `${(data.mean_recall * 100).toFixed(1)}%`;
        document.getElementById('pred-latency').textContent = `${data.mean_latency_ms.toFixed(2)} ms`;
      })
      .catch((err) => console.error('Tune error:', err));
  }

  // Degree Histogram Canvas Renderer
  function renderDegreeHistogram(topo) {
    const dCanvas = document.getElementById('degree-canvas');
    if (!dCanvas || !topo || !topo.layers || !topo.layers[0]) return;

    const dCtx = dCanvas.getContext('2d');
    const w = dCanvas.width;
    const h = dCanvas.height;

    dCtx.clearRect(0, 0, w, h);

    const layer0 = topo.layers[0];
    const hist = layer0.degree_histogram || {};
    const keys = Object.keys(hist).map(Number).sort((a, b) => a - b);

    document.getElementById('degree-gini-tag').textContent = `Gini: ${layer0.hub_congestion_index?.toFixed(2) || '0.00'}`;

    if (keys.length === 0) return;

    let maxCount = 0;
    keys.forEach((k) => {
      if (hist[k] > maxCount) maxCount = hist[k];
    });
    if (maxCount <= 0) maxCount = 1;

    const padL = 30;
    const padB = 20;
    const drawW = w - padL - 10;
    const drawH = h - padB - 10;

    const barW = Math.max(8, drawW / keys.length - 4);

    keys.forEach((k, i) => {
      const count = hist[k];
      const barH = (count / maxCount) * drawH;
      const x = padL + i * (barW + 4);
      const y = h - padB - barH;

      dCtx.fillStyle = '#3b82f6';
      dCtx.fillRect(x, y, barW, barH);

      dCtx.fillStyle = '#64748b';
      dCtx.font = '8px monospace';
      dCtx.fillText(`${k}`, x, h - 6);
    });
  }

  // Boot
  document.addEventListener('DOMContentLoaded', init);
})();
