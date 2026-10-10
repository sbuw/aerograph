/**
 * Canvas 2D overlay layer synchronized with Leaflet.
 * Renders planetary teleconnection Bézier arcs with antimeridian wrapping,
 * dynamic stroke width scaling, lag-type filtering, viewport culling, grid nodes, and SGWT wavelets.
 */
import L from 'leaflet';

export class TeleconnectionCanvasOverlay {
  constructor(map) {
    this.map = map;
    this.nodes = [];
    this.edges = [];
    this.waveletData = null; // { psi: Array(936), peak: number, centerNodeId: number }
    this.selectedNodeId = null;
    this.hoveredNodeId = null;
    this.minR = 0.80;
    this.maxGlobalArcs = 120; // Maximum arcs to draw in global overview mode
    this.lagFilter = 'all'; // 'all' | 'lead' | 'lag' | 'sync'
    this.showArcs = true;
    this.showNodes = true;
    this.showWavelet = true;
    this.showAllGlobalArcs = false; // Toggle between Top-120 vs All

    // Create Canvas element
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'teleconnection-canvas-overlay';
    this.canvas.style.position = 'absolute';
    this.canvas.style.top = '0';
    this.canvas.style.left = '0';
    this.canvas.style.pointerEvents = 'none';
    this.canvas.style.zIndex = '400';

    this.ctx = this.canvas.getContext('2d');

    // Attach to Leaflet overlay pane
    const pane = this.map.getPanes().overlayPane;
    pane.appendChild(this.canvas);

    // Event listeners for map transformations (синхронное обновление при зуме и панорамировании)
    this.map.on('zoomstart', () => this.render());
    this.map.on('zoom', () => this.render());
    this.map.on('zoomend', () => this.render());
    this.map.on('movestart', () => this.render());
    this.map.on('move', () => this.render());
    this.map.on('moveend', () => this.render());
    this.map.on('resize', () => this.resize());
    this.map.on('viewreset', () => this.render());

    this.resize();
  }

  resize() {
    const size = this.map.getSize();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = size.x * dpr;
    this.canvas.height = size.y * dpr;
    this.canvas.style.width = `${size.x}px`;
    this.canvas.style.height = `${size.y}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
  }

  setData({ nodes, edges }) {
    if (nodes) this.nodes = nodes;
    if (edges) {
      // Sort edges descending by absolute correlation
      this.edges = [...edges].sort((a, b) => b.abs_correlation - a.abs_correlation);
    }
    this.render();
  }

  setMinR(minR) {
    this.minR = minR;
    this.render();
  }

  setLagFilter(filter) {
    this.lagFilter = filter;
    this.render();
  }

  matchesLagFilter(edge) {
    if (this.lagFilter === 'lead') return edge.lag_days > 0;
    if (this.lagFilter === 'lag') return edge.lag_days < 0;
    if (this.lagFilter === 'sync') return edge.lag_days === 0;
    return true;
  }

  /**
   * Проверка видимости ребра (viewport culling) с учетом масштаба карты:
   * 1. Инцидентные дуги активного узла отображаются ВСЕГДА.
   * 2. Источник (source) находится в видимой области.
   * 3. Цель (target) находится в видимой области.
   * 4. Bounding box дуги с запасом на изгиб кривой пересекает границы карты (map.getBounds().intersects).
   * 5. Экранный AABB дуги пересекает видимый viewport Canvas.
   */
  isEdgeVisible(edge) {
    // 1. Для выбранного узла — рисуем ВСЕГДА без проверок на culling
    if (
      this.selectedNodeId !== null &&
      (edge.source_id === this.selectedNodeId || edge.target_id === this.selectedNodeId)
    ) {
      return true;
    }

    const mapBounds = this.map.getBounds();
    const sLat = edge.source_lat;
    const tLat = edge.target_lat;
    const sLon = edge.source_lon;
    const tLon = edge.target_lon;

    // 2. Источник в видимой области (с проверкой периодичности долготы ±360°)
    if (
      mapBounds.contains(L.latLng(sLat, sLon)) ||
      mapBounds.contains(L.latLng(sLat, sLon - 360.0)) ||
      mapBounds.contains(L.latLng(sLat, sLon + 360.0))
    ) {
      return true;
    }

    // 3. Цель в видимой области
    if (
      mapBounds.contains(L.latLng(tLat, tLon)) ||
      mapBounds.contains(L.latLng(tLat, tLon - 360.0)) ||
      mapBounds.contains(L.latLng(tLat, tLon + 360.0))
    ) {
      return true;
    }

    // 4. Bounding Box дуги с запасом на кривизну Безье пересекает границы экрана
    const minLat = Math.min(sLat, tLat);
    const maxLat = Math.min(90.0, Math.max(sLat, tLat) + 25.0); // Изгиб направлен к северу
    const minLon = Math.min(sLon, tLon) - 10.0;
    const maxLon = Math.max(sLon, tLon) + 10.0;

    const bbox = L.latLngBounds([minLat - 5.0, minLon], [maxLat, maxLon]);
    if (mapBounds.intersects(bbox)) {
      return true;
    }
    if (
      mapBounds.intersects(L.latLngBounds([minLat - 5.0, minLon - 360.0], [maxLat, maxLon - 360.0])) ||
      mapBounds.intersects(L.latLngBounds([minLat - 5.0, minLon + 360.0], [maxLat, maxLon + 360.0]))
    ) {
      return true;
    }

    // 5. Проверка экранного пересечения (Screen-space AABB)
    let curSLon = sLon;
    let curTLon = tLon;
    const mapCenterLon = this.map.getCenter().lng;
    while (curSLon - mapCenterLon > 180.0) curSLon -= 360.0;
    while (curSLon - mapCenterLon < -180.0) curSLon += 360.0;
    let dLon = curTLon - curSLon;
    while (dLon > 180.0) { curTLon -= 360.0; dLon = curTLon - curSLon; }
    while (dLon < -180.0) { curTLon += 360.0; dLon = curTLon - curSLon; }

    const p1 = this.map.latLngToContainerPoint([sLat, curSLon]);
    const p2 = this.map.latLngToContainerPoint([tLat, curTLon]);

    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const curvature = Math.min(dist * 0.16, 70);
    const cpY = (p1.y + p2.y) / 2.0 - curvature;

    const minX = Math.min(p1.x, p2.x);
    const maxX = Math.max(p1.x, p2.x);
    const minY = Math.min(p1.y, p2.y, cpY);
    const maxY = Math.max(p1.y, p2.y);

    const size = this.map.getSize();
    const margin = 150;
    if (
      maxX >= -margin &&
      minX <= size.x + margin &&
      maxY >= -margin &&
      minY <= size.y + margin
    ) {
      return true;
    }

    return false;
  }

  setSelectedNode(nodeId, waveletData = null) {
    this.selectedNodeId = nodeId;
    this.waveletData = waveletData;
    this.render();
  }

  setHoveredNode(nodeId) {
    if (this.hoveredNodeId !== nodeId) {
      this.hoveredNodeId = nodeId;
      this.render();
    }
  }

  findNodeAt(latlng, maxPixelDist = 20) {
    if (!this.nodes.length) return null;
    const targetPoint = this.map.latLngToContainerPoint(latlng);
    let closestNode = null;
    let minDistSq = maxPixelDist * maxPixelDist;
    const mapCenterLon = this.map.getCenter().lng;

    for (const node of this.nodes) {
      let nLon = node.lon;
      while (nLon - mapCenterLon > 180.0) nLon -= 360.0;
      while (nLon - mapCenterLon < -180.0) nLon += 360.0;

      const pt = this.map.latLngToContainerPoint([node.lat, nLon]);
      const dx = pt.x - targetPoint.x;
      const dy = pt.y - targetPoint.y;
      const distSq = dx * dx + dy * dy;
      if (distSq < minDistSq) {
        minDistSq = distSq;
        closestNode = node;
      }
    }
    return closestNode;
  }

  render() {
    const size = this.map.getSize();
    this.ctx.clearRect(0, 0, size.x, size.y);

    // Reposition canvas relative to Leaflet pane origin
    const topLeft = this.map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setPosition(this.canvas, topLeft);

    // 1. Draw Wavelet Heatmap Layer
    if (this.showWavelet && this.waveletData && this.waveletData.psi) {
      this.renderWaveletHeatmap();
    }

    // 2. Draw Teleconnection Bézier Arcs
    if (this.showArcs && this.edges.length > 0) {
      this.renderArcs();
    }

    // 3. Draw Grid Nodes
    if (this.showNodes && this.nodes.length > 0) {
      this.renderNodes();
    }
  }

  renderWaveletHeatmap() {
    const psi = this.waveletData.psi;
    const peak = this.waveletData.peak || 1e-4;
    const zoom = this.map.getZoom();
    const radius = Math.max(16, 12 + zoom * 5);
    const mapCenterLon = this.map.getCenter().lng;
    const size = this.map.getSize();

    for (let i = 0; i < this.nodes.length; i++) {
      const val = psi[i];
      if (val === undefined || Math.abs(val) < 1e-4) continue;

      const node = this.nodes[i];
      let nLon = node.lon;
      while (nLon - mapCenterLon > 180.0) nLon -= 360.0;
      while (nLon - mapCenterLon < -180.0) nLon += 360.0;

      const pt = this.map.latLngToContainerPoint([node.lat, nLon]);
      if (
        pt.x < -radius ||
        pt.x > size.x + radius ||
        pt.y < -radius ||
        pt.y > size.y + radius
      ) {
        continue;
      }

      const normIntensity = Math.min(Math.abs(val) / peak, 1.0);
      const alpha = Math.min(normIntensity * 0.75, 0.85);

      const grad = this.ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, radius);
      if (val > 0) {
        // Positive anomaly: Neon Cyan / Electric Blue
        grad.addColorStop(0, `rgba(0, 242, 254, ${alpha})`);
        grad.addColorStop(0.5, `rgba(79, 172, 254, ${alpha * 0.35})`);
        grad.addColorStop(1, 'rgba(0, 242, 254, 0)');
      } else {
        // Negative anomaly: Coral / Crimson
        grad.addColorStop(0, `rgba(255, 107, 107, ${alpha})`);
        grad.addColorStop(0.5, `rgba(255, 82, 82, ${alpha * 0.35})`);
        grad.addColorStop(1, 'rgba(255, 107, 107, 0)');
      }

      this.ctx.fillStyle = grad;
      this.ctx.beginPath();
      this.ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  renderArcs() {
    const hasSelection = this.selectedNodeId !== null;

    // Filter edges by both correlation threshold and lag-type filter
    const filteredEdges = this.edges.filter(
      e => e.abs_correlation >= this.minR && this.matchesLagFilter(e)
    );

    if (hasSelection) {
      // Focal Mode: Highlight incident arcs, dim non-incident arcs
      for (const edge of filteredEdges) {
        const isIncident =
          edge.source_id === this.selectedNodeId || edge.target_id === this.selectedNodeId;
        if (!isIncident && this.isEdgeVisible(edge)) {
          this.drawSingleArc(edge, 0.04, false);
        }
      }
      for (const edge of filteredEdges) {
        const isIncident =
          edge.source_id === this.selectedNodeId || edge.target_id === this.selectedNodeId;
        if (isIncident) {
          const alpha = 0.88 + 0.12 * ((edge.abs_correlation - 0.65) / 0.35);
          this.drawSingleArc(edge, alpha, true);
        }
      }
    } else {
      // Global Overview Mode: Draw top-K strongest arcs that intersect visible viewport
      let edgesToDraw;
      if (this.showAllGlobalArcs) {
        edgesToDraw = filteredEdges.filter(e => this.isEdgeVisible(e));
      } else {
        edgesToDraw = [];
        for (const edge of filteredEdges) {
          if (this.isEdgeVisible(edge)) {
            edgesToDraw.push(edge);
            if (edgesToDraw.length >= this.maxGlobalArcs) break;
          }
        }
      }

      for (const edge of edgesToDraw) {
        const strength = Math.max(0, (edge.abs_correlation - 0.65) / 0.35);
        const alpha = 0.35 + 0.5 * strength;
        this.drawSingleArc(edge, alpha, false);
      }
    }
  }

  drawSingleArc(edge, alpha, isHighlighted) {
    // 1. Antimeridian & Viewport Longitude Normalization
    let sLon = edge.source_lon;
    let tLon = edge.target_lon;

    const mapCenterLon = this.map.getCenter().lng;
    while (sLon - mapCenterLon > 180.0) sLon -= 360.0;
    while (sLon - mapCenterLon < -180.0) sLon += 360.0;

    let dLon = tLon - sLon;
    while (dLon > 180.0) { tLon -= 360.0; dLon = tLon - sLon; }
    while (dLon < -180.0) { tLon += 360.0; dLon = tLon - sLon; }

    const p1 = this.map.latLngToContainerPoint([edge.source_lat, sLon]);
    const p2 = this.map.latLngToContainerPoint([edge.target_lat, tLon]);

    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    // Skip degenerate zero-distance points or invalid values
    if (dist < 2 || !isFinite(dist)) return;

    // 2. Dynamic Stroke Width formula based on absolute correlation:
    // width = 1.0 + Math.pow((edge.abs_correlation - 0.65) / 0.30, 2) * 3.5
    const normR = Math.max(0, Math.min(1.0, (edge.abs_correlation - 0.65) / 0.30));
    let calculatedWidth = 1.0 + Math.pow(normR, 2) * 3.5;
    if (isHighlighted) {
      calculatedWidth += 1.5; // Emphasize incident arcs for selected node
    }

    // 3. Great-Circle Parabolic Arc Curvature (curving toward sub-polar latitudes)
    const midX = (p1.x + p2.x) / 2.0;
    const midY = (p1.y + p2.y) / 2.0;
    const curvature = Math.min(dist * 0.16, 70);
    const cpX = midX;
    const cpY = midY - curvature;

    this.ctx.beginPath();
    this.ctx.moveTo(p1.x, p1.y);
    this.ctx.quadraticCurveTo(cpX, cpY, p2.x, p2.y);

    // 4. Color by Lag Direction
    let strokeColor;
    if (edge.lag_days > 0) {
      // Source leads: Electric Cyan
      strokeColor = isHighlighted
        ? `rgba(0, 242, 254, ${alpha})`
        : `rgba(79, 172, 254, ${alpha})`;
    } else if (edge.lag_days < 0) {
      // Target leads: Warm Amber
      strokeColor = isHighlighted
        ? `rgba(249, 160, 63, ${alpha})`
        : `rgba(246, 211, 101, ${alpha})`;
    } else {
      // Synchronous: Emerald
      strokeColor = `rgba(32, 227, 178, ${alpha})`;
    }

    this.ctx.strokeStyle = strokeColor;
    this.ctx.lineWidth = calculatedWidth;
    this.ctx.lineCap = 'round';

    if (isHighlighted) {
      this.ctx.shadowColor = strokeColor;
      this.ctx.shadowBlur = 10;
    } else {
      this.ctx.shadowBlur = 0;
    }

    this.ctx.stroke();
    this.ctx.shadowBlur = 0;
  }

  renderNodes() {
    const mapCenterLon = this.map.getCenter().lng;
    const size = this.map.getSize();

    for (const node of this.nodes) {
      let nLon = node.lon;
      while (nLon - mapCenterLon > 180.0) nLon -= 360.0;
      while (nLon - mapCenterLon < -180.0) nLon += 360.0;

      const pt = this.map.latLngToContainerPoint([node.lat, nLon]);

      // Пропускаем точки далеко за пределами экрана для оптимизации
      if (pt.x < -30 || pt.x > size.x + 30 || pt.y < -30 || pt.y > size.y + 30) {
        continue;
      }

      const isSelected = node.id === this.selectedNodeId;
      const isHovered = node.id === this.hoveredNodeId;

      if (isSelected) {
        // Selected node: Golden pulsing double ring
        this.ctx.beginPath();
        this.ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(246, 211, 101, 0.4)';
        this.ctx.fill();
        this.ctx.strokeStyle = '#f6d365';
        this.ctx.lineWidth = 2.5;
        this.ctx.stroke();

        this.ctx.beginPath();
        this.ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fill();
      } else if (isHovered) {
        // Hovered node
        this.ctx.beginPath();
        this.ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        this.ctx.fillStyle = '#00f2fe';
        this.ctx.fill();
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 1.5;
        this.ctx.stroke();
      } else if (node.is_active) {
        // Active node
        this.ctx.beginPath();
        this.ctx.arc(pt.x, pt.y, 2.0, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(79, 172, 254, 0.45)';
        this.ctx.fill();
      } else {
        // Isolated node
        this.ctx.beginPath();
        this.ctx.arc(pt.x, pt.y, 1.2, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(88, 96, 105, 0.3)';
        this.ctx.fill();
      }
    }
  }
}
