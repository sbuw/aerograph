/**
 * Главный модуль веб-интерфейса.
 * Инициализирует карту Leaflet с чистым темным слоем Esri (без плашек копирайта),
 * оверлей Canvas 2D, график спектра uPlot и связь с API.
 */

import L from 'leaflet';
import { TeleconnectionCanvasOverlay } from './canvasOverlay.js';
import { SpectrumChart } from './spectrumChart.js';

// Состояние приложения
const appState = {
  nodes: [],
  edges: [],
  spectrum: null,
  selectedNode: null,
  incidentEdges: [],
  minR: 0.80,
  scale: 5.0,
  apiUrl: '/api',
};

// Ссылки на элементы DOM
const dom = {
  minRSlider: document.getElementById('min-r-slider'),
  minRVal: document.getElementById('min-r-val'),
  scaleSlider: document.getElementById('scale-slider'),
  scaleVal: document.getElementById('scale-val'),
  btnToggleArcs: document.getElementById('btn-toggle-arcs'),
  btnToggleNodes: document.getElementById('btn-toggle-nodes'),
  btnToggleWavelet: document.getElementById('btn-toggle-wavelet'),
  btnToggleDensity: document.getElementById('btn-toggle-density'),
  btnReset: document.getElementById('btn-reset'),
  btnExport: document.getElementById('btn-export'),
  statNodes: document.getElementById('stat-nodes'),
  statEdges: document.getElementById('stat-edges'),
  statFiedler: document.getElementById('stat-fiedler'),
  nodeInfoEmpty: document.getElementById('node-meta-empty'),
  nodeInfoContent: document.getElementById('node-meta-content'),
  nodeIdVal: document.getElementById('node-id-val'),
  nodeCoordsVal: document.getElementById('node-coords-val'),
  nodeDegreeVal: document.getElementById('node-degree-val'),
  nodeWaveletPeak: document.getElementById('node-wavelet-peak'),
  nodeWaveletEnergy: document.getElementById('node-wavelet-energy'),
  incidentLinksTbody: document.getElementById('incident-links-tbody'),
  spectrumContainer: document.getElementById('spectrum-chart'),
  btnCloseSidebar: document.getElementById('btn-close-sidebar'),
  btnOpenSidebar: document.getElementById('btn-open-sidebar'),
  sidebar: document.getElementById('sidebar'),
  sidebarResizer: document.getElementById('sidebar-resizer'),
  lagFilterGroup: document.getElementById('lag-filter-group'),
  globalTooltip: document.getElementById('global-tooltip'),
  hudHeader: document.querySelector('.hud-header'),
  statNaoVal: document.getElementById('stat-nao-val'),
  statAoVal: document.getElementById('stat-ao-val'),
  tabBtnLinks: document.getElementById('tab-btn-links'),
  tabBtnForecast: document.getElementById('tab-btn-forecast'),
  tabContentLinks: document.getElementById('tab-content-links'),
  tabContentForecast: document.getElementById('tab-content-forecast'),
  fcPredictors: document.getElementById('fc-predictors'),
  fcR2: document.getElementById('fc-r2'),
  fcVerdict: document.getElementById('fc-verdict'),
  fcChartContainer: document.getElementById('forecast-chart-container'),
};

// 1. Инициализация карты Leaflet с отключенной плашкой копирайта (attributionControl: false)
const map = L.map('map', {
  center: [52.0, 10.0],
  zoom: 3,
  minZoom: 2,
  maxZoom: 9,
  zoomControl: false,
  attributionControl: false,
});

L.control.zoom({ position: 'bottomleft' }).addTo(map);

// Esri World Dark Gray Base (Без водяных знаков, без API-ключа)
L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
  {
    maxZoom: 16,
  }
).addTo(map);

// Слой границ и меток Esri World Dark Gray Reference
L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
  {
    maxZoom: 16,
    opacity: 0.7,
  }
).addTo(map);

// 2. Инициализация слоя Canvas 2D и графика спектра
const canvasOverlay = new TeleconnectionCanvasOverlay(map);
canvasOverlay.setMinR(appState.minR);
const spectrumChart = new SpectrumChart(dom.spectrumContainer);

// 3. Загрузка данных через API
async function loadInitialData() {
  try {
    // 3A. Получение спектра Лапласиана
    const resSpec = await fetch(`${appState.apiUrl}/spectrum`);
    if (resSpec.ok) {
      appState.spectrum = await resSpec.json();
      dom.statFiedler.textContent = appState.spectrum.fiedler_value.toFixed(4);
      spectrumChart.render(appState.spectrum.eigenvalues, appState.spectrum.fiedler_value);
    }

    // 3B. Получение узлов сетки
    const resNodes = await fetch(`${appState.apiUrl}/nodes`);
    if (resNodes.ok) {
      appState.nodes = await resNodes.json();
      const activeCount = appState.nodes.filter(n => n.is_active).length;
      dom.statNodes.textContent = `${activeCount} / ${appState.nodes.length}`;
      canvasOverlay.setData({ nodes: appState.nodes });
    }

    // 3C. Получение дуг графа
    const resEdges = await fetch(`${appState.apiUrl}/edges?min_r=0.65&limit=5000`);
    if (resEdges.ok) {
      const edgeData = await resEdges.json();
      appState.edges = edgeData.edges;
      canvasOverlay.setData({ edges: appState.edges });
      updateVisibleEdgesCount();
    }

    // 3D. Получение макроклиматических индексов NAO и AO
    try {
      const resIdx = await fetch(`${appState.apiUrl}/climate-indices`);
      if (resIdx.ok) {
        const dataIdx = await resIdx.json();
        if (dom.statNaoVal && dataIdx.nao) {
          const v = dataIdx.nao.value;
          dom.statNaoVal.textContent = (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2));
        }
        if (dom.statAoVal && dataIdx.ao) {
          const v = dataIdx.ao.value;
          dom.statAoVal.textContent = (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2));
        }
      }
    } catch (e) {
      console.warn('Не удалось загрузить климатические индексы:', e);
    }
  } catch (err) {
    console.error('Ошибка загрузки данных климатического графа:', err);
  }
}

function updateVisibleEdgesCount() {
  const filtered = appState.edges.filter(
    e => e.abs_correlation >= appState.minR && canvasOverlay.matchesLagFilter(e)
  );
  if (!canvasOverlay.showAllGlobalArcs && !canvasOverlay.selectedNodeId) {
    dom.statEdges.textContent = `${Math.min(filtered.length, canvasOverlay.maxGlobalArcs)} из ${filtered.length}`;
  } else {
    dom.statEdges.textContent = `${filtered.length}`;
  }
}

// 4. Интерактивный выбор узла и генерация вейвлета
async function selectNode(node) {
  if (!node) {
    resetSelection();
    return;
  }

  appState.selectedNode = node;
  if (dom.sidebar.classList.contains('collapsed')) {
    openSidebar();
  }
  dom.nodeInfoEmpty.classList.add('hidden');
  dom.nodeInfoContent.classList.remove('hidden');

  dom.nodeIdVal.textContent = `#${node.id}`;
  dom.nodeCoordsVal.textContent = `${node.lat.toFixed(1)}°N, ${node.lon.toFixed(1)}°E`;
  dom.nodeDegreeVal.textContent = Math.round(node.degree);

  // Запрос инцидентных телеконнекций
  try {
    const resLinks = await fetch(`${appState.apiUrl}/node/${node.id}/teleconnections`);
    if (resLinks.ok) {
      const dataLinks = await resLinks.json();
      appState.incidentEdges = dataLinks.edges;
      renderIncidentLinksTable(dataLinks.edges);
    }
  } catch (err) {
    console.error('Ошибка получения связей узла:', err);
  }

  // Расчёт волнового отклика через вейвлет-оператор
  await fetchWaveletField(node.id, appState.scale);
  if (dom.tabBtnForecast && dom.tabBtnForecast.classList.contains('active')) {
    fetchNodeForecast(node.id);
  }
  updateVisibleEdgesCount();
}

async function fetchWaveletField(nodeId, scale) {
  try {
    const resWav = await fetch(`${appState.apiUrl}/wavelet?node_id=${nodeId}&scale=${scale}`);
    if (resWav.ok) {
      const wavData = await resWav.json();
      dom.nodeWaveletPeak.textContent = wavData.peak_value.toFixed(4);
      dom.nodeWaveletEnergy.textContent = wavData.l2_energy.toFixed(4);

      canvasOverlay.setSelectedNode(nodeId, {
        psi: wavData.psi,
        peak: wavData.peak_value,
        centerNodeId: nodeId,
      });
    }
  } catch (err) {
    console.error('Ошибка расчёта вейвлета:', err);
  }
}

function renderIncidentLinksTable(edges) {
  dom.incidentLinksTbody.innerHTML = '';
  if (!edges || edges.length === 0) {
    dom.incidentLinksTbody.innerHTML =
      '<tr><td colspan="4" style="text-align:center; color:#586069; padding:12px;">Нет связей выше выбранного порога</td></tr>';
    return;
  }

  const topEdges = edges.slice(0, 20);
  for (const e of topEdges) {
    const tr = document.createElement('tr');
    const lagStr = e.lag_days > 0 ? `+${e.lag_days} дн.` : `${e.lag_days} дн.`;
    const lagColor =
      e.lag_days > 0
        ? 'var(--color-cyan)'
        : e.lag_days < 0
        ? 'var(--color-amber)'
        : 'var(--color-emerald)';

    tr.innerHTML = `
      <td>#${e.peer_id} (${e.peer_lat.toFixed(1)}°, ${e.peer_lon.toFixed(1)}°)</td>
      <td style="color:#fff; font-weight:600;">${e.abs_correlation.toFixed(3)}</td>
      <td style="color:${lagColor};">${lagStr}</td>
      <td style="color:var(--text-muted);">${Math.round(e.distance_km)} км</td>
    `;
    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => {
      const targetNode = appState.nodes.find(n => n.id === e.peer_id);
      if (targetNode) selectNode(targetNode);
    });
    dom.incidentLinksTbody.appendChild(tr);
  }
}

let currentForecastRequestId = 0;

async function fetchNodeForecast(nodeId) {
  if (!dom.fcVerdict) return;
  const requestId = ++currentForecastRequestId;

  dom.fcVerdict.textContent = '⏳ Расчёт Lead-Lag авторегрессии...';
  if (dom.fcChartContainer) {
    dom.fcChartContainer.innerHTML = '<div style="color:var(--text-muted); font-size:11px; display:flex; align-items:center; justify-content:center; height:100%;">Загрузка расчёта...</div>';
  }

  try {
    const res = await fetch(`${appState.apiUrl}/node/${nodeId}/forecast`);
    if (requestId !== currentForecastRequestId) return; // Игнорируем устаревший ответ

    if (res.ok) {
      const data = await res.json();
      if (dom.fcPredictors) dom.fcPredictors.textContent = data.num_predictors;
      if (dom.fcR2) dom.fcR2.textContent = data.r2.toFixed(2);
      if (dom.fcVerdict) dom.fcVerdict.textContent = data.synoptic_verdict;
      renderForecastChart(data);
    } else {
      dom.fcVerdict.textContent = 'Не удалось построить прогноз для данного узла.';
      if (dom.fcChartContainer) {
        dom.fcChartContainer.innerHTML = `
          <div class="forecast-fallback-msg">
            <span class="fallback-icon">⚠️</span>
            <span>Недостаточно данных для прогноза</span>
          </div>
        `;
      }
    }
  } catch (err) {
    if (requestId !== currentForecastRequestId) return;
    console.error('Ошибка получения прогноза:', err);
    dom.fcVerdict.textContent = 'Ошибка соединения при получении прогноза.';
    if (dom.fcChartContainer) {
      dom.fcChartContainer.innerHTML = `
        <div class="forecast-fallback-msg">
          <span class="fallback-icon">⚠️</span>
          <span>Ошибка сети при получении прогноза</span>
        </div>
      `;
    }
  }
}

function renderForecastChart(fcData) {
  const container = dom.fcChartContainer;
  if (!container) return;

  // Чистая очистка контейнера перед отрисовкой
  container.innerHTML = '';

  const vals = fcData.forecast_values || [];
  const lowers = fcData.confidence_lower || [];
  const uppers = fcData.confidence_upper || [];
  const current = fcData.current_val;
  const numPredictors = fcData.num_predictors || 0;

  if (vals.length === 0 || current === undefined || current === null) {
    container.innerHTML = `
      <div class="forecast-fallback-msg">
        <span class="fallback-icon">⚠️</span>
        <span>Недостаточно опережающих связей для построения прогноза</span>
      </div>
    `;
    return;
  }

  // Если предикторы отсутствуют (num_predictors == 0), показываем чистый fallback UI персистенции
  const isPersistence = (numPredictors === 0);

  const allY = [current, ...vals, ...lowers, ...uppers];
  const minY = Math.min(...allY) - 5;
  const maxY = Math.max(...allY) + 5;
  const rangeY = (maxY - minY) || 1;

  const W = 310;
  const H = 125;
  const padL = 36;
  const padR = 18;
  const padT = 18;
  const padB = 24;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const getX = i => padL + (i / 5) * plotW;
  const getY = val => padT + (1 - (val - minY) / rangeY) * plotH;

  // Confidence band polygon (Day 1..5) (+-1 sigma corridor)
  let bandPoints = '';
  for (let i = 1; i <= 5; i++) {
    bandPoints += `${getX(i).toFixed(1)},${getY(uppers[i - 1]).toFixed(1)} `;
  }
  for (let i = 5; i >= 1; i--) {
    bandPoints += `${getX(i).toFixed(1)},${getY(lowers[i - 1]).toFixed(1)} `;
  }

  // Trajectory line
  const trajPoints = [`${getX(0).toFixed(1)},${getY(current).toFixed(1)}`];
  for (let i = 1; i <= 5; i++) {
    trajPoints.push(`${getX(i).toFixed(1)},${getY(vals[i - 1]).toFixed(1)}`);
  }

  // Dots & Labels
  let dotsSvg = '';
  const lineColor = isPersistence ? '#94a3b8' : '#00f2fe';
  const bandColor = isPersistence ? 'rgba(148, 163, 184, 0.12)' : 'rgba(0, 242, 254, 0.14)';
  const bandStroke = isPersistence ? 'rgba(148, 163, 184, 0.25)' : 'rgba(0, 242, 254, 0.35)';

  for (let i = 1; i <= 5; i++) {
    const px = getX(i);
    const py = getY(vals[i - 1]);
    const dVal = (vals[i - 1] - current);
    const sign = dVal >= 0 ? `+${dVal.toFixed(0)}` : `${dVal.toFixed(0)}`;
    const dotColor = isPersistence ? '#cbd5e1' : '#00f2fe';

    dotsSvg += `
      <circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="3.5" fill="${dotColor}" stroke="#0f172a" stroke-width="1.5" />
      <text x="${px.toFixed(1)}" y="${(py - 7).toFixed(1)}" text-anchor="middle" fill="#94a3b8" font-size="8.5" font-family="monospace">${sign}</text>
    `;
  }
  dotsSvg += `<circle cx="${getX(0).toFixed(1)}" cy="${getY(current).toFixed(1)}" r="3" fill="#f59e0b" />`;

  let xLabels = `<text x="${getX(0).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="#64748b" font-size="8.5">0д</text>`;
  for (let i = 1; i <= 5; i++) {
    xLabels += `<text x="${getX(i).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="#64748b" font-size="8.5">+${i}д</text>`;
  }

  // Badge for Persistence / Teleconnection mode in chart
  const modeBadge = isPersistence
    ? `<text x="${W - padR}" y="${padT - 4}" text-anchor="end" fill="#94a3b8" font-size="8" font-family="sans-serif">Инерционный режим (0 связей)</text>`
    : `<text x="${W - padR}" y="${padT - 4}" text-anchor="end" fill="#00f2fe" font-size="8" font-family="sans-serif">Коридор ±1σ Lead-Lag</text>`;

  container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" style="overflow: visible;">
      <line x1="${padL}" y1="${padT}" x2="${W - padR}" y2="${padT}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="2,2"/>
      <line x1="${padL}" y1="${padT + plotH/2}" x2="${W - padR}" y2="${padT + plotH/2}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="2,2"/>
      <line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.08)"/>
      <line x1="${padL}" y1="${getY(current).toFixed(1)}" x2="${W - padR}" y2="${getY(current).toFixed(1)}" stroke="rgba(245,158,11,0.4)" stroke-dasharray="3,3"/>
      <polygon points="${bandPoints.trim()}" fill="${bandColor}" stroke="${bandStroke}" stroke-dasharray="2,2" stroke-width="0.8" />
      <polyline points="${trajPoints.join(' ')}" fill="none" stroke="${lineColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
      ${modeBadge}
      ${dotsSvg}
      ${xLabels}
    </svg>
  `;
}

function resetSelection() {
  currentForecastRequestId++; // Отменяем ожидание любого текущего прогноза
  appState.selectedNode = null;
  appState.incidentEdges = [];
  dom.nodeInfoEmpty.classList.remove('hidden');
  dom.nodeInfoContent.classList.add('hidden');
  dom.incidentLinksTbody.innerHTML = '';
  if (dom.tabBtnLinks && dom.tabBtnForecast) {
    dom.tabBtnLinks.classList.add('active');
    dom.tabBtnForecast.classList.remove('active');
    dom.tabContentLinks.classList.remove('hidden');
    dom.tabContentForecast.classList.add('hidden');
  }
  if (dom.fcChartContainer) dom.fcChartContainer.innerHTML = '';
  if (dom.fcPredictors) dom.fcPredictors.textContent = '--';
  if (dom.fcR2) dom.fcR2.textContent = '--';
  if (dom.fcVerdict) dom.fcVerdict.textContent = 'Выберите узел на карте для расчёта прогноза.';
  canvasOverlay.setSelectedNode(null, null);
  updateVisibleEdgesCount();
}

// 5. Обработчики событий карты
map.on('click', e => {
  const clickedNode = canvasOverlay.findNodeAt(e.latlng, 22);
  if (clickedNode) {
    selectNode(clickedNode);
  } else {
    resetSelection();
  }
});

map.on('mousemove', e => {
  const hoveredNode = canvasOverlay.findNodeAt(e.latlng, 18);
  if (hoveredNode) {
    map.getContainer().style.cursor = 'pointer';
    canvasOverlay.setHoveredNode(hoveredNode.id);
  } else {
    map.getContainer().style.cursor = '';
    canvasOverlay.setHoveredNode(null);
  }
});

// 6. Обработчики контролов
dom.minRSlider.addEventListener('input', e => {
  const val = parseFloat(e.target.value);
  appState.minR = val;
  dom.minRVal.textContent = val.toFixed(2);
  canvasOverlay.setMinR(val);
  updateVisibleEdgesCount();
});

dom.scaleSlider.addEventListener('input', e => {
  const val = parseFloat(e.target.value);
  appState.scale = val;
  dom.scaleVal.textContent = val.toFixed(1);
  if (appState.selectedNode) {
    fetchWaveletField(appState.selectedNode.id, val);
  }
});

dom.btnToggleArcs.addEventListener('click', () => {
  canvasOverlay.showArcs = !canvasOverlay.showArcs;
  dom.btnToggleArcs.classList.toggle('active', canvasOverlay.showArcs);
  canvasOverlay.render();
});

dom.btnToggleNodes.addEventListener('click', () => {
  canvasOverlay.showNodes = !canvasOverlay.showNodes;
  dom.btnToggleNodes.classList.toggle('active', canvasOverlay.showNodes);
  canvasOverlay.render();
});

dom.btnToggleWavelet.addEventListener('click', () => {
  canvasOverlay.showWavelet = !canvasOverlay.showWavelet;
  dom.btnToggleWavelet.classList.toggle('active', canvasOverlay.showWavelet);
  canvasOverlay.render();
});

if (dom.btnToggleDensity) {
  dom.btnToggleDensity.addEventListener('click', () => {
    canvasOverlay.showAllGlobalArcs = !canvasOverlay.showAllGlobalArcs;
    dom.btnToggleDensity.classList.toggle('active', canvasOverlay.showAllGlobalArcs);
    dom.btnToggleDensity.textContent = canvasOverlay.showAllGlobalArcs ? 'Все связи' : 'Топ-120 связей';
    canvasOverlay.render();
    updateVisibleEdgesCount();
  });
}

dom.btnReset.addEventListener('click', () => {
  // 1. Сброс ползунка «Порог силы связи (|r|)» к дефолтному значению 0.65 с обновлением бейджа
  appState.minR = 0.65;
  dom.minRSlider.value = '0.65';
  dom.minRVal.textContent = '0.65';
  canvasOverlay.setMinR(0.65);

  // 2. Сброс ползунка «Масштаб волны (s)» к дефолтному значению 5.0 с обновлением бейджа
  appState.scale = 5.0;
  dom.scaleSlider.value = '5.0';
  dom.scaleVal.textContent = '5.0';

  // 3. Сброс фильтра лагов в активное состояние «Все»
  canvasOverlay.setLagFilter('all');
  if (dom.lagFilterGroup) {
    const lagButtons = dom.lagFilterGroup.querySelectorAll('button[data-lag]');
    lagButtons.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-lag') === 'all');
    });
  }

  // 4. Сброс тумблеров слоев: Дуги связей — ВКЛ, Узлы сетки — ВКЛ, Поле волн — ВКЛ
  canvasOverlay.showArcs = true;
  dom.btnToggleArcs.classList.add('active');

  canvasOverlay.showNodes = true;
  dom.btnToggleNodes.classList.add('active');

  canvasOverlay.showWavelet = true;
  dom.btnToggleWavelet.classList.add('active');

  if (dom.btnToggleDensity) {
    canvasOverlay.showAllGlobalArcs = false;
    dom.btnToggleDensity.classList.remove('active');
    dom.btnToggleDensity.textContent = 'Топ-120 связей';
  }

  // 5. Снятие фокуса с выбранного узла (selectedNode = null), очистка инцидентных связей и карточки
  resetSelection();

  // 6. Возврат камеры карты (map.setView([52.0, 10.0], 3)) к исходному масштабу
  map.setView([52.0, 10.0], 3);

  // 7. Полная перерисовка Canvas-слоя с пересчётом счетчиков «Видимые связи»
  canvasOverlay.render();
  updateVisibleEdgesCount();
});

// Всплывающее уведомление (Toast)
function showToast(message, type = 'info', duration = 3000) {
  const existing = document.querySelector('.export-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `export-toast toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 350);
  }, duration);
}

// Обработчик экспорта аналитического отчета
if (dom.btnExport) {
  dom.btnExport.addEventListener('click', async () => {
    const originalText = '📥 Экспорт отчёта';
    dom.btnExport.textContent = '⏳ Генерация...';
    dom.btnExport.disabled = true;

    try {
      const response = await fetch('/api/export/report');
      if (!response.ok) {
        throw new Error(`Ошибка сервера: ${response.status} ${response.statusText || ''}`.trim());
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = downloadUrl;
      a.download = 'climate_report.zip';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(downloadUrl);
      a.remove();

      dom.btnExport.textContent = '✅ Скачано!';
      showToast('Аналитический отчет успешно сформирован и скачан!', 'success', 2500);

      setTimeout(() => {
        dom.btnExport.textContent = originalText;
        dom.btnExport.disabled = false;
      }, 2500);
    } catch (err) {
      console.error('Ошибка при скачивании отчета:', err);
      showToast(`Ошибка экспорта: ${err.message}`, 'error', 4000);
      dom.btnExport.textContent = originalText;
      dom.btnExport.disabled = false;
    }
  });
}

// Переключение табов в карточке узла (Связи / Прогноз)
if (dom.tabBtnLinks && dom.tabBtnForecast) {
  dom.tabBtnLinks.addEventListener('click', () => {
    dom.tabBtnLinks.classList.add('active');
    dom.tabBtnForecast.classList.remove('active');
    if (dom.tabContentLinks) dom.tabContentLinks.classList.remove('hidden');
    if (dom.tabContentForecast) dom.tabContentForecast.classList.add('hidden');
  });

  dom.tabBtnForecast.addEventListener('click', () => {
    dom.tabBtnForecast.classList.add('active');
    dom.tabBtnLinks.classList.remove('active');
    if (dom.tabContentForecast) dom.tabContentForecast.classList.remove('hidden');
    if (dom.tabContentLinks) dom.tabContentLinks.classList.add('hidden');
    if (appState.selectedNode) {
      fetchNodeForecast(appState.selectedNode.id);
    }
  });
}

// 6B. Управление сворачиванием и восстановлением боковой панели
function closeSidebar() {
  dom.sidebar.classList.add('collapsed');
  if (dom.btnOpenSidebar) {
    dom.btnOpenSidebar.classList.remove('hidden');
  }
}

function openSidebar() {
  dom.sidebar.classList.remove('collapsed');
  if (dom.btnOpenSidebar) {
    dom.btnOpenSidebar.classList.add('hidden');
  }
  // Пересчитываем размер графика uPlot после возврата панели
  setTimeout(() => {
    spectrumChart.resize(dom.spectrumContainer.clientWidth);
  }, 120);
}

dom.btnCloseSidebar.addEventListener('click', closeSidebar);
if (dom.btnOpenSidebar) {
  dom.btnOpenSidebar.addEventListener('click', openSidebar);
}

// 6C. Динамическое измерение высоты верхнего бара (Multi-row Wrap Tracking)
function updateTopBarHeight() {
  if (dom.hudHeader) {
    const h = dom.hudHeader.offsetHeight;
    document.documentElement.style.setProperty('--topbar-height', `${h}px`);
  }
}

window.addEventListener('resize', updateTopBarHeight);
if (window.ResizeObserver && dom.hudHeader) {
  new ResizeObserver(() => updateTopBarHeight()).observe(dom.hudHeader);
}
updateTopBarHeight();

// 7. Фильтры по типу лага (Все / Опережающие / Запаздывающие / Синхронные)
if (dom.lagFilterGroup) {
  const lagButtons = dom.lagFilterGroup.querySelectorAll('button[data-lag]');
  lagButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      lagButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const lagType = btn.getAttribute('data-lag') || 'all';
      canvasOverlay.setLagFilter(lagType);
      updateVisibleEdgesCount();
    });
  });
}

// 8. Изменение ширины боковой панели мышью (Sidebar Resizer)
if (dom.sidebarResizer) {
  let isResizing = false;

  dom.sidebarResizer.addEventListener('mousedown', e => {
    isResizing = true;
    dom.sidebarResizer.classList.add('resizing');
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  window.addEventListener('mousemove', e => {
    if (!isResizing) return;
    const newWidth = Math.min(750, Math.max(320, window.innerWidth - e.clientX - 16));
    document.documentElement.style.setProperty('--sidebar-width', `${newWidth}px`);
    spectrumChart.resize(dom.spectrumContainer.clientWidth);
  });

  window.addEventListener('mouseup', () => {
    if (isResizing) {
      isResizing = false;
      dom.sidebarResizer.classList.remove('resizing');
      document.body.style.userSelect = '';
      spectrumChart.resize(dom.spectrumContainer.clientWidth);
    }
  });
}

// 9. Глобальные всплывающие подсказки (Tooltips с position: fixed)
const tooltip = dom.globalTooltip;
if (tooltip) {
  function positionTooltip(e) {
    const offset = 14;
    const margin = 10;
    const rect = tooltip.getBoundingClientRect();
    const tooltipWidth = rect.width || 220;
    const tooltipHeight = rect.height || 60;

    let x = e.clientX + offset;
    let y = e.clientY + offset;

    // Предотвращение выхода за правый край экрана
    if (x + tooltipWidth > window.innerWidth - margin) {
      x = e.clientX - tooltipWidth - offset;
    }
    // Предотвращение выхода за левый край экрана
    if (x < margin) {
      x = margin;
    }

    // Предотвращение выхода за нижний край экрана
    if (y + tooltipHeight > window.innerHeight - margin) {
      y = e.clientY - tooltipHeight - offset;
    }
    // Предотвращение выхода за верхний край экрана
    if (y < margin) {
      y = margin;
    }

    tooltip.style.left = `${Math.round(x)}px`;
    tooltip.style.top = `${Math.round(y)}px`;
  }

  document.addEventListener('mouseover', e => {
    const tipTarget = e.target.closest('[data-tooltip]');
    if (tipTarget) {
      const text = tipTarget.getAttribute('data-tooltip');
      if (text) {
        tooltip.textContent = text;
        tooltip.classList.remove('hidden');
        positionTooltip(e);
      }
    }
  });

  document.addEventListener('mousemove', e => {
    if (!tooltip.classList.contains('hidden')) {
      positionTooltip(e);
    }
  });

  document.addEventListener('mouseout', e => {
    const tipTarget = e.target.closest('[data-tooltip]');
    if (tipTarget) {
      tooltip.classList.add('hidden');
    }
  });
}

// Первоначальная загрузка
loadInitialData();
