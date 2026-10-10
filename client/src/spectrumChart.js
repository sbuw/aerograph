import uPlot from 'uplot';

/**
 * Плагин интерактивного тултипа для uPlot со стилем glassmorphism.
 */
function tooltipPlugin() {
  let tooltip;

  return {
    hooks: {
      init: [
        u => {
          tooltip = document.createElement('div');
          tooltip.className = 'uplot-tooltip hidden';
          u.over.appendChild(tooltip);

          u.over.addEventListener('mouseleave', () => {
            if (tooltip) tooltip.classList.add('hidden');
          });
        },
      ],
      setCursor: [
        u => {
          const idx = u.cursor.idx;
          if (idx === null || idx === undefined || idx < 0) {
            if (tooltip) tooltip.classList.add('hidden');
            return;
          }

          const val = u.data[1][idx];
          if (val === undefined || isNaN(val)) {
            if (tooltip) tooltip.classList.add('hidden');
            return;
          }

          let badgeHtml = '';
          if (idx === 0) {
            badgeHtml = '<div class="uplot-tooltip-badge trivial">Тривиальная нулевая мода</div>';
          } else if (idx === 1) {
            badgeHtml = '<div class="uplot-tooltip-badge fiedler">Число Фидлера λ₂ (алгебраическая связность)</div>';
          }

          tooltip.innerHTML = `
            <div class="uplot-tooltip-row">Мода k: #${idx + 1}</div>
            <div class="uplot-tooltip-row">Собственное значение λ: <strong>${val.toFixed(5)}</strong></div>
            ${badgeHtml}
          `;

          tooltip.classList.remove('hidden');

          const ptX = u.valToPos(u.data[0][idx], 'x');
          const ptY = u.valToPos(val, 'y');

          const tipWidth = tooltip.offsetWidth || 185;
          const tipHeight = tooltip.offsetHeight || 48;

          let posX = ptX - tipWidth / 2;
          let posY = ptY - tipHeight - 8;

          if (posY < 2) {
            posY = ptY + 12;
          }

          const plotWidth = u.over.clientWidth || 300;
          if (posX < 4) posX = 4;
          if (posX + tipWidth > plotWidth - 4) {
            posX = plotWidth - tipWidth - 4;
          }

          tooltip.style.left = `${posX}px`;
          tooltip.style.top = `${posY}px`;
        },
      ],
    },
  };
}

export class SpectrumChart {
  constructor(containerElement) {
    this.container = containerElement;
    this.chart = null;
  }

  render(eigenvalues, fiedlerValue) {
    if (!eigenvalues || eigenvalues.length === 0) return;

    // Clear previous chart
    this.container.innerHTML = '';

    const width = this.container.clientWidth || 340;
    const height = 130;

    const xValues = eigenvalues.map((_, i) => i + 1);
    const data = [xValues, eigenvalues];

    const opts = {
      width,
      height,
      plugins: [tooltipPlugin()],
      legend: { show: false },
      cursor: {
        show: true,
        points: {
          show: true,
          size: 6,
          fill: '#00f2fe',
        },
      },
      scales: {
        x: { time: false },
        y: { range: [0, Math.max(...eigenvalues) * 1.15] },
      },
      axes: [
        {
          stroke: '#8b949e',
          grid: { stroke: 'rgba(255, 255, 255, 0.05)' },
          ticks: { stroke: '#586069' },
          size: 22,
          font: '10px monospace',
        },
        {
          stroke: '#8b949e',
          grid: { stroke: 'rgba(255, 255, 255, 0.05)' },
          ticks: { stroke: '#586069' },
          size: 34,
          font: '10px monospace',
        },
      ],
      series: [
        { label: 'Мода k' },
        {
          label: 'Собственное значение λ',
          stroke: '#00f2fe',
          width: 2,
          fill: 'rgba(0, 242, 254, 0.1)',
          points: {
            show: true,
            size: 5,
            fill: (u, seriesIdx, dataIdx) => {
              const val = u.data[seriesIdx][dataIdx];
              if (Math.abs(val - fiedlerValue) < 1e-4) {
                return '#f6d365'; // Gold for Fiedler
              }
              if (val < 1e-5) {
                return '#586069'; // Muted for zero modes
              }
              return '#00f2fe';
            },
          },
        },
      ],
    };

    this.chart = new uPlot(opts, data, this.container);
  }

  resize(newWidth) {
    if (this.chart) {
      const width = Math.max(260, newWidth || this.container.clientWidth || 340);
      this.chart.setSize({ width, height: 130 });
    }
  }
}

