/* =====================================================================
   AutoML Studio — main.js  (Mobile-First Interactive Layer)
   ===================================================================== */

'use strict';

// ──────────────────────────────────────────────────────────────────────
// Loading Overlay
// ──────────────────────────────────────────────────────────────────────
function showLoading(title = 'Working…', subtitle = '') {
  let overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.remove();

  overlay = document.createElement('div');
  overlay.className = 'loading-overlay';
  overlay.id = 'loading-overlay';
  overlay.setAttribute('role', 'status');
  overlay.setAttribute('aria-live', 'polite');
  overlay.innerHTML = `
    <div class="spinner loading-spinner-lg"></div>
    <h3>${title}</h3>
    ${subtitle ? `<p>${subtitle}</p>` : ''}
  `;
  document.body.appendChild(overlay);
}

function hideLoading() {
  document.getElementById('loading-overlay')?.remove();
}

// ──────────────────────────────────────────────────────────────────────
// Toast Notifications
// ──────────────────────────────────────────────────────────────────────
function showToast(msg, type = 'info', duration = 4500) {
  let container = document.querySelector('.flash-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'flash-container';
    document.body.appendChild(container);
  }

  const el = document.createElement('div');
  el.className = `flash ${type}`;
  el.setAttribute('role', 'alert');

  const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };
  el.innerHTML = `<span style="flex-shrink:0;">${icons[type] || 'ℹ'}</span><span>${msg}</span>`;

  container.appendChild(el);
  el.addEventListener('click', () => el.remove());
  setTimeout(() => el?.remove(), duration);
}

// ──────────────────────────────────────────────────────────────────────
// Flash message auto-dismiss
// ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach(el => {
    el.addEventListener('click', () => el.remove());
    setTimeout(() => el?.remove(), 5000);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Tabs
// ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      if (!target) return;

      const scope = btn.closest('[data-tab-scope]') || btn.closest('.tabs')?.parentElement || document;

      scope.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      scope.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const panel = scope.querySelector(`.tab-panel[data-tab="${target}"]`);
      if (panel) panel.classList.add('active');
    });
  });
});

// ──────────────────────────────────────────────────────────────────────
// Accordion
// ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.accordion-header').forEach(header => {
    header.addEventListener('click', () => {
      header.closest('.accordion-item')?.classList.toggle('open');
    });
  });
});


// ──────────────────────────────────────────────────────────────────────
// Configure form: validate at least one model selected
// ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const configForm = document.querySelector('form[data-validate-models]');
  configForm?.addEventListener('submit', e => {
    const selected = document.querySelectorAll('.model-card.selected').length;
    if (selected === 0) {
      e.preventDefault();
      showToast('Please select at least one model to train.', 'error');
      return;
    }
    showLoading('Training models…', 'This may take 15–90 seconds depending on dataset size');
  });
});

// ──────────────────────────────────────────────────────────────────────
// Animate probability bars on page load
// ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.proba-fill').forEach(el => {
    const target = el.dataset.pct || '0';
    el.style.width = '0%';
    requestAnimationFrame(() => {
      setTimeout(() => { el.style.width = target + '%'; }, 120);
    });
  });
});

// ──────────────────────────────────────────────────────────────────────
// Chart.js helpers
// ──────────────────────────────────────────────────────────────────────

// Feature Importance (horizontal bar)
function renderFeatureImportanceChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || !data?.labels?.length) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Importance (%)',
        data: data.values,
        backgroundColor: data.labels.map((_, i) =>
          `hsla(${235 + i * 15}, 70%, 65%, 0.75)`
        ),
        borderRadius: 5,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => ` ${c.raw.toFixed(2)}%` } }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#a0a0c8', font: { size: 11 } }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#a0a0c8', font: { size: 10 } }
        }
      }
    }
  });
}

// Distribution histogram
function renderDistributionChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || !data?.length) return;

  const d = data[0];
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: d.labels,
      datasets: [{
        label: d.column,
        data: d.counts,
        backgroundColor: 'rgba(124,111,255,0.4)',
        borderColor: 'rgba(124,111,255,0.8)',
        borderWidth: 1,
        borderRadius: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#a0a0c8', font: { size: 10 }, maxRotation: 45 }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#a0a0c8', font: { size: 10 } }
        }
      }
    }
  });
}

// Model comparison bar
function renderComparisonChart(canvasId, results, metric, label) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || !results?.length) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: results.map(r => r.model),
      datasets: [{
        label,
        data: results.map(r => r[metric] ?? 0),
        backgroundColor: results.map((_, i) =>
          i === 0 ? 'rgba(124,111,255,0.85)' : 'rgba(124,111,255,0.3)'
        ),
        borderColor: results.map((_, i) =>
          i === 0 ? '#7c6fff' : 'rgba(124,111,255,0.4)'
        ),
        borderWidth: 1,
        borderRadius: 5,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#a0a0c8', font: { size: 10 } } },
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#a0a0c8', font: { size: 10 } },
          beginAtZero: true,
          max: ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'r2', 'silhouette'].includes(metric) ? 1 : undefined,
        }
      }
    }
  });
}

// Missing values bar
function renderMissingChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || !data?.length) return;

  const hasMissing = data.filter(d => d.missing > 0);
  if (!hasMissing.length) {
    const wrapper = ctx.closest('.chart-card');
    if (wrapper) {
      wrapper.innerHTML = '<p class="text-success" style="text-align:center;padding:32px;font-weight:600;">✓ No missing values detected</p>';
    }
    return;
  }

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: hasMissing.map(d => d.column),
      datasets: [{
        label: 'Missing %',
        data: hasMissing.map(d => d.missing_pct),
        backgroundColor: hasMissing.map(d =>
          d.missing_pct > 30 ? 'rgba(239,68,68,0.65)' :
          d.missing_pct > 10 ? 'rgba(245,158,11,0.65)' :
          'rgba(56,189,248,0.65)'
        ),
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#a0a0c8', font: { size: 9 }, maxRotation: 45 } },
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#a0a0c8', font: { size: 10 }, callback: v => v + '%' },
          max: 100,
        }
      }
    }
  });
}

// Confusion matrix CSS heatmap
function renderConfusionMatrix(containerId, data) {
  const container = document.getElementById(containerId);
  if (!container || !data?.matrix?.length) return;

  const { matrix, classes } = data;
  const maxVal = Math.max(...matrix.flat(), 1);

  let html = `<div style="overflow-x:auto;"><table style="border-collapse:separate;border-spacing:3px;margin:auto;">
    <tr><th style="padding:4px;color:var(--text-muted);font-size:0.65rem;"></th>`;
  classes.forEach(c => {
    html += `<th style="padding:5px 9px;color:var(--accent-light);font-size:0.68rem;font-weight:700;">${c}</th>`;
  });
  html += '</tr>';

  matrix.forEach((row, i) => {
    html += `<tr><td style="padding:5px 9px;color:var(--accent-light);font-size:0.68rem;font-weight:700;">${classes[i]}</td>`;
    row.forEach((val, j) => {
      const intensity = val / maxVal;
      const isCorrect = i === j;
      const bg = isCorrect
        ? `rgba(124,111,255,${0.12 + intensity * 0.72})`
        : `rgba(239,68,68,${intensity * 0.55})`;
      const txtColor = intensity > 0.5 ? 'white' : 'var(--text-muted)';
      html += `<td style="padding:10px 14px;background:${bg};color:${txtColor};text-align:center;border-radius:5px;font-weight:700;font-size:0.83rem;min-width:44px;">${val}</td>`;
    });
    html += '</tr>';
  });

  html += '</table></div>';
  container.innerHTML = html;
}
