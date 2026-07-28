const FEATURES = [
    { key: 'amt', label: 'Amount ($)', placeholder: '100.00' },
    { key: 'city_pop', label: 'City Population', placeholder: '50000' },
    { key: 'lat', label: 'Latitude', placeholder: '40.0' },
    { key: 'long', label: 'Longitude', placeholder: '-100.0' },
    { key: 'merch_lat', label: 'Merchant Lat', placeholder: '40.0' },
    { key: 'merch_long', label: 'Merchant Long', placeholder: '-100.0' },
    { key: 'unix_time', label: 'Unix Time', placeholder: '1371816893' },
    { key: 'distance', label: 'Distance', placeholder: '25.0' },
    { key: 'merchant', label: 'Merchant ID', placeholder: '50' },
    { key: 'category', label: 'Category', placeholder: '3' },
    { key: 'hour', label: 'Hour', placeholder: '14' },
    { key: 'day', label: 'Day', placeholder: '15' },
    { key: 'month', label: 'Month', placeholder: '6' },
    { key: 'gender', label: 'Gender', placeholder: '0' },
];

let rocChart = null;
let featureChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadModelInfo();
    loadEvaluation();
});

function initTabs() {
    const btns = document.querySelectorAll('.tab-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            btns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            document.getElementById(`tab-${tabId}`).classList.add('active');
        });
    });
}

function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const icons = { success: 'OK', error: 'X', info: 'i' };
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.innerHTML = `<span>${icons[type] || 'i'}</span> ${message}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

async function loadSample(type) {
    try {
        const res = await fetch(`/api/sample/${type}`);
        const data = await res.json();
        FEATURES.forEach(f => {
            const input = document.getElementById(`feat-${f.key}`);
            if (input && data[f.key] !== undefined) {
                input.value = data[f.key];
            }
        });
        showToast(`Da tao giao dich ${type.toUpperCase()} mau`, 'success');
    } catch (err) {
        showToast('Loi tao sample: ' + err.message, 'error');
    }
}

async function runPredict() {
    const row = {};
    let valid = true;
    FEATURES.forEach(f => {
        const input = document.getElementById(`feat-${f.key}`);
        const val = parseFloat(input.value);
        if (isNaN(val)) {
            input.style.borderColor = 'var(--accent-rose)';
            valid = false;
        } else {
            input.style.borderColor = '';
            row[f.key] = val;
        }
    });

    if (!valid) {
        showToast('Vui long nhap day du gia tri so!', 'error');
        return;
    }

    const btn = document.getElementById('btn-predict');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Dang du doan...';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(row),
        });
        const data = await res.json();

        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        displayPredictResult(data.proba, data.label, data.threshold);
    } catch (err) {
        showToast('Loi du doan: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Du doan giao dich';
    }
}

function displayPredictResult(proba, label, threshold) {
    const resultDiv = document.getElementById('predict-result');
    const isFraud = label === 1;

    drawGauge(proba);

    resultDiv.innerHTML = `
        <div class="result-banner ${isFraud ? 'result-banner--fraud' : 'result-banner--safe'}">
            <div class="result-banner__icon">${isFraud ? 'WARNING' : 'OK'}</div>
            <div>
                <div class="result-banner__title">
                    ${isFraud ? 'CANH BAO: Giao dich co kha nang GIAN LAN!' : 'Giao dich binh thuong'}
                </div>
                <div class="result-banner__desc">
                    Xac suat fraud: <strong>${(proba * 100).toFixed(2)}%</strong> | Threshold: ${(threshold * 100).toFixed(2)}%
                </div>
            </div>
        </div>
    `;
    resultDiv.classList.remove('hidden');
}

function drawGauge(value) {
    const canvas = document.getElementById('gauge-canvas');
    if (!canvas) return;

    const gaugeContainer = document.getElementById('gauge-section');
    gaugeContainer.classList.remove('hidden');

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h - 20;
    const radius = Math.min(cx, cy) - 20;
    const startAngle = Math.PI;
    const endAngle = 2 * Math.PI;
    const valueAngle = startAngle + (endAngle - startAngle) * value;

    ctx.clearRect(0, 0, w, h);

    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.stroke();

    const gradient = ctx.createLinearGradient(0, cy, w, cy);
    if (value < 0.3) {
        gradient.addColorStop(0, '#10b981');
        gradient.addColorStop(1, '#06b6d4');
    } else if (value < 0.6) {
        gradient.addColorStop(0, '#f59e0b');
        gradient.addColorStop(1, '#f97316');
    } else {
        gradient.addColorStop(0, '#f43f5e');
        gradient.addColorStop(1, '#ef4444');
    }

    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, valueAngle);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.stroke();

    ctx.shadowColor = value < 0.3 ? '#10b981' : value < 0.6 ? '#f59e0b' : '#f43f5e';
    ctx.shadowBlur = 15;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, valueAngle);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = '#f1f5f9';
    ctx.font = '700 36px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${(value * 100).toFixed(1)}%`, cx, cy - 20);

    ctx.fillStyle = '#64748b';
    ctx.font = '500 11px Inter, sans-serif';
    ctx.fillText('FRAUD PROBABILITY', cx, cy + 5);

    ctx.font = '400 10px Inter, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'left';
    ctx.fillText('0%', cx - radius - 5, cy + 20);
    ctx.textAlign = 'right';
    ctx.fillText('100%', cx + radius + 5, cy + 20);
}

async function loadModelInfo() {
    try {
        const res = await fetch('/api/model/info');
        const data = await res.json();
        renderModelInfo(data);
    } catch (err) {
        console.error('Failed to load model info:', err);
    }
}

function renderModelInfo(info) {
    const container = document.getElementById('model-info-content');
    if (!container || !info) return;

    const paramsList = Object.entries(info.params || {}).map(([k, v]) =>
        `<li><span class="info-list__key">${k}</span><span class="info-list__val">${v}</span></li>`
    ).join('');

    const featuresList = (info.features || []).map(f =>
        `<span class="badge badge--primary">${f}</span>`
    ).join(' ');

    container.innerHTML = `
        <div class="grid-2">
            <div class="glass-card">
                <div class="card-title">
                    <div class="card-title__icon card-title__icon--primary"></div>
                    Model Parameters
                </div>
                <ul class="info-list">${paramsList}</ul>
            </div>
            <div class="glass-card">
                <div class="card-title">
                    <div class="card-title__icon card-title__icon--cyan"></div>
                    Model Details
                </div>
                <ul class="info-list">
                    <li>
                        <span class="info-list__key">Model Type</span>
                        <span class="info-list__val">${info.model_type || 'LightGBM'}</span>
                    </li>
                    <li>
                        <span class="info-list__key">Threshold</span>
                        <span class="info-list__val">${info.threshold ? info.threshold.toFixed(4) : 'N/A'}</span>
                    </li>
                    <li>
                        <span class="info-list__key">Num Features</span>
                        <span class="info-list__val">${info.num_features || 'N/A'}</span>
                    </li>
                </ul>
                <hr class="section-divider">
                <div class="card-title" style="font-size:14px;">
                    <div class="card-title__icon card-title__icon--warning"></div>
                    Features
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:8px;">
                    ${featuresList}
                </div>
            </div>
        </div>
    `;
}

async function loadEvaluation() {
    try {
        const res = await fetch('/api/model/evaluate');
        const data = await res.json();
        if (data.error) {
            console.warn('Eval not available:', data.error);
            return;
        }
        renderEvaluation(data);
    } catch (err) {
        console.error('Failed to load evaluation:', err);
    }
}

function renderEvaluation(data) {
    renderStatCards(data);
    renderConfusionMatrix(data.confusion_matrix);
    renderClassReport(data.classification_report);
    renderROCChart(data.roc_data);
    renderFeatureImportance(data.feature_importance);
}

function renderStatCards(data) {
    const container = document.getElementById('eval-stats');
    if (!container || !data.classification_report) return;

    const report = data.classification_report;
    const fraud = report['1'] || {};
    const accuracy = report.accuracy || 0;

    container.innerHTML = `
        <div class="stat-card">
            <div class="stat-card__value">${(accuracy * 100).toFixed(1)}%</div>
            <div class="stat-card__label">Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-card__value stat-card__value--success">${((fraud.precision || 0) * 100).toFixed(1)}%</div>
            <div class="stat-card__label">Precision (Fraud)</div>
        </div>
        <div class="stat-card">
            <div class="stat-card__value stat-card__value--warning">${((fraud.recall || 0) * 100).toFixed(1)}%</div>
            <div class="stat-card__label">Recall (Fraud)</div>
        </div>
        <div class="stat-card">
            <div class="stat-card__value stat-card__value--danger">${((fraud['f1-score'] || 0) * 100).toFixed(1)}%</div>
            <div class="stat-card__label">F1-Score (Fraud)</div>
        </div>
    `;
}

function renderConfusionMatrix(cm) {
    const container = document.getElementById('confusion-matrix');
    if (!container || !cm) return;

    const [[tn, fp], [fn, tp]] = cm;
    container.innerHTML = `
        <div class="confusion-matrix">
            <div class="cm-header"></div>
            <div class="cm-header">Predicted 0</div>
            <div class="cm-header">Predicted 1</div>
            <div class="cm-header" style="writing-mode: vertical-rl; transform:rotate(180deg);">Actual 0</div>
            <div class="cm-cell cm-tn">${tn.toLocaleString()}<span class="cm-cell-label">True Neg</span></div>
            <div class="cm-cell cm-fp">${fp.toLocaleString()}<span class="cm-cell-label">False Pos</span></div>
            <div class="cm-header" style="writing-mode: vertical-rl; transform:rotate(180deg);">Actual 1</div>
            <div class="cm-cell cm-fn">${fn.toLocaleString()}<span class="cm-cell-label">False Neg</span></div>
            <div class="cm-cell cm-tp">${tp.toLocaleString()}<span class="cm-cell-label">True Pos</span></div>
        </div>
    `;
}

function renderClassReport(report) {
    const container = document.getElementById('class-report');
    if (!container || !report) return;

    const rows = ['0', '1', 'macro avg', 'weighted avg'].map(key => {
        const r = report[key];
        if (!r) return '';
        const labelMap = { '0': 'Normal', '1': 'Fraud', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg' };
        return `
            <tr>
                <td>${labelMap[key] || key} ${key === '1' ? '<span class="badge badge--danger">Target</span>' : ''}</td>
                <td class="highlight">${(r.precision * 100).toFixed(2)}%</td>
                <td class="highlight">${(r.recall * 100).toFixed(2)}%</td>
                <td class="highlight">${(r['f1-score'] * 100).toFixed(2)}%</td>
                <td>${r.support ? r.support.toLocaleString() : '-'}</td>
            </tr>
        `;
    }).join('');

    container.innerHTML = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function renderROCChart(rocData) {
    const canvas = document.getElementById('roc-chart');
    if (!canvas || !rocData) return;

    if (rocChart) rocChart.destroy();

    const points = rocData.fpr.map((fpr, i) => ({ x: fpr, y: rocData.tpr[i] }));

    rocChart = new Chart(canvas, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: `ROC Curve (AUC = ${rocData.auc.toFixed(4)})`,
                    data: points,
                    showLine: true,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'Random (AUC = 0.5)',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    showLine: true,
                    borderColor: 'rgba(255,255,255,0.15)',
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'False Positive Rate', color: '#94a3b8', font: { family: 'Inter' } },
                    min: 0, max: 1,
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                },
                y: {
                    title: { display: true, text: 'True Positive Rate', color: '#94a3b8', font: { family: 'Inter' } },
                    min: 0, max: 1,
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                },
            },
        },
    });
}

function renderFeatureImportance(fiData) {
    const canvas = document.getElementById('feature-chart');
    if (!canvas || !fiData) return;

    if (featureChart) featureChart.destroy();

    const sorted = Object.entries(fiData).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(e => e[0]);
    const values = sorted.map(e => e[1]);

    const maxVal = Math.max(...values);
    const colors = values.map(v => {
        const ratio = v / maxVal;
        if (ratio > 0.7) return 'rgba(99, 102, 241, 0.8)';
        if (ratio > 0.4) return 'rgba(139, 92, 246, 0.6)';
        return 'rgba(99, 102, 241, 0.35)';
    });

    featureChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Importance',
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.8', '1').replace('0.6', '0.9').replace('0.35', '0.6')),
                borderWidth: 1,
                borderRadius: 6,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                },
                y: {
                    ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
                    grid: { display: false },
                },
            },
        },
    });
}
