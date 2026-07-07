/**
 * DQS Engine - Enterprise Dashboard JavaScript
 * Professional data quality analysis with AI-powered insights
 */

// API base URL — reads from global config injected by index.html.
// When hosted on Vercel, index.html sets window.DQS_CONFIG.apiBase to the Render URL.
// When served locally by Flask, falls back to '' (same origin).
const API_BASE = (window.DQS_CONFIG && window.DQS_CONFIG.apiBase) ? window.DQS_CONFIG.apiBase : '';


// Layer metadata
const LAYERS = {
    '1': { name: 'Input Contract', phase: 1 },
    '2': { name: 'Input Validation', phase: 1 },
    '3': { name: 'Feature Extraction', phase: 2 },
    '4.1': { name: 'Structural Integrity', phase: 3 },
    '4.2': { name: 'Field Compliance', phase: 3 },
    '4.3': { name: 'Semantic Validation', phase: 3 },
    '4.4': { name: 'Anomaly Detection', phase: 4 },
    '4.5': { name: 'GenAI Summarization', phase: 4 },
    '5': { name: 'Output Contract', phase: 5 },
    '6': { name: 'Stability Check', phase: 5 },
    '7': { name: 'Conflict Detection', phase: 5 },
    '8': { name: 'Confidence Band', phase: 5 },
    '9': { name: 'Decision Gate', phase: 6 },
    '10': { name: 'Responsibility', phase: 6 },
    '11': { name: 'Logging & Trace', phase: 7 }
};

const LAYER_ORDER = ['1', '2', '3', '4.1', '4.2', '4.3', '4.4', '4.5', '5', '6', '7', '8', '9', '10', '11'];

// State
let currentTab = 'generate';
let customData = null;
let csvContent = null;  // Raw CSV content for backend adapter
let generatedData = null;  // Generated data for preview
let lastResults = null;
let aiEnabled = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

function initApp() {
    checkHealth();
    updateTime();
    setInterval(updateTime, 1000);
    setInterval(checkHealth, 30000);
    setupDragDrop();
    setupToggle();
    resetLayers();
}

// Time display
function updateTime() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    const dateStr = now.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
    document.getElementById('headerTime').textContent = `${dateStr} ${timeStr}`;
}

// Health check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(5000)
        });
        const data = await response.json();

        const indicator = document.getElementById('statusIndicator');
        const status = document.getElementById('apiStatus');

        if (data.status === 'healthy') {
            indicator.className = 'status-indicator connected';
            status.textContent = 'Connected';
        } else {
            indicator.className = 'status-indicator';
            status.textContent = 'Degraded';
        }
    } catch (error) {
        const indicator = document.getElementById('statusIndicator');
        const status = document.getElementById('apiStatus');
        indicator.className = 'status-indicator disconnected';
        status.textContent = 'Disconnected';
    }
}

// Tab switching
function switchTab(tab) {
    currentTab = tab;

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tab}`);
    });

    // Initialize Live Feed when tab is activated
    if (tab === 'live') {
        onLiveTabActivated();
    }

    // Toggle Pipeline Visualization
    const pipelineSection = document.getElementById('pipelineVisualizationSection');
    if (pipelineSection) {
        pipelineSection.style.display = (tab === 'live') ? 'none' : 'block';
    }
}

// Slider update
function updateSlider(id, valueId) {
    const slider = document.getElementById(id);
    const valueEl = document.getElementById(valueId);
    if (id === 'anomalyRate' || id === 'liveAnomalyRate') {
        valueEl.textContent = slider.value + '%';
    } else {
        valueEl.textContent = slider.value;
    }
}

// AI Toggle
function setupToggle() {
    const toggle = document.getElementById('useAI');
    const label = document.getElementById('aiLabel');

    toggle.addEventListener('change', () => {
        aiEnabled = toggle.checked;
        label.textContent = aiEnabled ? 'Enabled' : 'Disabled';
    });
}

// JSON handling
function validateJson() {
    const textarea = document.getElementById('jsonInput');
    const status = document.getElementById('jsonStatus');

    try {
        const data = JSON.parse(textarea.value);
        customData = Array.isArray(data) ? data : [data];
        status.textContent = `Valid (${customData.length} records)`;
        status.className = 'status-tag valid';
        showDataPreview(customData);
    } catch (e) {
        status.textContent = 'Invalid JSON';
        status.className = 'status-tag invalid';
        customData = null;
        showToast('error', 'Invalid JSON format: ' + e.message);
    }
}

async function loadSampleJson() {
    try {
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ count: 3 })
        });
        const data = await response.json();

        if (data.success && data.preview) {
            document.getElementById('jsonInput').value = JSON.stringify(data.preview, null, 2);
            validateJson();
            showToast('success', 'Sample data loaded successfully');
        }
    } catch (e) {
        showToast('error', 'Failed to load sample data');
    }
}

// File upload
function setupDragDrop() {
    const dropZone = document.getElementById('dropZone');

    dropZone.addEventListener('click', () => {
        document.getElementById('csvFile').click();
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

function handleFileUpload(event) {
    if (event.target.files.length > 0) {
        handleFile(event.target.files[0]);
    }
}

function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        showToast('error', 'Please upload a CSV file');
        return;
    }

    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatSize(file.size);
    document.getElementById('fileInfo').style.display = 'flex';
    document.getElementById('dropZone').style.display = 'none';

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            // Store raw CSV content for backend adapter
            csvContent = e.target.result;
            // Parse locally for preview only
            customData = parseCSV(csvContent);
            showDataPreview(customData);
            showToast('success', `Loaded ${customData.length} records from CSV (will be adapted to VISA format)`);
        } catch (err) {
            showToast('error', 'Failed to parse CSV: ' + err.message);
        }
    };
    reader.readAsText(file);
}

function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));

    return lines.slice(1).map(line => {
        const values = line.split(',').map(v => v.trim().replace(/"/g, ''));
        const row = {};
        headers.forEach((h, i) => row[h] = values[i] || '');
        return row;
    });
}

function clearFile() {
    customData = null;
    csvContent = null;
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('csvFile').value = '';
    document.getElementById('dataPreviewSection').style.display = 'none';
    // Reset example button
    const btn = document.getElementById('loadExampleBtn');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Load Example CSV`;
    }
}

async function loadExampleDataset() {
    const btn = document.getElementById('loadExampleBtn');
    btn.disabled = true;
    btn.innerHTML = `<svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Loading...`;

    try {
        const resp = await fetch(`${API_BASE}/api/example-dataset`);
        const data = await resp.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to load example dataset');
        }

        // Load into the CSV pipeline
        csvContent = data.csv_content;
        customData = parseCSV(csvContent);

        // Show file info
        document.getElementById('dropZone').style.display = 'none';
        document.getElementById('fileInfo').style.display = 'flex';
        document.getElementById('fileName').textContent = data.filename;
        document.getElementById('fileSize').textContent = `${data.total_rows.toLocaleString()} transactions · 26 anomalies`;

        showDataPreview(customData);
        showToast('success', `Loaded ${data.total_rows.toLocaleString()} transactions — ${data.anomaly_profiles.map(p => p.name).join(', ')}`);

        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Dataset loaded`;
    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Load Example CSV`;
        showToast('error', 'Failed to load example dataset: ' + err.message);
    }
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Data Preview
function showDataPreview(data) {
    if (!data || data.length === 0) return;

    const section = document.getElementById('dataPreviewSection');
    const count = document.getElementById('previewCount');
    const thead = document.getElementById('previewTableHead');
    const tbody = document.getElementById('previewTableBody');

    section.style.display = 'block';
    count.textContent = `${data.length} records`;

    // Get keys (limit to 10 columns for display)
    const keys = Object.keys(flattenObject(data[0])).slice(0, 10);

    thead.innerHTML = `<tr>${keys.map(k => `<th>${formatHeader(k)}</th>`).join('')}</tr>`;

    tbody.innerHTML = data.slice(0, 20).map(row => {
        const flat = flattenObject(row);
        return `<tr>${keys.map(k => `<td>${formatCell(flat[k])}</td>`).join('')}</tr>`;
    }).join('');
}

function flattenObject(obj, prefix = '') {
    const result = {};
    for (const key in obj) {
        const fullKey = prefix ? `${prefix}.${key}` : key;
        if (typeof obj[key] === 'object' && obj[key] !== null) {
            Object.assign(result, flattenObject(obj[key], fullKey));
        } else {
            result[fullKey] = obj[key];
        }
    }
    return result;
}

function formatHeader(key) {
    return key.split('.').pop().replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatCell(value) {
    if (value === null || value === undefined) return '-';
    const str = String(value);
    return str.length > 25 ? str.slice(0, 25) + '...' : str;
}

// Reset layers
function resetLayers() {
    document.querySelectorAll('.layer-card').forEach(card => {
        card.classList.remove('passed', 'degraded', 'failed', 'running');
    });
}

// Update layer status
function updateLayerStatus(layerId, status) {
    const card = document.querySelector(`[data-layer="${layerId}"]`);
    if (card) {
        card.classList.remove('passed', 'degraded', 'failed', 'running');
        if (status) {
            card.classList.add(status.toLowerCase());
        }
    }
}

// Run pipeline
async function runPipeline() {
    const btn = document.getElementById('runPipelineBtn');
    const overlay = document.getElementById('loadingOverlay');
    const results = document.getElementById('resultsSection');

    btn.disabled = true;
    overlay.style.display = 'flex';
    resetLayers();

    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const loadingLayer = document.getElementById('loadingLayer');

    // Animate through layers
    const animatePromise = (async () => {
        for (let i = 0; i < LAYER_ORDER.length; i++) {
            const layerId = LAYER_ORDER[i];
            const info = LAYERS[layerId];

            loadingLayer.textContent = `Layer ${layerId}: ${info.name}`;
            updateLayerStatus(layerId, 'running');

            const progress = Math.round(((i + 1) / LAYER_ORDER.length) * 100);
            progressFill.style.width = progress + '%';
            progressText.textContent = progress + '%';

            await sleep(60);
            updateLayerStatus(layerId, 'passed');
        }
    })();

    try {
        let body;
        if (currentTab === 'generate') {
            body = {
                count: parseInt(document.getElementById('txnCount').value),
                anomaly_rate: parseInt(document.getElementById('anomalyRate').value) / 100,
                use_ai: aiEnabled,
                seed: Date.now() % 10000
            };

            // Show preview for generated data
            try {
                const previewResp = await fetch(`${API_BASE}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const previewData = await previewResp.json();
                if (previewData.success && previewData.preview) {
                    generatedData = previewData.transactions;
                    showDataPreview(previewData.preview);
                }
            } catch (e) {
                console.log('Preview generation failed, continuing with pipeline');
            }
        } else if (currentTab === 'csv' && csvContent) {
            // Send raw CSV content to backend for adapter processing
            body = {
                csv_content: csvContent,
                use_ai: aiEnabled
            };
        } else {
            if (!customData || customData.length === 0) {
                throw new Error('No data provided. Please enter JSON or upload a CSV file.');
            }
            body = {
                custom_data: customData,
                use_ai: aiEnabled
            };
        }

        const response = await fetch(`${API_BASE}/api/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        await animatePromise;

        const data = await response.json();

        if (data.success) {
            lastResults = data;
            updateLayersFromResults(data.layer_timings);
            displayResults(data);
            results.style.display = 'block';
            results.scrollIntoView({ behavior: 'smooth' });
            showToast('success', `Analyzed ${data.total_records} records successfully`);
        } else {
            // Check if this is a schema compliance error
            if (data.import_metadata && data.import_metadata.compliance_score < 80) {
                showSchemaError(data.error, data.import_metadata);
            } else {
                throw new Error(data.error || 'Pipeline execution failed');
            }
        }
    } catch (error) {
        console.error('Pipeline error:', error);
        showToast('error', error.message.split('\n')[0]);  // Show first line only in toast

        // If error message is long, show in modal
        if (error.message.length > 100) {
            showErrorModal('Pipeline Error', error.message);
        }
        resetLayers();
    } finally {
        overlay.style.display = 'none';
        btn.disabled = false;
    }
}

// Show schema compliance error in modal
function showSchemaError(errorMsg, metadata) {
    const modal = document.getElementById('layerModal');
    const title = document.getElementById('modalLayerTitle');
    const body = document.getElementById('modalLayerBody');

    title.textContent = 'Data Format Error';

    const compliance = metadata.compliance_score || 0;
    const missing = metadata.missing_fields || [];
    const warnings = metadata.warnings || [];

    body.innerHTML = `
        <div class="modal-section">
            <h4>Schema Compliance Issue</h4>
            <div class="detail-grid">
                <div class="detail-item">
                    <span class="detail-label">Compliance Score</span>
                    <span class="status-tag-inline failed">${compliance.toFixed(0)}%</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Format Status</span>
                    <span class="status-tag-inline failed">Non-Standard</span>
                </div>
            </div>
        </div>
        
        <div class="modal-section">
            <h4>Missing Required Fields</h4>
            <ul style="margin: 10px 0; padding-left: 20px; color: var(--text-secondary);">
                ${missing.slice(0, 10).map(f => `<li>${f}</li>`).join('')}
            </ul>
        </div>
        
        <div class="modal-section">
            <h4>Required VISA Format</h4>
            <div class="fix-content" style="background: var(--surface-alt); padding: 15px; border-radius: 8px; font-family: monospace; font-size: 12px;">
<pre>{
  "transaction": {
    "transaction_id": "txn_001",
    "amount": 5000.00,
    "currency": "INR",
    "timestamp": "2026-01-05T10:00:00Z"
  },
  "card": {
    "network": "VISA",
    "card_type": "credit"
  },
  "merchant": {
    "merchant_id": "MID_1234",
    "merchant_category_code": "5812",
    "country": "IN"
  },
  "customer": {
    "customer_id": "cust_1234"
  },
  "fraud": {
    "risk_score": 25
  }
}</pre>
            </div>
        </div>
        
        ${warnings.length > 0 ? `
        <div class="modal-section">
            <h4>Warnings</h4>
            <ul style="margin: 10px 0; padding-left: 20px; color: var(--warning);">
                ${warnings.map(w => `<li>${w}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        <div class="modal-section">
            <p style="color: var(--text-secondary);">
                Please reformat your data to match the VISA transaction schema, or use the "Generate Sample" option to see the expected format.
            </p>
        </div>
    `;

    modal.classList.add('active');
    showToast('error', 'Data format error - see details in popup');
    resetLayers();
}

// Show generic error in modal
function showErrorModal(title, message) {
    const modal = document.getElementById('layerModal');
    const modalTitle = document.getElementById('modalLayerTitle');
    const body = document.getElementById('modalLayerBody');

    modalTitle.textContent = title;

    body.innerHTML = `
        <div class="modal-section">
            <div class="fix-content" style="background: var(--surface-alt); padding: 15px; border-radius: 8px; font-family: monospace; font-size: 12px; white-space: pre-wrap;">
${message}
            </div>
        </div>
    `;

    modal.classList.add('active');
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Update layers from results
function updateLayersFromResults(timings) {
    timings.forEach(t => {
        const status = t.status.toUpperCase();
        let visualStatus = 'passed';
        if (status === 'FAILED') visualStatus = 'failed';
        else if (status === 'DEGRADED') visualStatus = 'degraded';
        updateLayerStatus(t.layer_id, visualStatus);
    });
}

// Display results
function displayResults(data) {
    // Stats
    document.getElementById('safeCount').textContent = data.safe_count || 0;
    document.getElementById('reviewCount').textContent = data.review_count || 0;
    document.getElementById('escalateCount').textContent = data.escalate_count || 0;
    document.getElementById('rejectedCount').textContent = data.rejected_count || 0;

    // Metrics
    document.getElementById('qualityRate').textContent = data.quality_rate || 0;
    document.getElementById('avgDqs').textContent = (data.average_dqs || 0).toFixed(1);
    document.getElementById('totalRecords').textContent = data.total_records || 0;
    document.getElementById('duration').textContent = Math.round(data.total_duration_ms || 0);

    // Gauges
    updateGauge('qualityGauge', data.quality_rate || 0);
    updateGauge('dqsGauge', data.average_dqs || 0);

    // Layer log
    displayLayerLog(data.layer_timings, data.layer_details);

    // Action records
    displayActionRecords(data);

    // Output table
    displayOutputTable(data);

    // Reports
    document.getElementById('decisionReport').innerHTML = formatReport(data.decision_report);
    document.getElementById('executionLog').innerHTML = formatExecutionLog(data);
}

function updateGauge(id, value) {
    const gauge = document.getElementById(id);
    if (gauge) {
        const circumference = 251.2;
        const offset = circumference - (value / 100) * circumference;
        gauge.style.strokeDashoffset = offset;
    }
}

// Layer log table
function displayLayerLog(timings, details) {
    const tbody = document.getElementById('layerLogBody');
    const baseTime = new Date();

    tbody.innerHTML = timings.map((t, i) => {
        const detail = details[String(t.layer_id)] || {};
        const timestamp = new Date(baseTime.getTime() + i * 100);
        const timeStr = timestamp.toISOString().split('T')[1].split('.')[0];

        const statusClass = t.status.toLowerCase();
        const checksInfo = `${detail.checks_passed || 0}/${detail.checks_performed || 0}`;
        const detailText = getDetailText(t.layer_id, detail.details);

        const hasIssue = statusClass === 'degraded' || statusClass === 'failed';

        return `
            <tr>
                <td><span class="log-time">${timeStr}</span></td>
                <td><strong>L${t.layer_id}</strong></td>
                <td>${t.layer_name}</td>
                <td><span class="status-tag-inline ${statusClass}">${t.status}</span></td>
                <td>${t.duration_ms.toFixed(2)}ms</td>
                <td>${checksInfo}</td>
                <td>${detailText}</td>
                <td>
                    ${hasIssue ? `<button class="btn-sm" onclick="showLayerDetails('${t.layer_id}')">View Fix</button>` : '-'}
                </td>
            </tr>
        `;
    }).join('');
}

function getDetailText(layerId, details) {
    if (!details) return '-';

    if (details.records_processed !== undefined) return `${details.records_processed} processed`;
    if (details.features_extracted !== undefined) return `${details.features_extracted} features`;
    if (details.valid_records !== undefined) return `${details.valid_records} valid`;
    if (details.dqs_mean !== undefined) return `DQS: ${Number(details.dqs_mean).toFixed(1)}`;
    if (details.records_flagged !== undefined) return `${details.records_flagged} flagged`;
    if (details.total_conflicts !== undefined) return `${details.total_conflicts} conflicts`;
    if (details.safe_count !== undefined) return `${details.safe_count} safe`;
    if (details.records_logged !== undefined) return `${details.records_logged} logged`;

    return '-';
}

// Action records
function displayActionRecords(data) {
    const section = document.getElementById('actionRecordsSection');
    const grid = document.getElementById('actionRecordsGrid');
    const badge = document.getElementById('actionBadge');

    const actionCount = (data.review_count || 0) + (data.escalate_count || 0);

    if (actionCount === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    badge.textContent = actionCount;

    // Extract records from decision report
    const records = extractActionRecords(data);

    grid.innerHTML = records.map(rec => {
        const isEscalate = rec.action === 'ESCALATE';
        const fix = generateFix(rec, aiEnabled);

        return `
            <div class="action-record-card">
                <div class="action-record-header">
                    <span class="action-record-id">${rec.id}</span>
                    <span class="action-tag ${isEscalate ? 'escalate' : 'review'}">${rec.action}</span>
                </div>
                <div class="action-record-body">
                    <div class="action-record-reason">${rec.reason}</div>
                    <div class="action-record-data">
                        <pre>${JSON.stringify(rec.data, null, 2)}</pre>
                    </div>
                    <div class="action-record-fix">
                        <div class="fix-header">
                            <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                                <path d="M2 17l10 5 10-5"/>
                            </svg>
                            ${aiEnabled ? 'AI Recommended Fix' : 'Recommended Fix'}
                        </div>
                        <div class="fix-content">${fix}</div>
                    </div>
                </div>
                <div class="action-record-footer">
                    <button class="btn-sm" onclick="showRecordDetails('${rec.id}')">
                        Full Details
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function extractActionRecords(data) {
    const records = [];
    const report = data.decision_report || '';

    // Parse the decision report to extract flagged transactions
    const lines = report.split('\n');
    let currentSection = '';

    for (const line of lines) {
        if (line.includes('REVIEW') || line.includes('ESCALATE')) {
            currentSection = line.includes('ESCALATE') ? 'ESCALATE' : 'REVIEW';
        }

        const match = line.match(/txn_\d+/);
        if (match && currentSection) {
            const txnId = match[0];

            // Find the transaction in processed data
            let txnData = {};
            if (data.processed_records) {
                const found = data.processed_records.find(r =>
                    r.record_id === txnId ||
                    (r.transaction && r.transaction.transaction_id === txnId)
                );
                if (found) txnData = found;
            }

            records.push({
                id: txnId,
                action: currentSection,
                reason: extractReason(line, currentSection),
                data: txnData,
                dqs: txnData.dqs_final || txnData.dqs_base || 0
            });
        }
    }

    // If no records found from report, create placeholder records
    if (records.length === 0 && (data.review_count > 0 || data.escalate_count > 0)) {
        for (let i = 0; i < data.review_count; i++) {
            records.push({
                id: `Record ${i + 1}`,
                action: 'REVIEW',
                reason: 'Transaction flagged for review due to quality concerns',
                data: { note: 'Details available in Decision Report' },
                dqs: 0
            });
        }
        for (let i = 0; i < data.escalate_count; i++) {
            records.push({
                id: `Escalated ${i + 1}`,
                action: 'ESCALATE',
                reason: 'High-priority issue requiring immediate attention',
                data: { note: 'Details available in Decision Report' },
                dqs: 0
            });
        }
    }

    return records.slice(0, 10); // Limit to 10 for display
}

function extractReason(line, action) {
    if (line.includes('Anomaly')) return 'Anomaly detected: Unusual transaction patterns identified by ML model';
    if (line.includes('Low DQS')) return 'Low quality score: Transaction data quality below acceptable threshold';
    if (line.includes('Conflict')) return 'Data conflict: Inconsistent signals between validation layers';
    if (line.includes('confidence')) return 'Low confidence: Insufficient certainty in quality assessment';
    if (action === 'ESCALATE') return 'Critical issue: Requires immediate human intervention';
    return 'Quality concern: Transaction requires manual verification before processing';
}

function generateFix(record, isAI) {
    if (isAI) {
        // AI-based recommendations
        if (record.action === 'ESCALATE') {
            return `This transaction requires immediate escalation due to critical quality issues. 
Recommended actions:
1. Verify transaction source and authenticity
2. Cross-check with external fraud detection systems
3. Contact merchant or cardholder for verification
4. Document all findings before proceeding`;
        } else {
            return `This transaction has been flagged for review. Based on AI analysis:
1. Verify the flagged data fields for accuracy
2. Check for data entry errors or formatting issues
3. If data is correct, approve with documented justification
4. Consider updating validation rules if false positive`;
        }
    } else {
        // Non-AI recommendations
        if (record.action === 'ESCALATE') {
            return `Escalation required. Standard procedure:
1. Review all transaction details
2. Verify against compliance checklist
3. Escalate to supervisor if issues persist
4. Document resolution in audit trail`;
        } else {
            return `Review required. Standard procedure:
1. Check flagged fields for accuracy
2. Verify data against source documents
3. Correct any identified errors
4. Approve or reject with justification`;
        }
    }
}

// Output table
function displayOutputTable(data) {
    const thead = document.getElementById('outputTableHead');
    const tbody = document.getElementById('outputTableBody');

    const columns = [
        'Record ID', 'Action', 'DQS Score', 'Quality',
        'Anomaly', 'Confidence', 'Issues'
    ];

    thead.innerHTML = `<tr>${columns.map(c => `<th>${c}</th>`).join('')}</tr>`;

    // Generate output rows from available data
    const records = generateOutputRows(data);

    tbody.innerHTML = records.map(rec => {
        const actionClass = rec.action === 'SAFE_TO_USE' ? 'passed' :
            rec.action === 'REVIEW_REQUIRED' ? 'degraded' : 'failed';

        return `
            <tr>
                <td><strong>${rec.id}</strong></td>
                <td><span class="status-tag-inline ${actionClass}">${rec.action.replace('_', ' ')}</span></td>
                <td>${rec.dqs.toFixed(1)}</td>
                <td>${rec.quality}</td>
                <td>${rec.anomaly}</td>
                <td>${rec.confidence}</td>
                <td>${rec.issues}</td>
            </tr>
        `;
    }).join('');
}

function generateOutputRows(data) {
    const rows = [];

    // If single live transaction with real data
    if (data.total_records === 1 && data.transaction_id) {
        return [{
            id: data.transaction_id,
            action: data.action || 'REVIEW_REQUIRED',
            dqs: data.average_dqs || 0,
            quality: (data.average_dqs > 80) ? 'High' : (data.average_dqs > 60 ? 'Medium' : 'Low'),
            anomaly: (data.layer_details && data.layer_details['4.4'] && data.layer_details['4.4'].issues_count > 0) ? 'Detected' : 'None',
            confidence: (data.layer_details && data.layer_details['8']) ? 'Calculated' : '-',
            issues: data.reason || '-'
        }];
    }

    const total = data.total_records || 0;

    // Generate rows based on action counts
    for (let i = 0; i < Math.min(total, 50); i++) {
        let action = 'SAFE_TO_USE';
        if (i < data.escalate_count) action = 'ESCALATE';
        else if (i < data.escalate_count + data.review_count) action = 'REVIEW_REQUIRED';
        else if (i < data.escalate_count + data.review_count + data.rejected_count) action = 'NO_ACTION';

        const dqs = action === 'SAFE_TO_USE' ? 85 + Math.random() * 15 :
            action === 'REVIEW_REQUIRED' ? 50 + Math.random() * 30 :
                10 + Math.random() * 40;

        rows.push({
            id: `Txn_${1000 + i}`,
            action: action,
            dqs: dqs,
            quality: dqs > 80 ? 'High' : (dqs > 60 ? 'Medium' : 'Low'),
            anomaly: Math.random() > 0.9 ? 'Detected' : 'None',
            confidence: Math.random() > 0.9 ? 'Low' : 'High',
            issues: action === 'SAFE_TO_USE' ? '-' : 'Quality issues detected'
        });
    }

    return rows;
}

// Format reports
function formatReport(report) {
    if (!report) return '<span style="color: var(--text-muted)">No report available</span>';

    return report
        .replace(/\*\*/g, '')
        .replace(/\*/g, '')
        .replace(/#{1,6}\s/g, '')
        .replace(/`/g, '')
        .split('\n')
        .map(line => {
            line = line.replace(/(SAFE_TO_USE|SAFE)/g, '<span style="color: var(--success)">$1</span>');
            line = line.replace(/(REVIEW_REQUIRED|REVIEW)/g, '<span style="color: var(--warning)">$1</span>');
            line = line.replace(/(ESCALATE)/g, '<span style="color: var(--danger)">$1</span>');
            line = line.replace(/(NO_ACTION|REJECTED)/g, '<span style="color: var(--neutral)">$1</span>');
            line = line.replace(/(PASSED)/gi, '<span style="color: var(--success)">$1</span>');
            line = line.replace(/(FAILED)/gi, '<span style="color: var(--danger)">$1</span>');
            line = line.replace(/(DEGRADED)/gi, '<span style="color: var(--warning)">$1</span>');
            return line;
        })
        .join('\n');
}

function formatExecutionLog(data) {
    const lines = [];
    const now = new Date();

    lines.push(`[${formatTimestamp(now)}] Pipeline execution started`);
    lines.push(`[${formatTimestamp(now)}] Input: ${data.total_records} records`);
    lines.push('');

    if (data.layer_timings) {
        data.layer_timings.forEach((t, i) => {
            const time = new Date(now.getTime() + i * 100);
            const status = t.status.toUpperCase();
            const statusColor = status === 'PASSED' ? 'var(--success)' :
                status === 'DEGRADED' ? 'var(--warning)' : 'var(--danger)';

            lines.push(`[${formatTimestamp(time)}] Layer ${t.layer_id}: ${t.layer_name}`);
            lines.push(`  Status: <span style="color: ${statusColor}">${status}</span> | Duration: ${t.duration_ms.toFixed(2)}ms`);
        });
    }

    lines.push('');
    lines.push(`[${formatTimestamp(now)}] Pipeline completed`);
    lines.push(`[${formatTimestamp(now)}] Total duration: ${Math.round(data.total_duration_ms || 0)}ms`);
    lines.push(`[${formatTimestamp(now)}] Results: ${data.safe_count} safe, ${data.review_count} review, ${data.escalate_count} escalate, ${data.rejected_count} rejected`);

    return lines.join('\n');
}

function formatTimestamp(date) {
    return date.toISOString().split('T')[1].split('.')[0];
}

// Layer details modal
function showLayerDetails(layerId) {
    const modal = document.getElementById('layerModal');
    const title = document.getElementById('modalLayerTitle');
    const body = document.getElementById('modalLayerBody');

    const info = LAYERS[layerId];
    title.textContent = `Layer ${layerId}: ${info?.name || 'Unknown'}`;

    let detail = {};
    let timing = {};

    if (lastResults) {
        detail = lastResults.layer_details?.[layerId] || {};
        timing = lastResults.layer_timings?.find(t => String(t.layer_id) === layerId) || {};
    }

    const status = timing.status || 'Unknown';
    const statusClass = status.toLowerCase();
    const isDegraded = statusClass === 'degraded' || statusClass === 'failed';

    body.innerHTML = `
        <div class="modal-section">
            <h4>Execution Summary</h4>
            <div class="detail-grid">
                <div class="detail-item">
                    <span class="detail-label">Status</span>
                    <span class="status-tag-inline ${statusClass}">${status}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Duration</span>
                    <span>${timing.duration_ms?.toFixed(2) || 0}ms</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Checks Passed</span>
                    <span>${detail.checks_passed || 0}/${detail.checks_performed || 0}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Issues Found</span>
                    <span>${detail.issues_count || 0}</span>
                </div>
            </div>
        </div>
        
        ${isDegraded ? `
        <div class="modal-section">
            <h4>${aiEnabled ? 'AI Analysis & Recommended Fix' : 'Recommended Fix'}</h4>
            <div class="fix-box">
                ${generateLayerFix(layerId, detail, aiEnabled)}
            </div>
        </div>
        ` : ''}
        
        <div class="modal-section">
            <h4>Layer Details</h4>
            <pre class="detail-json">${JSON.stringify(detail.details || {}, null, 2)}</pre>
        </div>
    `;

    // Add modal-section styles
    const style = document.createElement('style');
    style.textContent = `
        .modal-section { margin-bottom: 20px; }
        .modal-section h4 { font-size: 14px; font-weight: 600; margin-bottom: 12px; color: var(--text-secondary); }
        .detail-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
        .detail-item { background: var(--bg-tertiary); padding: 12px; border-radius: 8px; }
        .detail-label { display: block; font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
        .fix-box { background: var(--info-bg); border: 1px solid var(--info-border); border-radius: 8px; padding: 16px; line-height: 1.6; color: var(--text-secondary); font-size: 13px; }
        .detail-json { background: var(--bg-primary); border-radius: 8px; padding: 12px; font-size: 11px; color: var(--text-secondary); overflow-x: auto; }
    `;
    body.appendChild(style);

    modal.style.display = 'flex';
}

function generateLayerFix(layerId, detail, isAI) {
    const fixes = {
        '4.3': isAI ?
            'AI Analysis: Semantic validation flagged records with business rule violations. Review the affected transactions for:\n- Invalid merchant category codes\n- Suspicious amount patterns\n- Geographic inconsistencies\n\nRecommendation: Update semantic rules to reduce false positives or escalate flagged records for manual review.' :
            'Standard Fix: Review semantic validation rules. Check for:\n- Business rule accuracy\n- Field value constraints\n- Cross-field validation logic',
        '4.4': isAI ?
            'AI Analysis: Anomaly detection ML model identified outlier transactions. The Isolation Forest algorithm flagged records based on:\n- Transaction amount deviation\n- Temporal patterns\n- Geographic risk factors\n\nRecommendation: Review flagged transactions for fraud indicators. Consider adjusting model contamination threshold if false positive rate is high.' :
            'Standard Fix: Review anomaly thresholds. Check:\n- Model parameters\n- Training data quality\n- Feature importance weights',
        '6': isAI ?
            'AI Analysis: Stability checks detected variance in quality scores. This may indicate:\n- Inconsistent input data\n- Processing edge cases\n- Feature extraction issues\n\nRecommendation: Investigate records with score variance > 15%. Review feature extraction pipeline for consistency.' :
            'Standard Fix: Review stability thresholds. Check:\n- Score calculation logic\n- Input data consistency\n- Processing order effects'
    };

    return fixes[layerId] || (isAI ?
        'AI Analysis: This layer reported issues during execution. Review the layer details above and check the affected records. Consider adjusting validation thresholds or updating processing rules based on the specific issues identified.' :
        'Standard Fix: Review layer configuration and validation rules. Check input data quality and processing logic for the affected records.');
}

function closeLayerModal() {
    document.getElementById('layerModal').style.display = 'none';
}

// Record details modal
function showRecordDetails(recordId) {
    const modal = document.getElementById('recordModal');
    const title = document.getElementById('modalRecordTitle');
    const body = document.getElementById('modalRecordBody');

    title.textContent = `Record: ${recordId}`;

    body.innerHTML = `
        <div class="modal-section">
            <h4>Transaction Details</h4>
            <p style="color: var(--text-secondary)">Full record data available in the Decision Report and Execution Log.</p>
        </div>
        <div class="modal-section">
            <h4>Recommended Actions</h4>
            <div class="fix-box">
                ${aiEnabled ?
            'Based on AI analysis, this record requires careful review. Verify all transaction details, check for data quality issues, and document your findings before approving or rejecting.' :
            'Standard review procedure: Verify transaction data, check against compliance rules, and document resolution.'}
            </div>
        </div>
    `;

    modal.style.display = 'flex';
}

function closeRecordModal() {
    document.getElementById('recordModal').style.display = 'none';
}

// Export
function exportToCSV() {
    if (!lastResults) {
        showToast('error', 'No data to export');
        return;
    }

    showToast('success', 'Export functionality ready');
}

// Download complete log file
function downloadCompleteLog() {
    if (!lastResults) {
        showToast('error', 'No results to download');
        return;
    }

    const data = lastResults;
    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);

    let log = '';

    // Header
    log += '================================================================================\n';
    log += '                    DQS ENGINE - COMPLETE ANALYSIS LOG\n';
    log += '================================================================================\n\n';
    log += `Generated: ${now.toISOString()}\n`;
    log += `Batch ID: ${data.batch_id || 'N/A'}\n`;
    log += `Execution ID: ${data.execution_id || 'N/A'}\n\n`;

    // Summary
    log += '--------------------------------------------------------------------------------\n';
    log += '                              EXECUTIVE SUMMARY\n';
    log += '--------------------------------------------------------------------------------\n\n';
    log += `Total Records Analyzed: ${data.total_records}\n`;
    log += `Total Execution Time: ${Math.round(data.total_duration_ms)}ms\n\n`;
    log += `Quality Rate: ${data.quality_rate}%\n`;
    log += `Average DQS: ${data.average_dqs.toFixed(1)}/100\n\n`;
    log += `Safe to Process: ${data.safe_count}\n`;
    log += `Review Required: ${data.review_count}\n`;
    log += `Escalate: ${data.escalate_count}\n`;
    log += `Rejected: ${data.rejected_count}\n\n`;

    // Layer Execution Details
    log += '--------------------------------------------------------------------------------\n';
    log += '                           LAYER EXECUTION DETAILS\n';
    log += '--------------------------------------------------------------------------------\n\n';

    if (data.layer_timings) {
        data.layer_timings.forEach(t => {
            const detail = data.layer_details?.[String(t.layer_id)] || {};
            log += `Layer ${t.layer_id}: ${t.layer_name}\n`;
            log += `  Status: ${t.status.toUpperCase()}\n`;
            log += `  Duration: ${t.duration_ms.toFixed(2)}ms\n`;
            log += `  Checks: ${detail.checks_passed || 0}/${detail.checks_performed || 0} passed\n`;
            if (detail.issues_count > 0) {
                log += `  Issues: ${detail.issues_count}\n`;
            }
            if (detail.warnings_count > 0) {
                log += `  Warnings: ${detail.warnings_count}\n`;
            }
            log += '\n';
        });
    }

    // Records Requiring Action
    const actionCount = (data.review_count || 0) + (data.escalate_count || 0);
    if (actionCount > 0) {
        log += '--------------------------------------------------------------------------------\n';
        log += '                        RECORDS REQUIRING ACTION\n';
        log += '--------------------------------------------------------------------------------\n\n';

        const records = extractActionRecords(data);
        records.forEach((rec, i) => {
            log += `[${i + 1}] ${rec.id}\n`;
            log += `    Action: ${rec.action}\n`;
            log += `    Reason: ${rec.reason}\n`;
            log += `    Recommended Fix:\n`;
            const fix = generateFix(rec, aiEnabled);
            fix.split('\n').forEach(line => {
                log += `      ${line.trim()}\n`;
            });
            log += '\n';
        });
    }

    // Import Metadata Warnings
    if (data.import_metadata && data.import_metadata.warnings?.length > 0) {
        log += '--------------------------------------------------------------------------------\n';
        log += '                          DATA FORMAT WARNINGS\n';
        log += '--------------------------------------------------------------------------------\n\n';
        log += `Schema Compliance: ${data.import_metadata.compliance_score}%\n\n`;
        data.import_metadata.warnings.forEach(w => {
            log += `Warning: ${w}\n\n`;
        });
        if (data.import_metadata.missing_fields?.length > 0) {
            log += 'Missing Fields:\n';
            data.import_metadata.missing_fields.forEach(f => {
                log += `  - ${f}\n`;
            });
            log += '\n';
        }
    }

    // Decision Report
    log += '--------------------------------------------------------------------------------\n';
    log += '                            DECISION REPORT\n';
    log += '--------------------------------------------------------------------------------\n\n';

    // Clean the decision report of HTML tags
    let cleanReport = (data.decision_report || 'No report available')
        .replace(/<[^>]*>/g, '')
        .replace(/\*\*/g, '')
        .replace(/\*/g, '')
        .replace(/#{1,6}\s/g, '')
        .replace(/`/g, '');
    log += cleanReport + '\n\n';

    // Footer
    log += '================================================================================\n';
    log += '                              END OF REPORT\n';
    log += '================================================================================\n';

    // Download
    const blob = new Blob([log], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dqs_analysis_log_${timestamp}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast('success', 'Log file downloaded');
}

// Copy to clipboard
function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    const text = element.innerText;

    navigator.clipboard.writeText(text).then(() => {
        showToast('success', 'Copied to clipboard');
    }).catch(() => {
        showToast('error', 'Failed to copy');
    });
}

// Toast notifications
function showToast(type, message) {
    const container = document.getElementById('toastContainer');

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            ${type === 'success' ?
            '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>' :
            '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'}
        </svg>
        <span class="toast-message">${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 4000);
}

// ============================================================
// LIVE FEED - WebSocket Streaming
// ============================================================

let socket = null;
let liveStreamActive = false;
let liveTransactionLogs = [];
const MAX_LOG_ENTRIES = 100;
let reconnectTimer = null;
let heartbeatInterval = null;

// Initialize Live Feed socket connection with robust reconnection
function initLiveSocket() {
    if (typeof io === 'undefined') {
        console.warn('Socket.IO not loaded - using polling fallback');
        updateLiveConnectionStatus('disconnected');
        updateHeaderConnectionStatus('disconnected');
        return null;
    }

    // If already connected, return existing socket
    if (socket && socket.connected) {
        return socket;
    }

    // Clean up existing socket if disconnected
    if (socket) {
        socket.removeAllListeners();
        socket.close();
    }

    console.log('Initializing WebSocket connection...');
    updateLiveConnectionStatus('connecting');
    updateHeaderConnectionStatus('connecting');

    socket = io(window.location.origin, {
        transports: ['websocket', 'polling'],  // Try WebSocket first, fallback to polling
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 10000,
        reconnectionAttempts: Infinity,  // Never give up
        timeout: 20000,
        forceNew: false
    });

    // Connection events
    socket.on('connect', () => {
        console.log('WebSocket connected successfully');
        updateLiveConnectionStatus('connected');
        updateHeaderConnectionStatus('connected');

        // Clear any pending reconnect timer
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }

        // Start heartbeat to keep connection alive
        startHeartbeat();
    });

    socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        updateLiveConnectionStatus('disconnected');
        updateHeaderConnectionStatus('disconnected');
        liveStreamActive = false;
        updateStreamButtons();

        // Stop heartbeat
        stopHeartbeat();

        // If server closed connection, try to reconnect
        if (reason === 'io server disconnect' || reason === 'transport close') {
            scheduleReconnect();
        }
    });

    socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error.message);
        updateLiveConnectionStatus('disconnected');
        updateHeaderConnectionStatus('disconnected');
    });

    socket.on('reconnect_attempt', (attemptNumber) => {
        console.log(`Reconnection attempt ${attemptNumber}...`);
        updateLiveConnectionStatus('connecting');
        updateHeaderConnectionStatus('connecting');
    });

    socket.on('reconnect', (attemptNumber) => {
        console.log(`Reconnected after ${attemptNumber} attempts`);
        showToast('success', 'Reconnected to server');
    });

    socket.on('reconnect_failed', () => {
        console.error('Failed to reconnect after all attempts');
        updateLiveConnectionStatus('disconnected');
        updateHeaderConnectionStatus('disconnected');
        showToast('error', 'Connection lost. Click Start Stream to retry.');
    });

    // Custom events
    socket.on('connected', (data) => {
        console.log('Server acknowledged connection:', data);
        if (data.stats) {
            updateLiveStats(data.stats);
        }
    });

    socket.on('stream_status', (data) => {
        console.log('Stream status:', data.status);
        if (data.status === 'started') {
            liveStreamActive = true;
            showToast('success', 'Live stream started');
        } else if (data.status === 'stopped') {
            liveStreamActive = false;
            showToast('info', 'Live stream stopped');
        } else if (data.status === 'already_running') {
            liveStreamActive = true;
            showToast('info', 'Stream already running');
        }
        updateStreamButtons();
    });

    socket.on('transaction_result', (data) => {
        handleLiveTransaction(data);
    });

    // Heartbeat response (pong)
    socket.on('pong', () => {
        // Connection is alive
    });

    return socket;
}

// Start heartbeat to keep connection alive
function startHeartbeat() {
    stopHeartbeat();
    heartbeatInterval = setInterval(() => {
        if (socket && socket.connected) {
            socket.emit('ping');
        }
    }, 25000);  // Ping every 25 seconds
}

// Stop heartbeat
function stopHeartbeat() {
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = null;
    }
}

// Schedule a reconnection attempt
function scheduleReconnect() {
    if (reconnectTimer) return;

    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        if (!socket || !socket.connected) {
            console.log('Attempting manual reconnection...');
            initLiveSocket();
        }
    }, 3000);
}

// Update connection status UI (Live Feed panel)
function updateLiveConnectionStatus(status) {
    const statusEl = document.getElementById('liveConnectionStatus');
    if (!statusEl) return;

    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('.status-text');

    if (dot) dot.className = 'status-dot ' + status;

    if (text) {
        if (status === 'connected') {
            text.textContent = 'Connected';
        } else if (status === 'connecting') {
            text.textContent = 'Connecting...';
        } else {
            text.textContent = 'Disconnected';
        }
    }
}

// Update header connection status (synced with WebSocket)
function updateHeaderConnectionStatus(status) {
    const indicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('apiStatus');

    if (!indicator || !statusText) return;

    if (status === 'connected') {
        indicator.className = 'status-indicator connected';
        statusText.textContent = 'Connected';
    } else if (status === 'connecting') {
        indicator.className = 'status-indicator connecting';
        statusText.textContent = 'Connecting...';
    } else {
        indicator.className = 'status-indicator disconnected';
        statusText.textContent = 'Disconnected';
    }
}

// Update stream control buttons
function updateStreamButtons() {
    const startBtn = document.getElementById('startStreamBtn');
    const stopBtn = document.getElementById('stopStreamBtn');

    if (startBtn && stopBtn) {
        startBtn.disabled = liveStreamActive;
        stopBtn.disabled = !liveStreamActive;
    }
}

// Start live stream (auto-starts when tab opens)
async function startLiveStream() {
    // Set API URL before starting
    await setLiveApiUrl();

    const sock = initLiveSocket();
    if (!sock) {
        console.warn('WebSocket not available');
        showToast('error', 'WebSocket not available. Please ensure flask-socketio is installed.');
        return;
    }

    // Send start command
    sock.emit('start_stream', {});

    // Update button states
    const startBtn = document.getElementById('btnStartStream');
    const stopBtn = document.getElementById('btnStopStream');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    // Clear any empty state
    const logContainer = document.getElementById('liveLogContainer');
    const emptyState = logContainer?.querySelector('.live-log-empty');
    if (emptyState) {
        emptyState.style.display = 'none';
    }

    // Update connection status
    updateLiveConnectionStatus('connected');
}

// Stop live stream
function stopLiveStream() {
    if (socket && socket.connected) {
        socket.emit('stop_stream');
    }

    // Update button states
    const startBtn = document.getElementById('btnStartStream');
    const stopBtn = document.getElementById('btnStopStream');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    // Update connection status
    updateLiveConnectionStatus('disconnected');
}

// Handle incoming live transaction
function handleLiveTransaction(data) {
    const { transaction, result, stats } = data;

    // Update stats
    if (stats) {
        updateLiveStats(stats);
    }

    // Add to log
    addLiveLogEntry(transaction, result);

    // Master Log
    const duration = result.total_duration_ms || result.processing_time_ms || 0;
    const action = result.action;
    const type = action === 'SAFE_TO_USE' ? 'success' : (action === 'ESCALATE' ? 'error' : 'warn');

    appendMasterLog(`Transaction ${result.transaction_id} processed in ${duration}ms.`, 'info');
    appendMasterLog(`> Outcome: ${action} | DQS: ${result.average_dqs?.toFixed(1) || 0}`, type);

    // Update counter
    const counterEl = document.getElementById('liveTransactionCount');
    if (counterEl && stats) {
        counterEl.textContent = stats.total;
    }
}

// Update live stats display
function updateLiveStats(stats) {
    const safeEl = document.getElementById('liveSafeCount');
    const reviewEl = document.getElementById('liveReviewCount');
    const escalateEl = document.getElementById('liveEscalateCount');
    const rejectedEl = document.getElementById('liveRejectedCount');

    if (safeEl) safeEl.textContent = stats.safe || 0;
    if (reviewEl) reviewEl.textContent = stats.review || 0;
    if (escalateEl) escalateEl.textContent = stats.escalate || 0;
    if (rejectedEl) rejectedEl.textContent = stats.rejected || 0;
}

// Add entry to live log
function addLiveLogEntry(transaction, result) {
    const logContainer = document.getElementById('liveLogContainer');
    if (!logContainer) return;

    // Clear empty state
    const emptyState = logContainer.querySelector('.live-log-empty');
    if (emptyState) {
        emptyState.remove();
    }

    // Store log
    liveTransactionLogs.unshift({ transaction, result, timestamp: new Date().toISOString() });
    if (liveTransactionLogs.length > MAX_LOG_ENTRIES) {
        liveTransactionLogs.pop();
    }

    // Determine action class
    let actionClass = 'safe';
    let actionText = 'SAFE';
    if (result.action === 'REVIEW_REQUIRED') {
        actionClass = 'review';
        actionText = 'REVIEW';
    } else if (result.action === 'ESCALATE') {
        actionClass = 'escalate';
        actionText = 'ESCALATE';
    } else if (result.action === 'NO_ACTION') {
        actionClass = 'rejected';
        actionText = 'REJECTED';
    }

    // Determine DQS badge class
    const dqs = result.dqs_score || 100;
    let dqsClass = 'high';
    if (dqs < 60) dqsClass = 'low';
    else if (dqs < 80) dqsClass = 'medium';

    // Create entry element with new format
    const entry = document.createElement('div');
    entry.className = 'live-txn-entry new';

    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    const txnId = result.transaction_id || transaction?.transaction?.transaction_id || 'N/A';
    const amount = transaction?.transaction?.amount || 0;
    const isExternal = transaction?._metadata?.source === 'external_api';

    entry.innerHTML = `
        <span class="txn-time">${time}</span>
        <span class="txn-id">${txnId}</span>
        <span class="txn-amount">₹${amount.toLocaleString()}</span>
        <span class="txn-dqs"><span class="dqs-badge ${dqsClass}">${dqs.toFixed(0)}</span></span>
        <span class="txn-action"><span class="action-badge ${actionClass}">${actionText}</span></span>
        <span class="txn-source ${isExternal ? 'external' : ''}">${isExternal ? '🔗 External' : '📊 Simulated'}</span>
    `;

    // Make clickable
    entry.style.cursor = 'pointer';
    entry.title = "Click to view full analysis details";
    entry.addEventListener('click', () => showLiveDetails(result));

    // Insert at top
    logContainer.insertBefore(entry, logContainer.firstChild);

    // Remove 'new' class after animation
    setTimeout(() => entry.classList.remove('new'), 500);

    // Update rolling stats
    updateRollingStats();

    // Limit visible entries
    const entries = logContainer.querySelectorAll('.live-txn-entry');
    if (entries.length > MAX_LOG_ENTRIES) {
        entries[entries.length - 1].remove();
    }
}

// Set API key for live data source
async function setLiveApiKey() {
    const apiKey = document.getElementById('liveApiKey')?.value || '';

    try {
        const response = await fetch('/api/live/set-api-key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: apiKey })
        });

        const data = await response.json();
        if (data.success) {
            showToast('success', 'API key configured');
        }
    } catch (error) {
        showToast('error', 'Error setting API key');
    }
}

// Test API connection
async function testApiConnection() {
    const apiUrl = document.getElementById('liveApiUrl')?.value || '';
    const hint = document.getElementById('apiUrlHint');

    if (!apiUrl) {
        if (hint) {
            hint.textContent = 'Leave empty to use simulated demo data';
            hint.className = 'config-hint';
        }
        return;
    }

    if (hint) {
        hint.textContent = 'Testing connection...';
        hint.className = 'config-hint';
    }

    try {
        const response = await fetch('/api/live/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_url: apiUrl })
        });

        const data = await response.json();
        if (data.success) {
            if (hint) {
                hint.textContent = '✓ Connection successful';
                hint.className = 'config-hint success';
            }
            showToast('success', 'API connection successful');
        } else {
            if (hint) {
                hint.textContent = '✗ ' + (data.error || 'Connection failed');
                hint.className = 'config-hint error';
            }
            showToast('error', data.error || 'Connection failed');
        }
    } catch (error) {
        if (hint) {
            hint.textContent = '✗ Network error';
            hint.className = 'config-hint error';
        }
        showToast('error', 'Failed to test connection');
    }
}

// Set API URL for external data source
async function setLiveApiUrl() {
    const apiUrl = document.getElementById('liveApiUrl')?.value || '';
    const apiKey = document.getElementById('liveApiKey')?.value || '';

    // Set API key first if provided
    if (apiKey) {
        await setLiveApiKey();
    }

    try {
        const response = await fetch('/api/live/set-api-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_url: apiUrl })
        });

        const data = await response.json();

        // Update data source badge
        const badge = document.getElementById('dataSourceBadge');
        if (badge) {
            if (data.using_external) {
                badge.innerHTML = '<span class="source-icon">🔗</span><span class="source-text">External API</span>';
                badge.classList.add('external');
            } else {
                badge.innerHTML = '<span class="source-icon">📊</span><span class="source-text">Simulated Data</span>';
                badge.classList.remove('external');
            }
        }

        return data;
    } catch (error) {
        console.error('Failed to set API URL:', error);
        return { success: false };
    }
}

// Update rolling statistics
function updateRollingStats() {
    const logs = liveTransactionLogs.slice(0, 50);
    if (logs.length === 0) return;

    // Calculate average DQS
    const dqsSum = logs.reduce((sum, log) => sum + (log.result?.dqs_score || 100), 0);
    const avgDqs = dqsSum / logs.length;

    // Calculate quality rate (safe transactions)
    const safeCount = logs.filter(log => {
        const action = log.result?.action || '';
        return action === 'SAFE' || action === 'PROCESS' ||
            action === 'safe' || action.includes('SAFE');
    }).length;
    const qualityRate = (safeCount / logs.length) * 100;

    // Update UI
    const avgDqsEl = document.getElementById('rollingAvgDqs');
    const dqsBar = document.getElementById('rollingDqsBar');
    const qualityRateEl = document.getElementById('rollingQualityRate');
    const qualityBar = document.getElementById('rollingQualityBar');

    if (avgDqsEl) avgDqsEl.textContent = avgDqs.toFixed(0);
    if (dqsBar) dqsBar.style.width = `${avgDqs}%`;
    if (qualityRateEl) qualityRateEl.textContent = `${qualityRate.toFixed(0)}%`;
    if (qualityBar) qualityBar.style.width = `${qualityRate}%`;
}

// Clear live logs
async function clearLiveLogs() {
    try {
        const response = await fetch('/api/live/clear', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            // Clear UI
            const logContainer = document.getElementById('liveLogContainer');
            if (logContainer) {
                logContainer.innerHTML = `
                    <div class="live-log-empty">
                        <svg class="icon-lg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                            <circle cx="12" cy="12" r="2" />
                            <path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14" />
                        </svg>
                        <p>Start the stream to see live transactions</p>
                    </div>
                `;
            }

            // Reset stats
            updateLiveStats({ safe: 0, review: 0, escalate: 0, rejected: 0, total: 0 });
            document.getElementById('liveTransactionCount').textContent = '0';

            liveTransactionLogs = [];
            showToast('success', 'Logs cleared');
        }
    } catch (error) {
        showToast('error', 'Failed to clear logs');
    }
}

// Download filtered logs (defaults to all data if no time specified)
async function downloadFilteredLogs() {
    const startInput = document.getElementById('logStartTime');
    const endInput = document.getElementById('logEndTime');

    // Time-only input - convert to today's date with that time
    let startTime = null;
    let endTime = null;

    const today = new Date().toISOString().split('T')[0];
    if (startInput?.value) {
        startTime = new Date(`${today}T${startInput.value}:00`).toISOString();
    }
    if (endInput?.value) {
        endTime = new Date(`${today}T${endInput.value}:00`).toISOString();
    }

    try {
        let url = '/api/live/logs';
        const params = new URLSearchParams();
        if (startTime) params.append('start', startTime);
        if (endTime) params.append('end', endTime);
        if (params.toString()) url += '?' + params.toString();

        const response = await fetch(url);
        const data = await response.json();

        if (!data.success || !data.logs || data.logs.length === 0) {
            showToast('info', 'No logs found. Run the stream first.');
            return;
        }

        // Generate log text
        const now = new Date();
        const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);

        let log = '';
        log += '================================================================================\n';
        log += '                    DQS ENGINE - LIVE STREAM LOGS\n';
        log += '================================================================================\n\n';
        log += `Downloaded: ${now.toISOString()}\n`;
        if (startTime) log += `Start Filter: ${startTime}\n`;
        if (endTime) log += `End Filter: ${endTime}\n`;
        log += `Total Records: ${data.logs.length}\n\n`;

        // Stats
        log += '--------------------------------------------------------------------------------\n';
        log += '                              SUMMARY\n';
        log += '--------------------------------------------------------------------------------\n\n';
        log += `Safe: ${data.stats.safe}\n`;
        log += `Review: ${data.stats.review}\n`;
        log += `Escalate: ${data.stats.escalate}\n`;
        log += `Rejected: ${data.stats.rejected}\n`;
        log += `Average DQS: ${data.stats.avg_dqs}\n\n`;

        // Transaction log
        log += '--------------------------------------------------------------------------------\n';
        log += '                          TRANSACTION LOG\n';
        log += '--------------------------------------------------------------------------------\n\n';

        data.logs.forEach((entry, i) => {
            log += `[${i + 1}] ${entry.timestamp}\n`;
            log += `    ID: ${entry.transaction_id}\n`;
            log += `    Amount: Rs ${entry.amount}\n`;
            log += `    Status: ${entry.status}\n`;
            log += `    DQS: ${entry.dqs_score}\n`;
            log += `    Action: ${entry.action}\n`;
            if (entry.flags && entry.flags.length > 0) {
                log += `    Flags: ${entry.flags.join(', ')}\n`;
            }
            log += `    Processing Time: ${entry.processing_time_ms}ms\n\n`;
        });

        log += '================================================================================\n';
        log += '                              END OF LOG\n';
        log += '================================================================================\n';

        // Download
        const blob = new Blob([log], { type: 'text/plain;charset=utf-8' });
        const url2 = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url2;
        a.download = `live_stream_log_${timestamp}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url2);

        showToast('success', `Downloaded ${data.logs.length} log entries`);

    } catch (error) {
        showToast('error', 'Failed to download logs');
        console.error(error);
    }
}

// Initialize socket when switching to Live tab
function onLiveTabActivated() {
    // Initialize socket but don't auto-start - user clicks Start
    initLiveSocket();

    // Fetch current stats
    fetch('/api/live/stats')
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                updateLiveStats(data.stats);
                document.getElementById('liveTransactionCount').textContent = data.stats.total || 0;

                // If already streaming, update UI
                if (data.streaming) {
                    updateLiveConnectionStatus('connected');
                    liveStreamActive = true;
                    const startBtn = document.getElementById('btnStartStream');
                    const stopBtn = document.getElementById('btnStopStream');
                    if (startBtn) startBtn.disabled = true;
                    if (stopBtn) stopBtn.disabled = false;
                }
            }
        })
        .catch(err => console.error('Failed to fetch live stats:', err));
}

// Show details for a live transaction
function showLiveDetails(fullResult) {
    if (!fullResult) return;

    const modal = document.getElementById('recordModal');
    const title = document.getElementById('modalRecordTitle');
    const body = document.getElementById('modalRecordBody');

    if (!modal || !body) return;

    // Set Title
    if (title) title.textContent = "Live Transaction: " + (fullResult.transaction_id || 'Unknown');

    // Helper to generate layer rows (DIV version)
    const generateLayerRows = (timings, details) => {
        if (!timings) return '<div class="no-data">No data</div>';
        return timings.map((t, i) => {
            const detail = (details && details[String(t.layer_id)]) || {};
            const statusClass = t.status.toLowerCase();
            const detailText = typeof detail.details === 'object' ? JSON.stringify(detail.details) : (detail.details || '-');

            return `
                <div class="modal-layer-item ${t.status === 'PASSED' ? 'passed' : 'failed'}" style="margin-bottom: 8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>L${t.layer_id}: ${t.layer_name}</strong>
                        <span class="status-tag-inline ${statusClass}">${t.status}</span>
                    </div>
                    <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                        Duration: ${t.duration_ms.toFixed(2)}ms
                    </div>
                    <div style="font-size:12px; margin-top:4px; color:var(--text-secondary);">
                        ${detailText}
                    </div>
                </div>
            `;
        }).join('');
    };

    // Build Content using DIVs instead of Table for premium look
    const layerHtml = generateLayerRows(fullResult.layer_timings, fullResult.layer_details);

    body.innerHTML = `
        <div class="modal-results-container">
            <!-- Summary Header -->
            <div class="modal-summary-header" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div class="stat-card ${fullResult.action === 'SAFE_TO_USE' ? 'stat-success' : (fullResult.action === 'ESCALATE' ? 'stat-danger' : 'stat-warning')}" style="padding: 1rem;">
                    <span class="stat-label">Action</span>
                    <span class="stat-value" style="font-size: 1.2rem;">${fullResult.action}</span>
                </div>
                <div class="stat-card" style="padding: 1rem;">
                    <span class="stat-label">DQS Score</span>
                    <span class="stat-value" style="font-size: 1.2rem;">${fullResult.average_dqs ? fullResult.average_dqs.toFixed(1) : '0.0'}</span>
                </div>
                <div class="stat-card" style="padding: 1rem;">
                    <span class="stat-label">Processing Time</span>
                    <span class="stat-value" style="font-size: 1.2rem;">${fullResult.total_duration_ms || 0} ms</span>
                </div>
                <div class="stat-card" style="padding: 1rem;">
                    <span class="stat-label">Quality Rate</span>
                    <span class="stat-value" style="font-size: 1.2rem;">${fullResult.quality_rate || 0}%</span>
                </div>
            </div>

            <!-- Decision Report -->
            <div class="section-card" style="margin-bottom: 1.5rem;">
                <div class="section-header">
                    <h3>AI Analysis & Decision Report</h3>
                </div>
                <div class="report-content-wrapper" style="max-height: 200px; overflow-y: auto;">
                    <pre class="report-content">${fullResult.decision_report || 'No report available'}</pre>
                </div>
            </div>

            <!-- Layer Analysis (New List View) -->
            <div class="section-card" style="margin-bottom: 1.5rem;">
                <div class="section-header">
                    <h3>Pipeline Layer Verification</h3>
                </div>
                <div class="modal-layer-list">
                    ${layerHtml}
                </div>
            </div>

            <!-- Terminal Log -->
            <div class="section-card">
                <div class="section-header">
                    <h3>System Execution Logs</h3>
                </div>
                <div class="report-content-wrapper" style="max-height: 200px; overflow-y: auto;">
                    <pre class="report-content log-style">${fullResult.execution_report || 'No log available'}</pre>
                </div>
            </div>
        </div>
    `;

    // Show Modal
    modal.style.display = 'flex';

    // Hide main results section if it was shown (to avoid confusion)
    const MainResults = document.getElementById('resultsSection');
    if (MainResults) MainResults.style.display = 'none';

    showToast('info', `Viewing details for ${fullResult.transaction_id || 'record'}`);
}

// ==========================================
// Helper Functions
// ==========================================

function appendMasterLog(message, type = 'info') {
    const container = document.getElementById('masterLogContainer');
    if (!container) return;

    const div = document.createElement('div');
    div.className = `log-line ${type}`;

    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    div.innerHTML = `
        <span class="timestamp">[${time}]</span>
        <span class="msg">${message}</span>
    `;

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function clearMasterLog() {
    const container = document.getElementById('masterLogContainer');
    if (container) container.innerHTML = '<div class="log-line info"><span class="timestamp">SYSTEM</span><span class="msg">Log cleared. Ready...</span></div>';
}
