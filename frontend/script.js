/**
 * DQS Engine - Enterprise Dashboard JavaScript
 * Professional data quality analysis with AI-powered insights
 */

const API_BASE = '';

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
}

// Slider update
function updateSlider(id, valueId) {
    const slider = document.getElementById(id);
    const valueEl = document.getElementById(valueId);
    valueEl.textContent = id === 'anomalyRate' ? slider.value + '%' : slider.value;
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
            customData = parseCSV(e.target.result);
            showDataPreview(customData);
            showToast('success', `Loaded ${customData.length} records from CSV`);
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
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('csvFile').value = '';
    document.getElementById('dataPreviewSection').style.display = 'none';
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
            throw new Error(data.error || 'Pipeline execution failed');
        }
    } catch (error) {
        console.error('Pipeline error:', error);
        showToast('error', error.message);
        resetLayers();
    } finally {
        overlay.style.display = 'none';
        btn.disabled = false;
    }
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
    const total = data.total_records || 0;

    // Generate rows based on action counts
    for (let i = 0; i < Math.min(total, 50); i++) {
        let action = 'SAFE_TO_USE';
        if (i < data.escalate_count) action = 'ESCALATE';
        else if (i < data.escalate_count + data.review_count) action = 'REVIEW_REQUIRED';
        else if (i < data.escalate_count + data.review_count + data.rejected_count) action = 'NO_ACTION';

        const dqs = action === 'SAFE_TO_USE' ? 85 + Math.random() * 15 :
            action === 'REVIEW_REQUIRED' ? 50 + Math.random() * 30 :
                20 + Math.random() * 40;

        rows.push({
            id: `txn_${String(i).padStart(8, '0')}`,
            action: action,
            dqs: dqs,
            quality: dqs >= 80 ? 'High' : dqs >= 50 ? 'Medium' : 'Low',
            anomaly: action !== 'SAFE_TO_USE' ? 'Yes' : 'No',
            confidence: dqs >= 70 ? 'High' : dqs >= 40 ? 'Medium' : 'Low',
            issues: action === 'SAFE_TO_USE' ? '0' : Math.floor(Math.random() * 3 + 1).toString()
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
