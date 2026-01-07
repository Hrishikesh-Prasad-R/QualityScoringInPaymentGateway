"""
Flask API Server for DQS Engine Frontend

Provides REST endpoints to interact with the Data Quality Scoring Engine.
Supports standard VISA format and graceful handling of non-standard CSV/JSON.
Includes WebSocket support for real-time streaming.
"""
import os
import sys
import json
import math
import time
import threading
from threading import Lock
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

# SocketIO imports
try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("Warning: flask-socketio not installed. Live streaming disabled.")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dqs_engine import DQSEngine
from src.data_generator import generate_visa_transactions
from src.csv_adapter import adapt_csv_to_visa, adapt_flat_json_to_visa
from src.config import Action
from src.live_data_generator import LiveDataGenerator, LiveLogStorage


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN, Infinity, and numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def sanitize_for_json(obj):
    """Recursively sanitize an object for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


app = Flask(__name__, static_folder='frontend', static_url_path='')
app.json_encoder = SafeJSONEncoder
app.config['SECRET_KEY'] = 'dqs-engine-secret-key'
CORS(app)

# Initialize SocketIO if available
if SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
else:
    socketio = None

# Thread-safe global state
_engine_lock = Lock()
_streaming_lock = Lock()

# Global instances (protected by locks)
engine = None
live_generator = LiveDataGenerator(anomaly_rate=0.15)
log_storage = LiveLogStorage(log_file="live_stream_logs.json")

# Streaming state (protected by _streaming_lock)
streaming_active = False
streaming_thread = None


def get_engine(use_ai=False):
    """Get or create engine instance (thread-safe singleton)."""
    global engine
    with _engine_lock:
        # Only create if AI setting matches or engine doesn't exist
        if engine is None:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            engine = DQSEngine(gemini_api_key=api_key, use_ai=use_ai and bool(api_key))
        return engine


@app.route('/')
def index():
    """Serve the main dashboard."""
    return send_from_directory('frontend', 'index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "layers": 15,
        "phases": 7
    })


@app.route('/api/generate', methods=['POST'])
def generate_data():
    """Generate sample VISA transactions with preview."""
    try:
        data = request.get_json() or {}
        n_transactions = min(int(data.get('count', 20)), 500)  # Max 500
        anomaly_rate = min(float(data.get('anomaly_rate', 0.15)), 0.5)  # Max 50%
        seed = data.get('seed')
        
        transactions = generate_visa_transactions(
            n_transactions=n_transactions,
            anomaly_rate=anomaly_rate,
            random_seed=seed
        )
        
        # Return full transactions for preview
        return jsonify({
            "success": True,
            "count": n_transactions,
            "anomaly_rate": anomaly_rate,
            "preview": transactions[:20],  # Preview up to 20
            "transactions": transactions,  # Full data for execution
            "message": f"Generated {n_transactions} transactions"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/run', methods=['POST'])
def run_pipeline():
    """Run the complete DQS pipeline with graceful CSV/JSON handling."""
    try:
        data = request.get_json() or {}
        
        # Check for various data sources
        custom_data = data.get('custom_data')
        csv_content = data.get('csv_content')
        use_ai = data.get('use_ai', False)
        
        import_metadata = None
        
        if csv_content:
            # CSV content provided - adapt to VISA format
            transactions, import_metadata = adapt_csv_to_visa(csv_content)
            if not transactions:
                return jsonify({
                    "success": False,
                    "error": "Failed to parse CSV content. Please check the format."
                }), 400
                
        elif custom_data:
            # JSON data provided - check if it needs adaptation
            if isinstance(custom_data, list):
                transactions, import_metadata = adapt_flat_json_to_visa(custom_data)
            else:
                transactions, import_metadata = adapt_flat_json_to_visa([custom_data])
        else:
            # Generate transactions
            n_transactions = min(int(data.get('count', 20)), 500)
            anomaly_rate = min(float(data.get('anomaly_rate', 0.15)), 0.5)
            seed = data.get('seed', 42)
            
            transactions = generate_visa_transactions(
                n_transactions=n_transactions,
                anomaly_rate=anomaly_rate,
                random_seed=seed
            )
        
        if not transactions:
            return jsonify({
                "success": False,
                "error": "No valid transactions to process"
            }), 400
        
        # Run engine with error handling that preserves import metadata
        try:
            eng = get_engine(use_ai=use_ai)
            result = eng.run(transactions)
        except Exception as engine_error:
            # Engine failed - provide helpful error message with schema info
            error_msg = str(engine_error)
            
            # Add schema compliance context if data was imported
            if import_metadata:
                compliance = import_metadata.get('compliance_score', 0)
                missing = import_metadata.get('missing_fields', [])[:5]
                warnings = import_metadata.get('warnings', [])
                
                error_msg = f"Pipeline execution failed: {error_msg}\n\n"
                error_msg += f"DATA FORMAT ISSUE DETECTED:\n"
                error_msg += f"Schema Compliance: {compliance:.0f}%\n"
                
                if missing:
                    error_msg += f"Missing required fields: {', '.join(missing)}\n"
                
                if compliance < 80:
                    error_msg += "\nYour data does not follow the required VISA transaction format.\n"
                    error_msg += "Please ensure your CSV/JSON includes these required fields:\n"
                    error_msg += "- transaction.transaction_id, transaction.amount, transaction.currency\n"
                    error_msg += "- card.network, card.card_type\n"
                    error_msg += "- merchant.merchant_id, merchant.country\n"
                    error_msg += "- customer.customer_id\n"
                    error_msg += "- fraud.risk_score\n"
                
                for w in warnings:
                    error_msg += f"\nWarning: {w}"
            
            return jsonify({
                "success": False,
                "error": error_msg,
                "import_metadata": {
                    "source": import_metadata.get("source", "unknown") if import_metadata else "unknown",
                    "compliance_score": import_metadata.get("compliance_score", 0) if import_metadata else 100,
                    "is_standard_format": import_metadata.get("is_standard_format", True) if import_metadata else True,
                    "missing_fields": import_metadata.get("missing_fields", [])[:10] if import_metadata else [],
                    "warnings": import_metadata.get("warnings", []) if import_metadata else []
                } if import_metadata else None
            }), 400
        
        # Build response
        layer_timings = []
        for t in result.layer_timings:
            layer_timings.append({
                "layer_id": t.layer_id,
                "layer_name": t.layer_name,
                "duration_ms": round(t.duration_ms, 2),
                "status": t.status,
                "start_time": t.start_time.isoformat(),
                "end_time": t.end_time.isoformat()
            })
        
        # Get detailed layer results
        layer_details = {}
        for lid, res in eng.layer_results.items():
            layer_details[str(lid)] = {
                "layer_id": res.layer_id,
                "layer_name": res.layer_name,
                "status": res.status.value,
                "checks_performed": res.checks_performed,
                "checks_passed": res.checks_passed,
                "issues_count": len(res.issues) if res.issues else 0,
                "warnings_count": len(res.warnings) if res.warnings else 0,
                "details": sanitize_for_json(res.details),
                "can_continue": res.can_continue
            }
        
        # Add import warnings to decision report if applicable
        decision_report = result.decision_report or ""
        if import_metadata and import_metadata.get("warnings"):
            warning_header = "\n\n--- DATA IMPORT NOTICE ---\n"
            warning_header += f"Schema Compliance: {import_metadata.get('compliance_score', 0):.0f}%\n"
            for warning in import_metadata.get("warnings", []):
                warning_header += f"Warning: {warning}\n"
            warning_header += "\nFor best results, use the standard VISA transaction format.\n"
            warning_header += "See documentation for required fields and structure.\n"
            warning_header += "-" * 30 + "\n\n"
            decision_report = warning_header + decision_report
        
        response = {
            "success": result.success,
            "batch_id": result.batch_id,
            "execution_id": result.execution_id,
            "total_records": result.total_records,
            "safe_count": result.safe_count,
            "review_count": result.review_count,
            "escalate_count": result.escalate_count,
            "rejected_count": result.rejected_count,
            "average_dqs": round(result.average_dqs, 2),
            "quality_rate": round(result.quality_rate, 2),
            "total_duration_ms": round(result.total_duration_ms, 2),
            "layer_timings": layer_timings,
            "layer_details": layer_details,
            "decision_report": decision_report,
            "execution_report": result.execution_report,
            "errors": result.errors
        }
        
        # Add import metadata if present
        if import_metadata:
            response["import_metadata"] = {
                "source": import_metadata.get("source", "unknown"),
                "compliance_score": import_metadata.get("compliance_score", 100),
                "is_standard_format": import_metadata.get("is_standard_format", True),
                "quality_penalty": import_metadata.get("quality_penalty", 0),
                "warnings": import_metadata.get("warnings", []),
                "missing_fields": import_metadata.get("missing_fields", [])[:10]
            }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        return jsonify({
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/layers')
def get_layers():
    """Get information about all layers."""
    layers = [
        {"id": 1, "name": "Input Contract", "phase": 1, "type": "Deterministic", "description": "Schema validation and manifest verification"},
        {"id": 2, "name": "Input Validation", "phase": 1, "type": "Deterministic", "description": "Parse and validate JSON transactions"},
        {"id": 3, "name": "Feature Extraction", "phase": 2, "type": "Deterministic", "description": "Extract 35+ quality dimensions"},
        {"id": 4.1, "name": "Structural Integrity", "phase": 3, "type": "Deterministic", "description": "Validate required fields and data types"},
        {"id": 4.2, "name": "Field Compliance", "phase": 3, "type": "Deterministic", "description": "Calculate Data Quality Score (DQS)"},
        {"id": 4.3, "name": "Semantic Validation", "phase": 3, "type": "Deterministic", "description": "Apply business rules and constraints"},
        {"id": 4.4, "name": "Anomaly Detection", "phase": 4, "type": "AI/ML", "description": "Isolation Forest anomaly detection"},
        {"id": 4.5, "name": "GenAI Summarization", "phase": 4, "type": "AI/GenAI", "description": "Generate quality summaries with Gemini"},
        {"id": 5, "name": "Output Contract", "phase": 5, "type": "Deterministic", "description": "Structure and validate outputs"},
        {"id": 6, "name": "Stability & Consistency", "phase": 5, "type": "Deterministic", "description": "Validate score distribution"},
        {"id": 7, "name": "Conflict Detection", "phase": 5, "type": "Deterministic", "description": "Identify signal conflicts"},
        {"id": 8, "name": "Confidence Band", "phase": 5, "type": "Deterministic", "description": "Calculate confidence levels"},
        {"id": 9, "name": "Decision Gate", "phase": 6, "type": "Deterministic", "description": "Make final action decisions"},
        {"id": 10, "name": "Responsibility Boundary", "phase": 6, "type": "Deterministic", "description": "Track decision ownership"},
        {"id": 11, "name": "Logging & Trace", "phase": 7, "type": "Deterministic", "description": "Complete audit trail"}
    ]
    
    phases = [
        {"id": 1, "name": "Foundation", "layers": [1, 2], "color": "#3498db"},
        {"id": 2, "name": "Feature Extraction", "layers": [3], "color": "#9b59b6"},
        {"id": 3, "name": "Deterministic Inference", "layers": [4.1, 4.2, 4.3], "color": "#2ecc71"},
        {"id": 4, "name": "AI Inference", "layers": [4.4, 4.5], "color": "#e74c3c"},
        {"id": 5, "name": "Output & Decision", "layers": [5, 6, 7, 8], "color": "#f39c12"},
        {"id": 6, "name": "Decision Gate", "layers": [9, 10], "color": "#1abc9c"},
        {"id": 7, "name": "Logging", "layers": [11], "color": "#34495e"}
    ]
    
    return jsonify({"layers": layers, "phases": phases})


@app.route('/api/schema')
def get_schema():
    """Get the expected VISA transaction schema."""
    schema = {
        "description": "VISA Transaction Schema for DQS Engine",
        "required_sections": [
            "transaction", "card", "merchant", "customer", 
            "authentication", "fraud", "network", "compliance", 
            "settlement", "business_metadata"
        ],
        "required_fields": {
            "transaction": ["transaction_id", "amount", "currency", "timestamp"],
            "card": ["network", "card_type"],
            "merchant": ["merchant_id", "merchant_category_code", "country"],
            "customer": ["customer_id"],
            "fraud": ["risk_score"]
        },
        "example": {
            "transaction": {
                "transaction_id": "txn_00000001",
                "amount": 5000.00,
                "currency": "INR",
                "timestamp": "2026-01-05T10:30:00Z",
                "status": "approved"
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
        }
    }
    return jsonify(schema)


# ============================================================
# LIVE STREAMING ENDPOINTS
# ============================================================

@app.route('/api/live/set-api-key', methods=['POST'])
def set_live_api_key():
    """Set API key for live data source."""
    data = request.json or {}
    api_key = data.get('api_key', '')
    live_generator.set_api_key(api_key)
    return jsonify({"success": True, "message": "API key configured"})


@app.route('/api/live/set-anomaly-rate', methods=['POST'])
def set_live_anomaly_rate():
    """Set anomaly rate for generated data."""
    data = request.json or {}
    rate = data.get('rate', 0.15)
    live_generator.set_anomaly_rate(rate)
    return jsonify({"success": True, "anomaly_rate": rate})


@app.route('/api/live/logs', methods=['GET'])
def get_live_logs():
    """Get filtered live stream logs."""
    start_time = request.args.get('start')
    end_time = request.args.get('end')
    
    logs = log_storage.get_logs(start_time, end_time)
    stats = log_storage.get_stats()
    
    return jsonify({
        "success": True,
        "logs": logs,
        "stats": stats,
        "count": len(logs)
    })


@app.route('/api/live/stats', methods=['GET'])
def get_live_stats():
    """Get current live stream statistics."""
    return jsonify({
        "success": True,
        "stats": log_storage.get_stats(),
        "streaming": streaming_active
    })


@app.route('/api/live/clear', methods=['POST'])
def clear_live_logs():
    """Clear all live stream logs."""
    log_storage.clear_logs()
    return jsonify({"success": True, "message": "Logs cleared"})


def process_single_transaction(transaction):
    """Process a single transaction through DQS engine."""
    try:
        start_time = time.time()
        
        # Flatten for DQS
        flat_txn = live_generator.flatten_for_dqs(transaction)
        
        # Get or create engine
        eng = get_engine(use_ai=False)
        
        # Process single record using run() method
        pipeline_result = eng.run([flat_txn])
        
        processing_time = (time.time() - start_time) * 1000
        
        # Extract result from PipelineResult
        if pipeline_result and pipeline_result.success:
            # Get counts from pipeline result
            if pipeline_result.safe_count > 0:
                action = 'SAFE_TO_USE'
            elif pipeline_result.escalate_count > 0:
                action = 'ESCALATE'
            elif pipeline_result.review_count > 0:
                action = 'REVIEW_REQUIRED'
            elif pipeline_result.rejected_count > 0:
                action = 'NO_ACTION'
            else:
                action = 'SAFE_TO_USE'
            
            dqs_score = pipeline_result.average_dqs
            reason = ''
            flags = []
            
            # Try to get more details from decision report
            if pipeline_result.decision_report:
                # Parse basic info from report
                if 'ESCALATE' in pipeline_result.decision_report.upper():
                    action = 'ESCALATE'
                elif 'REVIEW' in pipeline_result.decision_report.upper():
                    action = 'REVIEW_REQUIRED'
        else:
            action = 'UNKNOWN'
            dqs_score = 100
            reason = ''
            flags = pipeline_result.errors if pipeline_result else []
        
        return {
            "success": True,
            "transaction_id": transaction.get('transaction', {}).get('transaction_id'),
            "dqs_score": dqs_score,
            "action": action,
            "reason": reason,
            "flags": flags,
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "transaction_id": transaction.get('transaction', {}).get('transaction_id'),
            "dqs_score": 0,
            "action": "ERROR",
            "flags": [str(e)],
            "processing_time_ms": 0
        }


def streaming_worker():
    """Background worker for streaming transactions (thread-safe)."""
    global streaming_active
    
    while True:
        # Check if we should continue (with lock)
        with _streaming_lock:
            if not streaming_active:
                break
        
        try:
            # Generate transaction
            transaction = live_generator.generate_transaction()
            
            # Process it
            result = process_single_transaction(transaction)
            
            # Store log (already thread-safe)
            log_storage.add_log(transaction, result)
            
            # Emit via WebSocket (broadcast=True for background threads)
            # This doesn't require request context
            if socketio:
                socketio.emit('transaction_result', {
                    'transaction': transaction,
                    'result': result,
                    'stats': log_storage.get_stats()
                }, broadcast=True)
            
            # Wait 1 second
            time.sleep(1)
            
        except Exception as e:
            print(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
    
    # Flush logs on stop
    log_storage.flush()


if SOCKETIO_AVAILABLE:
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        emit('connected', {
            'message': 'Connected to DQS Live Stream',
            'stats': log_storage.get_stats()
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        pass
    
    @socketio.on('ping')
    def handle_ping():
        """Handle heartbeat ping from client."""
        emit('pong')
    
    @socketio.on('start_stream')
    def handle_start_stream(data=None):
        """Start the live transaction stream (thread-safe)."""
        global streaming_active, streaming_thread
        
        with _streaming_lock:
            if streaming_active:
                emit('stream_status', {'status': 'already_running'})
                return
            
            # Set anomaly rate if provided
            if data and 'anomaly_rate' in data:
                live_generator.set_anomaly_rate(data['anomaly_rate'])
            
            streaming_active = True
            streaming_thread = threading.Thread(target=streaming_worker, daemon=True)
            streaming_thread.start()
        
        emit('stream_status', {'status': 'started'})
    
    @socketio.on('stop_stream')
    def handle_stop_stream():
        """Stop the live transaction stream (thread-safe)."""
        global streaming_active
        
        with _streaming_lock:
            streaming_active = False
        emit('stream_status', {'status': 'stopped'})


if __name__ == '__main__':
    # Production-safe debug mode (default: False)
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    print("=" * 60)
    print("  DQS ENGINE - Enterprise Data Quality Platform")
    print(f"  Open http://localhost:{port} in your browser")
    if SOCKETIO_AVAILABLE:
        print("  Live streaming: ENABLED (WebSocket)")
    else:
        print("  Live streaming: DISABLED (install flask-socketio)")
    print(f"  Debug mode: {'ON' if debug_mode else 'OFF'}")
    print("=" * 60)
    
    if SOCKETIO_AVAILABLE:
        socketio.run(app, debug=debug_mode, port=port, host='0.0.0.0', allow_unsafe_werkzeug=debug_mode)
    else:
        app.run(debug=debug_mode, port=port, host='0.0.0.0')
