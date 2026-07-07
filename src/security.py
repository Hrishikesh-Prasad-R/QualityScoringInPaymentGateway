"""
DQS Engine Webhook Security Layer
==================================
Provides HMAC-SHA256 signature verification for secure API integrations.
"""
import hmac
import hashlib
import os
import logging
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger("DQS.Security")

# Secret key used for signing payloads.
# In a real environment, load this from environment variables.
WEBHOOK_SECRET = os.environ.get("DQS_WEBHOOK_SECRET", "dqs_secure_webhook_secret_key_2026").encode()

def require_hmac_signature(f):
    """
    Decorator to validate HMAC-SHA256 signature on requests.
    Validates if REQUIRE_HMAC environment variable is set, or if X-Signature header is provided.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        signature = request.headers.get("X-Signature")
        require_hmac = os.environ.get("REQUIRE_HMAC", "false").lower() == "true"
        
        # Enforce validation only if header is present or validation is explicitly required
        if require_hmac or signature:
            if not signature:
                logger.warning("Rejected request: Missing signature header (X-Signature)")
                return jsonify({
                    "success": False, 
                    "error": "Security validation active. Missing signature header (X-Signature)"
                }), 401
            
            payload = request.get_data()
            expected = hmac.new(WEBHOOK_SECRET, payload, hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(signature, expected):
                logger.warning("Rejected request: Invalid signature signature verification failed")
                return jsonify({
                    "success": False, 
                    "error": "Security validation failed. Invalid signature signature."
                }), 401
                
        return f(*args, **kwargs)
    return decorated
