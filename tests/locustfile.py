"""
Locust Performance Load Test Script for DQS Engine
===================================================
Simulates high-concurrency transactional checks on the DQS API.
Generates HMAC-SHA256 signatures to support secure request validation.
"""
import json
import random
import hmac
import hashlib
from locust import HttpUser, task, between

# Secret key must match WEBHOOK_SECRET in src/security.py
WEBHOOK_SECRET = b"dqs_secure_webhook_secret_key_2026"

class DQSEngineLoadTest(HttpUser):
    # Wait between 100ms and 500ms between consecutive requests
    wait_time = between(0.1, 0.5)

    @task
    def test_run_endpoint(self):
        """Simulates users sending batches of transactions for real-time quality scoring."""
        payload_data = {
            "count": random.choice([10, 50, 100]),
            "anomaly_rate": 0.15,
            "use_ai": False
        }
        payload_bytes = json.dumps(payload_data).encode()
        
        # Calculate HMAC signature
        signature = hmac.new(WEBHOOK_SECRET, payload_bytes, hashlib.sha256).hexdigest()
        
        headers = {
            "Content-Type": "application/json",
            "X-Signature": signature
        }
        
        # POST request to transaction pipeline execution API
        with self.client.post("/api/run", data=payload_bytes, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                resp_json = response.json()
                if resp_json.get("success"):
                    response.success()
                else:
                    response.failure(f"API returned failure: {resp_json.get('error')}")
            else:
                response.failure(f"HTTP error status code {response.status_code}: {response.text}")
