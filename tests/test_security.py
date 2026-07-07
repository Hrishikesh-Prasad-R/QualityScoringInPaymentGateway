"""
Tests API Webhook Security Signature Verification
"""
import sys
import os
import json
import hmac
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from src.security import WEBHOOK_SECRET

def test_webhook_security():
    print("=" * 60)
    print("  TESTING WEBHOOK SECURITY DECORATOR")
    print("=" * 60)

    # Enable signature validation in test mode
    os.environ["REQUIRE_HMAC"] = "true"
    
    client = app.test_client()
    payload = json.dumps({"count": 5, "use_ai": False})

    # Test 1: Missing X-Signature header
    print("Test 1: Requesting without signature header...")
    resp = client.post("/api/run", data=payload, content_type="application/json")
    print(f"  Status Code: {resp.status_code}")
    print(f"  Response: {resp.get_json()}")
    assert resp.status_code == 401
    assert "Missing signature" in resp.get_json()["error"]
    print("  -> Passed!")

    # Test 2: Invalid signature header
    print("\nTest 2: Requesting with invalid signature...")
    resp = client.post(
        "/api/run", 
        data=payload, 
        content_type="application/json",
        headers={"X-Signature": "invalid_hex_string"}
    )
    print(f"  Status Code: {resp.status_code}")
    print(f"  Response: {resp.get_json()}")
    assert resp.status_code == 401
    assert "Security validation failed" in resp.get_json()["error"]
    print("  -> Passed!")

    # Test 3: Valid signature header
    print("\nTest 3: Requesting with valid HMAC signature...")
    valid_sig = hmac.new(WEBHOOK_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    resp = client.post(
        "/api/run", 
        data=payload, 
        content_type="application/json",
        headers={"X-Signature": valid_sig}
    )
    print(f"  Status Code: {resp.status_code}")
    assert resp.status_code == 200
    print("  -> Passed!")
    
    # Clean up env
    os.environ["REQUIRE_HMAC"] = "false"
    print("\n[PASS] HMAC Webhook signature verification verified!")

if __name__ == "__main__":
    test_webhook_security()
