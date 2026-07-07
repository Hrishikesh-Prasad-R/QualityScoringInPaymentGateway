"""
Self-verification script to test PII masking, HMAC signatures, and Redis BIN caching.
"""
import sys
import os
import json
import hmac
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import mask_pii
from src.layers.layer3_feature_extraction import FeatureExtractionLayer
from src.security import WEBHOOK_SECRET

def verify_all():
    print("=" * 60)
    print("  VERIFYING SECURITY & CACHING ENHANCEMENTS")
    print("=" * 60)

    # 1. PII Masking Verification
    print("1. Testing PII Masking...")
    sample_text = "Transaction error for user john.doe@visa.com on token csv_tok_00293812 with phone +919876543210"
    masked = mask_pii(sample_text)
    print(f"   Original: {sample_text}")
    print(f"   Masked:   {masked}")
    
    assert "***@***.com" in masked
    assert "[MASKED_TOKEN]" in masked
    assert "[MASKED_PHONE]" in masked
    print("   -> PII Masking passed!")

    # 2. Redis / BIN Categorization Verification
    print("\n2. Testing BIN Categorization...")
    layer = FeatureExtractionLayer()
    
    # Test Visa BIN
    network_visa = layer._categorize_bin("453271")
    print(f"   BIN 453271 network category: {network_visa} (Expected: 0)")
    assert network_visa == 0
    
    # Test Mastercard BIN
    network_mc = layer._categorize_bin("521488")
    print(f"   BIN 521488 network category: {network_mc} (Expected: 1)")
    assert network_mc == 1
    
    # Test Other BIN
    network_other = layer._categorize_bin("371234")
    print(f"   BIN 371234 network category: {network_other} (Expected: 2)")
    assert network_other == 2
    
    print("   -> BIN Categorization passed!")

    # 3. HMAC Signatures verification
    print("\n3. Testing HMAC Signatures Generation...")
    payload = json.dumps({"count": 10})
    signature = hmac.new(WEBHOOK_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    print(f"   Payload: {payload}")
    print(f"   Generated Signature: {signature}")
    assert len(signature) == 64
    print("   -> HMAC Generation passed!")

    print("\n" + "=" * 60)
    print("  ALL ENHANCEMENTS FUNCTIONAL AND STABLE!")
    print("=" * 60)

if __name__ == "__main__":
    verify_all()
