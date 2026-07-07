"""
Validation test for GenAISchemaMapper
=====================================
Tests dynamic mapping of a completely non-standard CSV content (random names).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.csv_adapter import adapt_csv_to_visa

def test_mapper():
    print("=" * 60)
    print("  TESTING GENAI SCHEMA MAPPER")
    print("=" * 60)

    # Completely custom CSV content with zero exact matches to our schema
    custom_csv = """TxnRef,MoneySent,Denomination,TimestampOccurred,StatusState,CardBrand,CardClass,MerchantReference,StoreLocation,ClientCode,ClientMail,CustomerRiskFactor
TXN_998342,12500,INR,2026-07-07 12:30:15,Success,Visa,Credit,MERCH_8829,IN,CUST_442,test@customer.com,12
TXN_998343,450,INR,2026-07-07 12:32:00,Success,Mastercard,Debit,MERCH_8830,IN,CUST_443,test2@customer.com,45
"""
    # Check if API key is set
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[SKIP] GEMINI_API_KEY env var not set. Cannot run dynamic mapper test.")
        return

    print("Sending custom CSV to adapt_csv_to_visa...")
    txns, meta = adapt_csv_to_visa(custom_csv)

    print("\nMapping Metadata Results:")
    print(f"  Schema Compliance Score: {meta['compliance_score']:.1f}%")
    print(f"  AI Mapping Active: {meta['ai_mapped']}")
    print(f"  Mapped Fields: {meta['mapped_fields']}")
    print(f"  Warnings: {meta['warnings']}")

    if meta['ai_mapped'] and len(txns) > 0:
        print("\nSuccessfully parsed records! Sample of standard keys mapped:")
        t = txns[0]
        print(f"  transaction.transaction_id: {t['transaction']['transaction_id']} (from TxnRef)")
        print(f"  transaction.amount: {t['transaction']['amount']} (from MoneySent)")
        print(f"  transaction.currency: {t['transaction']['currency']} (from Denomination)")
        print(f"  card.network: {t['card']['network']} (from CardBrand)")
        print(f"  customer.customer_id: {t['customer']['customer_id']} (from ClientCode)")
        print(f"  customer.email: {t['customer']['email']} (from ClientMail)")
        print(f"  fraud.risk_score: {t['fraud']['risk_score']} (from CustomerRiskFactor)")
        print("\n[PASS] Dynamic column mapping functional!")
    else:
        print("\n[FAIL] Dynamic column mapping was not executed or failed to resolve keys.")

if __name__ == "__main__":
    test_mapper()
