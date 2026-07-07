"""
Tests DQSDatabase integration
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dqs_engine import DQSEngine
from src.data_generator import generate_visa_transactions
from src.database import DQSDatabase

def test_db():
    print("Initializing engine...")
    engine = DQSEngine(use_ai=False)
    
    print("Generating transactions...")
    txns = generate_visa_transactions(5)
    
    print("Running pipeline...")
    result = engine.run(txns)
    
    print("Checking database...")
    db = DQSDatabase()
    history = db.get_history()
    print(f"Total Runs in DB: {len(history)}")
    
    if len(history) > 0:
        run = history[0]
        print(f"Latest Run ID: {run['id']}")
        print(f"Batch ID: {run['batch_id']}")
        print(f"Timestamp: {run['timestamp']}")
        print(f"Record Count: {run['record_count']}")
        print(f"Average DQS: {run['average_dqs']}")
        print(f"Quality Rate: {run['quality_rate']}")
        print(f"Duration: {run['total_duration_ms']:.1f}ms")
        print("[PASS] Database Logging works!")
    else:
        print("[FAIL] No records found in database history.")

if __name__ == "__main__":
    test_db()
