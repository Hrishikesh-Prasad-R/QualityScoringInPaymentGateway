"""
DQS Engine History Database Manager
====================================
Manages the local SQLite history of DQS execution runs.
Stores metadata, record count, average DQS, and execution logs.
"""
import os
import sqlite3
import json
from datetime import datetime
import logging

logger = logging.getLogger("DQS.Database")

import re

def mask_pii(text: str) -> str:
    """Helper to redact sensitive data from strings (GDPR / PCI compliance)."""
    if not isinstance(text, str):
        return text
    # Mask emails
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '***@***.com',
        text
    )
    # Mask card pan tokens
    text = re.sub(
        r'\b(?:tok_|csv_tok_)\w+\b',
        '[MASKED_TOKEN]',
        text
    )
    # Mask phone numbers
    text = re.sub(
        r'(?:\+\d{1,3}[- ]?)?\d{10}\b',
        '[MASKED_PHONE]',
        text
    )
    return text

class DQSDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Save inside data/ relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "dqs_history.db")
        else:
            self.db_path = db_path
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates the run history table if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        execution_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        record_count INTEGER NOT NULL,
                        average_dqs REAL NOT NULL,
                        quality_rate REAL NOT NULL,
                        total_duration_ms REAL NOT NULL,
                        action_counts TEXT NOT NULL,
                        execution_report TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_batch_id ON execution_runs (batch_id)
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize SQLite DQS database: {e}")

    def save_run(
        self,
        batch_id: str,
        execution_id: str,
        timestamp: str,
        record_count: int,
        average_dqs: float,
        quality_rate: float,
        total_duration_ms: float,
        action_counts: dict,
        execution_report: str
    ) -> bool:
        """Persists a pipeline execution run summary."""
        try:
            # Scrub execution report of PII (GDPR / PCI compliance)
            masked_report = mask_pii(execution_report)
            
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO execution_runs (
                        batch_id, execution_id, timestamp, record_count,
                        average_dqs, quality_rate, total_duration_ms,
                        action_counts, execution_report
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    batch_id,
                    execution_id,
                    timestamp,
                    record_count,
                    average_dqs,
                    quality_rate,
                    total_duration_ms,
                    json.dumps(action_counts),
                    masked_report
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save execution run to database: {e}")
            return False

    def get_history(self, limit: int = 50) -> list:
        """Retrieves recent run history."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM execution_runs
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                history = []
                for row in rows:
                    run_dict = dict(row)
                    run_dict["action_counts"] = json.loads(run_dict["action_counts"])
                    history.append(run_dict)
                return history
        except Exception as e:
            logger.error(f"Failed to fetch execution run history: {e}")
            return []
