"""
Layer 2: Input Validation Layer

Purpose: Verify that actual data complies with declared contract
Type: 100% Deterministic executable checks
Failure Mode: VALIDATION_FAILURE → SAFE_STOP

Updated to support both:
- JSON format with nested VISA transaction structure
- CSV format with flattened columns
"""
import os
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import pandas as pd

from ..models.schema import (
    SchemaManifest, 
    DataType, 
    VisaTransaction, 
    parse_visa_transaction,
    flatten_transactions,
)
from ..config import (
    MAX_FILE_SIZE_BYTES,
    MIN_ROW_COUNT,
    MAX_ROW_COUNT,
    LayerStatus,
)
from .layer1_input_contract import LayerResult


class InputValidationLayer:
    """
    Layer 2: Input Validation Layer
    
    Validates that actual data complies with the declared schema contract.
    Supports both JSON (nested) and CSV (flat) formats.
    All checks are 100% deterministic.
    """
    
    LAYER_ID = 2
    LAYER_NAME = "input_validation"
    
    def __init__(self, schema: SchemaManifest):
        self.schema = schema
        self.dataframe: Optional[pd.DataFrame] = None
        self.raw_transactions: List[Dict[str, Any]] = []
        self.file_hash: Optional[str] = None
    
    def validate(
        self,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        json_data: Optional[Union[Dict, List[Dict]]] = None,
    ) -> LayerResult:
        """
        Validate the input data against the schema contract.
        
        Args:
            file_path: Path to the data file (CSV or JSON)
            dataframe: Already loaded pandas DataFrame
            json_data: Direct JSON data (single record or list)
            
        Returns:
            LayerResult with validation status
        """
        import time
        start_time = time.time()
        
        issues = []
        warnings = []
        checks_performed = 0
        checks_passed = 0
        
        try:
            # ================================================================
            # DETERMINE INPUT SOURCE
            # ================================================================
            
            if file_path:
                result = self._validate_from_file(file_path, issues, warnings)
                if result:
                    return result
                checks_performed += 3  # File exists, size, parse
                checks_passed += 3
                
            elif json_data is not None:
                result = self._validate_from_json(json_data, issues, warnings)
                if result:
                    return result
                checks_performed += 1
                checks_passed += 1
                
            elif dataframe is not None:
                self.dataframe = dataframe.copy()
                self.dataframe.columns = [col.lower().strip() for col in self.dataframe.columns]
                self.file_hash = hashlib.sha256(
                    pd.util.hash_pandas_object(dataframe).values.tobytes()
                ).hexdigest()
                checks_performed += 1
                checks_passed += 1
            else:
                issues.append({
                    "type": "VALIDATION_FAILURE",
                    "code": "NO_DATA_PROVIDED",
                    "message": "No data provided (file_path, dataframe, or json_data required)",
                    "severity": "critical",
                })
                return self._create_result(
                    status=LayerStatus.FAILED,
                    start_time=start_time,
                    checks_performed=checks_performed,
                    checks_passed=checks_passed,
                    issues=issues,
                    can_continue=False,
                )
            
            # ================================================================
            # ROW COUNT CHECKS
            # ================================================================
            checks_performed += 1
            row_count = len(self.dataframe)
            
            if row_count < self.schema.min_rows:
                issues.append({
                    "type": "VALIDATION_FAILURE",
                    "code": "TOO_FEW_ROWS",
                    "message": f"Row count ({row_count}) below minimum ({self.schema.min_rows})",
                    "severity": "critical",
                })
                return self._create_result(
                    status=LayerStatus.FAILED,
                    start_time=start_time,
                    checks_performed=checks_performed,
                    checks_passed=checks_passed,
                    issues=issues,
                    can_continue=False,
                )
            
            if row_count > self.schema.max_rows:
                issues.append({
                    "type": "VALIDATION_FAILURE",
                    "code": "TOO_MANY_ROWS",
                    "message": f"Row count ({row_count}) exceeds maximum ({self.schema.max_rows})",
                    "severity": "critical",
                })
                return self._create_result(
                    status=LayerStatus.FAILED,
                    start_time=start_time,
                    checks_performed=checks_performed,
                    checks_passed=checks_passed,
                    issues=issues,
                    can_continue=False,
                )
            checks_passed += 1
            
            # ================================================================
            # SCHEMA COMPLIANCE CHECKS
            # ================================================================
            actual_columns = set(self.dataframe.columns)
            required_columns = set(self.schema.get_required_columns())
            
            # Check: All required columns present
            checks_performed += 1
            missing_required = required_columns - actual_columns
            if missing_required:
                # For nested schema, some columns may have different prefixes
                # Do a more lenient check
                truly_missing = []
                for col in missing_required:
                    # Check if any column contains the base name
                    base_name = col.split("_")[-1] if "_" in col else col
                    if not any(base_name in actual_col for actual_col in actual_columns):
                        truly_missing.append(col)
                
                if truly_missing:
                    issues.append({
                        "type": "VALIDATION_FAILURE",
                        "code": "MISSING_REQUIRED_COLUMNS",
                        "message": f"Missing required columns: {sorted(truly_missing)}",
                        "severity": "critical",
                    })
                    return self._create_result(
                        status=LayerStatus.FAILED,
                        start_time=start_time,
                        checks_performed=checks_performed,
                        checks_passed=checks_passed,
                        issues=issues,
                        can_continue=False,
                    )
            checks_passed += 1
            
            # Check: No duplicate column names
            checks_performed += 1
            if len(self.dataframe.columns) != len(set(self.dataframe.columns)):
                duplicates = [col for col in self.dataframe.columns 
                             if list(self.dataframe.columns).count(col) > 1]
                issues.append({
                    "type": "VALIDATION_FAILURE",
                    "code": "DUPLICATE_COLUMNS",
                    "message": f"Duplicate column names found: {set(duplicates)}",
                    "severity": "critical",
                })
                return self._create_result(
                    status=LayerStatus.FAILED,
                    start_time=start_time,
                    checks_performed=checks_performed,
                    checks_passed=checks_passed,
                    issues=issues,
                    can_continue=False,
                )
            checks_passed += 1
            
            # Check: Primary key exists and is unique
            checks_performed += 1
            if self.schema.primary_key:
                pk_columns = [pk.lower() for pk in self.schema.primary_key]
                missing_pk = set(pk_columns) - actual_columns
                
                if not missing_pk:
                    # Check uniqueness
                    if len(pk_columns) == 1:
                        is_unique = self.dataframe[pk_columns[0]].is_unique
                    else:
                        is_unique = not self.dataframe.duplicated(subset=pk_columns).any()
                    
                    if not is_unique:
                        warnings.append(f"Primary key {pk_columns} contains duplicates")
            checks_passed += 1
            
            # ================================================================
            # ALL CHECKS PASSED
            # ================================================================
            
            return self._create_result(
                status=LayerStatus.PASSED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                warnings=warnings,
                can_continue=True,
                details={
                    "file_hash": self.file_hash,
                    "row_count": row_count,
                    "column_count": len(actual_columns),
                    "columns_found": sorted(actual_columns),
                    "format": "nested_json" if self.raw_transactions else "flat",
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "VALIDATION_FAILURE",
                "code": "UNEXPECTED_ERROR",
                "message": f"Unexpected error during validation: {str(e)}",
                "severity": "critical",
            })
            return self._create_result(
                status=LayerStatus.FAILED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                can_continue=False,
            )
    
    def _validate_from_file(
        self, 
        file_path: str, 
        issues: List[Dict], 
        warnings: List[str]
    ) -> Optional[LayerResult]:
        """Validate from file path. Returns LayerResult on failure, None on success."""
        import time
        start_time = time.time()
        
        # Check: File exists
        if not os.path.exists(file_path):
            issues.append({
                "type": "VALIDATION_FAILURE",
                "code": "FILE_NOT_FOUND",
                "message": f"Data file not found: {file_path}",
                "severity": "critical",
            })
            return self._create_result(
                status=LayerStatus.FAILED,
                start_time=start_time,
                checks_performed=1,
                checks_passed=0,
                issues=issues,
                can_continue=False,
            )
        
        # Check: File size
        file_size = os.path.getsize(file_path)
        max_size = int(self.schema.max_file_size_mb * 1024 * 1024)
        if file_size > max_size:
            issues.append({
                "type": "VALIDATION_FAILURE",
                "code": "FILE_TOO_LARGE",
                "message": f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds limit ({self.schema.max_file_size_mb} MB)",
                "severity": "critical",
            })
            return self._create_result(
                status=LayerStatus.FAILED,
                start_time=start_time,
                checks_performed=2,
                checks_passed=1,
                issues=issues,
                can_continue=False,
            )
        
        # Calculate file hash
        self.file_hash = self._calculate_file_hash(file_path)
        
        # Parse based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".json":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle single record or list
                if isinstance(data, dict):
                    data = [data]
                
                self.raw_transactions = data
                
                # Flatten for DataFrame
                flattened = flatten_transactions(data)
                self.dataframe = pd.DataFrame(flattened)
                self.dataframe.columns = [col.lower().strip() for col in self.dataframe.columns]
                
            except json.JSONDecodeError as e:
                issues.append({
                    "type": "VALIDATION_FAILURE",
                    "code": "INVALID_JSON",
                    "message": f"Failed to parse JSON: {str(e)}",
                    "severity": "critical",
                })
                return self._create_result(
                    status=LayerStatus.FAILED,
                    start_time=start_time,
                    checks_performed=3,
                    checks_passed=2,
                    issues=issues,
                    can_continue=False,
                )
        
        elif ext == ".csv":
            try:
                self.dataframe = pd.read_csv(file_path, encoding='utf-8')
                self.dataframe.columns = [col.lower().strip() for col in self.dataframe.columns]
            except Exception as e:
                issues.append({
                    "type": "VALIDATION_FAILURE",
                    "code": "PARSE_ERROR",
                    "message": f"Failed to parse CSV: {str(e)}",
                    "severity": "critical",
                })
                return self._create_result(
                    status=LayerStatus.FAILED,
                    start_time=start_time,
                    checks_performed=3,
                    checks_passed=2,
                    issues=issues,
                    can_continue=False,
                )
        else:
            issues.append({
                "type": "VALIDATION_FAILURE",
                "code": "UNSUPPORTED_FORMAT",
                "message": f"Unsupported file format: {ext}",
                "severity": "critical",
            })
            return self._create_result(
                status=LayerStatus.FAILED,
                start_time=start_time,
                checks_performed=3,
                checks_passed=2,
                issues=issues,
                can_continue=False,
            )
        
        return None  # Success - continue validation
    
    def _validate_from_json(
        self, 
        json_data: Union[Dict, List[Dict]], 
        issues: List[Dict], 
        warnings: List[str]
    ) -> Optional[LayerResult]:
        """Validate from direct JSON data. Returns LayerResult on failure, None on success."""
        import time
        start_time = time.time()
        
        try:
            # Handle single record or list
            if isinstance(json_data, dict):
                json_data = [json_data]
            
            self.raw_transactions = json_data
            
            # Flatten for DataFrame
            flattened = flatten_transactions(json_data)
            self.dataframe = pd.DataFrame(flattened)
            self.dataframe.columns = [col.lower().strip() for col in self.dataframe.columns]
            
            # Calculate hash from JSON string
            self.file_hash = hashlib.sha256(
                json.dumps(json_data, sort_keys=True).encode()
            ).hexdigest()
            
        except Exception as e:
            issues.append({
                "type": "VALIDATION_FAILURE",
                "code": "JSON_PROCESSING_ERROR",
                "message": f"Failed to process JSON data: {str(e)}",
                "severity": "critical",
            })
            return self._create_result(
                status=LayerStatus.FAILED,
                start_time=start_time,
                checks_performed=1,
                checks_passed=0,
                issues=issues,
                can_continue=False,
            )
        
        return None  # Success
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for traceability."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _create_result(
        self,
        status: LayerStatus,
        start_time: float,
        checks_performed: int,
        checks_passed: int,
        issues: List[Dict[str, Any]],
        warnings: List[str] = None,
        can_continue: bool = True,
        details: Dict[str, Any] = None,
    ) -> LayerResult:
        """Create a standardized layer result."""
        import time
        return LayerResult(
            layer_id=self.LAYER_ID,
            layer_name=self.LAYER_NAME,
            status=status,
            execution_time_ms=(time.time() - start_time) * 1000,
            checks_performed=checks_performed,
            checks_passed=checks_passed,
            issues=issues,
            warnings=warnings or [],
            details=details or {},
            can_continue=can_continue,
        )
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the validated DataFrame (flattened if from JSON)."""
        return self.dataframe
    
    def get_raw_transactions(self) -> List[Dict[str, Any]]:
        """Get raw transaction records (if loaded from JSON)."""
        return self.raw_transactions
    
    def get_file_hash(self) -> Optional[str]:
        """Get the file hash for traceability."""
        return self.file_hash
