"""
Layer 1: Input Contract Layer

Purpose: Define EXACTLY what data we accept (pre-flight checklist)
Type: Schema + Policy Definition
Failure Mode: CONTRACT_VIOLATION → SAFE_STOP

This layer validates that a schema manifest is provided and is valid.
It does NOT process data - it only validates the contract definition.
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..models.schema import SchemaManifest, create_default_transaction_schema
from ..config import (
    ACCEPTED_FORMATS,
    MAX_FILE_SIZE_BYTES,
    MIN_ROW_COUNT,
    MAX_ROW_COUNT,
    LayerStatus,
)


@dataclass
class LayerResult:
    """Standard result object for all layers."""
    layer_id: int
    layer_name: str
    status: LayerStatus
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    execution_time_ms: float = 0.0
    checks_performed: int = 0
    checks_passed: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    can_continue: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "execution_time_ms": self.execution_time_ms,
            "checks_performed": self.checks_performed,
            "checks_passed": self.checks_passed,
            "issues": self.issues,
            "warnings": self.warnings,
            "details": self.details,
            "can_continue": self.can_continue,
        }


class InputContractLayer:
    """
    Layer 1: Input Contract Layer
    
    Validates that a valid schema manifest is provided before any data processing.
    This layer acts as a gate - if no valid contract exists, processing stops.
    """
    
    LAYER_ID = 1
    LAYER_NAME = "input_contract"
    
    def __init__(self):
        self.schema: Optional[SchemaManifest] = None
    
    def validate_schema_manifest(
        self,
        schema_manifest: Optional[Dict[str, Any]] = None,
        schema_manifest_path: Optional[str] = None,
        use_default: bool = False,
    ) -> LayerResult:
        """
        Validate the provided schema manifest.
        
        Args:
            schema_manifest: Dictionary containing the schema manifest
            schema_manifest_path: Path to JSON file containing schema manifest
            use_default: If True, use the default transaction schema
            
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
            # Check 1: Schema source provided
            checks_performed += 1
            if schema_manifest is None and schema_manifest_path is None and not use_default:
                issues.append({
                    "type": "CONTRACT_VIOLATION",
                    "code": "SCHEMA_NOT_PROVIDED",
                    "message": "Schema manifest not provided. System cannot determine quality criteria without schema declaration.",
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
            
            # Load schema from appropriate source
            if use_default:
                self.schema = create_default_transaction_schema()
                warnings.append("Using default transaction schema")
            elif schema_manifest_path:
                # Check 2: File exists
                checks_performed += 1
                if not os.path.exists(schema_manifest_path):
                    issues.append({
                        "type": "CONTRACT_VIOLATION",
                        "code": "SCHEMA_FILE_NOT_FOUND",
                        "message": f"Schema manifest file not found: {schema_manifest_path}",
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
                
                # Check 3: Valid JSON
                checks_performed += 1
                try:
                    with open(schema_manifest_path, 'r', encoding='utf-8') as f:
                        schema_manifest = json.load(f)
                    checks_passed += 1
                except json.JSONDecodeError as e:
                    issues.append({
                        "type": "CONTRACT_VIOLATION",
                        "code": "INVALID_JSON",
                        "message": f"Schema manifest is not valid JSON: {str(e)}",
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
            
            # Check 4: Valid SchemaManifest structure
            if schema_manifest and not use_default:
                checks_performed += 1
                try:
                    self.schema = SchemaManifest(**schema_manifest)
                    checks_passed += 1
                except Exception as e:
                    issues.append({
                        "type": "CONTRACT_VIOLATION",
                        "code": "INVALID_SCHEMA_STRUCTURE",
                        "message": f"Schema manifest structure invalid: {str(e)}",
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
            
            # Check 5: For flat schema, at least one column defined
            #          For nested schema, at least one required section defined
            checks_performed += 1
            if self.schema.use_nested_schema:
                # Nested schema mode - check required_sections instead of columns
                if not self.schema.required_sections:
                    issues.append({
                        "type": "CONTRACT_VIOLATION",
                        "code": "NO_REQUIRED_SECTIONS",
                        "message": "Nested schema must define at least one required section",
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
            else:
                # Flat schema mode - check columns
                if not self.schema.columns:
                    issues.append({
                        "type": "CONTRACT_VIOLATION",
                        "code": "NO_COLUMNS_DEFINED",
                        "message": "Schema must define at least one column",
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
            
            # Check 6: Primary key validation (skip for nested schema as it's auto-derived)
            checks_performed += 1
            if not self.schema.use_nested_schema and self.schema.primary_key:
                column_names = {col.name for col in self.schema.columns}
                for pk_col in self.schema.primary_key:
                    if pk_col.lower() not in column_names:
                        issues.append({
                            "type": "CONTRACT_VIOLATION",
                            "code": "INVALID_PRIMARY_KEY",
                            "message": f"Primary key column '{pk_col}' not defined in schema",
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
            
            # Check 7: Quality thresholds are valid (0-100)
            checks_performed += 1
            thresholds = self.schema.quality_thresholds
            for dim_name in ["completeness", "accuracy", "validity", "uniqueness", "consistency", "timeliness", "integrity"]:
                value = getattr(thresholds, dim_name, 0)
                if not 0 <= value <= 100:
                    issues.append({
                        "type": "CONTRACT_VIOLATION",
                        "code": "INVALID_THRESHOLD",
                        "message": f"Quality threshold for '{dim_name}' must be between 0 and 100, got {value}",
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
            
            # All checks passed
            return self._create_result(
                status=LayerStatus.PASSED,
                start_time=start_time,
                checks_performed=checks_performed,
                checks_passed=checks_passed,
                issues=issues,
                warnings=warnings,
                can_continue=True,
                details={
                    "schema_name": self.schema.name,
                    "schema_version": self.schema.version,
                    "column_count": len(self.schema.columns),
                    "required_columns": self.schema.get_required_columns(),
                    "primary_key": self.schema.primary_key,
                    "business_rules_count": len(self.schema.business_rules),
                },
            )
            
        except Exception as e:
            issues.append({
                "type": "CONTRACT_VIOLATION",
                "code": "UNEXPECTED_ERROR",
                "message": f"Unexpected error during contract validation: {str(e)}",
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
    
    def get_schema(self) -> Optional[SchemaManifest]:
        """Get the validated schema manifest."""
        return self.schema
