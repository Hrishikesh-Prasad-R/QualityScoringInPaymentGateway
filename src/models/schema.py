"""
Schema Models for the Data Quality Scoring Engine.
Defines the contract for what data we accept and how it should be structured.

Updated to support the comprehensive VISA transaction format with nested objects.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime


class DataType(str, Enum):
    """Supported data types for columns."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    EMAIL = "email"
    CURRENCY = "currency"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    OBJECT = "object"  # For nested structures


# ============================================================================
# VISA TRANSACTION SCHEMA - Nested Models
# ============================================================================

class TransactionDetails(BaseModel):
    """Transaction-level details."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    merchant_order_id: Optional[str] = Field(None, description="Merchant's order ID")
    type: str = Field(..., description="Transaction type (authorization, capture, refund)")
    amount: float = Field(..., ge=0, description="Transaction amount")
    currency: str = Field(default="INR", description="Currency code")
    timestamp: str = Field(..., description="Transaction timestamp (ISO 8601)")
    status: str = Field(..., description="Transaction status")
    response_code: Optional[str] = Field(None, description="Response code")
    authorization_code: Optional[str] = Field(None, description="Authorization code")


class CardDetails(BaseModel):
    """Card-level details."""
    network: str = Field(..., description="Card network (VISA, Mastercard, etc.)")
    pan_token: str = Field(..., description="Tokenized PAN")
    bin: str = Field(..., min_length=6, max_length=6, description="Bank Identification Number")
    last4: str = Field(..., min_length=4, max_length=4, description="Last 4 digits")
    expiry_month: str = Field(..., description="Expiry month (MM)")
    expiry_year: str = Field(..., description="Expiry year (YYYY)")
    card_type: str = Field(..., description="Card type (credit, debit, prepaid)")
    funding_source: Optional[str] = Field(None, description="Funding source")
    issuer_bank: Optional[str] = Field(None, description="Issuing bank name")


class MerchantDetails(BaseModel):
    """Merchant-level details."""
    merchant_id: str = Field(..., description="Merchant identifier")
    terminal_id: Optional[str] = Field(None, description="Terminal identifier")
    merchant_name: str = Field(..., description="Merchant name")
    merchant_category_code: str = Field(..., description="MCC code")
    country: str = Field(..., description="Merchant country")
    acquirer_bank: Optional[str] = Field(None, description="Acquirer bank")
    settlement_account: Optional[str] = Field(None, description="Settlement account")


class AddressDetails(BaseModel):
    """Address structure."""
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None


class CustomerDetails(BaseModel):
    """Customer-level details."""
    customer_id: str = Field(..., description="Customer identifier")
    email: Optional[str] = Field(None, description="Customer email")
    phone: Optional[str] = Field(None, description="Customer phone")
    billing_address: Optional[AddressDetails] = None
    shipping_address: Optional[AddressDetails] = None
    ip_address: Optional[str] = Field(None, description="Customer IP address")
    device_fingerprint: Optional[str] = Field(None, description="Device fingerprint")
    user_agent: Optional[str] = Field(None, description="User agent string")


class AuthenticationDetails(BaseModel):
    """3DS Authentication details."""
    three_ds_version: Optional[str] = Field(None, description="3DS version")
    eci: Optional[str] = Field(None, description="ECI indicator")
    cavv: Optional[str] = Field(None, description="CAVV value")
    ds_transaction_id: Optional[str] = Field(None, description="DS Transaction ID")
    authentication_result: Optional[str] = Field(None, description="Authentication result")


class FraudDetails(BaseModel):
    """Fraud scoring details."""
    risk_score: Optional[float] = Field(None, ge=0, le=100, description="Risk score 0-100")
    risk_level: Optional[str] = Field(None, description="Risk level (low, medium, high)")
    velocity_check: Optional[str] = Field(None, description="Velocity check result")
    geo_check: Optional[str] = Field(None, description="Geo check result")


class NetworkDetails(BaseModel):
    """Network-level details."""
    network_transaction_id: Optional[str] = Field(None, description="Network transaction ID")
    acquirer_reference_number: Optional[str] = Field(None, description="ARN")
    routing_region: Optional[str] = Field(None, description="Routing region")
    interchange_category: Optional[str] = Field(None, description="Interchange category")


class ComplianceDetails(BaseModel):
    """Compliance and audit details."""
    sca_applied: Optional[bool] = Field(None, description="SCA applied flag")
    psd2_exemption: Optional[str] = Field(None, description="PSD2 exemption type")
    aml_screening: Optional[str] = Field(None, description="AML screening result")
    tax_reference: Optional[str] = Field(None, description="Tax reference")
    audit_log_id: Optional[str] = Field(None, description="Audit log ID")


class SettlementDetails(BaseModel):
    """Settlement details."""
    settlement_batch_id: Optional[str] = Field(None, description="Settlement batch ID")
    clearing_date: Optional[str] = Field(None, description="Clearing date")
    settlement_date: Optional[str] = Field(None, description="Settlement date")
    gross_amount: Optional[float] = Field(None, description="Gross amount")
    interchange_fee: Optional[float] = Field(None, description="Interchange fee")
    gateway_fee: Optional[float] = Field(None, description="Gateway fee")
    net_amount: Optional[float] = Field(None, description="Net amount")


class BusinessMetadata(BaseModel):
    """Business metadata."""
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    product_category: Optional[str] = Field(None, description="Product category")
    promo_code: Optional[str] = Field(None, description="Promo code used")
    campaign: Optional[str] = Field(None, description="Campaign name")
    notes: Optional[str] = Field(None, description="Internal notes")


class VisaTransaction(BaseModel):
    """
    Complete VISA Transaction Record.
    This is the primary data structure for quality assessment.
    """
    transaction: TransactionDetails
    card: CardDetails
    merchant: MerchantDetails
    customer: CustomerDetails
    authentication: Optional[AuthenticationDetails] = None
    fraud: Optional[FraudDetails] = None
    network: Optional[NetworkDetails] = None
    compliance: Optional[ComplianceDetails] = None
    settlement: Optional[SettlementDetails] = None
    business_metadata: Optional[BusinessMetadata] = None
    
    def flatten(self) -> Dict[str, Any]:
        """Flatten nested structure to single-level dict for DataFrame processing."""
        flat = {}
        
        # Transaction details
        for key, value in self.transaction.model_dump().items():
            flat[f"txn_{key}"] = value
        
        # Card details
        for key, value in self.card.model_dump().items():
            flat[f"card_{key}"] = value
        
        # Merchant details
        for key, value in self.merchant.model_dump().items():
            flat[f"merchant_{key}"] = value
        
        # Customer details (flatten addresses)
        customer_dict = self.customer.model_dump()
        for key, value in customer_dict.items():
            if key in ["billing_address", "shipping_address"] and value:
                for addr_key, addr_value in value.items():
                    flat[f"customer_{key}_{addr_key}"] = addr_value
            else:
                flat[f"customer_{key}"] = value
        
        # Optional sections
        if self.authentication:
            for key, value in self.authentication.model_dump().items():
                flat[f"auth_{key}"] = value
        
        if self.fraud:
            for key, value in self.fraud.model_dump().items():
                flat[f"fraud_{key}"] = value
        
        if self.network:
            for key, value in self.network.model_dump().items():
                flat[f"network_{key}"] = value
        
        if self.compliance:
            for key, value in self.compliance.model_dump().items():
                flat[f"compliance_{key}"] = value
        
        if self.settlement:
            for key, value in self.settlement.model_dump().items():
                flat[f"settlement_{key}"] = value
        
        if self.business_metadata:
            for key, value in self.business_metadata.model_dump().items():
                flat[f"biz_{key}"] = value
        
        return flat


# ============================================================================
# COLUMN DEFINITION FOR FLAT SCHEMA
# ============================================================================

class ColumnDefinition(BaseModel):
    """Definition for a single column in the schema."""
    name: str = Field(..., description="Column name")
    data_type: DataType = Field(..., description="Expected data type")
    required: bool = Field(default=True, description="Whether this column is required")
    nullable: bool = Field(default=False, description="Whether NULL values are allowed")
    unique: bool = Field(default=False, description="Whether values must be unique")
    min_value: Optional[float] = Field(default=None, description="Minimum value for numeric types")
    max_value: Optional[float] = Field(default=None, description="Maximum value for numeric types")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for validation")
    allowed_values: Optional[List[str]] = Field(default=None, description="Enum of allowed values")
    
    @field_validator('name')
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Column name cannot be empty')
        return v.strip().lower()


class QualityThresholds(BaseModel):
    """Quality thresholds per dimension."""
    completeness: float = Field(default=95.0, ge=0, le=100)
    accuracy: float = Field(default=90.0, ge=0, le=100)
    validity: float = Field(default=99.0, ge=0, le=100)
    uniqueness: float = Field(default=99.9, ge=0, le=100)
    consistency: float = Field(default=95.0, ge=0, le=100)
    timeliness: float = Field(default=90.0, ge=0, le=100)
    integrity: float = Field(default=95.0, ge=0, le=100)


class BusinessRule(BaseModel):
    """A business rule for semantic validation."""
    rule_id: str = Field(..., description="Unique rule identifier")
    description: str = Field(..., description="Human-readable description")
    expression: str = Field(..., description="Rule expression")
    severity: str = Field(default="warning", description="warning or critical")


class SchemaManifest(BaseModel):
    """
    The complete schema manifest that defines what data we accept.
    This is the INPUT CONTRACT (Layer 1).
    """
    name: str = Field(..., description="Name of this schema")
    version: str = Field(default="1.0", description="Schema version")
    description: Optional[str] = Field(default=None, description="Schema description")
    
    # Column definitions (for flat CSV format)
    columns: List[ColumnDefinition] = Field(default_factory=list, description="List of column definitions")
    
    # Use nested VISA schema
    use_nested_schema: bool = Field(default=True, description="Use VISA nested schema")
    
    # Required sections for nested schema
    required_sections: List[str] = Field(
        default=["transaction", "card", "merchant", "customer"],
        description="Required top-level sections"
    )
    
    # Primary key
    primary_key: Optional[List[str]] = Field(default=["txn_transaction_id"], description="Primary key column(s)")
    
    # Quality thresholds
    quality_thresholds: QualityThresholds = Field(
        default_factory=QualityThresholds,
        description="Quality thresholds per dimension"
    )
    
    # Business rules
    business_rules: List[BusinessRule] = Field(
        default_factory=list,
        description="Business rules for semantic validation"
    )
    
    # Format constraints
    accepted_formats: List[str] = Field(default=["json", "csv"], description="Accepted file formats")
    max_file_size_mb: float = Field(default=10.0, description="Maximum file size in MB")
    min_rows: int = Field(default=1, description="Minimum row count")
    max_rows: int = Field(default=50000, description="Maximum row count")
    
    def get_required_columns(self) -> List[str]:
        """Get list of required column names."""
        if self.use_nested_schema:
            # Core required fields from nested schema
            return [
                "txn_transaction_id",
                "txn_amount",
                "txn_currency",
                "txn_timestamp",
                "txn_status",
                "card_network",
                "card_pan_token",
                "card_bin",
                "merchant_merchant_id",
                "merchant_merchant_name",
                "merchant_merchant_category_code",
                "merchant_country",
                "customer_customer_id",
            ]
        return [col.name for col in self.columns if col.required]
    
    def get_column_by_name(self, name: str) -> Optional[ColumnDefinition]:
        """Get column definition by name."""
        for col in self.columns:
            if col.name == name.lower():
                return col
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()


# ============================================================================
# DEFAULT VISA TRANSACTION SCHEMA
# ============================================================================
def create_default_transaction_schema() -> SchemaManifest:
    """Create the default VISA transaction schema."""
    return SchemaManifest(
        name="visa_transaction_schema",
        version="2.0",
        description="Comprehensive VISA transaction data quality assessment schema",
        use_nested_schema=True,
        required_sections=["transaction", "card", "merchant", "customer"],
        primary_key=["txn_transaction_id"],
        quality_thresholds=QualityThresholds(
            completeness=95.0,
            accuracy=90.0,
            validity=99.0,
            uniqueness=99.9,
            consistency=95.0,
            timeliness=90.0,
            integrity=95.0,
        ),
        business_rules=[
            BusinessRule(
                rule_id="BR001",
                description="Settlement date must be after clearing date",
                expression="settlement_date >= clearing_date",
                severity="critical",
            ),
            BusinessRule(
                rule_id="BR002",
                description="Net amount must equal gross minus fees",
                expression="net_amount == gross_amount - interchange_fee - gateway_fee",
                severity="critical",
            ),
            BusinessRule(
                rule_id="BR003",
                description="Transaction amount must be positive",
                expression="amount > 0",
                severity="critical",
            ),
            BusinessRule(
                rule_id="BR004",
                description="Risk score must be 0-100",
                expression="0 <= risk_score <= 100",
                severity="warning",
            ),
            BusinessRule(
                rule_id="BR005",
                description="Card expiry must be in future",
                expression="expiry_date > current_date",
                severity="critical",
            ),
        ],
        accepted_formats=["json", "csv"],
        max_file_size_mb=10.0,
        min_rows=1,
        max_rows=50000,
    )


def parse_visa_transaction(data: Dict[str, Any]) -> VisaTransaction:
    """Parse a dictionary into a VisaTransaction object."""
    return VisaTransaction(**data)


def flatten_transactions(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten a list of nested transactions for DataFrame creation."""
    flattened = []
    for txn_data in transactions:
        txn = parse_visa_transaction(txn_data)
        flattened.append(txn.flatten())
    return flattened
