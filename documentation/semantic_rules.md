# Semantic Rules Reference — BR001 to BR015

Layer 4.3 (`SemanticValidationLayer`) evaluates 15 deterministic business rules on every record. **Critical** violations immediately escalate the transaction via the Decision Gate (Layer 9). **Warning** violations are flagged for review.

---

## Critical Rules (Violations → ESCALATE)

### BR001 — Amount Must Be Positive
- **Check:** `txn_amount > 0`
- **Rationale:** A zero or negative amount is structurally invalid for a payment transaction.
- **Violation message:** `"Transaction amount must be positive"`

### BR002 — Settlement Math Must Balance
- **Check:** `net_amount ≈ gross_amount − interchange_fee − gateway_fee` (±0.02 tolerance)
- **Rationale:** Discrepancies indicate processing errors or potential tampering.
- **Violation message:** `"Settlement math error: net {net} ≠ gross {gross} - fees {fees}"`

### BR003 — Settlement Date After Clearing Date
- **Check:** `settlement_date >= clearing_date`
- **Rationale:** Settlement cannot precede clearing — violates standard payment rail timing.
- **Violation message:** `"Settlement date before clearing date"`

### BR004 — Approved Transactions Must Have Auth Code
- **Check:** If `txn_status == "approved"`, then `authorization_code` must be non-empty.
- **Rationale:** An approved transaction without an auth code is a data integrity failure.
- **Violation message:** `"Approved transaction missing authorization code"`

### BR005 — Card Must Not Be Expired
- **Check:** `expiry_year > now.year OR (expiry_year == now.year AND expiry_month >= now.month)`
- **Rationale:** Transactions on expired cards should have been declined at the gateway.
- **Violation message:** `"Card expired: {month}/{year}"`

### BR008 — Geographic Consistency
- **Check:** `fraud_geo_passed == 1` AND (if domestic merchant: `customer_ip_is_domestic == 1`)
- **Rationale:** Flags impossible travel (e.g., merchant in US, IP in India with no travel time) and domestic merchants paired with foreign IPs.
- **Violation message:** `"Geographic inconsistency (failed geo check or domestic merchant with foreign IP)"`

### BR010 — Failed Velocity Must Correlate With Elevated Risk
- **Check:** If `fraud_velocity_passed == 0`, then `fraud_risk_score >= 40`
- **Rationale:** A velocity failure with a low risk score indicates inconsistent fraud signals — the risk model hasn't caught up with the velocity breach.
- **Violation message:** `"Failed velocity but low risk: {score}"`

### BR013 — BIN Prefix Must Match Card Network
- **Check:**
  - If `card_bin_category == 0` (VISA BIN): `card_network` must contain "visa"
  - If `card_bin_category == 1` (Mastercard BIN): `card_network` must contain "mastercard" or "mc"
- **Rationale:** A VISA BIN prefix (4xxxx) on a transaction claiming to be Mastercard is a clear data or fraud indicator.
- **Violation message:** `"BIN category {cat} but card network claims '{network}'"`

### BR014 — Critical Fields Must Match Format Requirements
- **Check (email):** If present, must match `^[^@]+@[^@]+\.[^@]+$`
- **Check (MCC):** If present, must match `^\d{4}$` (exactly 4 digits)
- **Check (TXN ID):** If present, must match `^[a-zA-Z0-9_-]+$` and be ≥ 5 chars
- **Rationale:** Malformed fields cannot be reliably processed or audited.
- **Violation message:** `"Invalid email format"` / `"Invalid MCC format"` / `"Invalid transaction ID format"`

### BR015 — Transaction Amount Must Not Be Extreme Outlier
- **Check:** `txn_amount <= 100,000` (INR)
- **Rationale:** Amounts above ₹1 lakh are statistical outliers (normal 99th percentile ≈ ₹8,800) requiring mandatory manual review.
- **Violation message:** `"Extreme transaction amount: {amount:,} (threshold: 100,000)"`

---

## Warning Rules (Violations → Flagged for Review)

### BR006 — Amount Should Be Rational for Merchant Category
- **Check:** Amount falls within expected range for MCC code (e.g., grocery ₹50–₹20,000)
- **Severity:** Warning
- **Note:** High amounts at low-ticket MCCs (e.g., ₹50,000 at a vending machine) are suspicious but not conclusive.

### BR007 — Risk Score Should Match Risk Level Label
- **Check:** `risk_level == "low"` → `risk_score < 40`; `risk_level == "high"` → `risk_score >= 70`
- **Severity:** Warning
- **Rationale:** Inconsistency between the ML risk score and its categorical label suggests a stale or misconfigured fraud system.

### BR009 — 3DS Required for High-Value Transactions
- **Check:** If `txn_amount > 10,000`, then `auth_result_encoded == 0` (authenticated)
- **Severity:** Warning
- **Rationale:** High-value transactions without 3DS authentication are a compliance risk under RBI and PSD2 guidelines.

### BR011 — Fee Ratio Should Be Reasonable
- **Check:** `settlement_fee_ratio < 0.05` (fees < 5% of gross amount)
- **Severity:** Warning
- **Rationale:** Unusually high fee ratios may indicate pricing errors or gateway misconfiguration.

### BR012 — Billing/Shipping Country Should Match Merchant Country (Domestic)
- **Check:** For IN merchants, billing and shipping addresses should be IN or blank.
- **Severity:** Warning
- **Rationale:** Domestic merchants with foreign billing addresses may indicate address fraud or data entry errors.

---

## How Rules Are Evaluated

```python
# In SemanticValidationLayer.validate()
for rule in self.rules:
    result = rule["check"](row, feat_row)
    result.severity = rule["severity"]   # orchestrator overrides hardcoded severity
    
    if not result.passed:
        if result.severity == "critical":
            critical_violations.append(result)
        else:
            warning_violations.append(result)
```

The orchestrator always overrides the severity returned by the check method with the severity declared in `_define_rules()`. This ensures rule metadata is the single source of truth.

---

## Adding a New Rule

1. Add entry to `_define_rules()` in `layer4_3_semantic.py`:
```python
{
    "rule_id": "BR016",
    "name": "Your rule description",
    "severity": "critical",   # or "warning"
    "check": self._check_your_rule,
},
```

2. Implement the check method:
```python
def _check_your_rule(self, row: pd.Series, feat_row: pd.Series) -> RuleResult:
    """BR016: Your rule docstring."""
    passed = <your_condition>
    return RuleResult(
        rule_id="BR016",
        rule_name="Your rule name",
        passed=passed,
        severity="critical",
        message="" if passed else "Violation description",
    )
```

3. Update `test_phase3.py` rule count assertion to `16`.
