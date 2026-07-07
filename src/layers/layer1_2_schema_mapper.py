"""
Layer 1.2: GenAI Schema Mapper
================================
Uses Gemini and Instructor to map raw columns of an unknown/custom CSV structure
to standard DQS/VISA schema target keys, with automatic error retry loops.
"""
import os
import json
import logging
from typing import List, Dict, Any

try:
    import google.generativeai as genai
    import instructor
    from pydantic import BaseModel, Field
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger("DQS.SchemaMapper")


if GEMINI_AVAILABLE:
    class SchemaMappingResult(BaseModel):
        """Pydantic model representing the mapping results to enforce strict schema structure."""
        mappings: Dict[str, str] = Field(
            ...,
            description="A dictionary mapping the standardized target field names (keys) to the matched raw CSV header names (values)."
        )


class GenAISchemaMapper:
    GEMINI_MODEL = "models/gemini-2.5-flash"

    # Canonical target fields we need for DQS evaluation
    TARGET_FIELDS = [
        "transaction_id", "amount", "currency", "timestamp", "status", "type",
        "network", "card_type", "bin", "last4",
        "merchant_id", "merchant_name", "merchant_category_code", "country",
        "customer_id", "email", "phone", "ip_address", "risk_score"
    ]

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.instructor_client = None
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                gemini_model = genai.GenerativeModel(self.GEMINI_MODEL)
                self.instructor_client = instructor.from_gemini(client=gemini_model)
            except Exception as e:
                logger.warning(f"Failed to configure Gemini model in schema mapper: {e}")

    def map_columns(self, raw_headers: List[str], sample_rows: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Queries Gemini to map raw headers to standard fields using Instructor for validation.
        Retries up to 3 times on schema failure, feeding error details back to the model.
        Returns a mapping dict: { "standard_field_name": "raw_csv_column_name" }
        """
        if not self.instructor_client or not raw_headers:
            return {}

        # Limit sample rows to avoid token waste
        samples = sample_rows[:3]
        
        # Build base system prompt
        base_prompt = f"""You are a payment data integration agent. Given a list of CSV headers and a few sample records from an unknown schema, map the raw columns to our standardized payment gateway schema.

Raw headers: {raw_headers}

Sample records:
{json.dumps(samples, indent=2)}

Our target standardized fields:
{self.TARGET_FIELDS}

You must map standard target fields to the matching raw column header names from the CSV input.
Only map headers where you are highly confident they correspond to the standardized field.
Do not invent mappings. Return a valid schema mapping dictionary."""

        errors_context = ""
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                prompt = base_prompt
                if errors_context:
                    prompt += f"\n\nCRITICAL: Your previous response failed validation with the following errors. Please correct them:\n{errors_context}"

                logger.info(f"GenAISchemaMapper call attempt {attempt}/{max_attempts}")
                
                # Request structured output via Instructor
                result = self.instructor_client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_model=SchemaMappingResult,
                )

                # Post-validate mapped fields against raw headers and valid target field checklist
                cleaned = {}
                for k, v in result.mappings.items():
                    k_clean = k.lower().strip()
                    if k_clean in self.TARGET_FIELDS and v in raw_headers:
                        cleaned[k_clean] = v
                    else:
                        raise ValueError(
                            f"Mapped key '{k_clean}' must be one of {self.TARGET_FIELDS} and mapped value '{v}' must be in raw headers {raw_headers}."
                        )

                logger.info(f"Gemini successfully mapped {len(cleaned)} headers on attempt {attempt}.")
                return cleaned

            except Exception as e:
                logger.warning(f"GenAISchemaMapper validation failed on attempt {attempt}: {str(e)}")
                errors_context = str(e)

        logger.error(f"GenAISchemaMapper failed completely after {max_attempts} attempts.")
        return {}
