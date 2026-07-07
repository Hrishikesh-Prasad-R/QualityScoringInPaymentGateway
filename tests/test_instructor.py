import unittest
from unittest.mock import MagicMock, patch
from src.layers.layer1_2_schema_mapper import GenAISchemaMapper, SchemaMappingResult
from src.layers.layer4_5_summarization import GenAISummarizationLayer, FieldCorrectionsBatch, FieldCorrection


class TestInstructorIntegration(unittest.TestCase):

    def test_schema_mapper_instantiation(self):
        """Test that the Schema Mapper instantiates with instructor patched."""
        with patch('google.generativeai.GenerativeModel') as mock_model, \
             patch('instructor.from_gemini') as mock_instructor:
            
            mapper = GenAISchemaMapper(api_key="mock_key")
            self.assertIsNotNone(mapper.api_key)
            mock_model.assert_called_once_with("models/gemini-2.5-flash")
            mock_instructor.assert_called_once()

    def test_schema_mapper_retry_loop_success(self):
        """Test that mapping succeeds if it returns correct schema on attempt 1."""
        mapper = GenAISchemaMapper(api_key="mock_key")
        mapper.instructor_client = MagicMock()
        
        # Mock structured response
        mock_result = SchemaMappingResult(mappings={
            "transaction_id": "raw_txn_id",
            "amount": "raw_amount"
        })
        mapper.instructor_client.chat.completions.create.return_value = mock_result
        
        raw_headers = ["raw_txn_id", "raw_amount"]
        result = mapper.map_columns(raw_headers, sample_rows=[])
        
        self.assertEqual(result, {"transaction_id": "raw_txn_id", "amount": "raw_amount"})
        self.assertEqual(mapper.instructor_client.chat.completions.create.call_count, 1)

    def test_schema_mapper_retry_loop_recovery(self):
        """Test that mapping retries up to 3 times if exceptions occur, then fails gracefully."""
        mapper = GenAISchemaMapper(api_key="mock_key")
        mapper.instructor_client = MagicMock()
        
        # Raise error on all attempts
        mapper.instructor_client.chat.completions.create.side_effect = ValueError("Parsing failed")
        
        raw_headers = ["raw_txn_id"]
        result = mapper.map_columns(raw_headers, sample_rows=[])
        
        # Should return empty and call exactly 3 times
        self.assertEqual(result, {})
        self.assertEqual(mapper.instructor_client.chat.completions.create.call_count, 3)

    def test_summarization_corrections_retry_loop(self):
        """Test that summarizer corrections retry loop triggers properly."""
        layer = GenAISummarizationLayer(api_key="mock_key", use_ai=True)
        layer.instructor_client = MagicMock()
        
        # Raise ValueError to trigger retries
        layer.instructor_client.chat.completions.create.side_effect = ValueError("Schema validation failed")
        
        result = layer._generate_field_corrections(
            record_id="txn_123",
            quality_data={"amount": 100, "risk_score": 10},
            key_issues=["issue 1"]
        )
        
        self.assertEqual(result, [])
        self.assertEqual(layer.instructor_client.chat.completions.create.call_count, 3)


if __name__ == "__main__":
    unittest.main()
