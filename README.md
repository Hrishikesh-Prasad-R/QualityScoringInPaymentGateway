# Data Quality Scoring Engine (Payment Gateway)

A comprehensive data quality scoring engine for payment gateway transactions, featuring deterministic rule-based validation and AI-powered anomaly detection.

## 🚀 Key Features

*   **15-Layer Integration Pipeline**: From input validation to final decision making.
*   **Hybrid Intelligence**: Combines deterministic business rules ("Rules Enforce") with Machine Learning and GenAI ("ML Informs").
*   **AI-Powered Summaries**: Uses Google Gemini API (Layer 4.5) to generate human-readable explanations for data quality issues.
*   **Comprehensive Audit Trail**: Full responsibility tracking and execution logging.
*   **Robust Architecture**:
    *   **Phase 1**: Foundation (Input Contract & Validation)
    *   **Phase 2**: Feature Extraction (35+ dimensions)
    *   **Phase 3**: Deterministic Inference (Structural, Compliance, Semantic)
    *   **Phase 4**: AI Inference (Anomaly Detection, GenAI Summarization)
    *   **Phase 5**: Output & Decision (Stability, Conflict, Confidence, Decision Gate)
    *   **Phase 6**: Responsibility & Logging
    *   **Phase 7**: Integration & Traceability

## 🛠️ Tech Stack

*   **Python 3.10+**
*   **Pandas & NumPy**: Data manipulation and analysis.
*   **Scikit-Learn**: Isolation Forest for anomaly detection.
*   **Google Generative AI**: Gemini API for summarization.
*   **Pytest**: Comprehensive test suite (88+ tests).

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Suraj-B12/QualityScoringInPaymentGateway.git
    cd QualityScoringInPaymentGateway
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Create a requirements.txt if not present)*

3.  Set up environment variables (Optional for AI features):
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```

## ⚡ Usage

### Running the Demo
Execute the full end-to-end pipeline demo:
```bash
python src/dqs_engine.py
```

### Basic Implementation
```python
from src.dqs_engine import DQSEngine
from src.data_generator import generate_visa_transactions

# Generate sample data
transactions = generate_visa_transactions(n_transactions=50)

# Initialize Engine
engine = DQSEngine(use_ai=True)  # Set use_ai=False to skip Gemini

# Run Pipeline
result = engine.run(transactions)

# View Results
print(f"Decision Report:\n{result.decision_report}")
print(f"Quality Rate: {result.quality_rate}%")
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
pytest tests/
```

## 🏗️ Architecture Overview

The engine follows a strict "Rules Enforce, ML Informs, Humans Decide" philosophy:
1.  **Layers 1-3**: Prepare and validate data structure.
2.  **Layers 4.1-4.3**: Apply hard business rules (can reject).
3.  **Layers 4.4-4.5**: Apply ML insights (can only flag).
4.  **Layers 5-9**: Synthesize signals into a final decision.
5.  **Layers 10-11**: Ensure accountability and traceability.

## 📄 License

MIT License
