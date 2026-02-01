# SentiSight - AI-Powered Customer Feedback Analyzer

<div align="center">

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Production-grade system for analyzing customer feedback using transformer models and ML algorithms**

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [API](#api-documentation) • [Architecture](#architecture)

</div>

---

## Overview

SentiSight is an enterprise-ready customer feedback analysis platform that leverages state-of-the-art AI models to extract actionable insights from customer communications. Built with scalability and performance in mind, it processes millions of feedback entries efficiently using advanced NLP techniques.

### Key Capabilities

- **Sentiment Analysis** - Deep learning-based emotion detection using DistilBERT transformers
- **Category Classification** - Multi-label categorization across 8 business domains
- **Anomaly Detection** - Isolation Forest algorithm to identify urgent/critical issues
- **Semantic Search** - Vector-based similarity search powered by PostgreSQL pgvector
- **Automated Insights** - LLM-powered report generation with trend analysis
- **REST API** - Production-ready FastAPI endpoints with auto-documentation
- **Interactive Dashboard** - Real-time analysis interface built with Streamlit

---

## Features

### 🤖 Machine Learning Models

| Model | Technology | Performance |
|-------|-----------|-------------|
| Sentiment Analysis | DistilBERT (HuggingFace) | ~95% accuracy |
| Category Classifier | TF-IDF + Hybrid ML | ~85% accuracy |
| Anomaly Detection | Isolation Forest | Configurable threshold |

### 📊 Supported Categories

- Product Quality
- Delivery & Shipping
- Customer Service
- Billing & Payment
- Technical Issues
- Account Management
- Returns & Refunds
- General Inquiry

### 🔍 Core Features

- **Batch Processing** - Handle thousands of feedbacks efficiently with chunked loading
- **Memory Optimization** - Stream processing for datasets with millions of records
- **Real-time Analysis** - Sub-200ms response time for single feedback
- **Scalable Architecture** - Docker containerization with microservices design
- **Vector Search** - Find similar complaints using semantic embeddings
- **LLM Integration** - Optional GPT-4/Gemini for advanced insights

---

## Installation

### Prerequisites

- Python 3.13 or higher
- PostgreSQL 14+ (optional, for production deployment)
- 4GB RAM minimum
- Docker (optional, for containerized deployment)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/sentisight.git
cd sentisight

# Install dependencies using uv (recommended)
pip install uv
uv sync

# Or use pip
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Optional: LLM Integration
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here

# Optional: Database (for production)
DATABASE_URL=postgresql://user:password@localhost:5432/sentisight

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Quick Start

### Option 1: Streamlit Dashboard (Recommended)

Launch the interactive web interface:

```bash
python scripts/run_dashboard.py
```

Access the dashboard at: **http://localhost:8501**

**Features:**
- Single feedback analysis with real-time results
- Batch processing via CSV upload or text paste
- Interactive analytics with visualizations
- Anomaly detection and flagging

### Option 2: REST API

Start the FastAPI server:

```bash
python api/main.py
```

API documentation available at: **http://localhost:8000/docs**

**Example API Usage:**

```bash
# Analyze single feedback
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product! Fast delivery and excellent quality."}'

# Batch analysis
curl -X POST "http://localhost:8000/batch-analyze" \
  -H "Content-Type: application/json" \
  -d '{"feedbacks": [{"text": "Good service"}, {"text": "Poor experience"}]}'

# Get insights
curl "http://localhost:8000/insights"

# Detect anomalies
curl "http://localhost:8000/anomalies?top_n=10"
```

### Option 3: Python Integration

```python
from src.models import SentimentAnalyzer, CategoryClassifier
from src.preprocessing import TextPreprocessor

# Initialize models
sentiment_analyzer = SentimentAnalyzer()
category_classifier = CategoryClassifier()
preprocessor = TextPreprocessor()

# Analyze feedback
text = "The product quality is excellent, but delivery was delayed."
cleaned = preprocessor.clean_text(text)

# Get sentiment
sentiment = sentiment_analyzer.analyze(cleaned)
print(f"Sentiment: {sentiment['sentiment']} ({sentiment['confidence']:.2%})")

# Get categories
categories = category_classifier.predict(cleaned)
print(f"Categories: {', '.join(categories)}")
```

---

## Performance & GPU (Measured)

**Measured on NVIDIA Quadro M1200 (4GB) with Python 3.12 + CUDA PyTorch**  
- DistilBERT model load (first warm run, GPU): **~9.2 s**
- Inference (single feedback, GPU): **~13.4 ms**
- Estimated Streamlit-ready time (models + framework): **~13 s**

**Notes:**
- First-time runs that download model weights may take longer (~45–80 s).
- To use GPU locally on Windows we installed Python **3.12** and `torch` with CUDA (cu121). See `STARTUP_GUIDE.md` for the downgrade steps and Docker GPU tips.

---

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed system status |
| `POST` | `/analyze` | Analyze single feedback |
| `POST` | `/batch-analyze` | Analyze multiple feedbacks |
| `GET` | `/insights` | Get aggregated insights |
| `GET` | `/anomalies` | Get detected anomalies |
| `POST` | `/clear` | Clear stored data |

### Request/Response Examples

**Single Analysis:**

```json
// Request
{
  "text": "Excellent customer service! Very helpful and responsive."
}

// Response
{
  "text": "Excellent customer service! Very helpful and responsive.",
  "sentiment": {
    "label": "positive",
    "confidence": 0.9876
  },
  "categories": ["Customer Service"],
  "is_anomaly": false,
  "processing_time_seconds": 0.143
}
```

**Batch Analysis:**

```json
// Request
{
  "feedbacks": [
    {"text": "Fast delivery"},
    {"text": "Product broken on arrival"}
  ]
}

// Response
{
  "results": [
    {
      "text": "Fast delivery",
      "sentiment": {"label": "positive", "confidence": 0.92},
      "categories": ["Delivery & Shipping"],
      "is_anomaly": false
    },
    {
      "text": "Product broken on arrival",
      "sentiment": {"label": "negative", "confidence": 0.88},
      "categories": ["Product Quality", "Delivery & Shipping"],
      "is_anomaly": true
    }
  ],
  "total_processed": 2,
  "total_time_seconds": 0.287
}
```

---

## Architecture

### System Design

```
┌─────────────────┐
│   Streamlit     │
│   Dashboard     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   FastAPI       │◄────►│  PostgreSQL  │
│   REST API      │      │  + pgvector  │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────────────────────┐
│      ML Models Layer            │
├─────────────────────────────────┤
│  • DistilBERT (Sentiment)       │
│  • TF-IDF Classifier (Category) │
│  • Isolation Forest (Anomaly)   │
└─────────────────────────────────┘
```

### Technology Stack

**Backend:**
- FastAPI - High-performance web framework
- Pydantic - Data validation
- SQLAlchemy - ORM for database operations

**ML/AI:**
- Transformers (HuggingFace) - DistilBERT model
- scikit-learn - ML algorithms
- sentence-transformers - Semantic embeddings
- PyTorch - Deep learning backend

**Frontend:**
- Streamlit - Interactive web interface
- Plotly - Data visualizations

**Database:**
- PostgreSQL 14+ - Primary data store
- pgvector - Vector similarity search

**DevOps:**
- Docker - Containerization
- Docker Compose - Multi-container orchestration
- pytest - Testing framework

---

## Docker Deployment

### Quick Deploy

```bash
# Windows
deployment\deploy.bat

# Linux/Mac
bash deployment/deploy.sh
```

### Manual Docker Setup

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- PostgreSQL: localhost:5432
- API Docs: http://localhost:8000/docs

### Production Deployment

```bash
# With nginx reverse proxy
docker-compose --profile production up -d
```

---

## Data Processing

### Handling Large Datasets

SentiSight is optimized for memory-efficient processing of large datasets:

```python
from src.preprocessing import DataLoader

# Load data in chunks to avoid memory issues
loader = DataLoader("data/large_dataset.csv")

for chunk in loader.load_chunks(chunk_size=1000):
    # Process each chunk
    results = process_batch(chunk)
    save_results(results)
```

### Sample Workflow

```python
from src.preprocessing import TextPreprocessor, DataLoader
from src.models import SentimentAnalyzer, CategoryClassifier, AnomalyDetector

# Initialize
preprocessor = TextPreprocessor()
sentiment_analyzer = SentimentAnalyzer()
category_classifier = CategoryClassifier()
anomaly_detector = AnomalyDetector()

# Load and process data
loader = DataLoader("customer_feedback.csv")
results = []

for chunk in loader.load_chunks(chunk_size=1000):
    # Clean text
    chunk['cleaned'] = chunk['text'].apply(preprocessor.clean_text)
    
    # Extract features
    features = chunk['cleaned'].apply(preprocessor.extract_features)
    
    # Analyze
    for idx, row in chunk.iterrows():
        sentiment = sentiment_analyzer.analyze(row['cleaned'])
        categories = category_classifier.predict(row['cleaned'])
        
        results.append({
            'text': row['text'],
            'sentiment': sentiment['sentiment'],
            'categories': categories
        })

# Train anomaly detector
anomaly_detector.fit(feature_matrix)
anomalies = anomaly_detector.predict(feature_matrix)
```

---

## Insights Engine

### Generate Automated Reports

```python
from src.insights_engine import InsightsEngine

# Initialize (with or without LLM)
engine = InsightsEngine(use_llm=False)  # Rule-based
# engine = InsightsEngine(use_llm=True, llm_provider="openai")  # LLM-powered

# Generate insights
insights = engine.generate_daily_insights(analyzed_feedbacks)

# Export report
markdown_report = engine.export_report(insights, format="markdown")
html_report = engine.export_report(insights, format="html")

print(markdown_report)
```

**Output:**
- Statistical summaries
- Trend analysis
- Top issues and categories
- Actionable recommendations
- Sentiment distribution

---

## Testing

Run the test suite:

```bash
# Run all tests
python scripts/run_tests.py

# Or use pytest directly
pytest tests/ -v

# With coverage report
pytest --cov=src --cov=api tests/
```

**Test Coverage:**
- Preprocessing and text cleaning
- Model predictions and accuracy
- API endpoints and responses
- Database operations

---

## Database Schema

### PostgreSQL Setup

```bash
# Create database
psql -U postgres -c "CREATE DATABASE sentisight;"

# Install pgvector extension
psql -U postgres -d sentisight -c "CREATE EXTENSION vector;"

# Run schema migration
psql -U postgres -d sentisight -f db/schema.sql
```

### Tables

- `feedback` - Customer feedback entries
- `sentiment_results` - Sentiment analysis results
- `categories` - Category classifications
- `anomalies` - Detected anomalies
- `insights` - Aggregated insights

---

## Performance Optimization

### Recommended Settings

**For Large Datasets (1M+ records):**
- Use chunked loading with `chunk_size=1000`
- Enable batch processing in API
- Configure PostgreSQL connection pooling
- Use Docker with increased memory limits

**For Real-time Analysis:**
- Keep models pre-loaded in memory
- Use GPU acceleration for transformers (if available)
- Enable API response caching
- Optimize database indexes

---

## Troubleshooting

### Common Issues

**Memory Errors:**
```bash
# Reduce chunk size
loader.load_chunks(chunk_size=500)
```

**Model Download Fails:**
```bash
# First run downloads DistilBERT (~250MB)
# Ensure stable internet connection
# Check disk space (1GB+ recommended)
```

**Port Already in Use:**
```bash
# API on different port
uvicorn api.main:app --port 8001

# Dashboard on different port
streamlit run frontend/app.py --server.port 8502
```

**Docker Issues:**
```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## Project Structure

```
sentisight/
├── api/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── schemas.py         # Pydantic models
│   └── __init__.py
├── src/                   # Core modules
│   ├── preprocessing.py   # Text processing
│   ├── db.py             # Database operations
│   ├── insights_engine.py # Report generation
│   └── models/           # ML models
│       ├── sentiment_analyzer.py
│       ├── category_classifier.py
│       └── anomaly_detector.py
├── frontend/             # Streamlit dashboard
│   └── app.py
├── tests/                # Test suite
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── notebooks/            # Jupyter notebooks
│   └── 01_eda.ipynb
├── db/                   # Database schemas
│   └── schema.sql
├── deployment/           # Deployment scripts
│   ├── deploy.sh
│   └── deploy.bat
├── scripts/              # Utility scripts
│   ├── run_dashboard.py
│   └── run_tests.py
├── Dockerfile            # API container
├── Dockerfile.frontend   # Frontend container
├── docker-compose.yml    # Multi-service setup
├── pyproject.toml        # Dependencies
└── README.md            # This file
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development Setup:**
```bash
# Install dev dependencies
uv sync --dev

# Run tests before committing
pytest tests/ -v

# Format code
black src/ api/ tests/
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **Dataset**: Twitter Customer Support Dataset (Kaggle)
- **Models**: HuggingFace Transformers
- **Frameworks**: FastAPI, Streamlit, scikit-learn
- **Infrastructure**: PostgreSQL, Docker

---

## Support

For issues, questions, or feature requests:

- Open an issue on GitHub
- Check existing documentation
- Review test files for usage examples

---

## Roadmap

- [ ] Multi-language support (Spanish, French, German)
- [ ] Real-time streaming analysis
- [ ] Advanced analytics dashboard
- [ ] Email alert notifications
- [ ] Custom model fine-tuning UI
- [ ] Export to Business Intelligence tools
- [ ] Mobile app integration

---

<div align="center">

**Built with ❤️ for better customer insights**

[⬆ Back to Top](#sentisight---ai-powered-customer-feedback-analyzer)

</div>
