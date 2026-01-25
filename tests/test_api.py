"""
Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app
import api.main as main_module

# Create mock models
mock_sentiment = Mock()
mock_sentiment.analyze.return_value = {
    'sentiment': 'positive',
    'confidence': 0.95
}

mock_category = Mock()
mock_category.predict.return_value = ['Product Quality', 'Customer Service']

mock_anomaly = Mock()
mock_anomaly.predict.return_value = [False]
mock_anomaly.predict_with_scores.return_value = ([False], [0.1])
mock_anomaly.is_fitted = True  # Add is_fitted attribute

# Replace the global model variables
main_module.sentiment_analyzer = mock_sentiment
main_module.category_classifier = mock_category
main_module.anomaly_detector = mock_anomaly

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
    
    def test_analyze_single(self):
        """Test single feedback analysis"""
        payload = {
            "text": "Great product! Love it!"
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "categories" in data
        assert "anomaly" in data
    
    def test_analyze_empty_text(self):
        """Test analysis with empty text"""
        payload = {"text": ""}
        response = client.post("/analyze", json=payload)
        # Should still return 200 but with different handling
        assert response.status_code in [200, 422]
    
    def test_batch_analyze(self):
        """Test batch analysis"""
        payload = {
            "feedbacks": [
                {"text": "Good service"},
                {"text": "Bad experience"}
            ]
        }
        response = client.post("/batch-analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
    
    def test_insights_no_data(self):
        """Test insights endpoint with no data"""
        # Clear data first
        client.delete("/clear")
        response = client.get("/insights")
        assert response.status_code == 404
    
    def test_clear_endpoint(self):
        """Test clear endpoint"""
        response = client.delete("/clear")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
