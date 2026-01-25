"""
Unit tests for ML models
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CategoryClassifier


class TestCategoryClassifier:
    """Test CategoryClassifier"""
    
    def test_initialization(self):
        """Test classifier initialization"""
        classifier = CategoryClassifier()
        assert len(classifier.categories) > 0
    
    def test_keyword_classification(self):
        """Test keyword-based classification"""
        classifier = CategoryClassifier()
        
        # Test shipping
        text = "Where is my package? It hasn't arrived yet."
        categories = classifier.predict(text)
        assert any('shipping' in cat.lower() for cat in categories)
        
        # Test billing
        text = "I was charged twice for my subscription"
        categories = classifier.predict(text)
        assert any('billing' in cat.lower() for cat in categories)
        
        # Test technical
        text = "The app keeps crashing with error 500"
        categories = classifier.predict(text)
        assert any('technical' in cat.lower() for cat in categories)
    
    def test_predict_empty_text(self):
        """Test prediction with empty text"""
        classifier = CategoryClassifier()
        categories = classifier.predict("")
        assert 'general_inquiry' in categories
    
    def test_category_distribution(self):
        """Test category distribution calculation"""
        classifier = CategoryClassifier()
        texts = [
            "Billing issue",
            "Shipping problem",
            "Billing error",
        ]
        distribution = classifier.get_category_distribution(texts)
        assert 'billing' in distribution
        assert distribution['billing'] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
