"""Test configuration and fixtures"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_feedback_texts():
    """Sample feedback texts for testing"""
    return [
        "This product is amazing! Great quality and fast shipping.",
        "I'm very disappointed with the service. Won't buy again.",
        "The item arrived broken. Requesting a refund.",
        "Average product, nothing special.",
        "URGENT! Package never arrived! Very frustrated!!!",
    ]


@pytest.fixture
def sample_analyzed_feedback():
    """Sample analyzed feedback data"""
    return {
        'text': 'Great product!',
        'sentiment': 'POSITIVE',
        'sentiment_score': 0.9,
        'confidence': 0.95,
        'categories': ['product_quality'],
        'is_anomaly': False,
        'anomaly_score': 0.0,
        'anomaly_reasons': [],
        'features': {
            'text_length': 14,
            'word_count': 2,
            'urgency_count': 0,
            'negative_emotion_count': 0,
            'positive_emotion_count': 1,
            'exclamation_count': 1,
            'caps_ratio': 0.0,
        }
    }
