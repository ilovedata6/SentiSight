"""
Unit tests for preprocessing module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import TextPreprocessor, DataLoader


class TestTextPreprocessor:
    """Test TextPreprocessor class"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "This is a TEST with CAPS"
        cleaned = TextPreprocessor.clean_text(text, lowercase=True)
        assert cleaned == "this is a test with caps"
    
    def test_clean_text_urls(self):
        """Test URL removal"""
        text = "Check this https://example.com link"
        cleaned = TextPreprocessor.clean_text(text, remove_urls=True)
        assert "https://example.com" not in cleaned
    
    def test_clean_text_emails(self):
        """Test email removal"""
        text = "Contact me at test@example.com"
        cleaned = TextPreprocessor.clean_text(text, remove_emails=True)
        assert "test@example.com" not in cleaned
    
    def test_clean_text_mentions(self):
        """Test mention removal"""
        text = "Hey @user thanks!"
        cleaned = TextPreprocessor.clean_text(text, remove_mentions=True)
        assert "@user" not in cleaned
    
    def test_extract_features(self):
        """Test feature extraction"""
        text = "This is urgent!! Please help ASAP!!!"
        features = TextPreprocessor.extract_features(text)
        
        assert 'text_length' in features
        assert 'word_count' in features
        assert 'urgency_count' in features
        assert 'exclamation_count' in features
        assert features['exclamation_count'] == 5  # Updated to actual count
        assert features['urgency_count'] > 0
    
    def test_extract_features_empty(self):
        """Test feature extraction with empty text"""
        features = TextPreprocessor.extract_features("")
        assert features['text_length'] == 0
        assert features['word_count'] == 0
    
    def test_extract_features_emotions(self):
        """Test emotion detection"""
        positive_text = "I love this amazing product!"
        positive_features = TextPreprocessor.extract_features(positive_text)
        assert positive_features['positive_emotion_count'] > 0
        
        negative_text = "Terrible awful horrible experience!"
        negative_features = TextPreprocessor.extract_features(negative_text)
        assert negative_features['negative_emotion_count'] > 0


class TestDataLoader:
    """Test DataLoader class"""
    
    def test_dataloader_init(self):
        """Test DataLoader initialization"""
        # This will fail if file doesn't exist
        with pytest.raises(FileNotFoundError):
            loader = DataLoader("nonexistent.csv")
    
    # Note: Actual file loading tests would require a test CSV file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
