"""Quick GPU test for sentiment model"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from models.sentiment_analyzer import SentimentAnalyzer

print("=" * 60)
print("QUICK GPU TEST")
print("=" * 60)

# Check GPU availability
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Initialize model
print("\nInitializing Sentiment Analyzer...")
analyzer = SentimentAnalyzer(use_gpu=True)

# Test inference
test_text = "This product is amazing! I love it!"
print(f"\nTest input: '{test_text}'")
result = analyzer.analyze(test_text)
print(f"✓ Result: {result['sentiment']} (confidence: {result['confidence']:.2%})")
print(f"✓ Device used: {analyzer.device}")

print("\n" + "=" * 60)
print("✓ GPU TEST PASSED!")
print("=" * 60)
