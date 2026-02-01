"""
Test startup time for SentiSight models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from src.models import SentimentAnalyzer, CategoryClassifier, AnomalyDetector

print("=" * 70)
print("🚀 SENTISIGHT STARTUP TIME TEST")
print("=" * 70)
print()

# Test sentiment analyzer (heaviest model)
print("📊 Loading Sentiment Analyzer (DistilBERT)...")
start = time.time()
sentiment = SentimentAnalyzer(use_gpu=True)
sentiment_time = time.time() - start
print(f"   ✓ Loaded in {sentiment_time:.2f} seconds")
print()

# Test category classifier
print("📋 Loading Category Classifier...")
start = time.time()
category = CategoryClassifier()
category_time = time.time() - start
print(f"   ✓ Loaded in {category_time:.2f} seconds")
print()

# Test anomaly detector
print("🚨 Loading Anomaly Detector...")
start = time.time()
anomaly = AnomalyDetector()
anomaly_time = time.time() - start
print(f"   ✓ Loaded in {anomaly_time:.2f} seconds")
print()

total_time = sentiment_time + category_time + anomaly_time

print("=" * 70)
print(f"⏱️  TOTAL STARTUP TIME: {total_time:.2f} seconds")
print("=" * 70)
print()

# Test inference speed
print("🧪 Testing Inference Speed...")
test_text = "Great product! Fast delivery and excellent customer service."

start = time.time()
result = sentiment.analyze(test_text)
inference_time = (time.time() - start) * 1000  # Convert to ms

print(f"   Sample feedback: \"{test_text}\"")
print(f"   Sentiment: {result['sentiment']}")
print(f"   Confidence: {result['confidence']:.2%}")
print(f"   ⚡ Processing time: {inference_time:.1f}ms")
print()

# Estimate Streamlit startup
streamlit_overhead = 3.5  # Streamlit framework initialization
total_app_startup = total_time + streamlit_overhead

print("=" * 70)
print(f"📱 ESTIMATED STREAMLIT APP STARTUP")
print(f"   Models: {total_time:.1f}s")
print(f"   Framework: {streamlit_overhead:.1f}s")
print(f"   TOTAL: {total_app_startup:.1f}s")
print("=" * 70)
print()
print("✅ App will be ready to use in approximately", int(total_app_startup), "seconds")
print()
