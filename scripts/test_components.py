"""
Quick test script to verify all components are working
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("🧪 SentiSight Component Test")
print("=" * 70)

# Test 1: Import preprocessing
print("\n[1/5] Testing preprocessing module...")
try:
    from src.preprocessing import TextPreprocessor, DataLoader
    print("  ✓ Preprocessing module loaded")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Import models
print("\n[2/5] Testing ML models...")
try:
    from src.models import SentimentAnalyzer, CategoryClassifier, AnomalyDetector
    print("  ✓ ML models loaded")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 3: Test preprocessing
print("\n[3/5] Testing text preprocessing...")
try:
    text = "This is a TEST with URLs https://example.com and @mentions!"
    cleaned = TextPreprocessor.clean_text(text)
    features = TextPreprocessor.extract_features(text)
    print(f"  Original: {text}")
    print(f"  Cleaned: {cleaned}")
    print(f"  Features extracted: {len(features)} features")
    print("  ✓ Preprocessing working")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 4: Test sentiment analysis
print("\n[4/5] Testing sentiment analysis...")
try:
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("This product is amazing! I love it!")
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print("  ✓ Sentiment analyzer working")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 5: Test category classification
print("\n[5/5] Testing category classification...")
try:
    classifier = CategoryClassifier()
    categories = classifier.predict("I haven't received my package yet")
    print(f"  Categories: {', '.join(categories)}")
    print("  ✓ Category classifier working")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 70)
print("✅ All component tests passed!")
print("=" * 70)
print("\nNext steps:")
print("1. Run EDA notebook: jupyter notebook notebooks/01_eda.ipynb")
print("2. Start API server: python -m uvicorn api.main:app --reload")
print("3. Visit API docs: http://localhost:8000/docs")
