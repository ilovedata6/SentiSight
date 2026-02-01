"""Startup timing with persistent logging to logs/startup_results.json"""
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import SentimentAnalyzer, CategoryClassifier, AnomalyDetector

LOG_PATH = Path(__file__).parent / ".." / "logs" / "startup_results.json"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

results = {"stages": [], "inference_ms": None}

print("Running instrumented startup test (writing results to logs/startup_results.json)")

# Sentiment
start = time.time()
sentiment = SentimentAnalyzer(use_gpu=True)
elapsed = time.time() - start
print(f"Sentiment loaded in {elapsed:.2f}s")
results["stages"].append({"name": "sentiment", "seconds": elapsed})

# Category
start = time.time()
category = CategoryClassifier()
elapsed = time.time() - start
print(f"Category loaded in {elapsed:.2f}s")
results["stages"].append({"name": "category", "seconds": elapsed})

# Anomaly
start = time.time()
anomaly = AnomalyDetector()
elapsed = time.time() - start
print(f"Anomaly loaded in {elapsed:.2f}s")
results["stages"].append({"name": "anomaly", "seconds": elapsed})

# Inference test
test_text = "Great product! Fast delivery and excellent customer service."
start = time.time()
res = sentiment.analyze(test_text)
inf_ms = (time.time() - start) * 1000
print(f"Inference: {res['sentiment']} ({res['confidence']:.2%}) in {inf_ms:.1f}ms")
results["inference_ms"] = inf_ms

# Totals
total = sum(s["seconds"] for s in results["stages"]) + 3.5
results["total_estimated_seconds"] = total

with open(LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Results written to:", LOG_PATH)
print(json.dumps(results, indent=2))
