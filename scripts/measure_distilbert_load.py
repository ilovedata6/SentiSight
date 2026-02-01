"""Measure DistilBERT load time (GPU)"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.sentiment_analyzer import SentimentAnalyzer

start = time.perf_counter()
print("Starting SentimentAnalyzer load...")
analyzer = SentimentAnalyzer(use_gpu=True)
load_time = time.perf_counter() - start
print(f"Loaded SentimentAnalyzer in {load_time:.2f} seconds")
print(f"Device used: {analyzer.device}")

# Run a quick inference to ensure warmup
print("Running warmup inference...")
res = analyzer.analyze("This is a quick warmup sentence.")
print(f"Warmup inference done, sentiment: {res['sentiment']}, confidence: {res['confidence']:.2%}")
