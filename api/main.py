"""
FastAPI application for SentiSight
Provides REST API endpoints for customer feedback analysis
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import time
import logging
from datetime import datetime
from collections import Counter
import json

from api.schemas import (
    FeedbackInput, BatchFeedbackInput, AnalysisResult, BatchAnalysisResult,
    InsightsResponse, AnomaliesResponse, AnomalyFeedback, HealthResponse, ErrorResponse,
    SentimentResult, CategoryResult, AnomalyResult
)
from src.models import SentimentAnalyzer, CategoryClassifier, AnomalyDetector
from src.preprocessing import TextPreprocessor
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom JSON Response with datetime handling
class DateTimeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o)
        ).encode("utf-8")

# Initialize FastAPI app
app = FastAPI(
    title="SentiSight API",
    description="Customer Feedback Analysis API - Sentiment, Classification & Anomaly Detection",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (loaded on startup)
sentiment_analyzer: Optional[SentimentAnalyzer] = None
category_classifier: Optional[CategoryClassifier] = None
anomaly_detector: Optional[AnomalyDetector] = None

# In-memory storage for demo (replace with database in production)
feedback_store: List[AnalysisResult] = []


@app.on_event("startup")
async def load_models():
    """Load ML models on startup with GPU optimization"""
    global sentiment_analyzer, category_classifier, anomaly_detector
    
    import time
    
    logger.info("="*50)
    logger.info("Starting SentiSight API Server")
    logger.info("="*50)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU Detected: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.info("💻 Running on CPU")
    
    logger.info("Loading models...")
    total_start = time.time()
    
    try:
        # Load sentiment analyzer with GPU support
        start = time.time()
        sentiment_analyzer = SentimentAnalyzer(use_gpu=True)
        logger.info(f"✅ Sentiment analyzer loaded ({time.time()-start:.1f}s)")
        
        # Load category classifier
        start = time.time()
        category_classifier = CategoryClassifier()
        logger.info(f"✅ Category classifier loaded ({time.time()-start:.1f}s)")
        
        # Initialize anomaly detector
        start = time.time()
        anomaly_detector = AnomalyDetector()
        logger.info(f"✅ Anomaly detector initialized ({time.time()-start:.1f}s)")
        
        total_time = time.time() - total_start
        logger.info("="*50)
        logger.info(f"🎉 All models loaded successfully! ({total_time:.1f}s total)")
        logger.info("🌐 API ready at http://localhost:8000")
        logger.info("📚 Docs at http://localhost:8000/docs")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to SentiSight API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded={
            "sentiment_analyzer": sentiment_analyzer is not None,
            "category_classifier": category_classifier is not None,
            "anomaly_detector": anomaly_detector is not None and anomaly_detector.is_fitted
        }
    )


@app.post("/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_feedback(feedback: FeedbackInput):
    """
    Analyze a single customer feedback
    
    - **text**: Feedback text content
    - **metadata**: Optional additional metadata
    """
    try:
        # Extract text features
        features = TextPreprocessor.extract_features(feedback.text)
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze(feedback.text)
        sentiment_score = SentimentAnalyzer.sentiment_to_score(
            sentiment_result['sentiment'],
            sentiment_result['confidence']
        )
        
        # Category classification
        categories = category_classifier.predict(feedback.text)
        
        # Anomaly detection (simplified without full dataset context)
        is_anomaly = (
            sentiment_score < -0.7 or
            features['urgency_count'] > 2 or
            features['negative_emotion_count'] > 3
        )
        anomaly_score = abs(sentiment_score) if is_anomaly else 0.0
        
        reasons = []
        if sentiment_score < -0.7:
            reasons.append("Very negative sentiment")
        if features['urgency_count'] > 2:
            reasons.append("High urgency indicators")
        if features['negative_emotion_count'] > 3:
            reasons.append("Multiple negative emotions")
        
        # Build result
        result = AnalysisResult(
            text=feedback.text,
            sentiment=SentimentResult(
                sentiment=sentiment_result['sentiment'],
                confidence=sentiment_result['confidence'],
                score=sentiment_score
            ),
            categories=CategoryResult(
                categories=categories
            ),
            anomaly=AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                reasons=reasons
            ),
            text_features=features
        )
        
        # Store for insights (in production, save to database)
        feedback_store.append(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/batch-analyze", response_model=BatchAnalysisResult, tags=["Analysis"])
async def batch_analyze(batch: BatchFeedbackInput):
    """
    Analyze multiple feedbacks in batch
    
    - **feedbacks**: List of feedback inputs
    """
    try:
        start_time = time.time()
        results = []
        
        # Process each feedback
        for feedback in batch.feedbacks:
            result = await analyze_feedback(feedback)
            results.append(result)
        
        processing_time = time.time() - start_time
        
        return BatchAnalysisResult(
            results=results,
            total_processed=len(results),
            processing_time_seconds=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/insights", response_model=InsightsResponse, tags=["Insights"])
async def get_insights(limit: Optional[int] = None):
    """
    Get aggregated insights from analyzed feedbacks
    
    - **limit**: Optional limit on number of recent feedbacks to analyze
    """
    try:
        if not feedback_store:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No feedbacks analyzed yet"
            )
        
        # Get feedbacks (optionally limited)
        feedbacks = feedback_store[-limit:] if limit else feedback_store
        
        # Calculate sentiment distribution
        sentiments = [f.sentiment.sentiment for f in feedbacks]
        sentiment_counts = Counter(sentiments)
        total = len(feedbacks)
        sentiment_dist = {
            k: round(v / total * 100, 2)
            for k, v in sentiment_counts.items()
        }
        
        # Category distribution
        all_categories = []
        for f in feedbacks:
            all_categories.extend(f.categories.categories)
        category_counts = Counter(all_categories)
        
        # Anomaly stats
        anomalies = [f for f in feedbacks if f.anomaly.is_anomaly]
        anomaly_count = len(anomalies)
        anomaly_pct = round(anomaly_count / total * 100, 2) if total > 0 else 0
        
        # Average sentiment score
        avg_sentiment = sum(f.sentiment.score for f in feedbacks) / total
        
        # Top categories
        top_categories = [
            {"category": cat, "count": count}
            for cat, count in category_counts.most_common(5)
        ]
        
        return InsightsResponse(
            total_feedbacks=total,
            sentiment_distribution=sentiment_dist,
            category_distribution=dict(category_counts),
            anomaly_count=anomaly_count,
            anomaly_percentage=anomaly_pct,
            avg_sentiment_score=round(avg_sentiment, 3),
            top_categories=top_categories
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/anomalies", response_model=AnomaliesResponse, tags=["Insights"])
async def get_anomalies(
    min_score: Optional[float] = 0.5,
    limit: Optional[int] = 50
):
    """
    Get flagged anomalous feedbacks
    
    - **min_score**: Minimum anomaly score threshold
    - **limit**: Maximum number of anomalies to return
    """
    try:
        # Filter anomalies
        anomalies = [
            f for f in feedback_store
            if f.anomaly.is_anomaly and f.anomaly.anomaly_score >= min_score
        ]
        
        # Sort by anomaly score
        anomalies.sort(key=lambda x: x.anomaly.anomaly_score, reverse=True)
        
        # Limit results
        anomalies = anomalies[:limit]
        
        # Format response
        anomaly_feedbacks = [
            AnomalyFeedback(
                text=a.text,
                sentiment=a.sentiment.sentiment,
                categories=a.categories.categories,
                anomaly_score=a.anomaly.anomaly_score,
                reasons=a.anomaly.reasons,
                timestamp=a.timestamp
            )
            for a in anomalies
        ]
        
        # Severity breakdown
        severity_breakdown = {
            "critical": len([a for a in anomalies if a.anomaly.anomaly_score > 0.8]),
            "high": len([a for a in anomalies if 0.6 <= a.anomaly.anomaly_score <= 0.8]),
            "medium": len([a for a in anomalies if a.anomaly.anomaly_score < 0.6])
        }
        
        return AnomaliesResponse(
            anomalies=anomaly_feedbacks,
            total_count=len(anomaly_feedbacks),
            severity_breakdown=severity_breakdown
        )
        
    except Exception as e:
        logger.error(f"Error retrieving anomalies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/clear", tags=["Management"])
async def clear_feedback_store():
    """Clear all stored feedback data (for demo purposes)"""
    global feedback_store
    count = len(feedback_store)
    feedback_store = []
    return {"message": f"Cleared {count} feedbacks from store"}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return DateTimeJSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return DateTimeJSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
