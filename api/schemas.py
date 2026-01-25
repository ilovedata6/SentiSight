"""
Pydantic models for request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class FeedbackInput(BaseModel):
    """Single feedback input"""
    text: str = Field(..., description="Feedback text content", min_length=1)
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")


class BatchFeedbackInput(BaseModel):
    """Batch feedback input"""
    feedbacks: List[FeedbackInput] = Field(..., description="List of feedbacks")


class SentimentResult(BaseModel):
    """Sentiment analysis result"""
    sentiment: str = Field(..., description="Sentiment label (POSITIVE/NEGATIVE/NEUTRAL)")
    confidence: float = Field(..., description="Confidence score", ge=0, le=1)
    score: float = Field(..., description="Numeric sentiment score", ge=-1, le=1)


class CategoryResult(BaseModel):
    """Category classification result"""
    categories: List[str] = Field(..., description="Predicted categories")
    category_scores: Optional[Dict[str, float]] = Field(default=None, description="Category confidence scores")


class AnomalyResult(BaseModel):
    """Anomaly detection result"""
    is_anomaly: bool = Field(..., description="Whether feedback is anomalous")
    anomaly_score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    reasons: List[str] = Field(default=[], description="Reasons for flagging as anomaly")


class AnalysisResult(BaseModel):
    """Complete analysis result for single feedback"""
    text: str = Field(..., description="Original feedback text")
    sentiment: SentimentResult
    categories: CategoryResult
    anomaly: AnomalyResult
    text_features: Dict = Field(..., description="Extracted text features")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchAnalysisResult(BaseModel):
    """Batch analysis results"""
    results: List[AnalysisResult]
    total_processed: int
    processing_time_seconds: float


class InsightsResponse(BaseModel):
    """Aggregated insights"""
    total_feedbacks: int
    sentiment_distribution: Dict[str, float]
    category_distribution: Dict[str, int]
    anomaly_count: int
    anomaly_percentage: float
    avg_sentiment_score: float
    top_categories: List[Dict[str, Any]]
    time_period: Optional[str] = None


class AnomalyFeedback(BaseModel):
    """Anomalous feedback details"""
    text: str
    sentiment: str
    categories: List[str]
    anomaly_score: float
    reasons: List[str]
    timestamp: datetime


class AnomaliesResponse(BaseModel):
    """List of anomalous feedbacks"""
    anomalies: List[AnomalyFeedback]
    total_count: int
    severity_breakdown: Dict[str, int]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
