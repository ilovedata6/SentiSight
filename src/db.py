"""
Database manager for SentiSight
Handles CRUD operations and semantic search
"""

import os
from typing import List, Dict, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager
import logging
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string or os.getenv(
            'DATABASE_URL',
            'postgresql://sentisight_user:password@localhost:5432/sentisight'
        )
        logger.info("DatabaseManager initialized")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def init_schema(self, schema_path: str = 'db/schema.sql'):
        """
        Initialize database schema
        
        Args:
            schema_path: Path to schema SQL file
        """
        logger.info(f"Initializing database schema from {schema_path}")
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
        
        logger.info("✓ Database schema initialized")
    
    def insert_feedback(self,
                       text: str,
                       sentiment: str,
                       sentiment_score: float,
                       sentiment_confidence: float,
                       categories: List[str],
                       is_anomaly: bool,
                       anomaly_score: float,
                       anomaly_reasons: List[str],
                       text_features: Dict,
                       embedding: Optional[np.ndarray] = None,
                       metadata: Optional[Dict] = None) -> int:
        """
        Insert analyzed feedback into database
        
        Args:
            text: Feedback text
            sentiment: Sentiment label
            sentiment_score: Sentiment score
            sentiment_confidence: Sentiment confidence
            categories: List of categories
            is_anomaly: Whether feedback is anomalous
            anomaly_score: Anomaly score
            anomaly_reasons: List of reasons
            text_features: Dictionary of text features
            embedding: Optional embedding vector
            metadata: Optional metadata
            
        Returns:
            Inserted feedback ID
        """
        query = """
        INSERT INTO feedback (
            text, sentiment, sentiment_score, sentiment_confidence,
            categories, is_anomaly, anomaly_score, anomaly_reasons,
            text_length, word_count, urgency_count, 
            negative_emotion_count, positive_emotion_count,
            exclamation_count, caps_ratio, embedding, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    text, sentiment, sentiment_score, sentiment_confidence,
                    categories, is_anomaly, anomaly_score, anomaly_reasons,
                    text_features.get('text_length'),
                    text_features.get('word_count'),
                    text_features.get('urgency_count', 0),
                    text_features.get('negative_emotion_count', 0),
                    text_features.get('positive_emotion_count', 0),
                    text_features.get('exclamation_count', 0),
                    text_features.get('caps_ratio', 0.0),
                    embedding.tolist() if embedding is not None else None,
                    metadata
                ))
                feedback_id = cur.fetchone()[0]
        
        logger.debug(f"Inserted feedback {feedback_id}")
        return feedback_id
    
    def get_feedback(self, 
                    feedback_id: Optional[int] = None,
                    limit: int = 100,
                    offset: int = 0,
                    sentiment: Optional[str] = None,
                    is_anomaly: Optional[bool] = None) -> List[Dict]:
        """
        Retrieve feedbacks from database
        
        Args:
            feedback_id: Specific feedback ID (optional)
            limit: Maximum number of results
            offset: Result offset for pagination
            sentiment: Filter by sentiment
            is_anomaly: Filter by anomaly status
            
        Returns:
            List of feedback dictionaries
        """
        query = "SELECT * FROM feedback WHERE 1=1"
        params = []
        
        if feedback_id is not None:
            query += " AND id = %s"
            params.append(feedback_id)
        
        if sentiment is not None:
            query += " AND sentiment = %s"
            params.append(sentiment)
        
        if is_anomaly is not None:
            query += " AND is_anomaly = %s"
            params.append(is_anomaly)
        
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                results = cur.fetchall()
        
        return [dict(row) for row in results]
    
    def search_similar_feedback(self,
                               query_embedding: np.ndarray,
                               match_threshold: float = 0.7,
                               match_count: int = 10) -> List[Dict]:
        """
        Search for similar feedback using embeddings
        
        Args:
            query_embedding: Query embedding vector
            match_threshold: Minimum similarity threshold
            match_count: Maximum number of results
            
        Returns:
            List of similar feedbacks
        """
        query = """
        SELECT * FROM search_similar_feedback(%s, %s, %s)
        """
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (
                    query_embedding.tolist(),
                    match_threshold,
                    match_count
                ))
                results = cur.fetchall()
        
        return [dict(row) for row in results]
    
    def get_insights(self,
                    period: str = 'daily',
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Dict:
        """
        Get aggregated insights
        
        Args:
            period: Time period ('daily', 'weekly', 'monthly')
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Dictionary with insights
        """
        query = """
        SELECT 
            COUNT(*) as total_feedbacks,
            COUNT(*) FILTER (WHERE sentiment = 'POSITIVE') as positive_count,
            COUNT(*) FILTER (WHERE sentiment = 'NEGATIVE') as negative_count,
            COUNT(*) FILTER (WHERE sentiment = 'NEUTRAL') as neutral_count,
            AVG(sentiment_score) as avg_sentiment_score,
            COUNT(*) FILTER (WHERE is_anomaly = TRUE) as anomaly_count,
            AVG(CASE WHEN is_anomaly THEN 1 ELSE 0 END) * 100 as anomaly_percentage
        FROM feedback
        WHERE 1=1
        """
        
        params = []
        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                result = cur.fetchone()
        
        return dict(result) if result else {}
    
    def create_alert(self,
                    feedback_id: int,
                    severity: str,
                    alert_type: str,
                    description: str) -> int:
        """
        Create an alert for anomalous feedback
        
        Args:
            feedback_id: ID of feedback
            severity: Alert severity
            alert_type: Type of alert
            description: Alert description
            
        Returns:
            Alert ID
        """
        query = """
        INSERT INTO alerts (feedback_id, severity, alert_type, description)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (feedback_id, severity, alert_type, description))
                alert_id = cur.fetchone()[0]
        
        return alert_id
    
    def get_category_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for each category
        
        Returns:
            Dictionary mapping category to stats
        """
        query = """
        SELECT 
            unnest(categories) as category,
            COUNT(*) as count,
            AVG(sentiment_score) as avg_sentiment
        FROM feedback
        GROUP BY category
        ORDER BY count DESC
        """
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                results = cur.fetchall()
        
        return {row['category']: dict(row) for row in results}


if __name__ == "__main__":
    # Example usage
    db = DatabaseManager()
    
    print("\n" + "="*60)
    print("DATABASE MANAGER TEST")
    print("="*60)
    
    # Note: This would require a running PostgreSQL instance
    print("\nTo initialize database:")
    print("1. Start PostgreSQL with pgvector extension")
    print("2. Create database: CREATE DATABASE sentisight;")
    print("3. Run: db.init_schema()")
    print("\nExample connection string:")
    print("postgresql://user:password@localhost:5432/sentisight")
