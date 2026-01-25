-- Database schema for SentiSight
-- PostgreSQL with pgvector extension for semantic search

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Feedback table (stores all analyzed feedback)
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    sentiment_score FLOAT NOT NULL,
    sentiment_confidence FLOAT NOT NULL,
    categories TEXT[] NOT NULL,
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    anomaly_score FLOAT DEFAULT 0.0,
    anomaly_reasons TEXT[],
    text_length INTEGER,
    word_count INTEGER,
    urgency_count INTEGER DEFAULT 0,
    negative_emotion_count INTEGER DEFAULT 0,
    positive_emotion_count INTEGER DEFAULT 0,
    exclamation_count INTEGER DEFAULT 0,
    caps_ratio FLOAT DEFAULT 0.0,
    embedding vector(768),  -- For semantic search
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_feedback_sentiment ON feedback(sentiment);
CREATE INDEX IF NOT EXISTS idx_feedback_is_anomaly ON feedback(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_categories ON feedback USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_feedback_embedding ON feedback USING ivfflat (embedding vector_cosine_ops);

-- Insights table (stores daily/weekly aggregated metrics)
CREATE TABLE IF NOT EXISTS insights (
    id SERIAL PRIMARY KEY,
    time_period VARCHAR(20) NOT NULL,  -- 'daily', 'weekly', 'monthly'
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    total_feedbacks INTEGER NOT NULL,
    positive_count INTEGER DEFAULT 0,
    negative_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,
    avg_sentiment_score FLOAT,
    anomaly_count INTEGER DEFAULT 0,
    anomaly_percentage FLOAT DEFAULT 0.0,
    top_categories JSONB,
    category_distribution JSONB,
    sentiment_trend FLOAT,  -- Change from previous period
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_insights_period ON insights(time_period, period_start);

-- Alerts table (stores flagged anomalies for review)
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    feedback_id INTEGER REFERENCES feedback(id) ON DELETE CASCADE,
    severity VARCHAR(20) NOT NULL,  -- 'critical', 'high', 'medium', 'low'
    alert_type VARCHAR(50) NOT NULL,
    description TEXT,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_is_resolved ON alerts(is_resolved);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

-- Categories table (tracks all categories and their metadata)
CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    keywords TEXT[],
    feedback_count INTEGER DEFAULT 0,
    avg_sentiment FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reports table (stores generated insight reports)
CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    report_type VARCHAR(50) NOT NULL,  -- 'daily', 'weekly', 'monthly', 'custom'
    content TEXT NOT NULL,  -- LLM-generated summary
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    key_findings JSONB,
    recommendations TEXT[],
    generated_by VARCHAR(50),  -- 'automated', 'manual'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_feedback_updated_at BEFORE UPDATE ON feedback
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_categories_updated_at BEFORE UPDATE ON categories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for semantic search
CREATE OR REPLACE FUNCTION search_similar_feedback(
    query_embedding vector(768),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id INTEGER,
    text TEXT,
    sentiment VARCHAR(20),
    categories TEXT[],
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.id,
        f.text,
        f.sentiment,
        f.categories,
        1 - (f.embedding <=> query_embedding) as similarity
    FROM feedback f
    WHERE 1 - (f.embedding <=> query_embedding) > match_threshold
    ORDER BY f.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Insert default categories
INSERT INTO categories (name, description, keywords) VALUES
    ('billing', 'Billing and payment related issues', ARRAY['bill', 'charge', 'payment', 'invoice', 'price']),
    ('technical_support', 'Technical issues and bugs', ARRAY['error', 'bug', 'crash', 'not working', 'broken']),
    ('shipping_delivery', 'Shipping and delivery inquiries', ARRAY['shipping', 'delivery', 'package', 'tracking']),
    ('product_quality', 'Product quality feedback', ARRAY['quality', 'defect', 'damaged', 'broken']),
    ('customer_service', 'Customer service experiences', ARRAY['support', 'service', 'representative', 'agent']),
    ('account_management', 'Account and profile management', ARRAY['account', 'profile', 'password', 'login']),
    ('refund_return', 'Refunds and returns', ARRAY['refund', 'return', 'money back', 'exchange']),
    ('general_inquiry', 'General questions and inquiries', ARRAY['question', 'inquiry', 'how to', 'information'])
ON CONFLICT (name) DO NOTHING;
