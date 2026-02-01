"""
SentiSight Streamlit Dashboard
Interactive web interface for customer feedback analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import SentimentAnalyzer, CategoryClassifier, AnomalyDetector
from src.preprocessing import TextPreprocessor, DataLoader
import torch
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SentiSight - Customer Feedback Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check GPU availability
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    GPU_NAME = "Not Available"
    GPU_MEMORY = 0

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_feedbacks' not in st.session_state:
    st.session_state.analyzed_feedbacks = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = None
if 'category_classifier' not in st.session_state:
    st.session_state.category_classifier = None
if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = None

# Check GPU availability
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    GPU_NAME = "Not Available"
    GPU_MEMORY = 0


@st.cache_resource
def load_models():
    """Load ML models with GPU support (cached)"""
    progress_placeholder = st.empty()
    
    with progress_placeholder.container():
        st.info("🚀 Loading AI models... This may take 15-30 seconds on first run.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        import time
        start_time = time.time()
        
        # Load sentiment analyzer (heaviest model - DistilBERT)
        progress_bar.progress(10)
        status_text.text("🤖 Loading Sentiment Analyzer (DistilBERT)...")
        model_start = time.time()
        sentiment_analyzer = SentimentAnalyzer(use_gpu=True)
        sentiment_time = time.time() - model_start
        
        # Load category classifier
        progress_bar.progress(50)
        status_text.text("📊 Loading Category Classifier...")
        model_start = time.time()
        category_classifier = CategoryClassifier()
        category_time = time.time() - model_start
        
        # Load anomaly detector
        progress_bar.progress(80)
        status_text.text("🚨 Initializing Anomaly Detector...")
        model_start = time.time()
        anomaly_detector = AnomalyDetector()
        anomaly_time = time.time() - model_start
        
        progress_bar.progress(100)
        total_time = time.time() - start_time
        
        device_info = f"GPU ({GPU_NAME})" if GPU_AVAILABLE else "CPU"
        status_text.success(f"✅ All models loaded in {total_time:.1f}s on {device_info}")
        time.sleep(1)
    
    progress_placeholder.empty()
    return sentiment_analyzer, category_classifier, anomaly_detector


def analyze_single_feedback(text, sentiment_analyzer, category_classifier):
    """Analyze a single feedback"""
    # Extract features
    features = TextPreprocessor.extract_features(text)
    
    # Sentiment analysis
    sentiment_result = sentiment_analyzer.analyze(text)
    sentiment_score = SentimentAnalyzer.sentiment_to_score(
        sentiment_result['sentiment'],
        sentiment_result['confidence']
    )
    
    # Category classification
    categories = category_classifier.predict(text)
    
    # Simple anomaly detection
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
    
    return {
        'text': text,
        'sentiment': sentiment_result['sentiment'],
        'sentiment_score': sentiment_score,
        'confidence': sentiment_result['confidence'],
        'categories': categories,
        'is_anomaly': is_anomaly,
        'anomaly_score': anomaly_score,
        'anomaly_reasons': reasons,
        'features': features,
        'timestamp': datetime.now()
    }


def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 SentiSight</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Customer Feedback Analysis</p>', unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.models_loaded:
        try:
            sentiment_analyzer, category_classifier, anomaly_detector = load_models()
            st.session_state.sentiment_analyzer = sentiment_analyzer
            st.session_state.category_classifier = category_classifier
            st.session_state.anomaly_detector = anomaly_detector
            st.session_state.models_loaded = True
            st.success("✓ Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # System info
    with st.sidebar.expander("💻 System Info", expanded=False):
        st.write("**Device:**")
        if GPU_AVAILABLE:
            st.success(f"🚀 GPU: {GPU_NAME}")
            st.info(f"📊 Memory: {GPU_MEMORY:.1f} GB")
        else:
            st.warning("💻 CPU Mode")
        st.write("**PyTorch:**", torch.__version__)
    
    # Mode selection
    mode = st.sidebar.radio(
        "Analysis Mode",
        ["📝 Single Feedback", "📊 Batch Analysis", "📈 Dashboard", "🚨 Anomalies"]
    )
    
    st.sidebar.markdown("---")
    
    # Stats
    if st.session_state.analyzed_feedbacks:
        st.sidebar.metric("Total Analyzed", len(st.session_state.analyzed_feedbacks))
        anomaly_count = sum(1 for f in st.session_state.analyzed_feedbacks if f['is_anomaly'])
        st.sidebar.metric("Anomalies", anomaly_count)
        
        if st.sidebar.button("Clear All Data"):
            st.session_state.analyzed_feedbacks = []

    
    # Main content based on mode
    if mode == "📝 Single Feedback":
        show_single_analysis()
    elif mode == "📊 Batch Analysis":
        show_batch_analysis()
    elif mode == "📈 Dashboard":
        show_dashboard()
    elif mode == "🚨 Anomalies":
        show_anomalies()


def show_single_analysis():
    """Single feedback analysis tab"""
    st.header("📝 Analyze Single Feedback")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input
        text_input = st.text_area(
            "Enter customer feedback:",
            height=150,
            placeholder="Type or paste customer feedback here..."
        )
        
        # Example feedbacks
        with st.expander("📌 Try Example Feedbacks"):
            examples = [
                "This product is absolutely amazing! Fast shipping and great quality!",
                "I'm very disappointed. The item arrived broken and customer service was rude.",
                "The package never arrived. I've been waiting for 2 weeks!",
                "Average product, nothing special but does the job.",
                "TERRIBLE EXPERIENCE! I want a refund immediately!!! Very frustrated!"
            ]
            for i, example in enumerate(examples):
                if st.button(f"Example {i+1}", key=f"example_{i}"):
                    st.session_state.example_text = example
                    st.rerun()
            
            if 'example_text' in st.session_state:
                text_input = st.session_state.example_text
                del st.session_state.example_text
        
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    with col2:
        st.info("💡 **Tips:**\n\n"
                "• Enter any customer feedback\n"
                "• Supports reviews, tickets, surveys\n"
                "• Click examples to try preset texts")
    
    # Analysis
    if analyze_button and text_input:
        with st.spinner("Analyzing feedback..."):
            result = analyze_single_feedback(
                text_input,
                st.session_state.sentiment_analyzer,
                st.session_state.category_classifier
            )
            st.session_state.analyzed_feedbacks.append(result)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sentiment_color = {
                'POSITIVE': '🟢',
                'NEGATIVE': '🔴',
                'NEUTRAL': '🟡'
            }.get(result['sentiment'], '⚪')
            st.metric(
                "Sentiment",
                f"{sentiment_color} {result['sentiment']}",
                f"{result['sentiment_score']:.2f}"
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{result['confidence']:.1%}"
            )
        
        with col3:
            st.metric(
                "Categories",
                len(result['categories'])
            )
        
        with col4:
            anomaly_icon = "🚨" if result['is_anomaly'] else "✅"
            st.metric(
                "Status",
                f"{anomaly_icon} {'Anomaly' if result['is_anomaly'] else 'Normal'}"
            )
        
        # Detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📂 Categories:**")
            for cat in result['categories']:
                st.markdown(f"- {cat.replace('_', ' ').title()}")
        
        with col2:
            st.markdown("**📏 Text Features:**")
            st.markdown(f"- Length: {result['features']['text_length']} chars")
            st.markdown(f"- Words: {result['features']['word_count']}")
            st.markdown(f"- Urgency indicators: {result['features']['urgency_count']}")
            st.markdown(f"- Negative emotions: {result['features']['negative_emotion_count']}")
        
        # Anomaly details
        if result['is_anomaly']:
            st.warning("🚨 **Anomaly Detected!**")
            st.markdown("**Reasons:**")
            for reason in result['anomaly_reasons']:
                st.markdown(f"- {reason}")


def show_batch_analysis():
    """Batch analysis tab"""
    st.header("📊 Batch Analysis")
    
    # Upload option
    upload_option = st.radio(
        "Choose input method:",
        ["📁 Upload CSV", "📝 Paste Multiple Texts"]
    )
    
    if upload_option == "📁 Upload CSV":
        st.markdown("Upload a CSV file with customer feedback")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="CSV should have a column with feedback text"
            )
        
        with col2:
            st.info("**CSV Requirements:**\n\n"
                   "• Must contain text column\n"
                   "• Common names: text, feedback, review, message")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✓ Loaded {len(df)} rows")
                
                # Select text column
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                text_col = st.selectbox("Select text column:", text_columns)
                
                # Limit rows
                max_rows = min(100, len(df))
                num_rows = st.slider("Number of rows to analyze:", 1, max_rows, min(10, max_rows))
                
                if st.button("🔍 Analyze Batch", type="primary"):
                    process_batch(df[text_col].head(num_rows).tolist())
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    else:  # Paste multiple texts
        st.markdown("Enter multiple feedbacks (one per line)")
        texts_input = st.text_area(
            "Feedback texts:",
            height=200,
            placeholder="Enter one feedback per line..."
        )
        
        if st.button("🔍 Analyze Texts", type="primary") and texts_input:
            texts = [t.strip() for t in texts_input.split('\n') if t.strip()]
            if texts:
                process_batch(texts)


def process_batch(texts):
    """Process batch of texts"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    for i, text in enumerate(texts):
        status_text.text(f"Processing {i+1}/{len(texts)}...")
        
        result = analyze_single_feedback(
            text,
            st.session_state.sentiment_analyzer,
            st.session_state.category_classifier
        )
        results.append(result)
        st.session_state.analyzed_feedbacks.append(result)
        
        progress_bar.progress((i + 1) / len(texts))
    
    status_text.text("✓ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # Show results summary
    st.success(f"✓ Analyzed {len(results)} feedbacks")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    sentiments = [r['sentiment'] for r in results]
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    with col1:
        st.metric("Positive", sentiment_counts.get('POSITIVE', 0))
    with col2:
        st.metric("Negative", sentiment_counts.get('NEGATIVE', 0))
    with col3:
        st.metric("Neutral", sentiment_counts.get('NEUTRAL', 0))
    with col4:
        anomalies = sum(1 for r in results if r['is_anomaly'])
        st.metric("Anomalies", anomalies)
    
    # Show results table
    st.markdown("### 📋 Results Preview")
    results_df = pd.DataFrame([
        {
            'Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
            'Sentiment': r['sentiment'],
            'Score': f"{r['sentiment_score']:.2f}",
            'Categories': ', '.join(r['categories'][:2]),
            'Anomaly': '🚨' if r['is_anomaly'] else '✅'
        }
        for r in results
    ])
    st.dataframe(results_df, use_container_width=True)


def show_dashboard():
    """Dashboard tab with visualizations"""
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.analyzed_feedbacks:
        st.info("👋 No data yet! Analyze some feedbacks to see visualizations.")
        return
    
    feedbacks = st.session_state.analyzed_feedbacks
    
    # Summary metrics
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedbacks", len(feedbacks))
    
    with col2:
        avg_sentiment = np.mean([f['sentiment_score'] for f in feedbacks])
        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
    
    with col3:
        anomaly_count = sum(1 for f in feedbacks if f['is_anomaly'])
        st.metric("Anomalies", anomaly_count)
    
    with col4:
        anomaly_pct = (anomaly_count / len(feedbacks)) * 100
        st.metric("Anomaly Rate", f"{anomaly_pct:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        st.subheader("🎭 Sentiment Distribution")
        sentiments = [f['sentiment'] for f in feedbacks]
        sentiment_df = pd.DataFrame({'Sentiment': sentiments})
        
        fig = px.pie(
            sentiment_df,
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'POSITIVE': '#2ecc71',
                'NEGATIVE': '#e74c3c',
                'NEUTRAL': '#f39c12'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment score distribution
        st.subheader("📊 Sentiment Score Distribution")
        scores = [f['sentiment_score'] for f in feedbacks]
        
        fig = go.Figure(data=[go.Histogram(
            x=scores,
            nbinsx=20,
            marker_color='#3498db'
        )])
        fig.update_layout(
            xaxis_title="Sentiment Score",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Category analysis
    st.subheader("📂 Category Analysis")
    
    all_categories = []
    for f in feedbacks:
        all_categories.extend(f['categories'])
    
    category_counts = pd.Series(all_categories).value_counts().head(10)
    
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={'x': 'Category', 'y': 'Count'},
        color=category_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment over time
    if len(feedbacks) > 5:
        st.subheader("📈 Sentiment Trend")
        
        timeline_df = pd.DataFrame([
            {
                'index': i,
                'sentiment_score': f['sentiment_score'],
                'timestamp': f['timestamp']
            }
            for i, f in enumerate(feedbacks)
        ])
        
        fig = px.line(
            timeline_df,
            x='index',
            y='sentiment_score',
            labels={'index': 'Feedback Number', 'sentiment_score': 'Sentiment Score'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


def show_anomalies():
    """Anomalies tab"""
    st.header("🚨 Anomaly Detection")
    
    if not st.session_state.analyzed_feedbacks:
        st.info("👋 No data yet! Analyze some feedbacks to detect anomalies.")
        return
    
    anomalies = [f for f in st.session_state.analyzed_feedbacks if f['is_anomaly']]
    
    if not anomalies:
        st.success("✅ No anomalies detected! All feedbacks look normal.")
        return
    
    # Summary
    st.subheader(f"Found {len(anomalies)} Anomalies")
    
    # Severity breakdown
    col1, col2, col3 = st.columns(3)
    
    critical = sum(1 for a in anomalies if a['anomaly_score'] > 0.8)
    high = sum(1 for a in anomalies if 0.6 <= a['anomaly_score'] <= 0.8)
    medium = sum(1 for a in anomalies if a['anomaly_score'] < 0.6)
    
    with col1:
        st.metric("🔴 Critical", critical)
    with col2:
        st.metric("🟠 High", high)
    with col3:
        st.metric("🟡 Medium", medium)
    
    st.markdown("---")
    
    # Filter
    severity_filter = st.selectbox(
        "Filter by severity:",
        ["All", "Critical (>0.8)", "High (0.6-0.8)", "Medium (<0.6)"]
    )
    
    # Apply filter
    filtered_anomalies = anomalies
    if severity_filter == "Critical (>0.8)":
        filtered_anomalies = [a for a in anomalies if a['anomaly_score'] > 0.8]
    elif severity_filter == "High (0.6-0.8)":
        filtered_anomalies = [a for a in anomalies if 0.6 <= a['anomaly_score'] <= 0.8]
    elif severity_filter == "Medium (<0.6)":
        filtered_anomalies = [a for a in anomalies if a['anomaly_score'] < 0.6]
    
    # Sort by score
    filtered_anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
    
    # Display anomalies
    st.subheader(f"📋 Showing {len(filtered_anomalies)} Anomalies")
    
    for i, anomaly in enumerate(filtered_anomalies):
        with st.expander(f"#{i+1} - Score: {anomaly['anomaly_score']:.2f} - {anomaly['sentiment']}"):
            st.markdown(f"**Text:** {anomaly['text']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Sentiment:** {anomaly['sentiment']} ({anomaly['sentiment_score']:.2f})")
                st.markdown(f"**Categories:** {', '.join(anomaly['categories'])}")
            
            with col2:
                st.markdown("**Anomaly Reasons:**")
                for reason in anomaly['anomaly_reasons']:
                    st.markdown(f"- {reason}")
            
            st.markdown(f"**Timestamp:** {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
