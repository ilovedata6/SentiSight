"""
Insights Engine
Generates automated reports and insights from customer feedback using LLMs
"""

import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Import LLM libraries if available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Gemini not available. Install with: pip install google-generativeai")


class InsightsEngine:
    """
    Generates insights and reports from analyzed feedback
    Supports rule-based insights and LLM-powered summaries
    """
    
    def __init__(self, 
                 use_llm: bool = False,
                 llm_provider: str = "openai",
                 api_key: Optional[str] = None):
        """
        Initialize insights engine
        
        Args:
            use_llm: Whether to use LLM for insights
            llm_provider: LLM provider ('openai' or 'gemini')
            api_key: API key for LLM provider
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        
        if use_llm:
            if llm_provider == "openai":
                if not OPENAI_AVAILABLE:
                    raise ValueError("OpenAI not available. Install: pip install openai")
                openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
                self.model = "gpt-4o-mini"
            elif llm_provider == "gemini":
                if not GEMINI_AVAILABLE:
                    raise ValueError("Gemini not available. Install: pip install google-generativeai")
                genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
                self.model = genai.GenerativeModel('gemini-pro')
            else:
                raise ValueError(f"Unknown LLM provider: {llm_provider}")
        
        logger.info(f"InsightsEngine initialized (LLM: {use_llm})")
    
    def generate_daily_insights(self, feedbacks: List[Dict]) -> Dict:
        """
        Generate daily insights from feedbacks
        
        Args:
            feedbacks: List of analyzed feedback dictionaries
            
        Returns:
            Dictionary with insights
        """
        if not feedbacks:
            return {"error": "No feedbacks to analyze"}
        
        # Calculate statistics
        stats = self._calculate_statistics(feedbacks)
        
        # Generate insights
        insights = {
            "period": "daily",
            "timestamp": datetime.now().isoformat(),
            "total_feedbacks": len(feedbacks),
            "statistics": stats,
            "top_issues": self._identify_top_issues(feedbacks),
            "trending_topics": self._identify_trends(feedbacks),
            "recommendations": self._generate_recommendations(stats),
        }
        
        # Add LLM summary if enabled
        if self.use_llm:
            insights["summary"] = self._generate_llm_summary(stats, feedbacks)
        else:
            insights["summary"] = self._generate_rule_based_summary(stats)
        
        return insights
    
    def _calculate_statistics(self, feedbacks: List[Dict]) -> Dict:
        """Calculate statistical metrics"""
        sentiments = [f.get('sentiment', f.get('sentiment_result', {}).get('sentiment')) for f in feedbacks]
        scores = [f.get('sentiment_score', 0) for f in feedbacks]
        anomalies = [f for f in feedbacks if f.get('is_anomaly', False)]
        
        # Sentiment distribution
        sentiment_counts = Counter(sentiments)
        total = len(feedbacks)
        
        # Category distribution
        all_categories = []
        for f in feedbacks:
            cats = f.get('categories', f.get('category_result', {}).get('categories', []))
            if isinstance(cats, list):
                all_categories.extend(cats)
        
        category_counts = Counter(all_categories)
        
        return {
            "sentiment_distribution": {
                "positive": sentiment_counts.get('POSITIVE', 0),
                "negative": sentiment_counts.get('NEGATIVE', 0),
                "neutral": sentiment_counts.get('NEUTRAL', 0),
                "positive_pct": round(sentiment_counts.get('POSITIVE', 0) / total * 100, 1),
                "negative_pct": round(sentiment_counts.get('NEGATIVE', 0) / total * 100, 1),
            },
            "average_sentiment_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "anomaly_count": len(anomalies),
            "anomaly_rate": round(len(anomalies) / total * 100, 1) if total > 0 else 0,
            "top_categories": [
                {"category": cat, "count": count}
                for cat, count in category_counts.most_common(5)
            ],
        }
    
    def _identify_top_issues(self, feedbacks: List[Dict]) -> List[Dict]:
        """Identify top issues from negative feedback"""
        # Filter negative feedback
        negative_feedbacks = [
            f for f in feedbacks
            if f.get('sentiment', '').upper() == 'NEGATIVE'
        ]
        
        if not negative_feedbacks:
            return []
        
        # Get categories from negative feedback
        negative_categories = []
        for f in negative_feedbacks:
            cats = f.get('categories', [])
            if isinstance(cats, list):
                negative_categories.extend(cats)
        
        category_counts = Counter(negative_categories)
        
        return [
            {
                "issue": cat.replace('_', ' ').title(),
                "count": count,
                "severity": "high" if count > len(negative_feedbacks) * 0.3 else "medium"
            }
            for cat, count in category_counts.most_common(3)
        ]
    
    def _identify_trends(self, feedbacks: List[Dict]) -> List[str]:
        """Identify trending topics"""
        trends = []
        
        # Check for urgency spike
        urgent_count = sum(
            1 for f in feedbacks
            if f.get('text_features', {}).get('urgency_count', 0) > 0
        )
        if urgent_count > len(feedbacks) * 0.2:
            trends.append(f"High urgency: {urgent_count} feedbacks show urgency indicators")
        
        # Check for negative emotions
        high_negative = sum(
            1 for f in feedbacks
            if f.get('text_features', {}).get('negative_emotion_count', 0) > 2
        )
        if high_negative > len(feedbacks) * 0.15:
            trends.append(f"Emotional intensity: {high_negative} feedbacks express strong negative emotions")
        
        # Check sentiment shift
        if len(feedbacks) > 10:
            first_half = feedbacks[:len(feedbacks)//2]
            second_half = feedbacks[len(feedbacks)//2:]
            
            avg_first = sum(f.get('sentiment_score', 0) for f in first_half) / len(first_half)
            avg_second = sum(f.get('sentiment_score', 0) for f in second_half) / len(second_half)
            
            change = avg_second - avg_first
            if abs(change) > 0.2:
                direction = "improved" if change > 0 else "declined"
                trends.append(f"Sentiment {direction} by {abs(change):.2f} points")
        
        return trends or ["No significant trends detected"]
    
    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High negative sentiment
        if stats['sentiment_distribution']['negative_pct'] > 30:
            recommendations.append(
                "⚠️ High negative sentiment detected. "
                "Review customer service processes and product quality."
            )
        
        # High anomaly rate
        if stats['anomaly_rate'] > 15:
            recommendations.append(
                "🚨 Elevated anomaly rate. "
                "Investigate urgent issues and prioritize high-severity cases."
            )
        
        # Category-specific
        for cat_info in stats['top_categories'][:3]:
            cat = cat_info['category']
            count = cat_info['count']
            
            if 'billing' in cat.lower() and count > 5:
                recommendations.append(
                    f"💰 Multiple billing issues ({count}). "
                    "Review payment processes and clarify pricing."
                )
            elif 'technical' in cat.lower() and count > 5:
                recommendations.append(
                    f"🔧 Technical issues reported ({count}). "
                    "Check system stability and provide better documentation."
                )
            elif 'shipping' in cat.lower() and count > 5:
                recommendations.append(
                    f"📦 Shipping concerns ({count}). "
                    "Review logistics partners and delivery timelines."
                )
        
        if not recommendations:
            recommendations.append("✅ Overall performance is healthy. Continue monitoring trends.")
        
        return recommendations
    
    def _generate_rule_based_summary(self, stats: Dict) -> str:
        """Generate summary using rules (no LLM)"""
        total = stats['sentiment_distribution']['positive'] + \
                stats['sentiment_distribution']['negative'] + \
                stats['sentiment_distribution']['neutral']
        
        positive_pct = stats['sentiment_distribution']['positive_pct']
        negative_pct = stats['sentiment_distribution']['negative_pct']
        
        # Overall sentiment
        if positive_pct > 60:
            sentiment_summary = "predominantly positive"
        elif negative_pct > 40:
            sentiment_summary = "concerning with high negative sentiment"
        else:
            sentiment_summary = "mixed"
        
        summary = f"""
**Daily Feedback Summary**

Analyzed {total} customer feedbacks with {sentiment_summary} results.

**Key Findings:**
- {stats['sentiment_distribution']['positive']} positive ({positive_pct}%)
- {stats['sentiment_distribution']['negative']} negative ({negative_pct}%)
- {stats['sentiment_distribution']['neutral']} neutral
- Average sentiment score: {stats['average_sentiment_score']}
- {stats['anomaly_count']} anomalies detected ({stats['anomaly_rate']}%)

**Top Categories:**
"""
        for cat_info in stats['top_categories'][:3]:
            summary += f"\n- {cat_info['category'].replace('_', ' ').title()}: {cat_info['count']} feedbacks"
        
        return summary.strip()
    
    def _generate_llm_summary(self, stats: Dict, feedbacks: List[Dict]) -> str:
        """Generate summary using LLM"""
        # Prepare context
        sample_negative = [
            f.get('text', '') for f in feedbacks
            if f.get('sentiment', '').upper() == 'NEGATIVE'
        ][:5]
        
        prompt = f"""
Analyze this customer feedback data and provide a concise executive summary.

Statistics:
- Total feedbacks: {stats['sentiment_distribution']['positive'] + stats['sentiment_distribution']['negative'] + stats['sentiment_distribution']['neutral']}
- Positive: {stats['sentiment_distribution']['positive_pct']}%
- Negative: {stats['sentiment_distribution']['negative_pct']}%
- Anomalies: {stats['anomaly_rate']}%
- Top categories: {', '.join([c['category'] for c in stats['top_categories'][:3]])}

Sample negative feedbacks:
{chr(10).join(f"- {fb}" for fb in sample_negative[:3])}

Provide:
1. Overall sentiment assessment
2. Key themes and patterns
3. Top 2-3 actionable recommendations

Keep it concise (3-4 sentences).
"""
        
        try:
            if self.llm_provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a customer feedback analyst providing concise insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            
            elif self.llm_provider == "gemini":
                response = self.model.generate_content(prompt)
                return response.text.strip()
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_rule_based_summary(stats)
    
    def export_report(self, insights: Dict, format: str = "markdown") -> str:
        """
        Export insights as formatted report
        
        Args:
            insights: Insights dictionary
            format: Export format ('markdown', 'html', 'text')
            
        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._format_markdown_report(insights)
        elif format == "html":
            return self._format_html_report(insights)
        else:
            return self._format_text_report(insights)
    
    def _format_markdown_report(self, insights: Dict) -> str:
        """Format as markdown"""
        stats = insights['statistics']
        
        report = f"""# Customer Feedback Report
**Period:** {insights['period'].title()}
**Generated:** {datetime.fromisoformat(insights['timestamp']).strftime('%Y-%m-%d %H:%M')}

---

## 📊 Summary

{insights['summary']}

## 📈 Statistics

- **Total Feedbacks:** {insights['total_feedbacks']}
- **Average Sentiment:** {stats['average_sentiment_score']}
- **Anomalies:** {stats['anomaly_count']} ({stats['anomaly_rate']}%)

### Sentiment Distribution
- ✅ Positive: {stats['sentiment_distribution']['positive']} ({stats['sentiment_distribution']['positive_pct']}%)
- ❌ Negative: {stats['sentiment_distribution']['negative']} ({stats['sentiment_distribution']['negative_pct']}%)
- ⚪ Neutral: {stats['sentiment_distribution']['neutral']}

## 🔍 Top Issues
"""
        
        for issue in insights['top_issues']:
            severity_icon = "🔴" if issue['severity'] == 'high' else "🟡"
            report += f"\n- {severity_icon} **{issue['issue']}**: {issue['count']} occurrences"
        
        report += "\n\n## 📈 Trends\n"
        for trend in insights['trending_topics']:
            report += f"\n- {trend}"
        
        report += "\n\n## 💡 Recommendations\n"
        for rec in insights['recommendations']:
            report += f"\n- {rec}"
        
        report += "\n\n---\n*Generated by SentiSight Analytics Engine*"
        
        return report
    
    def _format_text_report(self, insights: Dict) -> str:
        """Format as plain text"""
        md_report = self._format_markdown_report(insights)
        # Remove markdown formatting
        text_report = md_report.replace('#', '').replace('**', '').replace('*', '')
        return text_report
    
    def _format_html_report(self, insights: Dict) -> str:
        """Format as HTML"""
        md_report = self._format_markdown_report(insights)
        # Simple markdown to HTML conversion
        html = md_report.replace('# ', '<h1>').replace('\n## ', '</h1>\n<h2>')
        html = html.replace('\n### ', '</h2>\n<h3>').replace('\n- ', '<br>• ')
        html = html.replace('**', '<strong>').replace('*', '</strong>')
        return f"<html><body>{html}</body></html>"


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*60)
    print("INSIGHTS ENGINE TEST")
    print("="*60)
    
    # Sample feedbacks
    sample_feedbacks = [
        {
            'text': 'Great product!',
            'sentiment': 'POSITIVE',
            'sentiment_score': 0.9,
            'categories': ['product_quality'],
            'is_anomaly': False,
            'text_features': {'urgency_count': 0, 'negative_emotion_count': 0}
        },
        {
            'text': 'Terrible service, very frustrated!',
            'sentiment': 'NEGATIVE',
            'sentiment_score': -0.8,
            'categories': ['customer_service'],
            'is_anomaly': True,
            'text_features': {'urgency_count': 1, 'negative_emotion_count': 2}
        },
        {
            'text': 'Package never arrived',
            'sentiment': 'NEGATIVE',
            'sentiment_score': -0.7,
            'categories': ['shipping_delivery'],
            'is_anomaly': True,
            'text_features': {'urgency_count': 0, 'negative_emotion_count': 1}
        },
    ]
    
    # Generate insights (without LLM)
    engine = InsightsEngine(use_llm=False)
    insights = engine.generate_daily_insights(sample_feedbacks)
    
    print("\nInsights Generated:")
    print(f"Total Feedbacks: {insights['total_feedbacks']}")
    print(f"Anomaly Rate: {insights['statistics']['anomaly_rate']}%")
    
    print("\nTop Issues:")
    for issue in insights['top_issues']:
        print(f"  - {issue['issue']}: {issue['count']}")
    
    print("\nRecommendations:")
    for rec in insights['recommendations']:
        print(f"  - {rec}")
    
    # Export report
    print("\n" + "="*60)
    print("MARKDOWN REPORT")
    print("="*60)
    report = engine.export_report(insights, format="markdown")
    print(report)
