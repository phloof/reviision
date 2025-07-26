"""
Data Analysis and Recommendation Engine

Analyzes analytics data and generates actionable, data-driven recommendations
for business optimization and customer experience improvement.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import statistics

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Analyzes data and generates actionable recommendations"""
    
    # Industry benchmarks for comparison
    BENCHMARKS = {
        'conversion_rate': {'excellent': 15, 'good': 10, 'average': 5, 'poor': 2},
        'dwell_time': {'excellent': 300, 'good': 180, 'average': 120, 'poor': 60},  # seconds
        'gender_balance': {'balanced_threshold': 0.4},  # 40-60% is considered balanced
        'age_diversity': {'diverse_threshold': 0.3},  # At least 30% in multiple age groups
        'emotion_positive': {'excellent': 0.7, 'good': 0.5, 'average': 0.3, 'poor': 0.1}
    }
    
    def __init__(self):
        """Initialize the data analyzer"""
        pass
    
    def analyze_demographics_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze demographics data and generate insights
        
        Args:
            data: Demographics data from API
            
        Returns:
            Dict containing analysis results and recommendations
        """
        try:
            logger.info("Starting demographic data analysis...")

            logger.info("Analyzing demographic summary...")
            summary = self._analyze_demographic_summary(data)
            logger.info("Demographic summary analysis completed")

            logger.info("Analyzing age distribution...")
            age_analysis = self._analyze_age_distribution(data.get('age_groups', {}))
            logger.info("Age distribution analysis completed")

            logger.info("Analyzing gender distribution...")
            gender_analysis = self._analyze_gender_distribution(data.get('gender_distribution', {}))
            logger.info("Gender distribution analysis completed")

            logger.info("Analyzing emotions...")
            emotion_analysis = self._analyze_emotions(data.get('emotions', {}))
            logger.info("Emotion analysis completed")

            analysis = {
                'summary': summary,
                'age_analysis': age_analysis,
                'gender_analysis': gender_analysis,
                'emotion_analysis': emotion_analysis,
                'recommendations': []
            }

            # Generate recommendations based on analysis
            logger.info("Generating recommendations...")
            analysis['recommendations'] = self._generate_demographic_recommendations(analysis, data)
            logger.info("Recommendations generated")

            logger.info("Demographic data analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing demographics data: {e}")
            return {'error': str(e)}
    
    def analyze_historical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze historical data and generate insights
        
        Args:
            data: Historical data from API
            
        Returns:
            Dict containing analysis results and recommendations
        """
        try:
            analysis = {
                'summary': self._analyze_historical_summary(data),
                'traffic_analysis': self._analyze_traffic_patterns(data.get('traffic', {})),
                'performance_analysis': self._analyze_performance_metrics(data),
                'temporal_analysis': self._analyze_temporal_patterns(data),
                'recommendations': []
            }
            
            # Generate recommendations based on analysis
            analysis['recommendations'] = self._generate_historical_recommendations(analysis, data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing historical data: {e}")
            return {'error': str(e)}
    
    def _analyze_demographic_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall demographic summary"""
        total_visitors = data.get('total_visitors', 0)
        avg_age = data.get('avg_age', 0)
        
        return {
            'total_visitors': total_visitors,
            'avg_age': avg_age,
            'data_quality': 'Good' if total_visitors > 50 else 'Limited' if total_visitors > 10 else 'Insufficient',
            'sample_size_adequate': total_visitors >= 30
        }
    
    def _analyze_age_distribution(self, age_groups: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze age distribution patterns"""
        if not age_groups:
            return {'error': 'No age data available'}

        # Handle both formats: Dict[str, int] and Dict[str, Dict[str, Any]]
        counts = {}
        for group, value in age_groups.items():
            if isinstance(value, dict):
                # New format: {'display_name': '18-24', 'count': 20}
                counts[group] = value.get('count', 0)
            else:
                # Old format: direct integer values
                counts[group] = value

        total = sum(counts.values())
        if total == 0:
            return {'error': 'No age data available'}

        # Calculate percentages
        percentages = {group: (count / total) * 100 for group, count in counts.items()}
        
        # Find dominant age group
        dominant_group = max(percentages.items(), key=lambda x: x[1])
        
        # Calculate diversity index (how evenly distributed)
        diversity_score = self._calculate_diversity_index(list(percentages.values()))
        
        # Identify underrepresented groups
        underrepresented = [group for group, pct in percentages.items() if pct < 10]
        
        return {
            'distribution': percentages,
            'dominant_group': {'group': dominant_group[0], 'percentage': dominant_group[1]},
            'diversity_score': diversity_score,
            'diversity_level': 'High' if diversity_score > 0.7 else 'Medium' if diversity_score > 0.4 else 'Low',
            'underrepresented_groups': underrepresented,
            'total_groups_represented': len([p for p in percentages.values() if p > 5])
        }
    
    def _analyze_gender_distribution(self, gender_dist: Dict[str, int]) -> Dict[str, Any]:
        """Analyze gender distribution patterns"""
        if not gender_dist:
            return {'error': 'No gender data available'}
        
        total = sum(gender_dist.values())
        if total == 0:
            return {'error': 'No gender data available'}
        
        # Calculate percentages
        percentages = {gender: (count / total) * 100 for gender, count in gender_dist.items()}
        
        # Analyze balance
        male_pct = percentages.get('male', 0) / 100
        female_pct = percentages.get('female', 0) / 100
        
        balance_score = 1 - abs(male_pct - female_pct)  # 1 = perfect balance, 0 = completely skewed
        
        balance_level = (
            'Balanced' if balance_score >= self.BENCHMARKS['gender_balance']['balanced_threshold']
            else 'Slightly Skewed' if balance_score >= 0.2
            else 'Heavily Skewed'
        )
        
        return {
            'distribution': percentages,
            'balance_score': balance_score,
            'balance_level': balance_level,
            'dominant_gender': max(percentages.items(), key=lambda x: x[1])[0] if percentages else None,
            'skew_direction': 'male' if male_pct > female_pct else 'female' if female_pct > male_pct else 'balanced'
        }
    
    def _analyze_emotions(self, emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotion distribution and sentiment"""
        if not emotions:
            return {'error': 'No emotion data available'}

        # Handle both formats: Dict[str, int] and Dict[str, Dict[str, Any]]
        counts = {}
        for emotion, value in emotions.items():
            if isinstance(value, dict):
                # New format: {'display_name': 'Happy', 'count': 40}
                counts[emotion] = value.get('count', 0)
            else:
                # Old format: direct integer values
                counts[emotion] = value

        total = sum(counts.values())
        if total == 0:
            return {'error': 'No emotion data available'}

        # Calculate percentages
        percentages = {emotion: (count / total) * 100 for emotion, count in counts.items()}
        
        # Categorize emotions
        positive_emotions = ['happy', 'joy', 'surprise']
        negative_emotions = ['sad', 'angry', 'fear', 'disgust']
        neutral_emotions = ['neutral']
        
        positive_score = sum(percentages.get(emotion, 0) for emotion in positive_emotions) / 100
        negative_score = sum(percentages.get(emotion, 0) for emotion in negative_emotions) / 100
        neutral_score = sum(percentages.get(emotion, 0) for emotion in neutral_emotions) / 100
        
        # Overall sentiment
        sentiment_score = positive_score - negative_score
        sentiment_level = (
            'Very Positive' if sentiment_score > 0.3
            else 'Positive' if sentiment_score > 0.1
            else 'Neutral' if sentiment_score > -0.1
            else 'Negative' if sentiment_score > -0.3
            else 'Very Negative'
        )
        
        return {
            'distribution': percentages,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'neutral_score': neutral_score,
            'sentiment_score': sentiment_score,
            'sentiment_level': sentiment_level,
            'dominant_emotion': max(percentages.items(), key=lambda x: x[1])[0]
        }
    
    def _analyze_historical_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze historical summary metrics"""
        return {
            'total_visitors': data.get('total_visitors', 0),
            'avg_dwell_time': data.get('avg_dwell_time', 0),
            'conversion_rate': data.get('conversion_rate', 0),
            'peak_hour': data.get('peak_hour', 'Unknown'),
            'period_hours': data.get('period_hours', 24)
        }
    
    def _analyze_traffic_patterns(self, traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze traffic patterns and trends"""
        if not traffic_data or not traffic_data.get('data'):
            return {'error': 'No traffic data available'}
        
        data_points = traffic_data.get('data', [])
        labels = traffic_data.get('labels', [])
        
        if not data_points:
            return {'error': 'No traffic data points available'}
        
        # Calculate basic statistics
        total_traffic = sum(data_points)
        avg_traffic = statistics.mean(data_points)
        peak_traffic = max(data_points)
        min_traffic = min(data_points)
        
        # Find peak hours
        peak_indices = [i for i, val in enumerate(data_points) if val == peak_traffic]
        peak_hours = [labels[i] for i in peak_indices if i < len(labels)]
        
        # Calculate traffic distribution
        traffic_variance = statistics.variance(data_points) if len(data_points) > 1 else 0
        traffic_consistency = 1 - (traffic_variance / (avg_traffic ** 2)) if avg_traffic > 0 else 0
        
        return {
            'total_traffic': total_traffic,
            'avg_traffic': avg_traffic,
            'peak_traffic': peak_traffic,
            'min_traffic': min_traffic,
            'peak_hours': peak_hours,
            'traffic_variance': traffic_variance,
            'consistency_score': max(0, min(1, traffic_consistency)),
            'pattern_type': 'Consistent' if traffic_consistency > 0.7 else 'Variable' if traffic_consistency > 0.3 else 'Highly Variable'
        }
    
    def _analyze_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics against benchmarks"""
        conversion_rate = data.get('conversion_rate', 0)
        avg_dwell_time = data.get('avg_dwell_time', 0)
        
        # Benchmark conversion rate
        conv_benchmark = self._benchmark_metric(conversion_rate, self.BENCHMARKS['conversion_rate'])
        
        # Benchmark dwell time
        dwell_benchmark = self._benchmark_metric(avg_dwell_time, self.BENCHMARKS['dwell_time'])
        
        return {
            'conversion_performance': conv_benchmark,
            'dwell_time_performance': dwell_benchmark,
            'overall_performance': self._calculate_overall_performance([conv_benchmark, dwell_benchmark])
        }
    
    def _analyze_temporal_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in the data"""
        # This would be expanded with more sophisticated time series analysis
        period_hours = data.get('period_hours', 24)
        
        return {
            'analysis_period': f"{period_hours} hours",
            'period_type': 'Short-term' if period_hours <= 24 else 'Medium-term' if period_hours <= 168 else 'Long-term'
        }
    
    def _calculate_diversity_index(self, values: List[float]) -> float:
        """Calculate diversity index (Shannon diversity adapted for percentages)"""
        if not values or sum(values) == 0:
            return 0
        
        # Normalize to probabilities
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        
        # Calculate Shannon diversity
        import math
        diversity = -sum(p * math.log(p) for p in probabilities if p > 0)
        
        # Normalize to 0-1 scale
        max_diversity = math.log(len(probabilities)) if len(probabilities) > 1 else 1
        return diversity / max_diversity if max_diversity > 0 else 0
    
    def _benchmark_metric(self, value: float, benchmarks: Dict[str, float]) -> Dict[str, Any]:
        """Compare a metric against benchmarks"""
        if value >= benchmarks['excellent']:
            level = 'Excellent'
        elif value >= benchmarks['good']:
            level = 'Good'
        elif value >= benchmarks['average']:
            level = 'Average'
        else:
            level = 'Below Average'
        
        return {
            'value': value,
            'level': level,
            'benchmark_excellent': benchmarks['excellent'],
            'benchmark_good': benchmarks['good'],
            'benchmark_average': benchmarks['average']
        }
    
    def _calculate_overall_performance(self, performance_metrics: List[Dict[str, Any]]) -> str:
        """Calculate overall performance level"""
        levels = [metric.get('level', 'Below Average') for metric in performance_metrics]
        level_scores = {'Excellent': 4, 'Good': 3, 'Average': 2, 'Below Average': 1}
        
        avg_score = sum(level_scores.get(level, 1) for level in levels) / len(levels)
        
        if avg_score >= 3.5:
            return 'Excellent'
        elif avg_score >= 2.5:
            return 'Good'
        elif avg_score >= 1.5:
            return 'Average'
        else:
            return 'Below Average'

    def _generate_demographic_recommendations(self, analysis: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on demographic analysis"""
        recommendations = []

        # Age distribution recommendations
        age_analysis = analysis.get('age_analysis', {})
        if not age_analysis.get('error'):
            diversity_level = age_analysis.get('diversity_level', 'Low')
            dominant_group = age_analysis.get('dominant_group', {})
            underrepresented = age_analysis.get('underrepresented_groups', [])

            if diversity_level == 'Low':
                recommendations.append({
                    'title': 'Improve Age Demographic Diversity',
                    'description': f'Current customer base is heavily concentrated in {dominant_group.get("group", "one age group")} ({dominant_group.get("percentage", 0):.1f}%). Consider targeted marketing campaigns to attract other age demographics, particularly {", ".join(underrepresented[:2]) if underrepresented else "younger and older customers"}.',
                    'priority': 'High',
                    'category': 'Marketing'
                })

            if len(underrepresented) > 2:
                recommendations.append({
                    'title': 'Target Underrepresented Age Groups',
                    'description': f'Several age groups are underrepresented: {", ".join(underrepresented)}. Develop specific product offerings and marketing strategies to appeal to these demographics.',
                    'priority': 'Medium',
                    'category': 'Product Development'
                })

        # Gender distribution recommendations
        gender_analysis = analysis.get('gender_analysis', {})
        if not gender_analysis.get('error'):
            balance_level = gender_analysis.get('balance_level', 'Balanced')
            skew_direction = gender_analysis.get('skew_direction', 'balanced')

            if balance_level in ['Heavily Skewed', 'Slightly Skewed']:
                opposite_gender = 'female' if skew_direction == 'male' else 'male'
                recommendations.append({
                    'title': f'Address Gender Imbalance - Attract More {opposite_gender.title()} Customers',
                    'description': f'Customer base is skewed toward {skew_direction} customers. Implement targeted campaigns, adjust product mix, and review store environment to better appeal to {opposite_gender} demographics.',
                    'priority': 'High' if balance_level == 'Heavily Skewed' else 'Medium',
                    'category': 'Marketing'
                })

        # Emotion analysis recommendations
        emotion_analysis = analysis.get('emotion_analysis', {})
        if not emotion_analysis.get('error'):
            sentiment_level = emotion_analysis.get('sentiment_level', 'Neutral')
            sentiment_score = emotion_analysis.get('sentiment_score', 0)
            negative_score = emotion_analysis.get('negative_score', 0)

            if sentiment_level in ['Negative', 'Very Negative']:
                recommendations.append({
                    'title': 'Improve Customer Sentiment and Experience',
                    'description': f'Customer sentiment is {sentiment_level.lower()} with {negative_score*100:.1f}% negative emotions detected. Review customer service protocols, store environment, product quality, and staff training to address underlying issues.',
                    'priority': 'High',
                    'category': 'Operations'
                })
            elif sentiment_level == 'Neutral':
                recommendations.append({
                    'title': 'Enhance Customer Engagement and Satisfaction',
                    'description': 'Customer sentiment is neutral, indicating room for improvement. Consider implementing customer engagement initiatives, improving product presentation, and enhancing the overall shopping experience.',
                    'priority': 'Medium',
                    'category': 'Customer Experience'
                })

        # Data quality recommendations
        summary = analysis.get('summary', {})
        if summary.get('data_quality') == 'Limited':
            recommendations.append({
                'title': 'Increase Data Collection Coverage',
                'description': f'Current sample size ({summary.get("total_visitors", 0)} visitors) is limited. Consider extending analysis period or improving detection coverage to get more reliable insights.',
                'priority': 'Low',
                'category': 'Analytics'
            })

        return recommendations

    def _generate_historical_recommendations(self, analysis: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on historical analysis"""
        recommendations = []

        # Performance-based recommendations
        performance = analysis.get('performance_analysis', {})
        conv_perf = performance.get('conversion_performance', {})
        dwell_perf = performance.get('dwell_time_performance', {})

        if conv_perf.get('level') == 'Below Average':
            conv_rate = conv_perf.get('value', 0)
            target = conv_perf.get('benchmark_good', 10)
            recommendations.append({
                'title': 'Improve Conversion Rate Performance',
                'description': f'Current conversion rate ({conv_rate:.1f}%) is below industry standards. Target: {target}%. Implement staff training, optimize product placement, improve customer service, and review pricing strategy.',
                'priority': 'High',
                'category': 'Sales Optimization'
            })
        elif conv_perf.get('level') == 'Average':
            recommendations.append({
                'title': 'Optimize Conversion Rate to Exceed Industry Standards',
                'description': f'Conversion rate is average. Implement advanced sales techniques, personalized customer service, and strategic product recommendations to move into the "Good" performance tier.',
                'priority': 'Medium',
                'category': 'Sales Optimization'
            })

        if dwell_perf.get('level') == 'Below Average':
            dwell_time = dwell_perf.get('value', 0)
            target = dwell_perf.get('benchmark_good', 180)
            recommendations.append({
                'title': 'Increase Customer Dwell Time',
                'description': f'Average dwell time ({dwell_time:.0f} seconds) is below optimal levels. Target: {target} seconds. Improve store layout, add engaging displays, create comfortable seating areas, and enhance product discovery.',
                'priority': 'High',
                'category': 'Store Layout'
            })

        # Traffic pattern recommendations
        traffic_analysis = analysis.get('traffic_analysis', {})
        if not traffic_analysis.get('error'):
            pattern_type = traffic_analysis.get('pattern_type', 'Variable')
            peak_hours = traffic_analysis.get('peak_hours', [])
            consistency_score = traffic_analysis.get('consistency_score', 0)

            if pattern_type == 'Highly Variable':
                recommendations.append({
                    'title': 'Stabilize Traffic Patterns',
                    'description': f'Traffic patterns are highly variable (consistency score: {consistency_score:.2f}). Implement consistent marketing campaigns, regular promotions, and predictable operating hours to create more stable customer flow.',
                    'priority': 'Medium',
                    'category': 'Marketing'
                })

            if peak_hours:
                peak_hours_str = ', '.join(peak_hours[:3])
                recommendations.append({
                    'title': 'Optimize Staffing for Peak Hours',
                    'description': f'Peak traffic occurs at {peak_hours_str}. Ensure adequate staffing during these periods to maintain service quality and maximize conversion opportunities.',
                    'priority': 'High',
                    'category': 'Operations'
                })

        # Overall performance recommendations
        overall_perf = performance.get('overall_performance', 'Average')
        if overall_perf == 'Below Average':
            recommendations.append({
                'title': 'Comprehensive Performance Improvement Initiative',
                'description': 'Overall performance metrics are below industry standards. Implement a comprehensive improvement program focusing on staff training, customer experience enhancement, and operational efficiency.',
                'priority': 'High',
                'category': 'Strategic'
            })
        elif overall_perf == 'Excellent':
            recommendations.append({
                'title': 'Maintain Excellence and Scale Best Practices',
                'description': 'Performance metrics are excellent. Document current best practices, train staff on maintaining standards, and consider expanding successful strategies to other locations or time periods.',
                'priority': 'Low',
                'category': 'Strategic'
            })

        return recommendations
