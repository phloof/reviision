"""
Demographics Report Generator

Generates comprehensive PDF reports for demographic analysis including
age distribution, gender patterns, emotion analysis, and actionable recommendations.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from .pdf_generator import PDFGenerator
from .data_analyzer import DataAnalyzer
from .chart_generator import ChartGenerator
from reportlab.platypus import Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

logger = logging.getLogger(__name__)


class DemographicsReportGenerator:
    """Generates comprehensive demographics analysis reports"""
    
    def __init__(self):
        """Initialize the demographics report generator"""
        self.data_analyzer = DataAnalyzer()
        self.chart_generator = ChartGenerator()
    
    def generate_report(self, data: Dict[str, Any], time_period: str, output_path: Optional[str] = None) -> bytes:
        """
        Generate a comprehensive demographics report
        
        Args:
            data: Demographics data from API
            time_period: Time period description (e.g., "Last 24 hours")
            output_path: Optional file path to save PDF
            
        Returns:
            PDF bytes if output_path is None, otherwise path to saved file
        """
        try:
            # Analyze the data
            analysis = self.data_analyzer.analyze_demographics_data(data)
            
            if 'error' in analysis:
                logger.error(f"Data analysis failed: {analysis['error']}")
                return self._generate_error_report(analysis['error'], time_period, output_path)
            
            # Initialize PDF generator
            pdf_gen = PDFGenerator(
                title="Demographics Analysis Report",
                author="ReViision Analytics System"
            )
            
            # Build the report
            self._build_report_content(pdf_gen, data, analysis, time_period)
            
            # Generate PDF
            return pdf_gen.generate_pdf(output_path)
            
        except Exception as e:
            logger.error(f"Error generating demographics report: {e}")
            return self._generate_error_report(str(e), time_period, output_path)
    
    def _build_report_content(self, pdf_gen: PDFGenerator, data: Dict[str, Any], 
                            analysis: Dict[str, Any], time_period: str):
        """Build the complete report content"""
        
        # Title page
        pdf_gen.add_title_page("Demographics Analysis", time_period)
        
        # Executive summary
        self._add_executive_summary(pdf_gen, data, analysis)
        
        # Key metrics overview
        self._add_key_metrics_section(pdf_gen, data, analysis)
        
        # Age distribution analysis
        self._add_age_analysis_section(pdf_gen, data, analysis)
        
        # Gender distribution analysis
        self._add_gender_analysis_section(pdf_gen, data, analysis)
        
        # Emotion analysis
        self._add_emotion_analysis_section(pdf_gen, data, analysis)
        
        # Recommendations
        self._add_recommendations_section(pdf_gen, analysis)
        
        # Methodology and data quality
        self._add_methodology_section(pdf_gen, data, analysis)
    
    def _add_executive_summary(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add executive summary section"""
        pdf_gen.add_section_header("Executive Summary")
        
        summary = analysis.get('summary', {})
        total_visitors = summary.get('total_visitors', 0)
        avg_age = summary.get('avg_age', 0)
        data_quality = summary.get('data_quality', 'Unknown')
        
        # Age analysis summary
        age_analysis = analysis.get('age_analysis', {})
        dominant_age_group = age_analysis.get('dominant_group', {}).get('group', 'Unknown')
        age_diversity = age_analysis.get('diversity_level', 'Unknown')
        
        # Gender analysis summary
        gender_analysis = analysis.get('gender_analysis', {})
        gender_balance = gender_analysis.get('balance_level', 'Unknown')
        
        # Emotion analysis summary
        emotion_analysis = analysis.get('emotion_analysis', {})
        sentiment_level = emotion_analysis.get('sentiment_level', 'Unknown')
        
        summary_text = f"""
        This report analyzes demographic patterns from {total_visitors:,} customer interactions. 
        The analysis reveals key insights about customer composition and behavior patterns.
        
        <b>Key Findings:</b>
        • Average customer age: {avg_age:.1f} years
        • Dominant age group: {dominant_age_group}
        • Age diversity level: {age_diversity}
        • Gender balance: {gender_balance}
        • Overall customer sentiment: {sentiment_level}
        • Data quality assessment: {data_quality}
        
        The demographic composition shows {'strong' if age_diversity == 'High' else 'moderate' if age_diversity == 'Medium' else 'limited'} 
        age diversity and {'balanced' if gender_balance == 'Balanced' else 'imbalanced'} gender representation. 
        Customer sentiment analysis indicates {'positive' if 'Positive' in sentiment_level else 'neutral' if sentiment_level == 'Neutral' else 'concerning'} 
        emotional engagement levels.
        """
        
        pdf_gen.add_paragraph(summary_text)
        pdf_gen.add_spacer(20)
    
    def _add_key_metrics_section(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add key metrics overview section"""
        pdf_gen.add_section_header("Key Demographics Metrics")
        
        # Prepare metrics data
        metrics_data = []
        
        # Total visitors
        total_visitors = data.get('total_visitors', 0)
        metrics_data.append({
            'name': 'Total Visitors Analyzed',
            'value': f"{total_visitors:,}",
            'insight': 'Sample size for analysis'
        })
        
        # Average age
        avg_age = data.get('avg_age', 0)
        age_insight = (
            'Young demographic focus' if avg_age < 30 
            else 'Middle-aged demographic focus' if avg_age < 50 
            else 'Mature demographic focus'
        )
        metrics_data.append({
            'name': 'Average Age',
            'value': f"{avg_age:.1f} years",
            'insight': age_insight
        })
        
        # Gender distribution
        gender_dist = data.get('gender_distribution', {})
        if gender_dist:
            total_gender = sum(gender_dist.values())
            male_pct = (gender_dist.get('male', 0) / total_gender * 100) if total_gender > 0 else 0
            female_pct = (gender_dist.get('female', 0) / total_gender * 100) if total_gender > 0 else 0
            
            metrics_data.append({
                'name': 'Gender Split',
                'value': f"M: {male_pct:.1f}% / F: {female_pct:.1f}%",
                'insight': 'Gender representation balance'
            })
        
        # Age diversity
        age_analysis = analysis.get('age_analysis', {})
        diversity_score = age_analysis.get('diversity_score', 0)
        metrics_data.append({
            'name': 'Age Diversity Index',
            'value': f"{diversity_score:.2f}",
            'insight': f"{age_analysis.get('diversity_level', 'Unknown')} diversity"
        })
        
        # Sentiment score
        emotion_analysis = analysis.get('emotion_analysis', {})
        sentiment_score = emotion_analysis.get('sentiment_score', 0)
        metrics_data.append({
            'name': 'Customer Sentiment Score',
            'value': f"{sentiment_score:+.2f}",
            'insight': f"{emotion_analysis.get('sentiment_level', 'Unknown')} sentiment"
        })
        
        pdf_gen.add_key_metrics_table(metrics_data)
    
    def _add_age_analysis_section(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add detailed age distribution analysis"""
        pdf_gen.add_section_header("Age Distribution Analysis")
        
        age_groups = data.get('age_groups', {})
        age_analysis = analysis.get('age_analysis', {})
        
        if age_analysis.get('error'):
            pdf_gen.add_paragraph(f"Age analysis unavailable: {age_analysis['error']}")
            return
        
        # Add age distribution chart
        if age_groups:
            chart = self.chart_generator.create_age_distribution_chart(age_groups)
            pdf_gen.story.append(chart)
            pdf_gen.add_spacer(20)
        
        # Analysis text
        dominant_group = age_analysis.get('dominant_group', {})
        diversity_level = age_analysis.get('diversity_level', 'Unknown')
        underrepresented = age_analysis.get('underrepresented_groups', [])
        total_groups = age_analysis.get('total_groups_represented', 0)
        
        analysis_text = f"""
        <b>Age Distribution Insights:</b>
        
        The customer base shows {diversity_level.lower()} age diversity with {total_groups} age groups 
        having significant representation (>5% of total). The dominant age group is 
        {dominant_group.get('group', 'unknown')} representing {dominant_group.get('percentage', 0):.1f}% 
        of all customers.
        """
        
        if underrepresented:
            analysis_text += f"""
            
            <b>Underrepresented Groups:</b> {', '.join(underrepresented)} represent less than 10% 
            of the customer base each, indicating potential market expansion opportunities.
            """
        
        if diversity_level == 'Low':
            analysis_text += """
            
            <b>Diversity Concern:</b> The low age diversity suggests heavy concentration in one 
            demographic segment, which may indicate missed opportunities in other age markets.
            """
        
        pdf_gen.add_paragraph(analysis_text)
        pdf_gen.add_spacer(15)
    
    def _add_gender_analysis_section(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add detailed gender distribution analysis"""
        pdf_gen.add_section_header("Gender Distribution Analysis")
        
        gender_dist = data.get('gender_distribution', {})
        gender_analysis = analysis.get('gender_analysis', {})
        
        if gender_analysis.get('error'):
            pdf_gen.add_paragraph(f"Gender analysis unavailable: {gender_analysis['error']}")
            return
        
        # Add gender distribution chart
        if gender_dist:
            chart = self.chart_generator.create_gender_distribution_pie(gender_dist)
            pdf_gen.story.append(chart)
            pdf_gen.add_spacer(20)
        
        # Analysis text
        balance_level = gender_analysis.get('balance_level', 'Unknown')
        balance_score = gender_analysis.get('balance_score', 0)
        skew_direction = gender_analysis.get('skew_direction', 'balanced')
        
        analysis_text = f"""
        <b>Gender Distribution Insights:</b>
        
        The customer base shows {balance_level.lower()} gender representation with a balance 
        score of {balance_score:.2f} (1.0 = perfect balance). 
        """
        
        if skew_direction != 'balanced':
            analysis_text += f"""
            The distribution is skewed toward {skew_direction} customers, which may indicate 
            product positioning or marketing effectiveness differences between gender segments.
            """
        else:
            analysis_text += """
            The balanced gender distribution indicates broad market appeal across gender demographics.
            """
        
        if balance_level == 'Heavily Skewed':
            analysis_text += """
            
            <b>Balance Concern:</b> The heavy gender skew suggests potential missed opportunities 
            in the underrepresented gender segment and may indicate need for targeted marketing 
            or product adjustments.
            """
        
        pdf_gen.add_paragraph(analysis_text)
        pdf_gen.add_spacer(15)

    def _add_emotion_analysis_section(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add detailed emotion analysis section"""
        pdf_gen.add_section_header("Customer Emotion Analysis")

        emotions = data.get('emotions', {})
        emotion_analysis = analysis.get('emotion_analysis', {})

        if emotion_analysis.get('error'):
            pdf_gen.add_paragraph(f"Emotion analysis unavailable: {emotion_analysis['error']}")
            return

        # Add emotion distribution chart
        if emotions:
            chart = self.chart_generator.create_emotion_distribution_chart(emotions)
            pdf_gen.story.append(chart)
            pdf_gen.add_spacer(20)

        # Analysis text
        sentiment_level = emotion_analysis.get('sentiment_level', 'Unknown')
        sentiment_score = emotion_analysis.get('sentiment_score', 0)
        positive_score = emotion_analysis.get('positive_score', 0)
        negative_score = emotion_analysis.get('negative_score', 0)
        dominant_emotion = emotion_analysis.get('dominant_emotion', 'unknown')

        analysis_text = f"""
        <b>Customer Emotion Insights:</b>

        Customer sentiment analysis reveals {sentiment_level.lower()} emotional engagement with
        an overall sentiment score of {sentiment_score:+.2f}. The dominant emotion observed is
        {dominant_emotion}, representing the most common emotional state among customers.

        <b>Sentiment Breakdown:</b>
        • Positive emotions: {positive_score*100:.1f}%
        • Negative emotions: {negative_score*100:.1f}%
        • Neutral emotions: {(1-positive_score-negative_score)*100:.1f}%
        """

        if sentiment_level in ['Negative', 'Very Negative']:
            analysis_text += """

            <b>Concern:</b> The negative sentiment levels indicate potential issues with customer
            experience, product satisfaction, or service quality that require immediate attention.
            """
        elif sentiment_level in ['Very Positive', 'Positive']:
            analysis_text += """

            <b>Strength:</b> The positive sentiment levels indicate strong customer satisfaction
            and engagement, suggesting effective customer experience management.
            """

        pdf_gen.add_paragraph(analysis_text)
        pdf_gen.add_spacer(15)

    def _add_recommendations_section(self, pdf_gen: PDFGenerator, analysis: Dict[str, Any]):
        """Add actionable recommendations section"""
        recommendations = analysis.get('recommendations', [])

        if not recommendations:
            pdf_gen.add_section_header("Recommendations")
            pdf_gen.add_paragraph("No specific recommendations available based on current data.")
            return

        pdf_gen.add_recommendations_section(recommendations)

    def _add_methodology_section(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add methodology and data quality section"""
        pdf_gen.add_section_header("Methodology & Data Quality")

        summary = analysis.get('summary', {})
        total_visitors = summary.get('total_visitors', 0)
        data_quality = summary.get('data_quality', 'Unknown')
        sample_adequate = summary.get('sample_size_adequate', False)

        methodology_text = f"""
        <b>Analysis Methodology:</b>

        This report analyzes demographic data collected through computer vision and machine learning
        algorithms. The analysis includes age estimation, gender classification, and emotion recognition
        based on facial analysis of customer interactions.

        <b>Data Quality Assessment:</b>
        • Sample size: {total_visitors:,} customer interactions
        • Data quality rating: {data_quality}
        • Statistical significance: {'Adequate' if sample_adequate else 'Limited'}

        <b>Limitations:</b>
        • Age and gender estimates are algorithmic approximations with inherent uncertainty
        • Emotion detection reflects momentary expressions, not overall satisfaction
        • Sample may not represent all customer segments equally
        • Analysis period may not capture seasonal or cyclical patterns

        <b>Confidence Levels:</b>
        Recommendations are based on statistical analysis of available data. Higher sample sizes
        provide more reliable insights. Consider extending analysis periods for more robust conclusions.
        """

        pdf_gen.add_paragraph(methodology_text)

    def _generate_error_report(self, error_message: str, time_period: str, output_path: Optional[str] = None) -> bytes:
        """Generate an error report when data analysis fails"""
        try:
            pdf_gen = PDFGenerator(
                title="Demographics Analysis Report - Error",
                author="ReViision Analytics System"
            )

            pdf_gen.add_title_page("Demographics Analysis (Error)", time_period)

            pdf_gen.add_section_header("Report Generation Error")
            pdf_gen.add_paragraph(f"""
            An error occurred while generating the demographics analysis report:

            <b>Error Details:</b> {error_message}

            <b>Possible Causes:</b>
            • Insufficient data available for the selected time period
            • Database connectivity issues
            • Data processing errors

            <b>Recommended Actions:</b>
            • Try extending the analysis time period
            • Verify system connectivity and data availability
            • Contact system administrator if the problem persists
            """)

            return pdf_gen.generate_pdf(output_path)

        except Exception as e:
            logger.error(f"Error generating error report: {e}")
            # Return minimal error response
            return b"Error generating demographics report"
