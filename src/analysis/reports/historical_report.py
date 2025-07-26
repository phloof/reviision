"""
Historical Report Generator

Generates comprehensive PDF reports for historical analytics including
traffic patterns, performance metrics, temporal analysis, and actionable recommendations.
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


class HistoricalReportGenerator:
    """Generates comprehensive historical analysis reports"""
    
    def __init__(self):
        """Initialize the historical report generator"""
        self.data_analyzer = DataAnalyzer()
        self.chart_generator = ChartGenerator()
    
    def generate_report(self, data: Dict[str, Any], time_period: str, output_path: Optional[str] = None) -> bytes:
        """
        Generate a comprehensive historical analysis report
        
        Args:
            data: Historical data from API
            time_period: Time period description (e.g., "Last 24 hours")
            output_path: Optional file path to save PDF
            
        Returns:
            PDF bytes if output_path is None, otherwise path to saved file
        """
        try:
            # Analyze the data
            analysis = self.data_analyzer.analyze_historical_data(data)
            
            if 'error' in analysis:
                logger.error(f"Data analysis failed: {analysis['error']}")
                return self._generate_error_report(analysis['error'], time_period, output_path)
            
            # Initialize PDF generator
            pdf_gen = PDFGenerator(
                title="Historical Analytics Report",
                author="ReViision Analytics System"
            )
            
            # Build the report
            self._build_report_content(pdf_gen, data, analysis, time_period)
            
            # Generate PDF
            return pdf_gen.generate_pdf(output_path)
            
        except Exception as e:
            logger.error(f"Error generating historical report: {e}")
            return self._generate_error_report(str(e), time_period, output_path)
    
    def _build_report_content(self, pdf_gen: PDFGenerator, data: Dict[str, Any], 
                            analysis: Dict[str, Any], time_period: str):
        """Build the complete report content"""
        
        # Title page
        pdf_gen.add_title_page("Historical Analytics", time_period)
        
        # Executive summary
        self._add_executive_summary(pdf_gen, data, analysis)
        
        # Key performance metrics
        self._add_performance_metrics_section(pdf_gen, data, analysis)
        
        # Traffic pattern analysis
        self._add_traffic_analysis_section(pdf_gen, data, analysis)
        
        # Performance benchmarking
        self._add_performance_benchmarking_section(pdf_gen, data, analysis)
        
        # Temporal pattern analysis
        self._add_temporal_analysis_section(pdf_gen, data, analysis)
        
        # Recommendations
        self._add_recommendations_section(pdf_gen, analysis)
        
        # Methodology and data quality
        self._add_methodology_section(pdf_gen, data, analysis)
    
    def _add_executive_summary(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add executive summary section"""
        pdf_gen.add_section_header("Executive Summary")
        
        summary = analysis.get('summary', {})
        total_visitors = summary.get('total_visitors', 0)
        avg_dwell_time = summary.get('avg_dwell_time', 0)
        conversion_rate = summary.get('conversion_rate', 0)
        peak_hour = summary.get('peak_hour', 'Unknown')
        period_hours = summary.get('period_hours', 24)
        
        # Performance analysis
        performance = analysis.get('performance_analysis', {})
        overall_performance = performance.get('overall_performance', 'Unknown')
        
        # Traffic analysis
        traffic_analysis = analysis.get('traffic_analysis', {})
        pattern_type = traffic_analysis.get('pattern_type', 'Unknown')
        
        summary_text = f"""
        This report analyzes {period_hours} hours of operational data covering {total_visitors:,} 
        customer interactions. The analysis reveals key insights about traffic patterns, 
        customer engagement, and operational performance.
        
        <b>Key Performance Indicators:</b>
        • Total visitors: {total_visitors:,}
        • Average dwell time: {avg_dwell_time:.0f} seconds
        • Conversion rate: {conversion_rate:.1f}%
        • Peak activity hour: {peak_hour}
        • Overall performance rating: {overall_performance}
        • Traffic pattern: {pattern_type}
        
        The analysis shows {'strong' if overall_performance == 'Excellent' else 'good' if overall_performance == 'Good' else 'adequate' if overall_performance == 'Average' else 'concerning'} 
        operational performance with {'consistent' if pattern_type == 'Consistent' else 'variable'} 
        traffic patterns. Customer engagement levels are {'high' if avg_dwell_time > 180 else 'moderate' if avg_dwell_time > 120 else 'low'} 
        based on dwell time analysis.
        """
        
        pdf_gen.add_paragraph(summary_text)
        pdf_gen.add_spacer(20)
    
    def _add_performance_metrics_section(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add key performance metrics section"""
        pdf_gen.add_section_header("Key Performance Metrics")
        
        # Prepare metrics data
        metrics_data = []
        
        # Total visitors
        total_visitors = data.get('total_visitors', 0)
        visitor_rate = total_visitors / (data.get('period_hours', 24))
        metrics_data.append({
            'name': 'Total Visitors',
            'value': f"{total_visitors:,}",
            'insight': f"{visitor_rate:.1f} visitors/hour average"
        })
        
        # Conversion rate with benchmark
        conversion_rate = data.get('conversion_rate', 0)
        performance = analysis.get('performance_analysis', {})
        conv_perf = performance.get('conversion_performance', {})
        conv_level = conv_perf.get('level', 'Unknown')
        
        metrics_data.append({
            'name': 'Conversion Rate',
            'value': f"{conversion_rate:.1f}%",
            'insight': f"{conv_level} performance level"
        })
        
        # Dwell time with benchmark
        avg_dwell_time = data.get('avg_dwell_time', 0)
        dwell_perf = performance.get('dwell_time_performance', {})
        dwell_level = dwell_perf.get('level', 'Unknown')
        
        metrics_data.append({
            'name': 'Average Dwell Time',
            'value': f"{avg_dwell_time:.0f} seconds",
            'insight': f"{dwell_level} engagement level"
        })
        
        # Peak hour analysis
        peak_hour = data.get('peak_hour', 'Unknown')
        traffic_analysis = analysis.get('traffic_analysis', {})
        peak_traffic = traffic_analysis.get('peak_traffic', 0)
        
        metrics_data.append({
            'name': 'Peak Activity Hour',
            'value': peak_hour,
            'insight': f"{peak_traffic} visitors at peak"
        })
        
        # Traffic consistency
        consistency_score = traffic_analysis.get('consistency_score', 0)
        pattern_type = traffic_analysis.get('pattern_type', 'Unknown')
        
        metrics_data.append({
            'name': 'Traffic Consistency',
            'value': f"{consistency_score:.2f}",
            'insight': f"{pattern_type} traffic pattern"
        })
        
        pdf_gen.add_key_metrics_table(metrics_data)
    
    def _add_traffic_analysis_section(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add detailed traffic pattern analysis"""
        pdf_gen.add_section_header("Traffic Pattern Analysis")
        
        traffic_data = data.get('traffic', {})
        traffic_analysis = analysis.get('traffic_analysis', {})
        
        if traffic_analysis.get('error'):
            pdf_gen.add_paragraph(f"Traffic analysis unavailable: {traffic_analysis['error']}")
            return
        
        # Add traffic trend chart
        if traffic_data:
            chart = self.chart_generator.create_traffic_trend_chart(traffic_data)
            pdf_gen.story.append(chart)
            pdf_gen.add_spacer(20)
        
        # Analysis text
        total_traffic = traffic_analysis.get('total_traffic', 0)
        avg_traffic = traffic_analysis.get('avg_traffic', 0)
        peak_traffic = traffic_analysis.get('peak_traffic', 0)
        min_traffic = traffic_analysis.get('min_traffic', 0)
        peak_hours = traffic_analysis.get('peak_hours', [])
        pattern_type = traffic_analysis.get('pattern_type', 'Unknown')
        consistency_score = traffic_analysis.get('consistency_score', 0)
        
        analysis_text = f"""
        <b>Traffic Pattern Insights:</b>
        
        The analysis reveals {pattern_type.lower()} traffic patterns with a consistency score of 
        {consistency_score:.2f}. Total traffic volume reached {total_traffic:,} visitors with 
        an average of {avg_traffic:.1f} visitors per hour.
        
        <b>Traffic Distribution:</b>
        • Peak traffic: {peak_traffic} visitors
        • Minimum traffic: {min_traffic} visitors
        • Traffic range: {peak_traffic - min_traffic} visitors
        • Peak hours: {', '.join(peak_hours[:3]) if peak_hours else 'Not identified'}
        """
        
        if pattern_type == 'Highly Variable':
            analysis_text += """
            
            <b>Variability Concern:</b> The highly variable traffic pattern indicates unpredictable 
            customer flow, which may impact staffing efficiency and customer service quality during 
            unexpected peak periods.
            """
        elif pattern_type == 'Consistent':
            analysis_text += """
            
            <b>Consistency Strength:</b> The consistent traffic pattern enables predictable staffing 
            and resource allocation, supporting efficient operations and consistent customer service.
            """
        
        pdf_gen.add_paragraph(analysis_text)
        pdf_gen.add_spacer(15)

    def _add_performance_benchmarking_section(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add performance benchmarking section"""
        pdf_gen.add_section_header("Performance Benchmarking")

        performance = analysis.get('performance_analysis', {})
        conv_perf = performance.get('conversion_performance', {})
        dwell_perf = performance.get('dwell_time_performance', {})

        # Prepare performance comparison data
        performance_metrics = []

        if conv_perf:
            performance_metrics.append({
                'name': 'Conversion Rate',
                'current': conv_perf.get('value', 0),
                'benchmark': conv_perf.get('benchmark_good', 10)
            })

        if dwell_perf:
            performance_metrics.append({
                'name': 'Dwell Time',
                'current': dwell_perf.get('value', 0),
                'benchmark': dwell_perf.get('benchmark_good', 180)
            })

        # Add performance comparison chart
        if performance_metrics:
            chart = self.chart_generator.create_performance_comparison_chart(performance_metrics)
            pdf_gen.story.append(chart)
            pdf_gen.add_spacer(20)

        # Analysis text
        overall_performance = performance.get('overall_performance', 'Unknown')
        conv_level = conv_perf.get('level', 'Unknown')
        dwell_level = dwell_perf.get('level', 'Unknown')

        benchmarking_text = f"""
        <b>Performance Benchmarking Results:</b>

        Overall performance rating: {overall_performance}

        <b>Individual Metric Performance:</b>
        • Conversion Rate: {conv_level} ({conv_perf.get('value', 0):.1f}% vs {conv_perf.get('benchmark_good', 10):.1f}% benchmark)
        • Dwell Time: {dwell_level} ({dwell_perf.get('value', 0):.0f}s vs {dwell_perf.get('benchmark_good', 180):.0f}s benchmark)

        <b>Industry Comparison:</b>
        Performance levels are compared against retail industry benchmarks. "Good" performance
        indicates above-average results, while "Excellent" represents top-tier performance.
        """

        if overall_performance == 'Below Average':
            benchmarking_text += """

            <b>Performance Gap:</b> Current metrics fall below industry standards, indicating
            significant opportunities for improvement through operational optimization and
            customer experience enhancement.
            """
        elif overall_performance == 'Excellent':
            benchmarking_text += """

            <b>Performance Excellence:</b> Current metrics exceed industry benchmarks, indicating
            strong operational efficiency and customer engagement. Focus on maintaining these
            high standards and scaling best practices.
            """

        pdf_gen.add_paragraph(benchmarking_text)
        pdf_gen.add_spacer(15)

    def _add_temporal_analysis_section(self, pdf_gen: PDFGenerator, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Add temporal pattern analysis section"""
        pdf_gen.add_section_header("Temporal Pattern Analysis")

        temporal_analysis = analysis.get('temporal_analysis', {})
        period_hours = data.get('period_hours', 24)
        period_type = temporal_analysis.get('period_type', 'Unknown')

        # Calculate hourly averages and patterns
        traffic_data = data.get('traffic', {})
        traffic_values = traffic_data.get('data', [])

        temporal_text = f"""
        <b>Analysis Period:</b> {period_hours} hours ({period_type} analysis)

        <b>Temporal Insights:</b>
        """

        if period_type == 'Short-term':
            temporal_text += """
            This short-term analysis provides insights into immediate operational patterns and
            customer behavior. Results reflect current conditions but may not capture longer-term
            trends or seasonal variations.
            """
        elif period_type == 'Medium-term':
            temporal_text += """
            This medium-term analysis captures weekly patterns and provides reliable insights into
            operational trends. Results are suitable for tactical planning and operational adjustments.
            """
        else:
            temporal_text += """
            This long-term analysis reveals strategic patterns and seasonal trends. Results support
            strategic planning and long-term operational optimization.
            """

        if traffic_values:
            # Calculate basic temporal statistics
            peak_index = traffic_values.index(max(traffic_values))
            low_index = traffic_values.index(min(traffic_values))

            temporal_text += f"""

            <b>Pattern Characteristics:</b>
            • Peak activity occurs at position {peak_index + 1} in the analysis period
            • Lowest activity occurs at position {low_index + 1} in the analysis period
            • Traffic variation range: {max(traffic_values) - min(traffic_values)} visitors
            """

        pdf_gen.add_paragraph(temporal_text)
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

        total_visitors = data.get('total_visitors', 0)
        period_hours = data.get('period_hours', 24)

        methodology_text = f"""
        <b>Analysis Methodology:</b>

        This report analyzes historical operational data collected through computer vision and
        customer tracking systems. The analysis includes traffic pattern recognition, performance
        benchmarking, and temporal trend analysis.

        <b>Data Quality Assessment:</b>
        • Analysis period: {period_hours} hours
        • Total interactions: {total_visitors:,}
        • Data completeness: {'High' if total_visitors > 100 else 'Medium' if total_visitors > 50 else 'Limited'}
        • Temporal resolution: Hourly aggregation

        <b>Performance Benchmarks:</b>
        Industry benchmarks are based on retail analytics standards:
        • Conversion Rate: Excellent >15%, Good >10%, Average >5%
        • Dwell Time: Excellent >300s, Good >180s, Average >120s

        <b>Limitations:</b>
        • Analysis reflects the specific time period and may not represent long-term trends
        • Performance comparisons are based on general retail benchmarks
        • External factors (weather, events, promotions) may influence results
        • Data accuracy depends on system calibration and environmental conditions

        <b>Confidence Levels:</b>
        Recommendations are based on statistical analysis with confidence levels proportional
        to sample size and data quality. Larger datasets provide more reliable insights.
        """

        pdf_gen.add_paragraph(methodology_text)

    def _generate_error_report(self, error_message: str, time_period: str, output_path: Optional[str] = None) -> bytes:
        """Generate an error report when data analysis fails"""
        try:
            pdf_gen = PDFGenerator(
                title="Historical Analytics Report - Error",
                author="ReViision Analytics System"
            )

            pdf_gen.add_title_page("Historical Analytics (Error)", time_period)

            pdf_gen.add_section_header("Report Generation Error")
            pdf_gen.add_paragraph(f"""
            An error occurred while generating the historical analytics report:

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
            return b"Error generating historical report"
