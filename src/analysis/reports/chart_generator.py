"""
Chart Generation Utilities for PDF Reports

Provides functions to generate various types of charts and visualizations
for inclusion in PDF reports using ReportLab graphics.
"""

import logging
from typing import Dict, List, Any, Tuple
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.lib import colors
from reportlab.lib.units import inch

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generates charts for PDF reports"""
    
    # Color palette for charts
    COLORS = [
        colors.Color(0.2, 0.4, 0.8),  # Blue
        colors.Color(0.3, 0.6, 0.3),  # Green
        colors.Color(0.8, 0.4, 0.2),  # Orange
        colors.Color(0.6, 0.2, 0.6),  # Purple
        colors.Color(0.8, 0.6, 0.2),  # Yellow
        colors.Color(0.2, 0.6, 0.8),  # Light Blue
        colors.Color(0.8, 0.2, 0.4),  # Red
        colors.Color(0.4, 0.8, 0.4),  # Light Green
    ]
    
    def __init__(self):
        """Initialize chart generator"""
        pass
    
    def create_age_distribution_chart(self, age_data: Dict[str, int], width: float = 5*inch, height: float = 3*inch) -> Drawing:
        """Create age distribution bar chart"""
        try:
            if not age_data:
                return self._create_no_data_chart(width, height, "No age data available")
            
            drawing = Drawing(width, height)
            chart = VerticalBarChart()
            
            # Chart dimensions and position
            chart.x = 50
            chart.y = 50
            chart.width = width - 100
            chart.height = height - 100
            
            # Data preparation
            labels = list(age_data.keys())
            values = list(age_data.values())
            
            chart.data = [values]
            chart.categoryAxis.categoryNames = labels
            
            # Styling
            chart.bars[0].fillColor = self.COLORS[0]
            chart.valueAxis.valueMin = 0
            chart.valueAxis.valueMax = max(values) * 1.1 if values else 10
            
            # Labels and formatting
            chart.categoryAxis.labels.boxAnchor = 'ne'
            chart.categoryAxis.labels.dx = 8
            chart.categoryAxis.labels.dy = -2
            chart.categoryAxis.labels.angle = 30
            chart.categoryAxis.labels.fontSize = 8
            
            chart.valueAxis.labels.fontSize = 8
            chart.bars[0].strokeColor = colors.black
            chart.bars[0].strokeWidth = 0.5
            
            drawing.add(chart)
            return drawing
            
        except Exception as e:
            logger.error(f"Error creating age distribution chart: {e}")
            return self._create_error_chart(width, height, "Error creating chart")
    
    def create_gender_distribution_pie(self, gender_data: Dict[str, int], width: float = 4*inch, height: float = 3*inch) -> Drawing:
        """Create gender distribution pie chart"""
        try:
            if not gender_data:
                return self._create_no_data_chart(width, height, "No gender data available")
            
            drawing = Drawing(width, height)
            pie = Pie()
            
            # Chart dimensions and position
            pie.x = width/2 - 80
            pie.y = height/2 - 80
            pie.width = 160
            pie.height = 160
            
            # Data preparation
            labels = list(gender_data.keys())
            values = list(gender_data.values())
            
            pie.data = values
            pie.labels = [f"{label}\n({value})" for label, value in zip(labels, values)]
            
            # Styling with error handling
            try:
                pie.slices.strokeColor = colors.white
                pie.slices.strokeWidth = 1

                # Assign colors - handle ReportLab slice iteration issue
                try:
                    for i in range(len(values)):
                        if i < len(pie.slices):
                            pie.slices[i].fillColor = self.COLORS[i % len(self.COLORS)]
                except (IndexError, KeyError, TypeError) as e:
                    logger.warning(f"Could not set slice colors: {e}")
                    # Fallback: set colors directly on pie
                    pie.slices.fillColor = self.COLORS[0]

                # Labels
                pie.slices.labelRadius = 1.2
                pie.slices.fontName = "Helvetica"
                pie.slices.fontSize = 8

            except Exception as e:
                logger.warning(f"Could not apply pie chart styling: {e}")
                # Continue without styling
            
            drawing.add(pie)
            return drawing
            
        except Exception as e:
            logger.error(f"Error creating gender distribution pie chart: {e}")
            return self._create_error_chart(width, height, "Error creating chart")
    
    def create_emotion_distribution_chart(self, emotion_data: Dict[str, int], width: float = 5*inch, height: float = 3*inch) -> Drawing:
        """Create emotion distribution horizontal bar chart"""
        try:
            if not emotion_data:
                return self._create_no_data_chart(width, height, "No emotion data available")
            
            drawing = Drawing(width, height)
            chart = HorizontalBarChart()
            
            # Chart dimensions and position
            chart.x = 80
            chart.y = 50
            chart.width = width - 130
            chart.height = height - 100
            
            # Data preparation - sort by value for better visualization
            sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0].title() for item in sorted_emotions]
            values = [item[1] for item in sorted_emotions]
            
            chart.data = [values]
            chart.categoryAxis.categoryNames = labels
            
            # Styling
            chart.bars[0].fillColor = self.COLORS[2]  # Orange
            chart.valueAxis.valueMin = 0
            chart.valueAxis.valueMax = max(values) * 1.1 if values else 10
            
            # Labels and formatting
            chart.categoryAxis.labels.fontSize = 8
            chart.valueAxis.labels.fontSize = 8
            chart.bars[0].strokeColor = colors.black
            chart.bars[0].strokeWidth = 0.5
            
            drawing.add(chart)
            return drawing
            
        except Exception as e:
            logger.error(f"Error creating emotion distribution chart: {e}")
            return self._create_error_chart(width, height, "Error creating chart")
    
    def create_traffic_trend_chart(self, traffic_data: Dict[str, Any], width: float = 6*inch, height: float = 3*inch) -> Drawing:
        """Create traffic trend line chart"""
        try:
            labels = traffic_data.get('labels', [])
            values = traffic_data.get('data', [])
            
            if not labels or not values:
                return self._create_no_data_chart(width, height, "No traffic data available")
            
            drawing = Drawing(width, height)
            chart = HorizontalLineChart()
            
            # Chart dimensions and position
            chart.x = 50
            chart.y = 50
            chart.width = width - 100
            chart.height = height - 100
            
            # Data preparation
            chart.data = [values]
            chart.categoryAxis.categoryNames = labels
            
            # Styling
            chart.lines[0].strokeColor = self.COLORS[0]  # Blue
            chart.lines[0].strokeWidth = 2
            chart.lines[0].symbol.kind = 'Circle'
            chart.lines[0].symbol.size = 4
            chart.lines[0].symbol.fillColor = self.COLORS[0]
            
            chart.valueAxis.valueMin = 0
            chart.valueAxis.valueMax = max(values) * 1.1 if values else 10
            
            # Labels and formatting
            chart.categoryAxis.labels.fontSize = 8
            chart.categoryAxis.labels.angle = 45
            chart.valueAxis.labels.fontSize = 8
            
            drawing.add(chart)
            return drawing
            
        except Exception as e:
            logger.error(f"Error creating traffic trend chart: {e}")
            return self._create_error_chart(width, height, "Error creating chart")
    
    def create_performance_comparison_chart(self, metrics: List[Dict[str, Any]], width: float = 5*inch, height: float = 3*inch) -> Drawing:
        """Create performance comparison chart showing current vs benchmark"""
        try:
            if not metrics:
                return self._create_no_data_chart(width, height, "No performance data available")
            
            drawing = Drawing(width, height)
            chart = VerticalBarChart()
            
            # Chart dimensions and position
            chart.x = 50
            chart.y = 50
            chart.width = width - 100
            chart.height = height - 100
            
            # Data preparation
            labels = []
            current_values = []
            benchmark_values = []
            
            for metric in metrics:
                labels.append(metric.get('name', 'Unknown'))
                current_values.append(metric.get('current', 0))
                benchmark_values.append(metric.get('benchmark', 0))
            
            chart.data = [current_values, benchmark_values]
            chart.categoryAxis.categoryNames = labels
            
            # Styling
            chart.bars[0].fillColor = self.COLORS[0]  # Current - Blue
            chart.bars[1].fillColor = self.COLORS[1]  # Benchmark - Green
            
            chart.valueAxis.valueMin = 0
            all_values = current_values + benchmark_values
            chart.valueAxis.valueMax = max(all_values) * 1.1 if all_values else 10
            
            # Labels and formatting
            chart.categoryAxis.labels.fontSize = 8
            chart.valueAxis.labels.fontSize = 8
            chart.bars[0].strokeColor = colors.black
            chart.bars[1].strokeColor = colors.black
            chart.bars[0].strokeWidth = 0.5
            chart.bars[1].strokeWidth = 0.5
            
            drawing.add(chart)
            return drawing
            
        except Exception as e:
            logger.error(f"Error creating performance comparison chart: {e}")
            return self._create_error_chart(width, height, "Error creating chart")
    
    def _create_no_data_chart(self, width: float, height: float, message: str) -> Drawing:
        """Create a placeholder chart for no data scenarios"""
        from reportlab.graphics.shapes import String, Rect
        
        drawing = Drawing(width, height)
        
        # Background
        bg = Rect(0, 0, width, height)
        bg.fillColor = colors.lightgrey
        bg.strokeColor = colors.grey
        drawing.add(bg)
        
        # Message
        text = String(width/2, height/2, message)
        text.textAnchor = 'middle'
        text.fontSize = 12
        text.fillColor = colors.black
        drawing.add(text)
        
        return drawing
    
    def _create_error_chart(self, width: float, height: float, message: str) -> Drawing:
        """Create an error chart"""
        from reportlab.graphics.shapes import String, Rect
        
        drawing = Drawing(width, height)
        
        # Background
        bg = Rect(0, 0, width, height)
        bg.fillColor = colors.Color(1, 0.9, 0.9)  # Light red
        bg.strokeColor = colors.red
        drawing.add(bg)
        
        # Message
        text = String(width/2, height/2, message)
        text.textAnchor = 'middle'
        text.fontSize = 10
        text.fillColor = colors.red
        drawing.add(text)
        
        return drawing
