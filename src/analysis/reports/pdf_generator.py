"""
Base PDF Generator for ReViision Analytics Reports

Provides common PDF generation utilities, styling, and layout functions
for creating professional analytics reports.
"""

import io
import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

import logging
logger = logging.getLogger(__name__)


class PDFGenerator:
    """Base PDF generator with common styling and utilities"""
    
    # Color scheme
    PRIMARY_COLOR = colors.Color(0.2, 0.4, 0.8)  # Blue
    SECONDARY_COLOR = colors.Color(0.3, 0.6, 0.3)  # Green
    ACCENT_COLOR = colors.Color(0.8, 0.4, 0.2)  # Orange
    GRAY_COLOR = colors.Color(0.5, 0.5, 0.5)  # Gray
    LIGHT_GRAY = colors.Color(0.9, 0.9, 0.9)  # Light Gray
    
    def __init__(self, title="Analytics Report", author="ReViision Analytics"):
        """Initialize PDF generator with basic settings"""
        self.title = title
        self.author = author
        self.page_size = A4
        self.margin = 0.75 * inch
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Story elements (content)
        self.story = []
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=self.PRIMARY_COLOR,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            spaceBefore=20,
            textColor=self.PRIMARY_COLOR,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=18,
            textColor=self.SECONDARY_COLOR,
            fontName='Helvetica-Bold'
        ))
        
        # Body text with justification
        self.styles.add(ParagraphStyle(
            name='BodyJustified',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Recommendation style
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            spaceBefore=4,
            leftIndent=20,
            bulletIndent=10,
            fontName='Helvetica',
            textColor=self.ACCENT_COLOR
        ))
        
        # Key metric style
        self.styles.add(ParagraphStyle(
            name='KeyMetric',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            textColor=self.PRIMARY_COLOR
        ))
    
    def add_title_page(self, report_type, time_period, generation_time=None):
        """Add a professional title page"""
        if generation_time is None:
            generation_time = datetime.now()
            
        # Main title
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(self.title, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Report type and period
        self.story.append(Paragraph(f"Report Type: {report_type}", self.styles['CustomSubtitle']))
        self.story.append(Paragraph(f"Analysis Period: {time_period}", self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
        
        # Generation info
        self.story.append(Paragraph(f"Generated: {generation_time.strftime('%B %d, %Y at %I:%M %p')}", 
                                  self.styles['Normal']))
        self.story.append(Paragraph(f"Author: {self.author}", self.styles['Normal']))
        
        # Add separator
        self.story.append(Spacer(1, 1*inch))
        self.story.append(HRFlowable(width="100%", thickness=2, color=self.PRIMARY_COLOR))
        
        # Page break
        self.story.append(PageBreak())
    
    def add_section_header(self, title):
        """Add a section header with styling"""
        self.story.append(Paragraph(title, self.styles['SectionHeader']))
        self.story.append(HRFlowable(width="50%", thickness=1, color=self.SECONDARY_COLOR))
        self.story.append(Spacer(1, 12))
    
    def add_paragraph(self, text, style_name='BodyJustified'):
        """Add a paragraph with specified style"""
        self.story.append(Paragraph(text, self.styles[style_name]))
    
    def add_spacer(self, height=12):
        """Add vertical space"""
        self.story.append(Spacer(1, height))
    
    def add_key_metrics_table(self, metrics_data):
        """Add a table of key metrics"""
        if not metrics_data:
            return
            
        # Prepare table data
        table_data = [['Metric', 'Value', 'Insight']]
        
        for metric in metrics_data:
            table_data.append([
                metric.get('name', ''),
                metric.get('value', ''),
                metric.get('insight', '')
            ])
        
        # Create table
        table = Table(table_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.PRIMARY_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.LIGHT_GRAY])
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 20))
    
    def add_recommendations_section(self, recommendations):
        """Add a formatted recommendations section"""
        if not recommendations:
            return
            
        self.add_section_header("Key Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            priority = rec.get('priority', 'Medium')
            title = rec.get('title', f'Recommendation {i}')
            description = rec.get('description', '')
            
            # Priority indicator
            priority_color = {
                'High': self.ACCENT_COLOR,
                'Medium': colors.orange,
                'Low': self.GRAY_COLOR
            }.get(priority, colors.black)
            
            self.story.append(Paragraph(
                f"<b>{i}. {title}</b> <font color='{priority_color}'>[{priority} Priority]</font>",
                self.styles['Normal']
            ))
            
            self.story.append(Paragraph(description, self.styles['Recommendation']))
            self.story.append(Spacer(1, 8))
    
    def generate_pdf(self, output_path=None):
        """Generate the PDF document"""
        try:
            if output_path:
                doc = SimpleDocTemplate(
                    output_path,
                    pagesize=self.page_size,
                    rightMargin=self.margin,
                    leftMargin=self.margin,
                    topMargin=self.margin,
                    bottomMargin=self.margin,
                    title=self.title,
                    author=self.author
                )
                doc.build(self.story)
                return output_path
            else:
                # Return PDF as bytes
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(
                    buffer,
                    pagesize=self.page_size,
                    rightMargin=self.margin,
                    leftMargin=self.margin,
                    topMargin=self.margin,
                    bottomMargin=self.margin,
                    title=self.title,
                    author=self.author
                )
                doc.build(self.story)
                buffer.seek(0)
                return buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise
    
    def clear_story(self):
        """Clear the story elements to start fresh"""
        self.story = []
