"""
PDF report generation module
"""
import io
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from config import MODEL_ACCURACY_DESCRIPTIONS


def generate_pdf_report(stock_name: str, df: pd.DataFrame, predictions_dict: dict, 
                       signals: list, analysis_duration: float, 
                       data_start_date, data_end_date) -> io.BytesIO:
    """
    Generate professional PDF report with charts and analysis
    
    Args:
        stock_name: Name of the stock
        df: DataFrame with historical data
        predictions_dict: Dictionary of predictions
        signals: List of trading signals
        analysis_duration: Time taken for analysis
        data_start_date: Start date of data
        data_end_date: End date of data
    
    Returns:
        BytesIO buffer with PDF or None if error
    """
    try:
        # Create temporary file
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Title
        title = Paragraph(f"<b>Stock Analysis Report: {stock_name}</b>", title_style)
        story.append(title)
        
        # Report metadata
        today = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata_text = f"""
        <b>Generated:</b> {today}<br/>
        <b>Analysis Period:</b> {data_start_date.strftime('%Y-%m-%d')} to {data_end_date.strftime('%Y-%m-%d')}<br/>
        <b>Data Duration:</b> {(data_end_date - data_start_date).days} days<br/>
        <b>Processing Time:</b> {analysis_duration:.2f} seconds
        """
        story.append(Paragraph(metadata_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Current Metrics Section
        story.append(Paragraph("CURRENT METRICS", heading_style))
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Current Price', f'₹{current_price:.2f}'],
            ['Previous Close', f'₹{prev_close:.2f}'],
            ['Change', f'₹{change:.2f} ({change_pct:.2f}%)'],
            ['Day High', f'₹{df["High"].iloc[-1]:.2f}'],
            ['Day Low', f'₹{df["Low"].iloc[-1]:.2f}'],
            ['Volume', f'{df["Volume"].iloc[-1]:,.0f}'],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Technical Indicators Section
        story.append(Paragraph("TECHNICAL INDICATORS", heading_style))
        tech_data = [
            ['Indicator', 'Value', 'Meaning'],
            ['RSI (14)', f'{df["RSI"].iloc[-1]:.2f}', 'Momentum oscillator (0-100)'],
            ['MACD', f'{df["MACD"].iloc[-1]:.2f}', 'Trend following indicator'],
            ['MACD Signal', f'{df["MACD_Signal"].iloc[-1]:.2f}', 'MACD trigger line'],
            ['SMA 20', f'₹{df["SMA_20"].iloc[-1]:.2f}', '20-day average price'],
            ['SMA 50', f'₹{df["SMA_50"].iloc[-1]:.2f}', '50-day average price'],
            ['SMA 200', f'₹{df["SMA_200"].iloc[-1]:.2f}', '200-day average price'],
            ['ATR', f'{df["ATR"].iloc[-1]:.2f}', 'Average price volatility'],
        ]
        
        tech_table = Table(tech_data, colWidths=[1.8*inch, 1.5*inch, 3*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(tech_table)
        story.append(Spacer(1, 20))
        
        # Trading Signals
        story.append(Paragraph("TRADING SIGNALS", heading_style))
        if signals:
            signals_text = "<br/>".join([f"<b>{signal_type}:</b> {reason}" for signal_type, reason in signals])
        else:
            signals_text = "No strong signals. Hold position."
        story.append(Paragraph(signals_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Predictions Summary
        story.append(Paragraph("PREDICTION SUMMARY", heading_style))
        pred_data = [['Model', 'Predicted Price', 'Change %', 'Accuracy Rating']]
        
        for model_name, pred_df in predictions_dict.items():
            if pred_df is not None:
                if 'yhat' in pred_df.columns:
                    pred_price = pred_df['yhat'].iloc[-1]
                elif 'Predicted_Close' in pred_df.columns:
                    pred_price = pred_df['Predicted_Close'].iloc[-1]
                else:
                    continue
                
                pred_change = ((pred_price - current_price) / current_price) * 100
                
                pred_data.append([
                    model_name,
                    f'₹{pred_price:.2f}',
                    f'{pred_change:+.2f}%',
                    MODEL_ACCURACY_DESCRIPTIONS.get(model_name, '★★★★☆')
                ])
        
        pred_table = Table(pred_data, colWidths=[1.5*inch, 1.5*inch, 1.3*inch, 2*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5576c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 20))
        
        # Key Statistics
        story.append(Paragraph("KEY STATISTICS", heading_style))
        stats_data = [
            ['Statistic', 'Value'],
            ['52-Week High', f'₹{df["High"].tail(252).max():.2f}'],
            ['52-Week Low', f'₹{df["Low"].tail(252).min():.2f}'],
            ['Average Volume', f'{df["Volume"].mean():,.0f}'],
            ['7-Day Return', f'{((df["Close"].iloc[-1] - df["Close"].iloc[-7]) / df["Close"].iloc[-7]) * 100:.2f}%'],
            ['30-Day Return', f'{((df["Close"].iloc[-1] - df["Close"].iloc[-30]) / df["Close"].iloc[-30]) * 100:.2f}%'],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Create historical price chart using matplotlib
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index[-60:], df['Close'].iloc[-60:], label='Close Price', linewidth=2, color='#1f77b4')
        ax.plot(df.index[-60:], df['SMA_20'].iloc[-60:], label='SMA 20', linewidth=1, color='orange', linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₹)')
        ax.set_title(f'{stock_name} - Last 60 Days Price Movement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save historical chart to temporary file
        chart_path = tempfile.mktemp(suffix='.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Add historical chart to PDF
        story.append(PageBreak())
        story.append(Paragraph("HISTORICAL PRICE CHART (Last 60 Days)", heading_style))
        story.append(Image(chart_path, width=6.5*inch, height=3.5*inch))
        story.append(Spacer(1, 20))
        
        # Create prediction chart with historical + future data
        if predictions_dict:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Plot historical data (last 90 days)
            historical_days = min(90, len(df))
            ax.plot(df.index[-historical_days:], df['Close'].iloc[-historical_days:], 
                   label='Historical Price', linewidth=2, color='#1f77b4', marker='o', markersize=2)
            
            # Plot predictions from each model
            prediction_colors = ['#4caf50', '#00bcd4', '#9c27b0', '#673ab7', '#ff5722']
            color_idx = 0
            
            for model_name, pred_df in predictions_dict.items():
                if pred_df is not None and color_idx < len(prediction_colors):
                    if 'yhat' in pred_df.columns:
                        # Prophet predictions
                        future_data = pred_df[pred_df['ds'] > df.index[-1]]
                        if len(future_data) > 0:
                            ax.plot(future_data['ds'], future_data['yhat'], 
                                   label=f'{model_name} Prediction', 
                                   linewidth=2, linestyle='--', color=prediction_colors[color_idx],
                                   marker='x', markersize=3)
                    elif 'Predicted_Close' in pred_df.columns:
                        # ML model predictions
                        ax.plot(pred_df['Date'], pred_df['Predicted_Close'], 
                               label=f'{model_name} Prediction', 
                               linewidth=2, linestyle='--', color=prediction_colors[color_idx],
                               marker='x', markersize=3)
                    color_idx += 1
            
            # Add vertical line to separate historical from predictions
            ax.axvline(x=df.index[-1], color='red', linestyle=':', linewidth=2, 
                      label='Today', alpha=0.7)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (₹)')
            ax.set_title(f'{stock_name} - Historical Data & Future Predictions')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save prediction chart
            pred_chart_path = tempfile.mktemp(suffix='.png')
            plt.savefig(pred_chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Add prediction chart to PDF
            story.append(Paragraph("PREDICTION CHART (Historical + Future Trends)", heading_style))
            story.append(Image(pred_chart_path, width=6.5*inch, height=4*inch))
            story.append(Spacer(1, 20))
        
        # Disclaimer
        disclaimer = Paragraph(
            "<b>DISCLAIMER:</b> This report is for informational purposes only and should not be considered as financial advice. "
            "Please consult with a qualified financial advisor before making any investment decisions. "
            "Past performance does not guarantee future results.",
            styles['Normal']
        )
        story.append(disclaimer)
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        
        return pdf_buffer
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return None
