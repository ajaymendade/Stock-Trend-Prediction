"""
Chart creation module
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_fetcher import fetch_stock_data


def create_main_chart(df: pd.DataFrame, predictions_prophet=None, predictions_ml=None, 
                     predictions_ensemble=None, predictions_xgboost=None, 
                     predictions_arima=None, stock_name: str = "") -> go.Figure:
    """
    Create interactive chart with historical and predicted data
    
    Args:
        df: DataFrame with historical data
        predictions_*: Prediction DataFrames from different models
        stock_name: Name of the stock
    
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{stock_name} - Price & Predictions with Trendlines', 'MACD', 'RSI', 'Volume'),
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                            line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                            line=dict(color='#2196f3', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200',
                            line=dict(color='#f44336', width=1.5)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB High',
                            line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Low',
                            line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'),
                            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # Prophet predictions
    if predictions_prophet is not None:
        fig.add_trace(
            go.Scatter(
                x=predictions_prophet['ds'],
                y=predictions_prophet['yhat'],
                name='Prophet Forecast',
                line=dict(color='#4caf50', width=3, dash='dot'),
                mode='lines'
            ),
            row=1, col=1
        )
        # Confidence interval
        fig.add_trace(
            go.Scatter(
                x=predictions_prophet['ds'],
                y=predictions_prophet['yhat_upper'],
                fill=None,
                mode='lines',
                line=dict(color='rgba(76,175,80,0)', width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=predictions_prophet['ds'],
                y=predictions_prophet['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(76,175,80,0)', width=0),
                fillcolor='rgba(76,175,80,0.2)',
                name='Prophet Range'
            ),
            row=1, col=1
        )
    
    # ARIMA predictions
    if predictions_arima is not None:
        fig.add_trace(
            go.Scatter(
                x=predictions_arima['Date'],
                y=predictions_arima['Predicted_Close'],
                name='ARIMA Forecast',
                line=dict(color='#00bcd4', width=2.5, dash='dashdot')
            ),
            row=1, col=1
        )
    
    # XGBoost predictions
    if predictions_xgboost is not None:
        fig.add_trace(
            go.Scatter(
                x=predictions_xgboost['Date'],
                y=predictions_xgboost['Predicted_Close'],
                name='XGBoost Forecast',
                line=dict(color='#9c27b0', width=3, dash='solid')
            ),
            row=1, col=1
        )
    
    # ML predictions
    if predictions_ml is not None:
        fig.add_trace(
            go.Scatter(
                x=predictions_ml['Date'],
                y=predictions_ml['Predicted_Close'],
                name='Random Forest',
                line=dict(color='#673ab7', width=2, dash='dashdot')
            ),
            row=1, col=1
        )
    
    # Ensemble predictions
    if predictions_ensemble is not None:
        fig.add_trace(
            go.Scatter(
                x=predictions_ensemble['Date'],
                y=predictions_ensemble['Predicted_Close'],
                name='Ensemble (Best)',
                line=dict(color='#ff5722', width=3.5, dash='solid')
            ),
            row=1, col=1
        )
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                            line=dict(color='#2196f3', width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                            line=dict(color='#f44336', width=1.5)), row=2, col=1)
    colors_macd = ['#26a69a' if val >= 0 else '#ef5350' for val in df['MACD_Diff']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Diff'], name='MACD Histogram',
                        marker_color=colors_macd), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                            line=dict(color='#9c27b0', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f44336", row=3, col=1,
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#4caf50", row=3, col=1,
                  annotation_text="Oversold (30)")
    
    # Volume
    colors_vol = ['#ef5350' if row['Close'] < row['Open'] else '#26a69a' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors_vol), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_SMA'], name='Vol SMA',
                            line=dict(color='#ff9800', width=2)), row=4, col=1)
    
    # Update layout
    fig.update_layout(
        height=1400,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1, gridcolor='#e0e0e0')
    fig.update_yaxes(title_text="MACD", row=2, col=1, gridcolor='#e0e0e0')
    fig.update_yaxes(title_text="RSI", row=3, col=1, gridcolor='#e0e0e0')
    fig.update_yaxes(title_text="Volume", row=4, col=1, gridcolor='#e0e0e0')
    fig.update_xaxes(gridcolor='#e0e0e0')
    
    return fig


def create_simple_price_chart(df: pd.DataFrame, stock_name: str = "") -> go.Figure:
    """
    Create a simple time-value chart
    
    Args:
        df: DataFrame with historical data
        stock_name: Name of the stock
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add close price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Add high and low
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['High'],
        mode='lines',
        name='High',
        line=dict(color='#2ca02c', width=1, dash='dash'),
        opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Low'],
        mode='lines',
        name='Low',
        line=dict(color='#d62728', width=1, dash='dash'),
        opacity=0.5
    ))
    
    fig.update_layout(
        title=f'{stock_name} - Simple Price Chart (Time vs Value)',
        xaxis_title='Time',
        yaxis_title='Price (₹)',
        height=500,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def create_comparison_chart(stocks_dict: dict, period: str = "1y") -> go.Figure:
    """
    Create normalized comparison chart for multiple stocks
    
    Args:
        stocks_dict: Dictionary of {name: symbol}
        period: Time period for comparison
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    for name, symbol in stocks_dict.items():
        df = fetch_stock_data(symbol, period)
        if df is not None and not df.empty:
            normalized = (df['Close'] / df['Close'].iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized,
                name=name,
                mode='lines',
                line=dict(width=2.5)
            ))
    
    fig.update_layout(
        title='Stock Performance Comparison (Normalized to 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig
