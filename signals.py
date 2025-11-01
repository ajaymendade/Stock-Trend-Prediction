"""
Trading signals calculation module
"""
import pandas as pd
from typing import List, Tuple


def calculate_signals(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Calculate buy/sell signals based on technical indicators
    
    Args:
        df: DataFrame with technical indicators
    
    Returns:
        List of tuples (signal_type, reason)
    """
    signals = []
    
    # RSI signals
    if df['RSI'].iloc[-1] < 30:
        signals.append(("BUY", "RSI is oversold (< 30) - Potential buying opportunity"))
    elif df['RSI'].iloc[-1] > 70:
        signals.append(("SELL", "RSI is overbought (> 70) - Consider taking profits"))
    
    # Moving average crossover
    if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] and df['SMA_20'].iloc[-2] <= df['SMA_50'].iloc[-2]:
        signals.append(("BUY", "Golden Cross: SMA 20 crossed above SMA 50 - Bullish signal"))
    elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] and df['SMA_20'].iloc[-2] >= df['SMA_50'].iloc[-2]:
        signals.append(("SELL", "Death Cross: SMA 20 crossed below SMA 50 - Bearish signal"))
    
    # MACD signals
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
        signals.append(("BUY", "MACD crossed above signal line - Bullish momentum"))
    elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
        signals.append(("SELL", "MACD crossed below signal line - Bearish momentum"))
    
    # Price vs Bollinger Bands
    if df['Close'].iloc[-1] < df['BB_Low'].iloc[-1]:
        signals.append(("BUY", "Price below lower Bollinger Band - Potentially oversold"))
    elif df['Close'].iloc[-1] > df['BB_High'].iloc[-1]:
        signals.append(("SELL", "Price above upper Bollinger Band - Potentially overbought"))
    
    return signals
