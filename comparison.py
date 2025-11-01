"""
Stock comparison module
"""
import pandas as pd
import numpy as np
from data_fetcher import fetch_stock_data
from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE


def compare_stocks(stocks_dict: dict, period: str = "1y") -> pd.DataFrame:
    """
    Compare multiple stocks performance
    
    Args:
        stocks_dict: Dictionary of {name: symbol}
        period: Time period for comparison
    
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = {}
    
    for name, symbol in stocks_dict.items():
        df = fetch_stock_data(symbol, period)
        if df is not None and not df.empty:
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            total_return = ((end_price - start_price) / start_price) * 100
            
            volatility = df['Close'].pct_change().std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
            
            returns = df['Close'].pct_change()
            excess_returns = returns.mean() * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE
            sharpe = excess_returns / (returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if returns.std() > 0 else 0
            
            comparison_data[name] = {
                'Symbol': symbol,
                'Start Price': start_price,
                'Current Price': end_price,
                'Total Return (%)': total_return,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe,
                'Max Price': df['High'].max(),
                'Min Price': df['Low'].min(),
                'Avg Volume': df['Volume'].mean()
            }
    
    return pd.DataFrame(comparison_data).T
