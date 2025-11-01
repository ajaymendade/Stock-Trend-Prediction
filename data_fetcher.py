"""
Improved data fetching and technical indicator calculation module
With enhanced error handling, validation, and logging
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings
import time
import re
from typing import Optional, Tuple
from config import (MAX_MISSING_DATA_PCT, MIN_DATA_POINTS, MAX_OUTLIER_STD,
                   API_RETRY_ATTEMPTS, API_RETRY_DELAY)

warnings.filterwarnings('ignore')


def sanitize_stock_symbol(symbol: str) -> Optional[str]:
    """
    Sanitize and validate stock symbol input
    
    Args:
        symbol: Raw stock symbol input
    
    Returns:
        Sanitized symbol or None if invalid
    """
    if not symbol:
        return None
    
    # Remove whitespace and convert to uppercase
    symbol = symbol.strip().upper()
    
    # Allow only alphanumeric, dots, hyphens, and carets
    if not re.match(r'^[A-Z0-9.\-^]+$', symbol):
        st.error(f"Invalid symbol format: {symbol}. Only letters, numbers, dots, hyphens, and carets allowed.")
        return None
    
    # Ensure it has .NS or .BO suffix for Indian stocks (if not an index)
    if not symbol.startswith('^') and not symbol.endswith(('.NS', '.BO')):
        st.warning(f"Adding .NS suffix to {symbol} for NSE listing")
        symbol = f"{symbol}.NS"
    
    return symbol


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_stock_data(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance with retry logic and validation
    
    Args:
        symbol: Stock symbol (e.g., 'TCS.NS')
        period: Time period ('1y', '2y', '5y', 'max')
    
    Returns:
        DataFrame with stock data or None if error
    """
    # Sanitize symbol
    symbol = sanitize_stock_symbol(symbol)
    if not symbol:
        return None
    
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            st.info(f"Fetching data for {symbol}... (Attempt {attempt + 1}/{API_RETRY_ATTEMPTS})")
            
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                if attempt < API_RETRY_ATTEMPTS - 1:
                    st.warning(f"No data received. Retrying in {API_RETRY_DELAY} seconds...")
                    time.sleep(API_RETRY_DELAY)
                    continue
                else:
                    st.error(f"No data available for {symbol}. Please check the symbol.")
                    return None
            
            # Remove timezone to prevent issues
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Validate data quality
            if not validate_stock_data(df, symbol):
                return None
            
            # Clean data
            df = clean_stock_data(df)
            
            st.success(f"✅ Successfully fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            if attempt < API_RETRY_ATTEMPTS - 1:
                st.warning(f"Error: {str(e)}. Retrying in {API_RETRY_DELAY} seconds...")
                time.sleep(API_RETRY_DELAY)
            else:
                st.error(f"Failed to fetch data for {symbol} after {API_RETRY_ATTEMPTS} attempts: {str(e)}")
                return None
    
    return None


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean stock data by handling missing values and outliers
    
    Args:
        df: DataFrame with stock data
    
    Returns:
        Cleaned DataFrame
    """
    # Forward fill missing values (use previous day's data)
    df = df.fillna(method='ffill')
    
    # Backward fill any remaining missing values at the start
    df = df.fillna(method='bfill')
    
    # Remove extreme outliers (beyond 5 standard deviations)
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(
                lower=mean - MAX_OUTLIER_STD * std,
                upper=mean + MAX_OUTLIER_STD * std
            )
    
    # Ensure High >= Low
    if 'High' in df.columns and 'Low' in df.columns:
        df['High'] = df[['High', 'Low']].max(axis=1)
        df['Low'] = df[['High', 'Low']].min(axis=1)
    
    # Ensure Close is within High-Low range
    if all(col in df.columns for col in ['Close', 'High', 'Low']):
        df['Close'] = df['Close'].clip(lower=df['Low'], upper=df['High'])
    
    return df


def validate_stock_data(df: pd.DataFrame, symbol: str = "") -> bool:
    """
    Comprehensive validation of stock data quality
    
    Args:
        df: DataFrame with stock data
        symbol: Stock symbol for error messages
    
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        st.error(f"❌ No data available for {symbol}")
        return False
    
    # Check for minimum data points
    if len(df) < MIN_DATA_POINTS:
        st.error(f"❌ Insufficient data: Only {len(df)} points available. Need at least {MIN_DATA_POINTS}.")
        return False
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return False
    
    # Check for excessive missing values
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    
    if missing_pct > MAX_MISSING_DATA_PCT:
        st.error(f"❌ Data quality issue: {missing_pct:.1f}% missing values (max allowed: {MAX_MISSING_DATA_PCT}%)")
        return False
    elif missing_pct > 0:
        st.warning(f"⚠️ Data contains {missing_pct:.1f}% missing values. Will be cleaned automatically.")
    
    # Check for data integrity (High >= Low)
    if 'High' in df.columns and 'Low' in df.columns:
        invalid_rows = (df['High'] < df['Low']).sum()
        if invalid_rows > 0:
            st.warning(f"⚠️ Found {invalid_rows} rows where High < Low. Will be corrected.")
    
    # Check for negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns and (df[col] <= 0).any():
            st.warning(f"⚠️ Found negative or zero values in {col}. Data may be unreliable.")
    
    # Check for data gaps (missing dates)
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    missing_dates = len(date_range) - len(df)
    if missing_dates > len(df) * 0.5:  # More than 50% dates missing
        st.warning(f"⚠️ Large data gaps detected: {missing_dates} missing dates")
    
    return True


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe with error handling
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added technical indicators
    """
    try:
        df = df.copy()  # Avoid modifying original
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()

        # Volume indicators
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)

        # ATR (Average True Range)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Fill any NaN values created by indicators
        df = df.fillna(method='bfill')

        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return df


def get_data_quality_score(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Calculate a data quality score (0-100)
    
    Args:
        df: DataFrame with stock data
    
    Returns:
        Tuple of (score, description)
    """
    score = 100.0
    issues = []
    
    # Penalize for missing data
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    score -= missing_pct * 2
    if missing_pct > 0:
        issues.append(f"{missing_pct:.1f}% missing")
    
    # Penalize for data gaps
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    gap_pct = (len(date_range) - len(df)) / len(date_range) * 100
    score -= gap_pct * 0.5
    if gap_pct > 20:
        issues.append(f"{gap_pct:.0f}% date gaps")
    
    # Penalize for short history
    if len(df) < 500:
        score -= (500 - len(df)) * 0.1
        issues.append("limited history")
    
    score = max(0, min(100, score))
    
    if score >= 90:
        quality = "Excellent"
    elif score >= 75:
        quality = "Good"
    elif score >= 60:
        quality = "Fair"
    else:
        quality = "Poor"
    
    description = f"{quality} ({score:.0f}/100)"
    if issues:
        description += f" - Issues: {', '.join(issues)}"
    
    return score, description
