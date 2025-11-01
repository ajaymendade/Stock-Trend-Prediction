"""
Configuration file for Stock Analysis Application
Contains constants, stock symbols, and term explanations
"""

# Popular Indian stocks with NSE/BSE symbols
INDIAN_STOCKS = {
    # IT Sector
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Wipro": "WIPRO.NS",
    "HCL Tech": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    # Banking & Finance
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra": "KOTAKBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    # Automobile
    "Tata Motors": "TATAMOTORS.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    # Pharmaceuticals
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Labs": "DIVISLAB.NS",
    # Consumer Goods
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia": "BRITANNIA.NS",
    # Telecom
    "Bharti Airtel": "BHARTIARTL.NS",
    "Reliance Industries": "RELIANCE.NS",
    # Metals & Mining
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Hindalco": "HINDALCO.NS",
    # Energy
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    # Conglomerates
    "Larsen & Toubro": "LT.NS",
    "Adani Enterprises": "ADANIENT.NS",
    # Cement
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Ambuja Cements": "AMBUJACEM.NS",
    # Indices
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
}

# Technical term explanations dictionary
TERM_EXPLANATIONS = {
    "RSI": "Relative Strength Index - Momentum indicator measuring overbought (>70) or oversold (<30) conditions",
    "MACD": "Moving Average Convergence Divergence - Shows relationship between two moving averages to identify trend changes",
    "Volume": "Number of shares traded - High volume confirms price movements and trends",
    "SMA": "Simple Moving Average - Average price over a specific period, smooths out price fluctuations",
    "EMA": "Exponential Moving Average - Gives more weight to recent prices, responds faster to price changes",
    "Bollinger Bands": "Volatility bands around price - Helps identify overbought/oversold levels and potential breakouts",
    "ATR": "Average True Range - Measures market volatility and potential price movement range",
    "Sharpe Ratio": "Risk-adjusted return metric - Higher values indicate better risk-adjusted performance",
    "Volatility": "Price fluctuation measure - Higher volatility means greater price swings and risk",
    "Golden Cross": "Bullish signal when short-term MA crosses above long-term MA",
    "Death Cross": "Bearish signal when short-term MA crosses below long-term MA",
    "Support": "Price level where buying pressure prevents further decline",
    "Resistance": "Price level where selling pressure prevents further rise",
}

# Model configuration
MODEL_CONFIGS = {
    'prophet': {
        'daily_seasonality': False,
        'weekly_seasonality': True,
        'yearly_seasonality': True,
        'changepoint_prior_scale': 0.05
    },
    'arima': {
        'order': (5, 1, 0)
    },
    'xgboost': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 7,
        'random_state': 42,
        'n_jobs': -1
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'random_state': 42
    }
}

# Model characteristics (NOT accuracy ratings - actual accuracy varies by stock)
MODEL_ACCURACY_RATINGS = {
    'Prophet': '📊 Time Series',
    'ARIMA': '📈 Statistical',
    'XGBoost': '🤖 Advanced ML',
    'Random Forest': '🌲 Ensemble ML',
    'Ensemble': '⚡ Combined'
}

MODEL_ACCURACY_DESCRIPTIONS = {
    'Prophet': 'Time Series (Good for trends)',
    'ARIMA': 'Statistical (Simple & stable)',
    'XGBoost': 'Advanced ML (Complex patterns)',
    'Random Forest': 'Ensemble ML (General purpose)',
    'Ensemble': 'Combined (Multiple models)'
}

# Stock sectors for display
STOCK_SECTORS = {
    "💻 IT": ["TCS", "Infosys", "Wipro", "HCL Tech", "Tech Mahindra"],
    "🏦 Banking": ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra"],
    "🚗 Auto": ["Tata Motors", "Maruti Suzuki", "M&M", "Bajaj Auto", "Hero MotoCorp"],
    "💊 Pharma": ["Sun Pharma", "Dr Reddy's", "Cipla", "Divi's Labs"],
    "🛒 Consumer": ["Hindustan Unilever", "ITC", "Nestle India", "Britannia"],
    "📡 Others": ["Reliance", "L&T", "Bharti Airtel", "UltraTech", "NTPC"]
}

# Trading days per year
TRADING_DAYS_PER_YEAR = 252

# Risk-free rate for Sharpe ratio calculation (6% annual)
RISK_FREE_RATE = 0.06

# Prediction period options (in days)
PREDICTION_PERIODS = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "3 Years": 1095,
    "4 Years": 1460,
    "5 Years": 1825
}

# Data validation thresholds
MAX_MISSING_DATA_PCT = 10  # Maximum 10% missing data allowed
MIN_DATA_POINTS = 100  # Minimum data points required
MAX_OUTLIER_STD = 5  # Maximum standard deviations for outlier detection

# API rate limiting
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2  # seconds
