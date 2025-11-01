# 📈 Indian Stock Market Analyzer & Predictor Pro

[![Version](https://img.shields.io/badge/version-2.0-blue.svg)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.50.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![Educational](https://img.shields.io/badge/purpose-educational-yellow.svg)]()

A **modular**, AI-powered stock analysis application built with Streamlit that provides advanced technical analysis and price predictions for Indian stocks listed on NSE/BSE.

> ⚠️ **Educational Tool Only** - Not for real trading decisions. See [Disclaimer](#️-disclaimer) below.

---

## ✨ Features

### 📊 Data & Analysis
- **40+ Indian Stocks** across all major sectors (IT, Banking, Auto, Pharma, Consumer Goods)
- **Real-time Data** from NSE/BSE via Yahoo Finance API (15-minute delay on free tier)
- **Advanced Technical Analysis**:
  - RSI (Relative Strength Index) - 14-period momentum oscillator
  - MACD (Moving Average Convergence Divergence) - 12/26 EMA with 9-period signal
  - Bollinger Bands - 20-period SMA with 2 standard deviations
  - Moving Averages (SMA 20, 50, 200)
  - ATR (Average True Range) - 14-period volatility measure
  - Volume Analysis
- **Interactive Charts** powered by Plotly with candlestick patterns and technical indicators
- **Stock Comparison** - Compare 2-6 stocks side-by-side with comparative metrics
- **Term Explanations** - Built-in tooltips for every technical term

### 🤖 AI Prediction Models

| Model | Type | Speed | Best For |
|-------|------|-------|----------|
| **Prophet** | Time Series | ⚡⚡ | Long-term trends, seasonal patterns |
| **ARIMA** | Statistical | ⚡⚡⚡ | Stable stocks, short-term |
| **XGBoost** | ML (200 estimators) | ⚡ | Complex patterns, high volatility |
| **Random Forest** | ML (100 trees) | ⚡⚡ | General purpose, balanced accuracy |
| **Ensemble** | Combined | ⚡ | Maximum accuracy, all scenarios |

- **Customizable Prediction Period**: 30 to 180 days
- **Model Recommendations**:
  - Long-term (>90 days): Prophet or Ensemble
  - Short-term (<30 days): XGBoost or Random Forest
  - Maximum accuracy: Ensemble
  - Stable stocks: ARIMA or Prophet

### 📈 Trading Signals
- **Automated Buy/Sell Recommendations** based on:
  - RSI overbought/oversold conditions
  - Moving average crossovers (Golden Cross/Death Cross)
  - MACD crossovers
  - Bollinger Band breakouts
- **Real-time Analysis** with detailed explanations for each signal

### 📄 Professional Reports
- **PDF Reports** with embedded charts, metrics, and predictions (ReportLab)
- **Text Reports** for quick reference
- **Download & Share** analysis results

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.8 or higher**
- Stable internet connection
- Modern web browser

### Quick Start

```bash
# 1. Clone or download the repository
git clone https://github.com/ajaymendade/Stock-Trend-Prediction.git

# 2. Navigate to directory
cd "Stock trend prediction"

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The application will automatically open at `http://localhost:8501`

### Troubleshooting
- **Permission errors**: Try `pip install --user -r requirements.txt`
- **Prophet issues on Windows**: Install C++ build tools
- **Module not found**: Ensure all packages installed successfully

---

## 🎯 Usage Guide

### Single Stock Analysis

1. **Select Stock**: Choose from 40+ pre-configured stocks or enter custom NSE symbol
2. **Configure Analysis**:
   - Historical period: 1y, 2y, 5y, or max
   - Prediction days: 30-180 days
   - AI models: Select one or multiple
3. **Analyze**: Click "Analyze Stock" to generate:
   - Interactive price charts with predictions
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Trading signals and recommendations
   - Downloadable PDF/text reports

### Stock Comparison Mode

1. Switch to "Compare Stocks" tab
2. Select 2-6 stocks from the same or different sectors
3. View comparative analysis:
   - Total returns and performance charts
   - Volatility comparison
   - Sharpe ratio (risk-adjusted returns)
   - Side-by-side metrics

---

## 📊 Available Stocks

### 💻 IT Sector
TCS, Infosys, Wipro, HCL Tech, Tech Mahindra

### 🏦 Banking & Finance
HDFC Bank, ICICI Bank, SBI, Axis Bank, Kotak Mahindra, Bajaj Finance

### 🚗 Automobile
Tata Motors, Maruti Suzuki, M&M, Bajaj Auto, Hero MotoCorp

### 💊 Pharmaceuticals
Sun Pharma, Dr Reddy's, Cipla, Divi's Labs

### 🛒 Consumer Goods
Hindustan Unilever, ITC, Nestle India, Britannia

### 📡 Other Sectors
Reliance Industries, Bharti Airtel, L&T, UltraTech Cement, NTPC, and more

---

## 🔧 Technical Architecture

### Data Processing
- **Source**: Yahoo Finance API via yfinance library
- **Caching**: 1-hour TTL for performance optimization
- **Feature Engineering**:
  - Price returns and percentage changes
  - High-Low ranges
  - Lag features (1-5 days)
  - Volume moving averages
  - MinMax scaling for ML models

### Visualization
- **Library**: Plotly (interactive charts)
- **Chart Types**: Candlestick, line, bar, scatter
- **Features**: Unified hover, zoom, pan, export
- **Layout**: 4-panel view (Price, MACD, RSI, Volume)

### Performance Metrics
- **Analysis Time**: 10-20 seconds (model-dependent)
- **Data Points**: Up to 5+ years of historical data
- **Chart Rendering**: Instant with Plotly optimization
- **Cache Duration**: 1 hour per stock

---

## 💡 Best Practices

1. **Use 2-year historical data** for balanced context between recent trends and long-term patterns
2. **Enable multiple models** to cross-validate predictions and identify consensus
3. **Always check trading signals** before making any decisions
4. **Review technical indicators** together, not in isolation
5. **Monitor volatility metrics** (ATR, Bollinger width) for risk assessment
6. **Compare similar stocks** within the same sector for relative analysis
7. **Download PDF reports** for offline review and record-keeping

---

## ⚠️ Production Readiness Assessment

### Current Status: **NOT Production-Ready**

This is an **educational/demonstration tool** with these limitations:

**Critical Gaps:**
- ❌ No model validation or backtesting framework
- ❌ Iterative forecasting compounds prediction errors
- ❌ Missing API rate limit handling
- ❌ No data validation or quality checks
- ❌ Absent logging and monitoring systems
- ❌ Accuracy claims are not independently validated

**Appropriate Use Cases:**
- ✅ Learning financial analysis concepts
- ✅ Experimenting with ML models
- ✅ Educational demonstrations
- ✅ Personal portfolio tracking and research

**Not Suitable For:**
- ❌ Real trading decisions
- ❌ Providing financial advice
- ❌ Commercial use without thorough validation
- ❌ Production trading systems

---

## ⚠️ Disclaimer

**CRITICAL NOTICE**: This application is strictly for educational and informational purposes.

- 📉 **Not Financial Advice**: Predictions are based on historical data and mathematical models
- ⚠️ **Past Performance**: Does not guarantee future results
- 💰 **Market Risk**: Stock investments carry inherent risk of loss
- 👨‍💼 **Professional Consultation**: Always consult qualified financial advisors before investing
- 📊 **Accuracy**: Predictions are NOT validated and should not be relied upon for real trading
- ⚖️ **Liability**: Authors are NOT responsible for any financial losses

**By using this software, you acknowledge it is NOT to be used for real trading without professional guidance.**

---

## 🤝 Contributing

We welcome contributions to improve this educational tool!

### How to Contribute
- 🐛 Report bugs via GitHub Issues
- 💡 Suggest new features or improvements
- 📝 Enhance documentation
- 🔧 Submit pull requests with code improvements
- ⭐ Star the repository to show support

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📚 Additional Resources

- **[QUICK_START.md](QUICK_START.md)** - Fast setup guide
- **[FEATURES_DOCUMENTATION.md](FEATURES_DOCUMENTATION.md)** - Detailed feature explanations
- **In-app tooltips** - Hover over any term for instant definitions

---

## 📧 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review in-app help tooltips

---

## 🙏 Acknowledgments

Built with excellent open-source tools:
- **Streamlit** - Web framework
- **Facebook Prophet** - Time series forecasting
- **XGBoost** - Gradient boosting
- **Scikit-learn** - Machine learning
- **Plotly** - Interactive visualizations
- **ReportLab** - PDF generation
- **yFinance** - Stock data API
- **Yahoo Finance** - Data provider

---

## 📄 License

MIT License - Free for educational and personal use

---

## 📊 Project Stats

![AI Models](https://img.shields.io/badge/AI%20Models-5-blue)
![Stocks](https://img.shields.io/badge/Stocks-40+-green)
![Indicators](https://img.shields.io/badge/Indicators-10+-orange)
![Charts](https://img.shields.io/badge/Charts-Interactive-purple)
![Reports](https://img.shields.io/badge/Reports-PDF%20%2B%20Text-red)

**Version:** 2.0 Pro  
**Last Updated:** November 2025  
**Status:** Educational Use Only

---

<div align="center">

### **Happy Learning! 📈**

**Explore AI-powered stock analysis for educational purposes**

[Get Started](#-installation--setup) • [Features](#-features) • [Usage Guide](#-usage-guide)

</div>

---

**Remember:** This tool is for learning only. Always conduct thorough research and consult financial professionals before making investment decisions.