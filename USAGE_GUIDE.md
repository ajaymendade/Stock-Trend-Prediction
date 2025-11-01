# 📖 Usage Guide - Improved Stock Analysis App

## 🚀 Quick Start

### 1. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🎯 Features Overview

### **Extended Prediction Periods**
You can now predict stock prices for:
- 1 Month (30 days)
- 3 Months (90 days) 
- 6 Months (180 days)
- 1 Year (365 days)
- 2 Years (730 days)
- 3 Years (1,095 days)
- 4 Years (1,460 days)
- 5 Years (1,825 days)

### **Data Quality Indicators**
- **Quality Score**: 0-100 rating (Excellent/Good/Fair/Poor)
- **Data Points**: Number of historical records
- **Data Duration**: Time span of historical data
- **Missing Data**: Percentage of missing values

### **Prediction Validation**
Each model now shows:
- **MAE** (Mean Absolute Error): Average prediction error in ₹
- **RMSE** (Root Mean Squared Error): Prediction accuracy metric
- **MAPE** (Mean Absolute Percentage Error): Error as percentage
- **Prediction Confidence**: How much models agree (Very High/High/Medium/Low)

---

## 📋 Step-by-Step Guide

### Step 1: Select a Stock
1. Open the sidebar (left panel)
2. Choose from 40+ pre-configured Indian stocks
3. OR enter a custom NSE symbol (e.g., `RELIANCE.NS`)

### Step 2: Configure Analysis
1. **Historical Data Period**: Choose 1y, 2y, 5y, or max
2. **Prediction Period**: Select from 1 month to 5 years
3. **AI Models**: Select which models to use
   - Prophet (Time Series)
   - ARIMA (Statistical)
   - XGBoost (Advanced ML)
   - Random Forest (ML)
   - Ensemble (Combined)

### Step 3: Run Analysis
1. Click **"🚀 Analyze Stock"** button
2. Wait for data fetching and validation
3. Models will train and show validation metrics

### Step 4: Review Results

#### **Data Quality Section**
- Check the quality score (aim for 75+)
- Review any warnings about missing data
- Verify sufficient data points

#### **Current Metrics**
- Current price and daily change
- Day high/low
- Trading volume
- RSI indicator

#### **Charts**
- **Advanced Chart**: Candlestick with predictions and indicators
- **Simple Chart**: Clean time-value visualization

#### **Trading Signals**
- Buy/Sell recommendations
- Based on RSI, MACD, Moving Averages, Bollinger Bands
- Explanations provided

#### **Prediction Summary**
- Overall trend (Upward/Downward)
- Average predicted change percentage
- **Prediction Confidence** level
- Individual model predictions

#### **Technical Analysis**
- 52-week high/low
- Average volume
- RSI, MACD, ATR values
- 7-day, 30-day, period returns

### Step 5: Download Reports
- **PDF Report**: Professional report with charts
- **Text Report**: Quick summary

---

## 🔍 Understanding the Metrics

### Data Quality Score
- **90-100**: Excellent - Reliable data, minimal issues
- **75-89**: Good - Minor issues, generally reliable
- **60-74**: Fair - Some concerns, use with caution
- **<60**: Poor - Significant issues, results may be unreliable

### Prediction Confidence
- **Very High**: Models agree closely (CV < 2%)
- **High**: Good agreement (CV < 5%)
- **Medium**: Some disagreement (CV < 10%)
- **Low**: Models disagree significantly (CV > 10%)

### Validation Metrics
- **MAE**: Lower is better (measures average error in ₹)
- **RMSE**: Lower is better (penalizes large errors more)
- **MAPE**: Lower is better (error as percentage)

---

## ⚠️ Important Notes

### **Disclaimers**
1. This is an **educational tool** only
2. Predictions are **NOT guaranteed** to be accurate
3. **DO NOT** use for real trading without professional advice
4. Past performance ≠ future results
5. Always consult a financial advisor

### **Data Limitations**
- Data comes from Yahoo Finance (free, no API key needed)
- Some stocks may have limited historical data
- Market holidays cause data gaps
- Real-time data may have 15-minute delay

### **Prediction Limitations**
- Longer predictions (3-5 years) are less reliable
- Models cannot predict black swan events
- Market sentiment not captured
- News and events not factored in

---

## 🛠️ Troubleshooting

### "No data available for symbol"
- Check if symbol is correct (e.g., `TCS.NS` not just `TCS`)
- Verify stock is listed on NSE/BSE
- Try a different symbol

### "Insufficient data points"
- Stock may be newly listed
- Try a longer historical period
- Choose a different stock

### "Data quality issue: X% missing values"
- Data has gaps (holidays, trading halts)
- App will attempt to clean automatically
- If quality score < 60, consider different stock

### Models taking too long
- First run takes longer (no cache)
- Subsequent runs are faster (cached)
- Longer prediction periods take more time
- Consider selecting fewer models

### Predictions seem unrealistic
- Check prediction confidence level
- Review validation metrics (MAE, RMSE)
- Compare multiple models
- Consider shorter prediction period

---

## 💡 Tips for Best Results

1. **Start with shorter periods**
   - Try 3-6 months first
   - Validate against known data
   - Then try longer periods

2. **Use multiple models**
   - Compare predictions
   - Look for consensus
   - Check confidence level

3. **Check data quality**
   - Aim for 75+ quality score
   - Review warnings
   - Ensure sufficient history

4. **Understand the indicators**
   - Read term explanations
   - Learn RSI, MACD, etc.
   - Understand what signals mean

5. **Compare stocks**
   - Use comparison mode
   - Check relative performance
   - Identify trends

---

## 📊 Example Workflow

### Analyzing TCS for 1 Year Prediction

1. **Select**: TCS from dropdown
2. **Configure**: 
   - Historical: 2y
   - Prediction: 1 Year
   - Models: Prophet, XGBoost, Ensemble
3. **Analyze**: Click button
4. **Review**:
   - Data quality: 92/100 (Excellent) ✅
   - Current price: ₹3,450
   - Validation MAE: ₹45 (good)
   - Prediction confidence: High ✅
5. **Results**:
   - Prophet: ₹3,650 (+5.8%)
   - XGBoost: ₹3,720 (+7.8%)
   - Ensemble: ₹3,685 (+6.8%)
   - Average: ₹3,685 (+6.8%)
   - Confidence: High (models agree)
6. **Decision**: Positive outlook, but verify with other analysis

---

## 🔄 Comparison Mode

### Comparing Multiple Stocks

1. Switch to **"Compare Stocks"** mode
2. Select 2-6 stocks (e.g., TCS, Infosys, Wipro, HCL Tech)
3. Click **"Analyze Stock"**
4. Review:
   - **Performance table**: Returns, volatility, Sharpe ratio
   - **Best return**: Highest percentage gain
   - **Best risk-adjusted**: Highest Sharpe ratio
   - **Lowest volatility**: Most stable
   - **Comparison chart**: Normalized performance

---

## 📈 Advanced Features

### Custom Stock Symbols
- Enter any NSE symbol with `.NS` suffix
- Example: `RELIANCE.NS`, `TATAMOTORS.NS`
- App will validate and sanitize input

### PDF Reports
- Comprehensive analysis document
- Includes charts and tables
- Metrics and predictions
- Downloadable for records

### Technical Indicators
- RSI: Momentum (overbought/oversold)
- MACD: Trend changes
- Bollinger Bands: Volatility
- Moving Averages: Trend direction
- ATR: Volatility measure

---

## 🎓 Learning Resources

### Understanding Predictions
- Models use historical patterns
- No model is perfect
- Ensemble combines multiple approaches
- Validation shows past performance

### Technical Analysis
- RSI > 70: Overbought (consider selling)
- RSI < 30: Oversold (consider buying)
- Golden Cross: Bullish signal
- Death Cross: Bearish signal

### Risk Management
- Never invest more than you can afford to lose
- Diversify across stocks
- Use stop-loss orders
- Consult professionals

---

## 📞 Support

### Issues or Questions?
- Check this guide first
- Review error messages carefully
- Try with different stocks/settings
- Refer to IMPROVEMENTS_SUMMARY.md

---

**Remember: This is an educational tool. Always do your own research and consult financial advisors before investing!**
