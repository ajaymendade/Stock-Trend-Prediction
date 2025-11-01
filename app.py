"""
Main Streamlit Application
Modular Stock Analysis and Prediction App
"""
import streamlit as st
import time
from datetime import datetime

# Import custom modules
from config import INDIAN_STOCKS, STOCK_SECTORS, MODEL_ACCURACY_RATINGS, PREDICTION_PERIODS
from data_fetcher import fetch_stock_data, add_technical_indicators, validate_stock_data, get_data_quality_score
from predictors import (prophet_prediction, arima_prediction, xgboost_prediction, 
                       ml_prediction, ensemble_prediction, calculate_trendline, get_prediction_confidence)
from signals import calculate_signals
from comparison import compare_stocks
from charts import create_main_chart, create_simple_price_chart, create_comparison_chart
from pdf_report import generate_pdf_report
from ui_components import apply_custom_css, show_term_explanation, display_header

# Page configuration
st.set_page_config(
    page_title="Indian Stock Market Analyzer Pro", 
    layout="wide", 
    page_icon="📈"
)

# Apply custom styling
apply_custom_css()


def display_welcome_screen():
    """Display welcome screen with features"""
    st.info("👈 Select a stock from the sidebar and click 'Analyze Stock' to get started!")
    
    # Important disclaimer
    st.warning("""
    ⚠️ **IMPORTANT DISCLAIMER**: This is an educational tool for learning purposes only. 
    - Predictions are NOT guaranteed to be accurate
    - Do NOT use for real trading decisions without professional advice
    - Past performance does not indicate future results
    - Always consult a qualified financial advisor before investing
    """)
    
    st.markdown("### ✨ Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Data & Analysis**
        - **40+ Indian Stocks** across all major sectors
        - **Real-time Data** from NSE/BSE via Yahoo Finance
        - **Advanced Technical Analysis** (RSI, MACD, Bollinger Bands, ATR)
        - **Interactive Charts** with candlestick patterns
        - **Stock Comparison** - Compare multiple stocks side-by-side
        - **Term Explanations** - Understand every technical term
        """)
    
    with col2:
        st.markdown("""
        **🤖 AI & Predictions**
        - **Prophet** - Facebook's time series forecasting
        - **ARIMA** - Statistical forecasting model
        - **XGBoost** - Advanced gradient boosting
        - **Random Forest** - ML ensemble learning
        - **Ensemble Model** - Combines multiple models
        - **Extended Predictions** - From 1 month to 5 years
        - **Model Validation** - Cross-validation metrics shown
        - **Prediction Confidence** - Model agreement indicators
        - **Professional PDF Reports** - Downloadable analysis
        """)
    
    st.markdown("---")
    st.markdown("### 📋 Available Stocks by Sector")
    
    cols = st.columns(3)
    sector_items = list(STOCK_SECTORS.items())
    
    for i, (sector, stocks) in enumerate(sector_items):
        with cols[i % 3]:
            st.markdown(f"**{sector}**")
            for stock in stocks:
                st.write(f"• {stock}")


def display_stock_comparison(period: str):
    """Display stock comparison analysis"""
    st.subheader("📊 Multi-Stock Comparison Analysis")
    
    stocks_to_compare = st.multiselect(
        "Select stocks to compare (2-6 stocks)",
        options=list(INDIAN_STOCKS.keys()),
        default=["TCS", "Infosys", "Wipro", "HCL Tech"],
        max_selections=6
    )
    
    if len(stocks_to_compare) >= 2:
        with st.spinner("Analyzing and comparing stocks..."):
            stocks_dict = {name: INDIAN_STOCKS[name] for name in stocks_to_compare}
            comparison_df = compare_stocks(stocks_dict, period)
            
            if not comparison_df.empty:
                st.markdown("### Performance Metrics Comparison")
                st.dataframe(comparison_df.style.format({
                    'Start Price': '₹{:.2f}',
                    'Current Price': '₹{:.2f}',
                    'Total Return (%)': '{:.2f}%',
                    'Volatility (%)': '{:.2f}%',
                    'Sharpe Ratio': '{:.2f}',
                    'Max Price': '₹{:.2f}',
                    'Min Price': '₹{:.2f}',
                    'Avg Volume': '{:,.0f}'
                }), width='stretch')
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_return = comparison_df['Total Return (%)'].idxmax()
                    st.metric(
                        "🏆 Best Return",
                        best_return,
                        f"{comparison_df.loc[best_return, 'Total Return (%)']:.2f}%"
                    )
                
                with col2:
                    best_sharpe = comparison_df['Sharpe Ratio'].idxmax()
                    st.metric(
                        "⚖️ Best Risk-Adjusted",
                        best_sharpe,
                        f"Sharpe: {comparison_df.loc[best_sharpe, 'Sharpe Ratio']:.2f}"
                    )
                
                with col3:
                    lowest_vol = comparison_df['Volatility (%)'].idxmin()
                    st.metric(
                        "🛡️ Lowest Volatility",
                        lowest_vol,
                        f"{comparison_df.loc[lowest_vol, 'Volatility (%)']:.2f}%"
                    )
                
                st.markdown("---")
                st.markdown("### Performance Comparison Chart")
                fig_comparison = create_comparison_chart(stocks_dict, period)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 💡 Investment Insights")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Best for Growth:** {best_return} with {comparison_df.loc[best_return, 'Total Return (%)']:.2f}% return")
                    st.info(f"**Most Stable:** {lowest_vol} with {comparison_df.loc[lowest_vol, 'Volatility (%)']:.2f}% volatility")
                
                with col2:
                    st.success(f"**Best Risk-Adjusted:** {best_sharpe} with Sharpe ratio {comparison_df.loc[best_sharpe, 'Sharpe Ratio']:.2f}")
                    show_term_explanation("Sharpe Ratio")
    else:
        st.warning("Please select at least 2 stocks to compare.")


def display_single_stock_analysis(selected_stock_name: str, selected_symbol: str, 
                                  period: str, prediction_days: int, 
                                  prediction_model: list, start_time: float):
    """Display single stock analysis"""
    with st.spinner(f"Fetching data for {selected_stock_name}..."):
        df = fetch_stock_data(selected_symbol, period)
    
    if df is not None and not df.empty and validate_stock_data(df):
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Calculate analysis duration
        data_start_date = df.index[0]
        data_end_date = df.index[-1]
        data_duration_days = (data_end_date - data_start_date).days
        
        # Display data info with quality score
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"📊 **Data Period:** {data_start_date.strftime('%Y-%m-%d')} to {data_end_date.strftime('%Y-%m-%d')}")
        with col2:
            st.info(f"📆 **Data Duration:** {data_duration_days} days")
        with col3:
            st.info(f"📈 **Total Data Points:** {len(df)} records")
        with col4:
            quality_score, quality_desc = get_data_quality_score(df)
            st.info(f"✅ **Data Quality:** {quality_desc}")
        
        st.markdown("---")
        
        # Display current metrics
        st.subheader("📊 Current Market Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        
        col1.metric("Current Price", f"₹{current_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
        col1.caption("Latest closing price")
        
        col2.metric("Day High", f"₹{df['High'].iloc[-1]:.2f}")
        col2.caption("Highest price today")
        
        col3.metric("Day Low", f"₹{df['Low'].iloc[-1]:.2f}")
        col3.caption("Lowest price today")
        
        col4.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
        col4.caption("Shares traded")
        
        rsi_val = df['RSI'].iloc[-1]
        rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
        col5.metric("RSI", f"{rsi_val:.2f}", rsi_status)
        col5.caption("Momentum indicator")
        
        st.markdown("---")
        
        # Run predictions
        predictions_dict = {}
        
        if "Prophet (Time Series)" in prediction_model:
            with st.spinner("🔮 Running Prophet prediction..."):
                predictions_dict['Prophet'] = prophet_prediction(df, prediction_days)
        
        if "ARIMA (Statistical)" in prediction_model:
            with st.spinner("📊 Running ARIMA prediction..."):
                predictions_dict['ARIMA'] = arima_prediction(df, prediction_days)
        
        if "XGBoost (Advanced ML)" in prediction_model:
            with st.spinner("🤖 Running XGBoost prediction..."):
                predictions_dict['XGBoost'] = xgboost_prediction(df, prediction_days)
        
        if "Random Forest (ML)" in prediction_model:
            with st.spinner("🌲 Running Random Forest prediction..."):
                predictions_dict['Random Forest'] = ml_prediction(df, prediction_days)
        
        if "Ensemble (Combined)" in prediction_model:
            with st.spinner("⚡ Running Ensemble prediction..."):
                predictions_dict['Ensemble'] = ensemble_prediction(df, prediction_days)
        
        # Create and display main chart
        st.subheader("📈 Advanced Price Chart with Predictions & Trendlines")
        fig = create_main_chart(
            df,
            predictions_dict.get('Prophet'),
            predictions_dict.get('Random Forest'),
            predictions_dict.get('Ensemble'),
            predictions_dict.get('XGBoost'),
            predictions_dict.get('ARIMA'),
            selected_stock_name
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Simple time-value chart
        st.subheader("📊 Simple Price Chart (Time vs Value)")
        simple_fig = create_simple_price_chart(df, selected_stock_name)
        st.plotly_chart(simple_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Trading signals
        st.subheader("🎯 Trading Signals & Recommendations")
        signals = calculate_signals(df)
        
        if signals:
            cols = st.columns(len(signals))
            for i, (signal_type, reason) in enumerate(signals):
                with cols[i]:
                    if signal_type == "BUY":
                        st.markdown(f'<div class="signal-card-buy"><h3>🟢 {signal_type}</h3><p>{reason}</p></div>',
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="signal-card-sell"><h3>🔴 {signal_type}</h3><p>{reason}</p></div>',
                                  unsafe_allow_html=True)
        else:
            st.info("ℹ️ No strong signals at the moment. Consider holding your position.")
        
        st.markdown("---")
        
        # Prediction summary
        if predictions_dict:
            st.subheader("🔮 Prediction Summary")
            
            # Calculate overall trend
            trend, trend_change, avg_pred = calculate_trendline(df, predictions_dict)
            
            if trend:
                trend_color = "green" if trend == "Upward" else "red"
                st.markdown(f"### 📊 Overall Market Trend: <span style='color:{trend_color}'>{trend} Trend</span>",
                          unsafe_allow_html=True)
                st.markdown(f"**Average Predicted Change:** {trend_change:+.2f}% | **Target Price:** ₹{avg_pred:.2f}")
                
                # Show prediction confidence
                confidence = get_prediction_confidence(predictions_dict, current_price)
                st.markdown(f"**Prediction Confidence:** {confidence}")
                
                st.progress(min(abs(trend_change) / 10, 1.0))
            
            st.markdown("---")
            
            cols = st.columns(len(predictions_dict))
            
            for idx, (model_name, pred_df) in enumerate(predictions_dict.items()):
                if pred_df is not None:
                    with cols[idx]:
                        if 'yhat' in pred_df.columns:
                            pred_price = pred_df['yhat'].iloc[-1]
                        elif 'Predicted_Close' in pred_df.columns:
                            pred_price = pred_df['Predicted_Close'].iloc[-1]
                        else:
                            continue
                        
                        pred_change = ((pred_price - current_price) / current_price) * 100
                        
                        st.markdown(f'<div class="prediction-card">', unsafe_allow_html=True)
                        st.markdown(f"**{model_name}**")
                        st.metric(
                            f"Price in {prediction_days} days",
                            f"₹{pred_price:.2f}",
                            f"{pred_change:+.2f}%"
                        )
                        st.caption(f"Type: {MODEL_ACCURACY_RATINGS.get(model_name, '📊 Model')}")
                        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key statistics
        st.subheader("📈 Detailed Technical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Price Statistics**")
            st.write(f"52-Week High: ₹{df['High'].tail(252).max():.2f}")
            st.write(f"52-Week Low: ₹{df['Low'].tail(252).min():.2f}")
            st.write(f"Average Volume: {df['Volume'].mean():,.0f}")
            show_term_explanation("Volume")
        
        with col2:
            st.markdown("**🔧 Technical Indicators**")
            st.write(f"RSI (14): {df['RSI'].iloc[-1]:.2f}")
            st.write(f"MACD: {df['MACD'].iloc[-1]:.2f}")
            st.write(f"ATR: {df['ATR'].iloc[-1]:.2f}")
            show_term_explanation("ATR")
        
        with col3:
            st.markdown("**💰 Returns Analysis**")
            returns_7d = ((df['Close'].iloc[-1] - df['Close'].iloc[-7]) / df['Close'].iloc[-7]) * 100
            returns_30d = ((df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]) * 100
            returns_ytd = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            st.write(f"7-Day Return: {returns_7d:+.2f}%")
            st.write(f"30-Day Return: {returns_30d:+.2f}%")
            st.write(f"Period Return: {returns_ytd:+.2f}%")
        
        st.markdown("---")
        
        # Calculate total analysis duration
        analysis_duration = time.time() - start_time
        st.success(f"✅ **Analysis completed in {analysis_duration:.2f} seconds**")
        
        # Download reports
        st.subheader("📥 Download Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate PDF report
            with st.spinner("Generating professional PDF report..."):
                pdf_buffer = generate_pdf_report(
                    selected_stock_name, df, predictions_dict, signals,
                    analysis_duration, data_start_date, data_end_date
                )
                
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download PDF Report (Professional)",
                        data=pdf_buffer,
                        file_name=f"{selected_stock_name}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        width='stretch'
                    )
        
        with col2:
            # Quick text report
            st.download_button(
                label="📝 Download Quick Text Report",
                data=f"Stock Analysis: {selected_stock_name}\nCurrent Price: ₹{current_price:.2f}\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d')}",
                file_name=f"{selected_stock_name}_quick_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                width='stretch'
            )
        
        st.markdown("---")
        
        # Comprehensive explanation section at the bottom
        st.subheader("📚 Understanding All Features & Indicators")
        
        with st.expander("🎯 Trading Signals Explained", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                show_term_explanation("RSI")
                show_term_explanation("MACD")
                show_term_explanation("Golden Cross")
            with col2:
                show_term_explanation("Death Cross")
                show_term_explanation("Bollinger Bands")
                show_term_explanation("Support")
                show_term_explanation("Resistance")
        
        with st.expander("📊 Technical Indicators Explained", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                show_term_explanation("SMA")
                show_term_explanation("EMA")
                show_term_explanation("Volume")
            with col2:
                show_term_explanation("ATR")
                show_term_explanation("Volatility")
                show_term_explanation("Sharpe Ratio")
        
        with st.expander("🤖 AI Prediction Models Explained", expanded=False):
            st.markdown("""
            **Prophet (Time Series)**
            - Facebook's time series forecasting algorithm
            - Excellent for capturing seasonal patterns and trends
            - Handles missing data and outliers well
            - Best for: Long-term trends, seasonal stocks
            
            **ARIMA (Statistical)**
            - AutoRegressive Integrated Moving Average
            - Classical statistical approach
            - Simple and stable predictions
            - Best for: Stable stocks with clear patterns
            
            **XGBoost (Advanced ML)**
            - Gradient boosting machine learning
            - Captures complex non-linear patterns
            - Uses multiple features and lags
            - Best for: Volatile stocks, complex patterns
            
            **Random Forest (ML)**
            - Ensemble of decision trees
            - Robust to overfitting
            - Good general-purpose predictor
            - Best for: Balanced predictions across stocks
            
            **Ensemble (Combined)**
            - Combines Random Forest, Gradient Boosting, and Linear Regression
            - Weighted average of multiple models
            - Most reliable overall predictions
            - Best for: Maximum accuracy and reliability
            """)
        
        with st.expander("📈 Chart Features Explained", expanded=False):
            st.markdown("""
            **Candlestick Chart**
            - Green candles: Price increased (Close > Open)
            - Red candles: Price decreased (Close < Open)
            - Wicks show high and low prices
            
            **Moving Averages**
            - SMA 20 (Orange): Short-term trend (20 days)
            - SMA 50 (Blue): Medium-term trend (50 days)
            - SMA 200 (Red): Long-term trend (200 days)
            
            **Bollinger Bands**
            - Upper and lower bands show volatility
            - Price near upper band: Potentially overbought
            - Price near lower band: Potentially oversold
            
            **MACD Histogram**
            - Green bars: Bullish momentum
            - Red bars: Bearish momentum
            - Crossing zero line: Potential trend change
            
            **RSI Indicator**
            - Above 70: Overbought (consider selling)
            - Below 30: Oversold (consider buying)
            - 50: Neutral
            
            **Volume Bars**
            - Green: Volume on up day
            - Red: Volume on down day
            - High volume confirms price movements
            """)
        
        with st.expander("💡 How to Use This App", expanded=False):
            st.markdown("""
            **Step 1: Select Stock**
            - Choose from 40+ Indian stocks
            - Or enter custom NSE/BSE symbol
            
            **Step 2: Configure Analysis**
            - Historical period: 1y, 2y, 5y, or max
            - Prediction period: 1 month to 5 years
            - Select AI models to use
            
            **Step 3: Analyze**
            - Click "🚀 ANALYZE STOCK" button
            - Wait for data fetching and model training
            - Review validation metrics (MAE, RMSE, MAPE)
            
            **Step 4: Interpret Results**
            - Check data quality score (aim for 75+)
            - Review current metrics and indicators
            - Look at trading signals (Buy/Sell/Hold)
            - Compare predictions from different models
            - Check prediction confidence level
            
            **Step 5: Make Decisions**
            - ⚠️ This is educational only - NOT financial advice
            - Use predictions as ONE input among many
            - Consider multiple models and their agreement
            - Always consult professional financial advisors
            - Never invest more than you can afford to lose
            
            **Step 6: Download Reports**
            - PDF report: Comprehensive analysis with charts
            - Text report: Quick summary for reference
            """)
        
        with st.expander("⚠️ Important Disclaimers & Limitations", expanded=False):
            st.warning("""
            **CRITICAL DISCLAIMERS:**
            
            🚫 **NOT Financial Advice**
            - This is an educational tool for learning purposes only
            - Do NOT use for real trading without professional consultation
            - Always consult qualified financial advisors before investing
            
            📊 **Prediction Limitations**
            - Predictions are based on historical patterns only
            - Cannot predict black swan events or market crashes
            - Longer predictions (3-5 years) are less reliable
            - Market sentiment and news are not factored in
            - Models can be wrong - no guarantees
            
            📉 **Data Limitations**
            - Data from Yahoo Finance (free, 15-min delay)
            - Some stocks have limited historical data
            - Market holidays cause data gaps
            - Data quality varies by stock
            
            🎓 **Educational Purpose**
            - Learn about technical analysis
            - Understand ML/AI prediction models
            - Practice stock analysis skills
            - NOT for professional trading
            
            ⚖️ **Legal Notice**
            - Authors not liable for financial losses
            - Use at your own risk
            - Past performance ≠ future results
            - Invest responsibly
            """)
    
    else:
        st.error(f"❌ Unable to fetch data for {selected_stock_name}. Please check the symbol and try again.")


def main():
    """Main application function"""
    display_header()
    
    # Display today's date
    col_date1, col_date2 = st.columns([3, 1])
    with col_date1:
        st.info(f"📅 **Today's Date:** {datetime.now().strftime('%A, %B %d, %Y')}")
    with col_date2:
        st.info(f"⏰ **Time:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Sidebar with compact layout
    st.sidebar.title("⚙️ Configuration")
    
    # Analyze button at the top
    analyze_button = st.sidebar.button("🚀 ANALYZE STOCK", type="primary", width='stretch', 
                                       help="Click to start analysis")
    
    st.sidebar.markdown("---")
    
    # App mode at top
    app_mode = st.sidebar.radio(
        "📊 Mode",
        options=["Single Stock", "Compare Stocks"],
        index=0,
        help="Choose analysis mode"
    )
    
    st.sidebar.markdown("---")
    
    # Stock selection - compact
    selected_stock_name = st.sidebar.selectbox(
        "🏢 Stock",
        options=list(INDIAN_STOCKS.keys()),
        index=0,
        help="Select from 40+ Indian stocks"
    )
    selected_symbol = INDIAN_STOCKS[selected_stock_name]
    
    # Custom stock option - compact
    custom_stock = st.sidebar.text_input("Custom Symbol", placeholder="e.g., RELIANCE.NS",
                                         help="Enter any NSE/BSE symbol")
    if custom_stock:
        selected_symbol = custom_stock
        selected_stock_name = custom_stock
    
    # Time period - compact
    col1, col2 = st.sidebar.columns(2)
    with col1:
        period = st.selectbox(
            "📅 History",
            options=["1y", "2y", "5y", "max"],
            index=1,
            help="Historical data period"
        )
    
    with col2:
        prediction_period_name = st.selectbox(
            "🔮 Predict",
            options=list(PREDICTION_PERIODS.keys()),
            index=1,
            help="Prediction period"
        )
    
    prediction_days = PREDICTION_PERIODS[prediction_period_name]
    
    # Prediction models - compact with expander
    with st.sidebar.expander("🤖 AI Models", expanded=False):
        prediction_model = st.multiselect(
            "Select Models",
            options=["Prophet (Time Series)", "ARIMA (Statistical)", "XGBoost (Advanced ML)",
                    "Random Forest (ML)", "Ensemble (Combined)"],
            default=["Prophet (Time Series)", "XGBoost (Advanced ML)", "Ensemble (Combined)"],
            help="Choose prediction models to use"
        )
    
    # Main content
    if analyze_button:
        start_time = time.time()
        
        if app_mode == "Compare Stocks":
            display_stock_comparison(period)
        else:
            display_single_stock_analysis(
                selected_stock_name, selected_symbol, period, 
                prediction_days, prediction_model, start_time
            )
    else:
        display_welcome_screen()


if __name__ == "__main__":
    main()
