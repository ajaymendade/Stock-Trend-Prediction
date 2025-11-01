"""
Improved prediction models module with validation and better forecasting
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from config import MODEL_CONFIGS
from typing import Optional, Tuple, Dict


@st.cache_data(ttl=3600)
def prophet_prediction(df: pd.DataFrame, periods: int = 90) -> Optional[pd.DataFrame]:
    """
    Use Prophet for time series forecasting with validation
    
    Args:
        df: DataFrame with historical data
        periods: Number of days to predict
    
    Returns:
        DataFrame with predictions or None if error
    """
    try:
        # Prepare data for Prophet
        prophet_df = df.reset_index()[['Date', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Ensure datetime column has no timezone
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
        
        # Create and fit model
        model = Prophet(**MODEL_CONFIGS['prophet'])
        
        with st.spinner("Training Prophet model..."):
            model.fit(prophet_df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Calculate validation metrics on historical data
        historical_predictions = forecast[forecast['ds'].isin(prophet_df['ds'])]
        if len(historical_predictions) > 0:
            actual = prophet_df['y'].values
            predicted = historical_predictions['yhat'].values[:len(actual)]
            
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            st.info(f"📊 Prophet Validation - MAE: ₹{mae:.2f}, RMSE: ₹{rmse:.2f}, MAPE: {mape:.2f}%")
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        st.error(f"Prophet prediction error: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def arima_prediction(df: pd.DataFrame, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Use ARIMA for time series prediction with validation
    
    Args:
        df: DataFrame with historical data
        days: Number of days to predict
    
    Returns:
        DataFrame with predictions or None if error
    """
    try:
        # Prepare data
        close_prices = df['Close'].values
        
        # Fit ARIMA model
        with st.spinner("Training ARIMA model..."):
            model = ARIMA(close_prices, order=MODEL_CONFIGS['arima']['order'])
            fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=days)
        
        # Get confidence intervals
        forecast_result = fitted_model.get_forecast(steps=days)
        conf_int = forecast_result.conf_int()
        
        # Create prediction dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': forecast,
            'Lower_Bound': conf_int.iloc[:, 0] if len(conf_int) > 0 else forecast * 0.95,
            'Upper_Bound': conf_int.iloc[:, 1] if len(conf_int) > 0 else forecast * 1.05
        })
        
        # Calculate in-sample metrics
        fitted_values = fitted_model.fittedvalues
        actual = close_prices[1:]  # ARIMA starts from second value
        mae = mean_absolute_error(actual, fitted_values)
        rmse = np.sqrt(mean_squared_error(actual, fitted_values))
        
        st.info(f"📊 ARIMA Validation - MAE: ₹{mae:.2f}, RMSE: ₹{rmse:.2f}")
        
        return pred_df
    except Exception as e:
        st.error(f"ARIMA prediction error: {str(e)}")
        return None


def prepare_ml_features(df: pd.DataFrame, n_lags: int = 10) -> Tuple:
    """
    Prepare enhanced features for ML models with infinity/NaN handling
    
    Args:
        df: DataFrame with historical data
        n_lags: Number of lag features to create
    
    Returns:
        Tuple of (X, y, scaler, feature_cols, X_original)
    """
    df_ml = df.copy()
    
    # Basic features
    df_ml['Returns'] = df_ml['Close'].pct_change()
    df_ml['High_Low'] = df_ml['High'] - df_ml['Low']
    df_ml['Price_Change'] = df_ml['Close'] - df_ml['Open']
    df_ml['Volume_Change'] = df_ml['Volume'].pct_change()
    
    # Moving averages as features
    df_ml['SMA_5'] = df_ml['Close'].rolling(window=5).mean()
    df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean()
    df_ml['SMA_20'] = df_ml['Close'].rolling(window=20).mean()
    
    # Volatility features
    df_ml['Volatility_5'] = df_ml['Returns'].rolling(window=5).std()
    df_ml['Volatility_20'] = df_ml['Returns'].rolling(window=20).std()
    
    # Create lag features
    for i in range(1, n_lags + 1):
        df_ml[f'Lag_{i}'] = df_ml['Close'].shift(i)
    
    # Drop NaN values
    df_ml = df_ml.dropna()
    
    # Prepare features
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'Returns', 'High_Low',
                   'Price_Change', 'Volume_Change', 'SMA_5', 'SMA_10', 'SMA_20',
                   'Volatility_5', 'Volatility_20'] + [f'Lag_{i}' for i in range(1, n_lags + 1)]
    
    X = df_ml[feature_cols].copy()
    y = df_ml['Close']
    
    # Replace infinity with NaN, then fill with column mean
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with column mean (or 0 if all NaN)
    for col in X.columns:
        col_mean = X[col].mean()
        if np.isnan(col_mean):
            X[col] = X[col].fillna(0)
        else:
            X[col] = X[col].fillna(col_mean)
    
    # Final check: ensure no infinity or NaN values
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    
    # Clip extreme values to prevent overflow
    X = X.clip(-1e10, 1e10)
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols, X


def validate_ml_model(model, X, y, model_name: str):
    """
    Validate ML model using time series cross-validation
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        model_name: Name of the model
    """
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        scores.append(mae)
    
    avg_mae = np.mean(scores)
    st.info(f"📊 {model_name} Cross-Validation - Average MAE: ₹{avg_mae:.2f}")


@st.cache_data(ttl=3600)
def xgboost_prediction(df: pd.DataFrame, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Use XGBoost for prediction with improved forecasting
    
    Args:
        df: DataFrame with historical data
        days: Number of days to predict
    
    Returns:
        DataFrame with predictions or None if error
    """
    try:
        X_scaled, y, scaler, feature_cols, X_original = prepare_ml_features(df)
        
        # Train XGBoost model
        with st.spinner("Training XGBoost model..."):
            model = XGBRegressor(**MODEL_CONFIGS['xgboost'])
            model.fit(X_scaled, y)
        
        # Validate model
        validate_ml_model(model, X_scaled, y, "XGBoost")
        
        # Multi-step ahead prediction using recursive strategy
        predictions = []
        last_window = X_original.iloc[-1:].copy()
        
        for step in range(days):
            # Clean window before prediction
            last_window_clean = last_window[feature_cols].copy()
            last_window_clean = last_window_clean.replace([np.inf, -np.inf], np.nan)
            last_window_clean = last_window_clean.fillna(method='ffill').fillna(0)
            last_window_clean = last_window_clean.clip(-1e10, 1e10)
            
            # Predict next value
            last_scaled = scaler.transform(last_window_clean)
            pred = model.predict(last_scaled)[0]
            
            # Validate prediction
            if np.isnan(pred) or np.isinf(pred):
                pred = predictions[-1] if predictions else df['Close'].iloc[-1]
            
            predictions.append(pred)
            
            # Update window for next prediction - only update features in feature_cols
            # Shift lag features
            for i in range(10, 1, -1):
                if f'Lag_{i}' in feature_cols:
                    last_window.loc[:, f'Lag_{i}'] = last_window[f'Lag_{i-1}'].values[0]
            
            if 'Lag_1' in feature_cols:
                last_window.loc[:, 'Lag_1'] = pred
            
            # Update other features that are in feature_cols
            if 'Returns' in feature_cols and step > 0 and len(predictions) > 1:
                prev_pred = predictions[-2]
                if prev_pred != 0:
                    last_window.loc[:, 'Returns'] = (pred - prev_pred) / prev_pred
                else:
                    last_window.loc[:, 'Returns'] = 0
            elif 'Returns' in feature_cols:
                last_window.loc[:, 'Returns'] = 0
        
        # Create prediction dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions
        })
        
        return pred_df
    except Exception as e:
        st.error(f"XGBoost prediction error: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def ml_prediction(df: pd.DataFrame, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Use Random Forest for prediction
    
    Args:
        df: DataFrame with historical data
        days: Number of days to predict
    
    Returns:
        DataFrame with predictions or None if error
    """
    try:
        X_scaled, y, scaler, feature_cols, X_original = prepare_ml_features(df)
        
        # Train model
        with st.spinner("Training Random Forest model..."):
            model = RandomForestRegressor(**MODEL_CONFIGS['random_forest'])
            model.fit(X_scaled, y)
        
        # Validate model
        validate_ml_model(model, X_scaled, y, "Random Forest")
        
        # Multi-step ahead prediction
        predictions = []
        last_window = X_original.iloc[-1:].copy()
        
        for step in range(days):
            # Clean window before prediction - only use features from feature_cols
            last_window_clean = last_window[feature_cols].copy()
            last_window_clean = last_window_clean.replace([np.inf, -np.inf], np.nan)
            last_window_clean = last_window_clean.fillna(method='ffill').fillna(0)
            last_window_clean = last_window_clean.clip(-1e10, 1e10)
            
            last_scaled = scaler.transform(last_window_clean)
            pred = model.predict(last_scaled)[0]
            
            # Validate prediction
            if np.isnan(pred) or np.isinf(pred):
                pred = predictions[-1] if predictions else df['Close'].iloc[-1]
            
            predictions.append(pred)
            
            # Update window - only features in feature_cols
            for i in range(10, 1, -1):
                if f'Lag_{i}' in feature_cols:
                    last_window.loc[:, f'Lag_{i}'] = last_window[f'Lag_{i-1}'].values[0]
            
            if 'Lag_1' in feature_cols:
                last_window.loc[:, 'Lag_1'] = pred
        
        # Create prediction dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions
        })
        
        return pred_df
    except Exception as e:
        st.error(f"ML prediction error: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def ensemble_prediction(df: pd.DataFrame, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Ensemble prediction using multiple ML models with validation
    
    Args:
        df: DataFrame with historical data
        days: Number of days to predict
    
    Returns:
        DataFrame with predictions or None if error
    """
    try:
        X_scaled, y, scaler, feature_cols, X_original = prepare_ml_features(df)
        
        # Train multiple models
        with st.spinner("Training Ensemble models..."):
            rf_model = RandomForestRegressor(**MODEL_CONFIGS['random_forest'])
            gb_model = GradientBoostingRegressor(**MODEL_CONFIGS['gradient_boosting'])
            lr_model = LinearRegression()
            
            rf_model.fit(X_scaled, y)
            gb_model.fit(X_scaled, y)
            lr_model.fit(X_scaled, y)
        
        # Validate ensemble
        validate_ml_model(rf_model, X_scaled, y, "Ensemble (RF)")
        
        # Multi-step ahead prediction with ensemble
        predictions = []
        last_window = X_original.iloc[-1:].copy()
        
        for step in range(days):
            # Clean window before prediction - only use features from feature_cols
            last_window_clean = last_window[feature_cols].copy()
            last_window_clean = last_window_clean.replace([np.inf, -np.inf], np.nan)
            last_window_clean = last_window_clean.fillna(method='ffill').fillna(0)
            last_window_clean = last_window_clean.clip(-1e10, 1e10)
            
            last_scaled = scaler.transform(last_window_clean)
            
            # Get predictions from all models
            rf_pred = rf_model.predict(last_scaled)[0]
            gb_pred = gb_model.predict(last_scaled)[0]
            lr_pred = lr_model.predict(last_scaled)[0]
            
            # Validate individual predictions
            if np.isnan(rf_pred) or np.isinf(rf_pred):
                rf_pred = predictions[-1] if predictions else df['Close'].iloc[-1]
            if np.isnan(gb_pred) or np.isinf(gb_pred):
                gb_pred = predictions[-1] if predictions else df['Close'].iloc[-1]
            if np.isnan(lr_pred) or np.isinf(lr_pred):
                lr_pred = predictions[-1] if predictions else df['Close'].iloc[-1]
            
            # Weighted ensemble (RF and GB get more weight)
            ensemble_pred = (0.4 * rf_pred + 0.4 * gb_pred + 0.2 * lr_pred)
            
            # Final validation
            if np.isnan(ensemble_pred) or np.isinf(ensemble_pred):
                ensemble_pred = predictions[-1] if predictions else df['Close'].iloc[-1]
            
            predictions.append(ensemble_pred)
            
            # Update window - only features in feature_cols
            for i in range(10, 1, -1):
                if f'Lag_{i}' in feature_cols:
                    last_window.loc[:, f'Lag_{i}'] = last_window[f'Lag_{i-1}'].values[0]
            
            if 'Lag_1' in feature_cols:
                last_window.loc[:, 'Lag_1'] = ensemble_pred
        
        # Create prediction dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions
        })
        
        return pred_df
    except Exception as e:
        st.error(f"Ensemble prediction error: {str(e)}")
        return None


def calculate_trendline(df: pd.DataFrame, predictions_dict: Dict) -> Tuple:
    """
    Calculate overall trendline from predictions
    
    Args:
        df: DataFrame with historical data
        predictions_dict: Dictionary of predictions from different models
    
    Returns:
        Tuple of (trend, change_pct, avg_prediction)
    """
    try:
        all_predictions = []
        
        for pred_df in predictions_dict.values():
            if pred_df is not None:
                if 'yhat' in pred_df.columns:
                    all_predictions.append(pred_df['yhat'].values[-1])
                elif 'Predicted_Close' in pred_df.columns:
                    all_predictions.append(pred_df['Predicted_Close'].values[-1])
        
        if all_predictions:
            avg_prediction = np.mean(all_predictions)
            current_price = df['Close'].iloc[-1]
            trend = "Upward" if avg_prediction > current_price else "Downward"
            change_pct = ((avg_prediction - current_price) / current_price) * 100
            return trend, change_pct, avg_prediction
        
        return None, None, None
    except:
        return None, None, None


def get_prediction_confidence(predictions_dict: Dict, current_price: float) -> str:
    """
    Calculate confidence level based on model agreement
    
    Args:
        predictions_dict: Dictionary of predictions
        current_price: Current stock price
    
    Returns:
        Confidence level string
    """
    if not predictions_dict:
        return "Unknown"
    
    predictions = []
    for pred_df in predictions_dict.values():
        if pred_df is not None:
            if 'yhat' in pred_df.columns:
                predictions.append(pred_df['yhat'].values[-1])
            elif 'Predicted_Close' in pred_df.columns:
                predictions.append(pred_df['Predicted_Close'].values[-1])
    
    if len(predictions) < 2:
        return "Low (Single Model)"
    
    # Calculate standard deviation of predictions
    std_dev = np.std(predictions)
    mean_pred = np.mean(predictions)
    cv = (std_dev / mean_pred) * 100  # Coefficient of variation
    
    if cv < 2:
        return "Very High (Models Agree)"
    elif cv < 5:
        return "High (Good Agreement)"
    elif cv < 10:
        return "Medium (Some Disagreement)"
    else:
        return "Low (Models Disagree)"
