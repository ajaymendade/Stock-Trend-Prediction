"""
UI components and styling module
"""
import streamlit as st
from config import TERM_EXPLANATIONS


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
        /* Main header with gradient */
        .main-header {
            font-size: 3.5rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(120deg, #4fc3f7 0%, #66bb6a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            padding: 1rem 0;
            text-shadow: 0 0 30px rgba(79, 195, 247, 0.3);
        }

        /* Sub-header with better contrast */
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #b0b0b0;
            margin-bottom: 2rem;
        }

        /* Metric cards with dark theme optimization */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 0.5rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Info tooltip for dark theme */
        .info-tooltip {
            background: linear-gradient(135deg, rgba(30, 30, 40, 0.9) 0%, rgba(40, 40, 50, 0.9) 100%);
            border-left: 4px solid #4fc3f7;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            color: #e0e0e0;
            border: 1px solid rgba(79, 195, 247, 0.3);
        }

        /* Prediction cards with glow effect */
        .prediction-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 25px rgba(245, 87, 108, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Buy signal cards with green glow */
        .signal-card-buy {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            box-shadow: 0 8px 25px rgba(56, 239, 125, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Sell signal cards with red glow */
        .signal-card-sell {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            box-shadow: 0 8px 25px rgba(244, 92, 67, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Metric styling for dark theme */
        .stMetric {
            background: linear-gradient(135deg, rgba(40, 40, 50, 0.6) 0%, rgba(50, 50, 60, 0.6) 100%);
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Enhanced styling for Streamlit components in dark mode */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(20, 20, 30, 0.95) 0%, rgba(30, 30, 40, 0.95) 100%);
            border-right: 1px solid rgba(79, 195, 247, 0.2);
        }

        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, #4fc3f7 0%, #66bb6a 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(79, 195, 247, 0.3);
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            box-shadow: 0 6px 20px rgba(79, 195, 247, 0.5);
            transform: translateY(-2px);
        }

        /* Download button styling */
        .stDownloadButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #4fc3f7 0%, #66bb6a 100%);
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)


def show_term_explanation(term: str):
    """
    Display explanation for a technical term
    
    Args:
        term: Technical term to explain
    """
    if term in TERM_EXPLANATIONS:
        st.markdown(
            f'<div class="info-tooltip"><strong>{term}:</strong> {TERM_EXPLANATIONS[term]}</div>', 
            unsafe_allow_html=True
        )


def display_header():
    """Display the main application header"""
    st.markdown('<p class="main-header">Indian Stock Market Analyzer & Predictor Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-Powered Stock Analysis with Multiple Prediction Models</p>', unsafe_allow_html=True)
