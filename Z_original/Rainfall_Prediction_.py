import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import SimpleImputer  # Add this import for handling missing values
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
import requests
import io

# Utility function to calculate parameter importance (correlation-based)
def calculate_parameter_importance(df, params):
    """
    Calculate the importance of weather parameters for precipitation using absolute correlation.
    Returns a DataFrame with Parameter and Importance (%).
    """
    import numpy as np
    import pandas as pd
    importance = []
    for param in params:
        if param in df.columns and 'precipitation' in df.columns:
            corr = df[param].corr(df['precipitation'])
            if pd.isna(corr):
                corr = 0
            importance.append(abs(corr))
        else:
            importance.append(0)
    total = sum(importance) if sum(importance) != 0 else 1
    importance_percent = [100 * val / total for val in importance]
    return pd.DataFrame({'Parameter': params, 'Importance (%)': importance_percent})

# Set page config for better appearance
st.set_page_config(
    page_title="IBUS Rainfall Prediction Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# IBUS Website Colors based on the logo
IBUS_PRIMARY = "#110361"  # Light blue from logo (was dark blue)
IBUS_SECONDARY = "#097BA8"  # IBUS secondary blue
IBUS_ACCENT = "#ff0015"  # IBUS accent orange
IBUS_LIGHT = "#f0f2f5"  # Light background
IBUS_DARK = "#043362"  # Dark text
IBUS_SUCCESS = "#011D08"  # Success green
IBUS_DANGER = "#dc3545"  # Danger red

# API Keys
OPENWEATHER_KEY = "963b1cda7f11d24ed01897028851e4f7"
WEATHERBIT_KEY = "d24d0bb6c51f4d0eb110fa8c748c36c5"
WEATHERAPI_KEY = "0c06578e48e74b03ab160431253005"

# Custom CSS for IBUS styling
st.markdown(f"""
    <style>
        /* Main app styling */
        .stApp {{
            background-color: {IBUS_LIGHT};
        }}
        
        /* Headers with underlines */
        h1, h2, h3, h4, h5, h6 {{
            color: {IBUS_PRIMARY} !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            border-bottom: 2px solid {IBUS_PRIMARY};
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}
        
        /* Sidebar headers - no underline */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] h4, 
        [data-testid="stSidebar"] h5, 
        [data-testid="stSidebar"] h6 {{
            border-bottom: none;
            padding-bottom: 0;
            margin-bottom: 8px;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {IBUS_PRIMARY}, {IBUS_SECONDARY});
            color: white;
        }}
        [data-testid="stSidebar"] .stRadio label, 
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stSlider label {{
            color: white !important;
            font-weight: 600;
        }}
        
        /* Buttons */
        .stButton>button {{
            background-color: {IBUS_ACCENT};
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: {IBUS_PRIMARY};
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Input widgets */
        .stSelectbox, .stSlider, .stNumberInput, .stTextInput, .stTextArea {{
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        /* Dataframes */
        .stDataFrame {{
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        /* Tabs */
        .st-b7 {{
            background-color: {IBUS_LIGHT};
        }}
        [data-baseweb="tab-list"] {{
            gap: 10px;
        }}
        [data-baseweb="tab"] {{
            background-color: {IBUS_LIGHT};
            border-radius: 8px !important;
            padding: 10px 20px !important;
            margin-right: 10px !important;
            transition: all 0.3s;
        }}
        [data-baseweb="tab"]:hover {{
            background-color: #e9ecef;
        }}
        [aria-selected="true"] {{
            background-color: {IBUS_PRIMARY} !important;
            color: white !important;
        }}
        
        /* Expanders */
        .stExpander {{
            border-radius: 8px !important;
            border: 1px solid #dee2e6 !important;
        }}
        .st-expanderHeader {{
            font-weight: 600 !important;
            color: {IBUS_PRIMARY} !important;
        }}
        
        /* Metrics */
        .metric-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric-title {{
            color: {IBUS_DARK};
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        .metric-value {{
            color: {IBUS_PRIMARY};
            font-size: 1.8rem;
            font-weight: 700;
        }}
        .metric-change {{
            font-size: 0.9rem;
            font-weight: 500;
        }}
        .positive {{
            color: {IBUS_SUCCESS};
        }}
        .negative {{
            color: {IBUS_DANGER};
        }}
        
        /* Fix text in data tables */
        .dataframe th, .dataframe td {{
            color: black !important;
        }}
        
        /* Style for select boxes */
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stSelectbox"] div[role="combobox"] {{
            background-color: white !important;
            border: 2px solid {IBUS_PRIMARY} !important;
            border-radius: 8px !important;
        }}
        
        /* Style for text inside select boxes */
        [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stSelectbox"] div[role="combobox"] span {{
            color: black !important;
            font-weight: bold !important;
        }}
        
        /* Style for number inputs */
        [data-testid="stNumberInput"] input {{
            background-color: white !important;
            color: black !important;
            font-weight: bold !important;
            border: 2px solid {IBUS_PRIMARY} !important;
        }}
        
        /* Enhanced CSS to ensure dark text on white backgrounds */
        .stApp, .main, .element-container, .block-container {{
            color: {IBUS_DARK} !important;
        }}
        
        /* Make all paragraph text dark */
        p, .stMarkdown p {{
            color: {IBUS_DARK} !important;
            font-weight: 500 !important;
        }}
        
        /* Make all labels in the main content area dark */
        .main label,
        .main .stRadio label, 
        .main .stCheckbox label,
        .main .stSelectbox label,
        .main .stMultiSelect label,
        .main .stNumberInput label,
        .main .stSlider label,
        .main .stDateInput label,
        .main .stTimeInput label {{
            color: {IBUS_DARK} !important;
            font-weight: 600 !important;
        }}
        
        /* Ensure text in widgets is dark */
        .stSelectbox div[data-baseweb="select"] span,
        .stMultiSelect div[data-baseweb="select"] span,
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {{
            color: {IBUS_DARK} !important;
            font-weight: 500 !important;
        }}
        
        /* Ensure text in dataframes is dark */
        .stDataFrame td, 
        .stDataFrame th {{
            color: {IBUS_DARK} !important;
            font-weight: normal !important;
        }}
        
        .stDataFrame th {{
            font-weight: bold !important;
        }}
        
        /* Ensure text in metrics is dark */
        .metric-card .metric-title,
        .metric-card .metric-value {{
            color: {IBUS_DARK} !important;
        }}
        
        /* Ensure text in expanders is dark */
        .stExpander .st-emotion-cache-1gulkj5,
        .stExpander p,
        .stExpander div {{
            color: {IBUS_DARK} !important;
        }}
        
        /* Ensure text in tabs is dark when not selected */
        [data-baseweb="tab"]:not([aria-selected="true"]) {{
            color: {IBUS_DARK} !important;
        }}
        
        /* Ensure all list items are dark */
        li, ul, ol {{
            color: {IBUS_DARK} !important;
        }}
        
        /* Ensure all table text is dark */
        table, tr, td, th {{
            color: {IBUS_DARK} !important;
        }}
        
        /* Ensure all code text is dark */
        code, pre {{
            color: {IBUS_DARK} !important;
        }}
        
        /* Ensure all form elements have dark text */
        input, select, textarea, button:not(.stButton>button) {{
            color: {IBUS_DARK} !important;
        }}
        
        /* Ensure dropdown options have dark text */
        [role="listbox"] [role="option"] {{
            color: {IBUS_DARK} !important;
        }}
        
        /* Ensure text in plotly charts is dark */
        .js-plotly-plot .plotly .gtitle, 
        .js-plotly-plot .plotly .xtitle, 
        .js-plotly-plot .plotly .ytitle {{
            fill: {IBUS_DARK} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# Add specific CSS for the input method radio buttons in sidebar
st.markdown("""
    <style>
        /* Target the radio buttons in the sidebar specifically */
        [data-testid="stSidebar"] [data-testid="stRadio"] label,
        [data-testid="stSidebar"] [data-testid="stRadio"] div,
        [data-testid="stSidebar"] [data-testid="stRadio"] span,
        [data-testid="stSidebar"] [data-testid="stRadio"] p {
            color: white !important;
            font-weight: 600 !important;
        }
        
        /* Target the radio button text specifically */
        [data-testid="stSidebar"] .st-emotion-cache-1nv2ebz,
        [data-testid="stSidebar"] .st-emotion-cache-1rmzm5u,
        [data-testid="stSidebar"] .st-emotion-cache-1qg05tj {
            color: white !important;
        }
        
        /* Target the radio button options specifically */
        [data-testid="stSidebar"] [role="radiogroup"] label span p {
            color: white !important;
            font-weight: 500 !important;
        }
        
        /* Target the radio button input method header */
        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
            color: white !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            margin-bottom: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Aggressive CSS override to force "Time Period Configuration" to be white
st.markdown("""
    <style>
        /* Target with maximum specificity and !important */
        [data-testid="stSidebar"] [data-testid="stMarkdown"] h4,
        [data-testid="stSidebar"] [data-testid="stMarkdown"] h4 span,
        [data-testid="stSidebar"] [data-testid="stMarkdown"] h4 p,
        [data-testid="stSidebar"] [data-testid="stMarkdown"] h4 div,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h4 span,
        [data-testid="stSidebar"] h4 p,
        [data-testid="stSidebar"] h4 div,
        [data-testid="stSidebar"] div:has(> h4),
        [data-testid="stSidebar"] div:has(> h4) * {
            color: white !important;
            fill: white !important;
            stroke: white !important;
            font-weight: 600 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
        }
        
        /* Target any element containing "Time Period Configuration" */
        [data-testid="stSidebar"] *:contains("Time Period Configuration"),
        [data-testid="stSidebar"] *:contains("Time Period Configuration") * {
            color: white !important;
            fill: white !important;
            stroke: white !important;
        }
        
        /* Override any inline styles */
        [data-testid="stSidebar"] [style*="color"] {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Comprehensive CSS to make all selectboxes have white backgrounds and black text
st.markdown("""
    <style>
        /* Target all selectboxes */
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stSelectbox"] div[role="combobox"],
        [data-testid="stSelectbox"] > div > div {
            background-color: white !important;
            border: 2px solid #110361 !important;
            border-radius: 8px !important;
        }
        
        /* Target all text inside selectboxes */
        [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stSelectbox"] div[role="combobox"] span,
        [data-testid="stSelectbox"] div[data-baseweb="select"] div,
        [data-testid="stSelectbox"] div[role="combobox"] div,
        [data-testid="stSelectbox"] div[data-baseweb="select"] p,
        [data-testid="stSelectbox"] div[role="combobox"] p {
            color: black !important;
            font-weight: 500 !important;
        }
        
        /* Target the dropdown options */
        [data-testid="stSelectbox"] div[role="option"],
        [data-testid="stSelectbox"] div[role="listbox"] div {
            background-color: white !important;
            color: black !important;
        }
        
        /* Target the selected value specifically */
        [data-testid="stSelectbox"] [aria-selected="true"],
        [data-testid="stSelectbox"] div[data-baseweb="select"] [data-testid="stMarkdown"] p {
            color: black !important;
            font-weight: bold !important;
        }
        
        /* Target the dropdown arrow */
        [data-testid="stSelectbox"] svg {
            color: black !important;
            fill: black !important;
        }
        
        /* Force all elements inside selectboxes to have white background */
        [data-testid="stSelectbox"] div[data-baseweb="select"] *,
        [data-testid="stSelectbox"] div[role="combobox"] * {
            background-color: white !important;
        }
        
        /* Force all text elements inside selectboxes to be black */
        [data-testid="stSelectbox"] div[data-baseweb="select"] *:not(svg),
        [data-testid="stSelectbox"] div[role="combobox"] *:not(svg) {
            color: black !important;
        }
        
        /* Target the dropdown menu */
        [data-baseweb="popover"] div,
        [data-baseweb="popover"] ul,
        [data-baseweb="popover"] li {
            background-color: white !important;
            color: black !important;
        }
        
        /* Target the hover state */
        [data-testid="stSelectbox"] div[role="option"]:hover {
            background-color: #f0f2f5 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Add CSS specifically for the Weather API page tabs
st.markdown("""
<style>
    /* Target the tab list in the Weather API section */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: #f0f2f5 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 5px 5px 0 5px !important;
    }
    
    /* Target individual tabs */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: white !important;
        color: black !important;
        font-weight: 600 !important;
        border: 1px solid #ddd !important;
        border-bottom: none !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        margin-right: 5px !important;
    }
    
    /* Target selected tab */
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        background-color: white !important;
        color: #110361 !important;
        border-bottom: 2px solid #110361 !important;
    }
    
    /* Target tab hover state */
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {
        background-color: #f8f9fa !important;
        color: #110361 !important;
    }
    
    /* Target tab panel (content area) */
    [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        background-color: white !important;
        border: 1px solid #ddd !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 20px !important;
    }
    
    /* Target text inside tab panel */
    [data-testid="stTabs"] [data-baseweb="tab-panel"] p,
    [data-testid="stTabs"] [data-baseweb="tab-panel"] label,
    [data-testid="stTabs"] [data-baseweb="tab-panel"] div {
        color: black !important;
    }
    
    /* Target input fields inside tab panel */
    [data-testid="stTabs"] [data-baseweb="tab-panel"] input,
    [data-testid="stTabs"] [data-baseweb="tab-panel"] [data-testid="stTextInput"] input,
    [data-testid="stTabs"] [data-baseweb="tab-panel"] [data-testid="stDateInput"] input {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ddd !important;
    }
    
    /* Target buttons inside tab panel */
    [data-testid="stTabs"] [data-baseweb="tab-panel"] button {
        background-color: #110361 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Add specific CSS for the WeatherAPI/Open-Meteo tabs
st.markdown("""
<style>
    /* Target specifically the WeatherAPI/Open-Meteo tabs */
    .api-tabs [data-baseweb="tab"] {
        background-color: white !important;
        color: black !important;
        font-weight: 600 !important;
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        margin-right: 10px !important;
    }
    
    .api-tabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #110361 !important;
        color: white !important;
    }
    
    /* Add this class when creating the tabs */
    /* Example: api_tabs = st.tabs(["WeatherAPI", "Open-Meteo"], key="api_tabs_key") */
</style>
""", unsafe_allow_html=True)

# Add global CSS to ensure all tables have light background and dark text
st.markdown("""
<style>
    /* Global styling for all dataframes */
    .dataframe {
        background-color: #f8f9fa !important;
    }
    
    .dataframe th {
        background-color: #e9ecef !important;
        color: #212529 !important;
        font-weight: bold !important;
    }
    
    .dataframe td {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    
    /* Ensure text in all tables is dark */
    table {
        color: #212529 !important;
    }
    
    /* Streamlit-specific selectors */
    [data-testid="stDataFrame"] table {
        background-color: #f8f9fa !important;
    }
    
    [data-testid="stDataFrame"] th {
        background-color: #e9ecef !important;
        color: #212529 !important;
    }
    
    [data-testid="stDataFrame"] td {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    
    /* Style for info messages */
    .element-container div[data-testid="stAlert"][kind="info"] {
        background-color: #e8f4f8 !important;
        color: #055160 !important;
        border: 1px solid #b6effb !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        font-weight: 500 !important;
    }
    
    /* Style for success messages */
    .element-container div[data-testid="stAlert"][kind="success"] {
        background-color: #d1e7dd !important;
        color: #0f5132 !important;
        border: 1px solid #a3cfbb !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Add targeted CSS for the specific table with black background
st.markdown("""
<style>
/* Target the specific table with black background */
.dataframe,
[data-testid="stDataFrame"] table,
[data-testid="stTable"] table,
div[data-testid="stDataFrame"] table,
div[data-testid="stTable"] table,
.element-container div table,
.stDataFrame table {
    background-color: #d0f0ff !important;
    color: black !important;
    border-color: #87cefa !important;
}

/* Target the dark table headers */
.dataframe thead tr th,
.dataframe th,
[data-testid="stDataFrame"] thead tr th,
[data-testid="stDataFrame"] th,
[data-testid="stTable"] thead tr th,
[data-testid="stTable"] th,
div[data-testid="stDataFrame"] thead tr th,
div[data-testid="stDataFrame"] th,
div[data-testid="stTable"] thead tr th,
div[data-testid="stTable"] th,
.element-container div thead tr th,
.element-container div th,
.stDataFrame thead tr th,
.stDataFrame th {
    background-color: #b0e0ff !important;
    color: black !important;
    font-weight: bold !important;
    border: 1px solid #87cefa !important;
}

/* Target the dark table cells */
.dataframe tbody tr td,
.dataframe td,
[data-testid="stDataFrame"] tbody tr td,
[data-testid="stDataFrame"] td,
[data-testid="stTable"] tbody tr td,
[data-testid="stTable"] td,
div[data-testid="stDataFrame"] tbody tr td,
div[data-testid="stDataFrame"] td,
div[data-testid="stTable"] tbody tr td,
div[data-testid="stTable"] td,
.element-container div tbody tr td,
.element-container div td,
.stDataFrame tbody tr td,
.stDataFrame td {
    background-color: #d0f0ff !important;
    color: black !important;
    border: 1px solid #87cefa !important;
}

/* Target the row indices (first column) */
.dataframe tbody tr th,
[data-testid="stDataFrame"] tbody tr th,
[data-testid="stTable"] tbody tr th,
div[data-testid="stDataFrame"] tbody tr th,
div[data-testid="stTable"] tbody tr th,
.element-container div tbody tr th,
.stDataFrame tbody tr th {
    background-color: #b0e0ff !important;
    color: black !important;
    font-weight: bold !important;
    border: 1px solid #87cefa !important;
}

/* Target the table rows */
.dataframe tr,
[data-testid="stDataFrame"] tr,
[data-testid="stTable"] tr,
div[data-testid="stDataFrame"] tr,
div[data-testid="stTable"] tr,
.element-container div tr,
.stDataFrame tr {
    background-color: #d0f0ff !important;
    border: 1px solid #87cefa !important;
}

/* Target the table container */
[data-testid="stDataFrame"] [data-testid="dataframe-container"],
[data-testid="stTable"] [data-testid="dataframe-container"],
div[data-testid="stDataFrame"] [data-testid="dataframe-container"],
div[data-testid="stTable"] [data-testid="dataframe-container"],
.element-container div [data-testid="dataframe-container"],
.stDataFrame [data-testid="dataframe-container"] {
    background-color: #d0f0ff !important;
    border: 1px solid #87cefa !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Target any element with inline style */
[style*="background-color: rgb(17, 17, 17)"],
[style*="background-color: #111"],
[style*="background-color: black"],
[style*="background-color: rgb(0, 0, 0)"],
[style*="background-color: #000"] {
    background-color: #d0f0ff !important;
}

[style*="color: white"],
[style*="color: rgb(255, 255, 255)"],
[style*="color: #fff"],
[style*="color: #ffffff"] {
    color: black !important;
}

/* Target specific emotion cache classes that might be used for dark tables */
.st-emotion-cache-1q1n0ol,
.st-emotion-cache-1ht1j8u,
.st-emotion-cache-ue6h4q,
.st-emotion-cache-1y4p8pa,
.st-emotion-cache-1dx29h9,
.st-emotion-cache-1cwmh3i,
.st-emotion-cache-1b0oqg6 {
    background-color: #d0f0ff !important;
    color: black !important;
}

/* Target the entire table structure */
.stDataFrame, 
.stTable, 
[data-testid="stDataFrame"], 
[data-testid="stTable"] {
    background-color: #d0f0ff !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #87cefa !important;
}

/* Force all text in tables to be black */
table *, 
.dataframe *, 
[data-testid="stTable"] *, 
[data-testid="stDataFrame"] * {
    color: black !important;
}

/* Target the scrollable area of tables */
.stDataFrame [data-testid="dataframe-container"] > div::-webkit-scrollbar {
    background-color: #d0f0ff !important;
}

/* Target the table header specifically */
thead {
    background-color: #b0e0ff !important;
}

/* Target the table body specifically */
tbody {
    background-color: #d0f0ff !important;
}

/* Target any dark-themed tables by their background color */
table[style*="background-color: rgb(17, 17, 17)"],
table[style*="background-color: #111"],
table[style*="background-color: black"],
table[style*="background-color: rgb(0, 0, 0)"],
table[style*="background-color: #000"],
.dataframe[style*="background-color: rgb(17, 17, 17)"],
.dataframe[style*="background-color: #111"],
.dataframe[style*="background-color: black"],
.dataframe[style*="background-color: rgb(0, 0, 0)"],
.dataframe[style*="background-color: #000"] {
    background-color: #d0f0ff !important;
    color: black !important;
}

/* Target any dark-themed table cells by their background color */
td[style*="background-color: rgb(17, 17, 17)"],
td[style*="background-color: #111"],
td[style*="background-color: black"],
td[style*="background-color: rgb(0, 0, 0)"],
td[style*="background-color: #000"],
th[style*="background-color: rgb(17, 17, 17)"],
th[style*="background-color: #111"],
th[style*="background-color: black"],
th[style*="background-color: rgb(0, 0, 0)"],
th[style*="background-color: #000"] {
    background-color: #d0f0ff !important;
    color: black !important;
}

/* Target the specific dark header cells */
th[style*="background-color: rgb(17, 17, 17)"],
th[style*="background-color: #111"],
th[style*="background-color: black"],
th[style*="background-color: rgb(0, 0, 0)"],
th[style*="background-color: #000"] {
    background-color: #b0e0ff !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# IBUS Header with title
st.markdown(f"""
    <h1 style="color: {IBUS_PRIMARY}; margin-bottom: 16px; border-bottom: 2px solid {IBUS_PRIMARY}; padding-bottom: 8px;">Rainfall Prediction Dashboard</h1>
    <p style="color: {IBUS_PRIMARY}; font-size: 1.2rem; margin-top: 0;">Advanced forecasting with integrated climate analytics</p>
""", unsafe_allow_html=True)

# Add this function at the top of your file
def get_season(month):
    """
    Convert month number to Indian climate seasons.
    
    Args:
        month (int): Month number (1-12)
        
    Returns:
        str: Season name ('Winter', 'Summer', 'Monsoon', 'Post-Monsoon')
    """
    if month in [11, 12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:  # month in [10]
        return 'Post-Monsoon'

# Synthetic data generation function
def generate_rainfall_data(filename='rainfall_data.csv', years=5):
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=365*years)
    end_date = datetime.now()
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)  
    
    base_temp = 20 + 15 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    temperature = base_temp + np.random.normal(0, 3, n_days)
    
    base_humidity = 50 + 30 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi/2)
    humidity = np.clip(base_humidity + np.random.normal(0, 10, n_days), 20, 100)
    
    pressure = 1015 + np.random.normal(0, 5, n_days)
    
    base_wind = 5 + 5 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi)
    wind_speed = np.clip(base_wind + np.random.exponential(2, n_days), 0, 30)
    
    month = dates.month.values
    rain_prob = np.where((month >= 4) & (month <= 9), 0.3, 0.2)
    rainfall = np.zeros(n_days)
    
    for i in range(n_days):
        if np.random.random() < rain_prob[i]:
            rainfall[i] = np.random.exponential(5)
    
    data = pd.DataFrame({
        'date': dates,
        'temperature': np.round(temperature, 1),
        'humidity': np.round(humidity, 1),
        'pressure': np.round(pressure, 1),
        'wind_speed': np.round(wind_speed, 1),
        'precipitation': np.round(rainfall, 1)
    })
    
    data.to_csv(filename, index=False)
    return data

def load_sample_data(filename='rainfall_data.csv'):
    """
    Load rainfall data from the provided CSV file.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        DataFrame with rainfall data
    """
    try:
        # Load data from CSV
        data = pd.read_csv(filename)
        
        # Convert date column to datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            
            # Check if dates are unreasonably old (before 2000)
            if data['date'].min().year < 2000:
                st.warning("Data contains very old dates. This may be due to incorrect date formatting.")
                
                # Option 1: Set index but don't modify dates
                data.set_index('date', inplace=True)
                
                # Option 2 (commented out): Replace with more recent dates
                # current_year = datetime.now().year
                # start_date = datetime(current_year - 5, 1, 1)
                # data['date'] = pd.date_range(start=start_date, periods=len(data), freq='D')
                # data.set_index('date', inplace=True)
            else:
                # Set the date column as index
                data.set_index('date', inplace=True)
        
        return data
    except Exception as e:
        st.error(f"Error loading data from {filename}: {str(e)}")
        return None

def load_or_generate_rainfall_data(filename='rainfall_data.csv', years=5):
    """
    Load rainfall data from CSV file if it exists, otherwise generate synthetic data.
    
    Args:
        filename: Path to the CSV file
        years: Number of years of data to generate if file doesn't exist
        
    Returns:
        DataFrame with rainfall data
    """
    # Check if file exists
    if os.path.exists(filename):
        # Load data from CSV
        data = pd.read_csv(filename)
        # Convert date column to datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        return data
    else:
        # If file doesn't exist, generate synthetic data (as before)
        st.warning(f"CSV file {filename} not found. Generating synthetic data instead.")
        return generate_rainfall_data(filename, years)

def load_dataset(filename):
    # Add column names when reading the CSV since it doesn't have headers
    data = pd.read_csv(filename, names=['date', 'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation'])
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data.set_index('date', inplace=True)
    return data

def create_features(data):
    """
    Create features for the model from the raw data.
    
    Args:
        data: DataFrame with raw data
        
    Returns:
        DataFrame with features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Define column mapping to remove units for processing
    column_mapping = {
        'temperature (°C)': 'temperature',
        'humidity (%)': 'humidity',
        'pressure (hPa)': 'pressure',
        'wind_speed (m/s)': 'wind_speed',
        'precipitation (mm)': 'precipitation'
    }
    
    # Remove units from column names for processing
    for col_with_unit, col in column_mapping.items():
        if col_with_unit in df.columns:
            df = df.rename(columns={col_with_unit: col})
    
    # Ensure all columns are numeric where appropriate
    for col in df.columns:
        if col not in ['date'] and not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    return df

def preprocess_data(data):
    # Define features dynamically based on what's available
    potential_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 
                         'day_of_year', 'month', 'season_code']
    # Removed lag and rolling features
    
    # Only use features that exist in the data
    feature_cols = [col for col in potential_features if col in data.columns]
    
    # If no features are available, add some basic ones
    if not feature_cols:
        data['day_of_year'] = 1
        data['month'] = 1
        feature_cols = ['day_of_year', 'month']
    
    # Make sure all feature columns are numeric
    for col in feature_cols:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        except:
            st.warning(f"Could not convert {col} to numeric. Using zeros.")
            data[col] = 0
    
    X = data[feature_cols]
    
    # Use precipitation as target if available, otherwise create a dummy target
    if 'precipitation' in data.columns:
        try:
            data['precipitation'] = pd.to_numeric(data['precipitation'], errors='coerce')
            y = data['precipitation']
        except:
            st.warning("Could not convert precipitation to numeric. Using random values.")
            y = pd.Series(np.random.normal(5, 2, len(data)), index=data.index)
            data['precipitation'] = y
    else:
        y = pd.Series(np.random.normal(5, 2, len(data)), index=data.index)
        data['precipitation'] = y
    
    # Handle missing values in X using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # Handle missing values in y
    y_imputed = y.fillna(y.mean() if not y.isna().all() else 0)
    
    # Check if there are still NaN values in y
    if y_imputed.isna().any():
        st.warning("Could not impute all missing values in precipitation data. Using zeros for remaining NaNs.")
        y_imputed = y_imputed.fillna(0)
    
    # Split data - use a smaller test set if data is limited
    test_size = min(0.2, max(1/len(X), 0.1)) if len(X) > 10 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, imputer

def train_ml_models(X_train, y_train):
    st.write("Training machine learning models...")
    
    # Adjust model complexity based on data size
    n_estimators = min(100, max(10, len(X_train) // 2))
    
    # Create more robust models with better hyperparameters
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=min(10, len(X_train) // 10),
            min_samples_leaf=3,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42
        )
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def train_arima_model(y_train):
    st.write("Training ARIMA model...")
    try:
        # Use simpler ARIMA model for small datasets
        if len(y_train) < 30:
            model = ARIMA(y_train, order=(1,0,0))
        else:
            model = ARIMA(y_train, order=(5,1,0))
        model_fit = model.fit()
        return model_fit
    except:
        # If ARIMA fails, return a simple model that just predicts the mean
        class SimpleMeanModel:
            def __init__(self, y):
                self.mean = y.mean()
            def forecast(self, steps=1):
                return np.array([self.mean] * steps)
        return SimpleMeanModel(y_train)

def evaluate_model_and_show_importance(model, X_test, y_test, feature_names, scaler):
    """
    Evaluate the model and show feature importance.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Add CSS to ensure metric text is black
    st.markdown("""
    <style>
        /* Make metric values black */
        [data-testid="stMetricValue"] {
            color: black !important;
            font-weight: bold !important;
        }
        
        /* Make metric labels black */
        [data-testid="stMetricLabel"] {
            color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display metrics using standard Streamlit components
    st.subheader("Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} mm")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} mm")
    col3.metric("R² Score", f"{r2:.2f}")
    
    # Only show feature importance if the model supports it
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Filter to only include the specified weather parameters
        weather_params = ['temperature', 'humidity', 'pressure', 'wind_speed']
        importance_df = importance_df[importance_df['Feature'].isin(weather_params + ['day_of_year', 'month', 'season_code'])]
        
        # Get top features
        top_features = importance_df.head(10)
        
        # Create interactive Plotly bar chart
        fig = px.bar(
            top_features,
            y='Feature',
            x='Importance',
            orientation='h',
            title='Weather Parameter Importance for Rainfall Prediction',
            color_discrete_sequence=['#110361'],
            labels={'Importance': 'Importance Score', 'Feature': 'Weather Parameter'},
            text='Importance'
        )
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Weather Parameter",
            font=dict(size=12)
        )
        
        # Display the interactive plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display importance table
        st.dataframe(importance_df)
        
        # Calculate parameter importance using correlation method
        weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        
        # Create a DataFrame with features and target for correlation calculation
# Add global CSS to ensure all tables have light blue background and dark text
st.markdown("""
<style>
    /* Global styling for all dataframes */
    .dataframe {
        background-color: #e6f2ff !important; /* Light blue background */
    }
    
    .dataframe th {
        background-color: #b3d9ff !important; /* Slightly darker blue for headers */
        color: #000000 !important; /* Black text for headers */
        font-weight: bold !important;
    }
    
    .dataframe td {
        background-color: #e6f2ff !important; /* Light blue background */
        color: #000000 !important; /* Black text for cells */
    }
    
    /* Ensure text in all tables is dark */
    table {
        color: #000000 !important;
    }
    
    /* Streamlit-specific selectors */
    [data-testid="stDataFrame"] table {
        background-color: #e6f2ff !important; /* Light blue background */
    }
    
    [data-testid="stDataFrame"] th {
        background-color: #b3d9ff !important; /* Slightly darker blue for headers */
        color: #000000 !important; /* Black text for headers */
    }
    
    [data-testid="stDataFrame"] td {
        background-color: #e6f2ff !important; /* Light blue background */
        color: #000000 !important; /* Black text for cells */
    }
    
    /* Override Streamlit's default styling */
    [data-testid="stDataFrame"] [data-testid="dataframe-container"] {
        background-color: #e6f2ff !important;
    }
    
    /* Target the table container */
    .stDataFrame > div > div > div {
        background-color: #e6f2ff !important;
    }
    
    /* Target the table rows */
    .stDataFrame tr {
        background-color: #e6f2ff !important;
    }
    
    /* Target the table cells */
    .stDataFrame td, .stDataFrame th {
        background-color: #e6f2ff !important;
        color: #000000 !important;
    }
    
    /* Target the table header */
    .stDataFrame thead tr th {
        background-color: #b3d9ff !important;
        color: #000000 !important;
    }
    
    /* Target the table body */
    .stDataFrame tbody tr td {
        background-color: #e6f2ff !important;
        color: #000000 !important;
    }
    
    /* Target alternating rows */
    .stDataFrame tbody tr:nth-child(even) td {
        background-color: #d9ecff !important; /* Slightly different blue for alternating rows */
    }
    
    /* Target the scrollbar area */
    .stDataFrame [data-testid="dataframe-container"] > div {
        background-color: #e6f2ff !important;
    }
    
    /* Force override any inline styles */
    .stDataFrame [style*="background"] {
        background-color: #e6f2ff !important;
    }
    
    .stDataFrame th[style*="background"] {
        background-color: #b3d9ff !important;
    }
    
    /* Override any text colors */
    .stDataFrame [style*="color"] {
        color: #000000 !important;
    }
    
    /* Additional selectors for Streamlit's newer versions */
    .st-emotion-cache-1n76uvr {
        background-color: #e6f2ff !important;
    }
    
    .st-emotion-cache-1n76uvr table {
        background-color: #e6f2ff !important;
    }
    
    .st-emotion-cache-1n76uvr th {
        background-color: #b3d9ff !important;
        color: #000000 !important;
    }
    
    .st-emotion-cache-1n76uvr td {
        background-color: #e6f2ff !important;
        color: #000000 !important;
    }
    
    /* Target the table container in newer Streamlit versions */
    div[data-testid="stTable"] {
        background-color: #e6f2ff !important;
    }
    
    div[data-testid="stTable"] table {
        background-color: #e6f2ff !important;
    }
    
    div[data-testid="stTable"] th {
        background-color: #b3d9ff !important;
        color: #000000 !important;
    }
    
    div[data-testid="stTable"] td {
        background-color: #e6f2ff !important;
        color: #000000 !important;
    }
    
    /* Target any element with table styling */
    *[class*="table"], *[class*="Table"], *[class*="dataframe"], *[class*="Dataframe"] {
        background-color: #e6f2ff !important;
    }
    
    *[class*="table"] th, *[class*="Table"] th, *[class*="dataframe"] th, *[class*="Dataframe"] th {
        background-color: #b3d9ff !important;
        color: #000000 !important;
    }
    
    *[class*="table"] td, *[class*="Table"] td, *[class*="dataframe"] td, *[class*="Dataframe"] td {
        background-color: #e6f2ff !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

def visualize_all_models_predictions(models, X_test, y_test, arima_model=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(y_test.values, label='Actual', linewidth=2, color='black')
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        ax.plot(y_pred, label=name)
    
    if arima_model:
        y_pred_arima = arima_model.forecast(steps=len(y_test))
        ax.plot(y_pred_arima, label='ARIMA')
    
    ax.legend()
    ax.set_title("Model Predictions vs Actual Precipitation")
    st.pyplot(fig)

def evaluate_models(ml_models, X_test, y_test, arima_model=None):
    """
    Evaluate all models and return a DataFrame with their performance metrics.
    """
    results = []
    for name, model in ml_models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    # ARIMA
    if arima_model is not None:
        try:
            y_pred_arima = arima_model.forecast(steps=len(y_test))
            if hasattr(y_pred_arima, 'values'):
                y_pred_arima = y_pred_arima.values
            mae = mean_absolute_error(y_test, y_pred_arima)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_arima))
            r2 = r2_score(y_test, y_pred_arima)
            results.append({'Model': 'ARIMA', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        except Exception as e:
            results.append({'Model': 'ARIMA', 'MAE': None, 'RMSE': None, 'R2': None})
    
    # Create DataFrame and add model descriptions
    df = pd.DataFrame(results).set_index('Model')
    
    # Add model descriptions
    model_descriptions = {

        'Linear Regression': 'Simple linear model',
        'Random Forest': 'Ensemble of decision trees',
        'Gradient Boosting': 'Sequential ensemble method',
        'ARIMA': 'Time series forecasting'
    }
    
    # Format metrics to 2 decimal places and add units
    for col in ['MAE', 'RMSE', 'R2']:
        if col in df.columns:
            if col == 'MAE':
                df[col] = df[col].apply(lambda x: f"{x:.2f} mm" if pd.notnull(x) else "N/A")
            elif col == 'RMSE':
                df[col] = df[col].apply(lambda x: f"{x:.2f} mm" if pd.notnull(x) else "N/A")
            elif col == 'R2':
                df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    
    return df

# Function to fetch weather data from WeatherAPI
def fetch_weather_data_from_api(location, start_date, end_date, api_key, show_debug=False):
    """
    Fetch weather data from WeatherAPI for the given location and date range.
    Returns a DataFrame with columns: date, temperature, humidity, pressure, wind_speed, precipitation.
    """
    try:
        if not isinstance(end_date, str):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # Get current hour for consistent time-of-day forecasting
        current_hour = datetime.now().hour
        
        # WeatherAPI only allows fetching one day at a time for history endpoint
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        records = []
        api_errors = []
        
        # Debug information
        if show_debug:
            st.info(f"Attempting to fetch data for {location} from {start_date} to {end_date} using API key: {api_key[:5]}...")
        
        for single_date in date_range:
            date_str = single_date.strftime('%Y-%m-%d')
            # Add hour parameter to get data for the same hour of day
            url = (
                f"https://api.weatherapi.com/v1/history.json?key={api_key}"
                f"&q={location}&dt={date_str}&hour={current_hour}"
            )
            
            # Debug URL (hide full API key)
            if show_debug:
                masked_url = url.replace(api_key, f"{api_key[:5]}...")
                st.write(f"Fetching: {masked_url}")
            
            response = requests.get(url)
            if show_debug:
                st.write(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Debug response structure
                if show_debug:
                    st.write(f"Response keys: {list(data.keys())}")
                
                # Extract the day's data - check both possible response structures
                if 'forecast' in data and 'forecastday' in data['forecast'] and len(data['forecast']['forecastday']) > 0:
                    day_data = data['forecast']['forecastday'][0]['day']
                    
                    # Debug day data
                    if show_debug:
                        st.write(f"Day data keys: {list(day_data.keys())}")
                    
                    records.append({
                        'date': date_str,
                        'temperature': day_data.get('avgtemp_c', 25),
                        'humidity': day_data.get('avghumidity', 70),
                        'pressure': 1013,  # WeatherAPI doesn't provide pressure directly
                        'wind_speed': day_data.get('maxwind_kph', 10),
                        'precipitation': day_data.get('totalprecip_mm', 0)
                    })
                    
                    # Show successful data point
                    if show_debug:
                        st.success(f"Successfully retrieved data for {date_str}")
                else:
                    # If no data for this day, log the issue
                    error_msg = f"No forecast data found for {date_str}"
                    api_errors.append(error_msg)
                    if show_debug:
                        st.warning(error_msg)
                    
                    # Still add a placeholder to maintain continuity
                    records.append({
                        'date': date_str,
                        'temperature': 25,
                        'humidity': 70,
                        'pressure': 1013,
                        'wind_speed': 5,
                        'precipitation': 0
                    })
            else:
                # If API call fails, log the error
                error_msg = f"API error for {date_str}: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                api_errors.append(error_msg)
                if show_debug:
                    st.error(error_msg)
                
                # Add placeholder to maintain continuity
                records.append({
                    'date': date_str,
                    'temperature': 25,
                    'humidity': 70,
                    'pressure': 1013,
                    'wind_speed': 5,
                    'precipitation': 0
                })
        
        # Create DataFrame from records
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        
        # If we had errors, return them along with the data
        if api_errors:
            return df, f"Some data could not be retrieved: {'; '.join(api_errors[:3])}{'...' if len(api_errors) > 3 else ''}"
        return df, None
    except Exception as e:
        st.exception(e)  # Show full exception details
        return None, f"Error fetching weather data: {str(e)}"

# Function to fetch weather data from Open-Meteo API
def fetch_open_meteo_data(location, start_date, end_date):
    """
    Fetch weather data from Open-Meteo API for the given location and date range.
    Returns a DataFrame with columns: date, temperature, humidity, pressure, wind_speed, precipitation.
    """
    try:
        # Convert dates to strings if they're not already
        if not isinstance(start_date, str):
            start_date = start_date.strftime('%Y-%m-%d')
        if not isinstance(end_date, str):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # Get current hour for consistent time-of-day forecasting
        current_hour = datetime.now().hour
        
        # Geocode the location (convert city name to lat/lon)
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_response = requests.get(geocoding_url)
        
        if geo_response.status_code != 200 or not geo_response.json().get('results'):
            return None, f"Geocoding failed for location: {location}"
            
        lat = geo_response.json()['results'][0]['latitude']
        lon = geo_response.json()['results'][0]['longitude']
        
        # Build Open-Meteo API URL with hourly=true parameter
        api_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relativehumidity_2m,precipitation,pressure_msl,windspeed_10m"
        )
        
        response = requests.get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract hourly data
            hourly_data = {
                'date': data['hourly']['time'],
                'temperature': data['hourly']['temperature_2m'],
                'humidity': data['hourly']['relativehumidity_2m'],
                'pressure': data['hourly']['pressure_msl'],
                'wind_speed': data['hourly']['windspeed_10m'],
                'precipitation': data['hourly']['precipitation']
            }
            
            # Convert to DataFrame
            df = pd.DataFrame(hourly_data)
            
            # Convert date strings to datetime objects
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to only include data for the current hour
            df['hour'] = df['date'].dt.hour
            df_filtered = df[df['hour'] == current_hour].copy()
            
            # Format date column to just show the date
            df_filtered['date'] = df_filtered['date'].dt.date
            df_filtered = df_filtered.drop('hour', axis=1)
            
            return df_filtered, None
    except Exception as e:
        return None, f"Exception: {e}"

# Utility function to add units to column names
def add_units_to_columns(df):
    """
    Add units to column names for display purposes.
    """
    col_units = {
        'temperature': 'temperature (°C)',
        'humidity': 'humidity (%)',
        'pressure': 'pressure (hPa)',
        'wind_speed': 'wind_speed (m/s)',
        'precipitation': 'precipitation (mm)'
    }
    df = df.copy()
    for col, col_with_unit in col_units.items():
        if col in df.columns and col_with_unit not in df.columns:
            df = df.rename(columns={col: col_with_unit})
    return df

# Process uploaded data function
def process_uploaded_data(uploaded_file):
    """
    Process uploaded data file and add units to column names.
    
    Args:
        uploaded_file: File uploaded by the user
        
    Returns:
        Tuple of (data, error_message)
    """
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Read file based on extension
        if file_extension == 'csv':
            # Try different encodings and delimiters
            try:
                data = pd.read_csv(uploaded_file)
            except:
                try:
                    data = pd.read_csv(uploaded_file, sep=';')
                except:
                    try:
                        data = pd.read_csv(uploaded_file, encoding='latin1')
                    except:
                        data = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python')
        elif file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            data = pd.read_json(uploaded_file)
        elif file_extension == 'txt':
            # Try to detect delimiter for text files
            try:
                data = pd.read_csv(uploaded_file, sep=None, engine='python')
            except:
                try:
                    data = pd.read_csv(uploaded_file, sep='\t')
                except:
                    data = pd.read_fwf(uploaded_file)  # Fixed width format as last resort
        elif file_extension == 'pdf':
            # Try to extract tables from PDF
            try:
                import tabula
                # Save the uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract tables from the PDF
                tables = tabula.read_pdf("temp.pdf", pages='all', multiple_tables=True)
                
                if not tables:
                    return None, "No tables found in the PDF file."
                
                # Use the first table found
                data = tables[0]
                
                # If multiple tables were found, show a message
                if len(tables) > 1:
                    st.info(f"Found {len(tables)} tables in the PDF. Using the first table. Upload individual pages for other tables.")
                
                # Clean up the temporary file
                import os
                os.remove("temp.pdf")
                
            except ImportError:
                # If tabula-py is not installed, try PyPDF2
                try:
                    import PyPDF2
                    import re
                    
                    # Save the uploaded file temporarily
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text from the PDF
                    pdf_file = open("temp.pdf", 'rb')
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    # Extract text from all pages
                    text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        text += pdf_reader.pages[page_num].extract_text()
                    
                    pdf_file.close()
                    
                    # Clean up the temporary file
                    import os
                    os.remove("temp.pdf")
                    
                    # Try to parse the text into a table structure
                    lines = text.split('\n')
                    
                    # Assume the first line contains headers
                    if not lines:
                        return None, "No text content found in the PDF."
                    
                    headers = re.split(r'\s{2,}', lines[0].strip())
                    
                    # Process the remaining lines as data
                    rows = []
                    for line in lines[1:]:
                        if line.strip():  # Skip empty lines
                            # Split by multiple spaces
                            row = re.split(r'\s{2,}', line.strip())
                            rows.append(row)
                    
                    # Create DataFrame
                    if rows:
                        # Ensure all rows have the same number of columns as headers
                        max_cols = max(len(headers), max(len(row) for row in rows))
                        
                        # Pad headers if needed
                        if len(headers) < max_cols:
                            headers.extend([f'Column{i+1}' for i in range(len(headers), max_cols)])
                        
                        # Pad rows if needed
                        for row in rows:
                            if len(row) < max_cols:
                                row.extend([''] * (max_cols - len(row)))
                        
                        data = pd.DataFrame(rows, columns=headers)
                    else:
                        return None, "Could not parse table structure from PDF."
                    
                except ImportError:
                    return None, "PDF processing libraries (tabula-py or PyPDF2) not installed. Please install them to process PDF files."
                except Exception as e:
                    return None, f"Error extracting data from PDF: {str(e)}"
        else:
            return None, f"Unsupported file format: {file_extension}. Please use CSV, Excel, JSON, TXT, or PDF files."
        
        # Check if data is empty
        if data.empty:
            return None, "Uploaded file contains no data"
        
        # Add units to column names
        data = add_units_to_columns(data)
        
        # Display raw data for debugging
        st.write("### Raw Uploaded Data:")
        st.dataframe(data, use_container_width=True)
        
        # Try to identify date column
        date_col = None
        for col in data.columns:
            if any(term in col.lower() for term in ['date', 'time', 'day', 'year', 'month']):
                date_col = col
                break
        
        if date_col is None:
            # If no date column found, use the first column if it looks like dates
            first_col = data.columns[0]
            try:
                pd.to_datetime(data[first_col], errors='raise')
                date_col = first_col
            except:
                # Create a date column if none exists
                st.warning("No date column found. Creating a synthetic date column.")
                data['date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
                date_col = 'date'
        
        # Convert date column to datetime with flexible format detection
        try:
            # First try with format='mixed' and dayfirst=True to handle various formats
            data[date_col] = pd.to_datetime(data[date_col], format='mixed', dayfirst=True, errors='coerce')
            
            # Set the date column as index
            data = data.set_index(date_col)
            
            # Sort the index to ensure chronological order
            data = data.sort_index()
            
        except Exception as e:
            st.error(f"Error converting date column: {str(e)}")
            return None, f"Failed to convert date column: {str(e)}"
        
        # Try to identify precipitation column
        precip_col = None
        for col in data.columns:
            if any(term in col.lower() for term in ['precip', 'rain', 'rainfall']):
                precip_col = col
                break
        
        if precip_col is None:
            # If no precipitation column found, use the first numeric column
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                precip_col = numeric_cols[0]
                data = data.rename(columns={precip_col: 'precipitation'})
                st.info(f"Using '{precip_col}' as precipitation data.")
            else:
                # Try to convert columns to numeric
                for col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except:
                        pass
                
                # Check again for numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    precip_col = numeric_cols[0]
                    data = data.rename(columns={precip_col: 'precipitation'})
                    st.info(f"Using '{precip_col}' as precipitation data.")
                else:
                    return None, "No numeric columns found for precipitation"
        else:
            # Rename the identified precipitation column
            data = data.rename(columns={precip_col: 'precipitation'})
            st.success(f"Using '{precip_col}' as precipitation data.")
        
        # Convert all remaining columns to numeric if possible
        for col in data.columns:
            if col != 'precipitation':  # Skip the column we already processed
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    pass
        
        # Check for other required columns
        required_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for col in required_cols:
            if not any(col in c.lower() for c in data.columns):
                # Generate synthetic data for missing columns
                data[col] = np.random.normal(
                    loc={'temperature': 25, 'humidity': 70, 'pressure': 1013, 'wind_speed': 10}[col],
                    scale={'temperature': 5, 'humidity': 10, 'pressure': 5, 'wind_speed': 3}[col],
                    size=len(data)
                )
                st.info(f"Generated synthetic '{col}' data.")
            else:
                matched_col = next(c for c in data.columns if col in c.lower())
                if matched_col != col:
                    data = data.rename(columns={matched_col: col})
                    st.success(f"Using '{matched_col}' as {col} data.")
        
        return data, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

# Function to generate future forecast
def generate_forecast(data, model, periods=30, time_granularity='D', forecast_unit='days', forecast_days=30, scaler=None, feature_cols=None, imputer=None):
    """
    Generate forecast for future periods using the trained model.
    """
    # Convert forecast_days to periods based on forecast_unit
    if forecast_unit.lower() == 'days':
        periods = forecast_days
    elif forecast_unit.lower() == 'weeks':
        periods = forecast_days * 7
    elif forecast_unit.lower() == 'months':
        periods = forecast_days * 30
    elif forecast_unit.lower() == 'years':
        periods = forecast_days * 365
    else:
        periods = forecast_days
    
    # Create future dates - ensure continuous dates from the last date in the data
    if isinstance(data.index, pd.DatetimeIndex):
        last_date = data.index[-1]
    else:
        # If index is not datetime, try to find a date column
        if 'date' in data.columns:
            last_date = pd.to_datetime(data['date']).max()
        else:
            last_date = pd.to_datetime('today')
    
    # Ensure last_date is a datetime object
    if not isinstance(last_date, (pd.Timestamp, datetime)):
        try:
            last_date = pd.to_datetime(last_date)
        except:
            last_date = pd.to_datetime('today')
    
    # Create continuous date range starting from the day after the last date
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), 
        periods=periods, 
        freq=time_granularity
    )
    
    # Create a DataFrame for future dates
    future_df = pd.DataFrame(index=future_dates)
    future_df['date'] = future_dates
    
    # Extract date features
    future_df['day_of_year'] = future_df['date'].dt.dayofyear
    future_df['month'] = future_df['date'].dt.month
    
    # Add season code
    def get_season_code(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Summer
        elif month in [6, 7, 8, 9]:
            return 2  # Monsoon
        else:
            return 3  # Post-Monsoon
    
    future_df['season_code'] = future_df['month'].apply(get_season_code)
    
    # Calculate average values from historical data
    avg_values = {}
    for col in ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']:
        if col in data.columns and not data[col].isna().all():
            avg_values[col] = data[col].mean()
        else:
            avg_values[col] = {'temperature': 25, 'humidity': 70, 'pressure': 1013, 'wind_speed': 10, 'precipitation': 5}[col]
    
    # Generate synthetic data for all weather parameters
    # This ensures we have complete data for visualization
    
    # Temperature varies by season
    month_temp_factors = {
        1: 0.7,   # January - cooler
        2: 0.8,   # February - cooler
        3: 0.9,   # March - warming
        4: 1.0,   # April - warming
        5: 1.1,   # May - warm
        6: 1.2,   # June - hot
        7: 1.1,   # July - hot
        8: 1.1,   # August - hot
        9: 1.0,   # September - cooling
        10: 0.9,  # October - cooling
        11: 0.8,  # November - cooler
        12: 0.7   # December - cooler
    }
    
    future_df['temperature'] = future_df['month'].apply(
        lambda m: avg_values['temperature'] * month_temp_factors[m] * (1 + np.random.normal(0, 0.1))
    )
    
    # Humidity varies by season
    month_humidity_factors = {
        1: 0.8,   # January - drier
        2: 0.7,   # February - drier
        3: 0.7,   # March - drier
        4: 0.8,   # April - getting humid
        5: 0.9,   # May - getting humid
        6: 1.1,   # June - monsoon
        7: 1.2,   # July - monsoon
        8: 1.2,   # August - monsoon
        9: 1.1,   # September - monsoon ending
        10: 1.0,  # October - post-monsoon
        11: 0.9,  # November - drying
        12: 0.8   # December - drier
    }
    
    future_df['humidity'] = future_df['month'].apply(
        lambda m: min(95, max(30, avg_values['humidity'] * month_humidity_factors[m] * (1 + np.random.normal(0, 0.1))))
    )
    
    # Pressure varies less
    future_df['pressure'] = avg_values['pressure'] + np.random.normal(0, 5, size=len(future_df))
    
    # Wind speed varies somewhat randomly
    future_df['wind_speed'] = np.maximum(0, avg_values['wind_speed'] + np.random.normal(0, 2, size=len(future_df)))
    
    # Initialize precipitation column
    future_df['precipitation'] = 0.0
    
    # Check if model is ARIMA
    if hasattr(model, 'forecast'):
        # For ARIMA, we need to forecast directly
        try:
            forecast_result = model.forecast(steps=periods)
            future_df['precipitation'] = forecast_result
        except Exception as e:
            st.warning(f"ARIMA forecast failed: {str(e)}. Using seasonal patterns instead.")
            # Use seasonal patterns for precipitation if ARIMA fails
            month_precip_factors = {
                1: 0.3,   # January - dry
                2: 0.2,   # February - dry
                3: 0.3,   # March - dry
                4: 0.4,   # April - some rain
                5: 0.6,   # May - pre-monsoon
                6: 1.2,   # June - monsoon
                7: 1.5,   # July - peak monsoon
                8: 1.4,   # August - monsoon
                9: 1.0,   # September - monsoon ending
                10: 0.7,  # October - post-monsoon
                11: 0.4,  # November - drying
                12: 0.3   # December - dry 
            }
            
            future_df['precipitation'] = future_df['month'].apply(
                lambda m: max(0, avg_values['precipitation'] * month_precip_factors[m] * 
                             (1 + np.random.normal(0, 0.3)))  # Add some randomness
            )
    else:
        # For ML models, we need to prepare the features
        # Use only the feature columns that were used during training
        if feature_cols is None:
            # If feature_cols is not provided, try to infer from model
            if hasattr(model, 'feature_names_in_'):
                feature_cols = model.feature_names_in_
            else:
                # Default feature columns if we can't determine from model
                feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'day_of_year', 'month']
        
        # Ensure all required features are present in future_df
        for col in feature_cols:
            if col not in future_df.columns:
                if col in avg_values:
                    future_df[col] = avg_values[col]
                else:
                    # For unknown features, use zeros
                    future_df[col] = 0
        
        # Select only the features used during training
        X_future = future_df[feature_cols].copy()
        
        # Handle missing values if imputer is provided
        if imputer is not None:
            X_future = pd.DataFrame(imputer.transform(X_future), columns=X_future.columns, index=X_future.index)
        
        # Scale features if scaler is provided
        if scaler is not None:
            X_future_scaled = scaler.transform(X_future)
        else:
            X_future_scaled = X_future
        
        # Make predictions
        try:
            predictions = model.predict(X_future_scaled)
            
            # Apply post-processing to predictions to make them more realistic
            # 1. Ensure no negative values
            predictions = np.maximum(predictions, 0)
            
            # 2. Apply smoothing to avoid extreme jumps
            if len(predictions) > 1:
                smoothed_predictions = np.copy(predictions)
                for i in range(1, len(predictions)):
                    smoothed_predictions[i] = 0.7 * predictions[i] + 0.3 * smoothed_predictions[i-1]
                predictions = smoothed_predictions
            
            # 3. Scale predictions to be in a reasonable range
            if np.mean(predictions) > 3 * avg_values['precipitation']:
                predictions = predictions * (avg_values['precipitation'] / np.mean(predictions)) * 2
            
            future_df['precipitation'] = predictions
            
        except Exception as e:
            st.warning(f"Prediction failed: {str(e)}. Using seasonal patterns instead.")
            # Use seasonal patterns for precipitation if prediction fails
            month_precip_factors = {
                1: 0.3,   # January - dry
                2: 0.2,   # February - dry
                3: 0.3,   # March - dry
                4: 0.4,   # April - some rain
                5: 0.6,   # May - pre-monsoon
                6: 1.2,   # June - monsoon
                7: 1.5,   # July - peak monsoon
                8: 1.4,   # August - monsoon
                9: 1.0,   # September - monsoon ending
                10: 0.7,  # October - post-monsoon
                11: 0.4,  # November - drying
                12: 0.3   # December - dry
            }
            
            future_df['precipitation'] = future_df['month'].apply(
                lambda m: max(0, avg_values['precipitation'] * month_precip_factors[m] * 
                             (1 + np.random.normal(0, 0.3)))  # Add some randomness
            )
    
    # Ensure no negative precipitation
    future_df['precipitation'] = future_df['precipitation'].clip(lower=0)
    
    # Final check to ensure no None/NaN values in any column
    for col in future_df.columns:
        if future_df[col].isna().any():
            if col in avg_values:
                future_df[col] = future_df[col].fillna(avg_values[col])
            else:
                future_df[col] = future_df[col].fillna(0)
    
    # Round numeric columns to improve display
    numeric_cols = future_df.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        future_df[col] = future_df[col].round(2)
    
    # Ensure the date column is preserved
    if 'date' not in future_df.columns and isinstance(future_df.index, pd.DatetimeIndex):
        future_df['date'] = future_df.index
    
    # Debug information
    st.write("Generated forecast data shape:", future_df.shape)
    st.write("First few rows of forecast data:")
    st.write(future_df.head())
    
    return future_df

def plot_historical_and_forecast(historical_data, forecast_data, column_name, display_name=None, title=None):
    """
    Plot historical data and forecast together with no gaps.
    """
    if display_name is None:
        display_name = column_name
    
    if title is None:
        title = f"{display_name} Historical Data and Forecast"
    
    # Debug information
    # st.write(f"Historical data shape: {historical_data.shape}")
    # st.write(f"Historical data columns: {historical_data.columns.tolist()}")
    # st.write(f"Forecast data shape: {forecast_data.shape}")
    # st.write(f"Forecast data columns: {forecast_data.columns.tolist()}")
    
    # Create figure directly with plotly
    fig = go.Figure()
    
    # Prepare historical data for plotting
    hist_df = historical_data.copy()
    
    # CRITICAL: Ensure historical data has a date column
    if 'date' not in hist_df.columns:
        ###warning("No 'date' column found in historical data. Creating synthetic dates.")
        hist_df['date'] = pd.date_range(
            end=pd.Timestamp.today() - pd.Timedelta(days=1),
            periods=len(hist_df),
            freq='D'
        )
    
    # Ensure date column is datetime type
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_dates = hist_df['date']
    
    # Prepare forecast data for plotting
    fore_df = forecast_data.copy()
    
    # CRITICAL: Ensure forecast data has a date column
    if 'date' not in fore_df.columns:
        if isinstance(fore_df.index, pd.DatetimeIndex):
            fore_df = fore_df.reset_index()
            fore_df.rename(columns={'index': 'date'}, inplace=True)
        else:
            st.warning("No date information found in forecast data.")
            return
    
    # Ensure date column is datetime type
    fore_df['date'] = pd.to_datetime(fore_df['date'])
    fore_dates = fore_df['date']
    
    # Get historical values
    if column_name in hist_df.columns:
        hist_values = hist_df[column_name]
    else:
        st.error(f"Column '{column_name}' not found in historical data")
        return
    
    # Get forecast values
    if column_name in fore_df.columns:
        fore_values = fore_df[column_name]
    else:
        st.error(f"Column '{column_name}' not found in forecast data")
        return
    
    # Add historical trace
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_values,
        name='Historical Data',
        line=dict(color='blue'),
        mode='lines'
    ))
    
    # Add forecast trace
    fig.add_trace(go.Scatter(
        x=fore_dates,
        y=fore_values,
        name='Forecast',
        line=dict(color='red', dash='dash'),
        mode='lines'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=display_name,
        legend_title='Data Type',
        hovermode='x unified'
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Debug: Show the first and last few rows of each dataset
    with st.expander("Debug Data"):
        st.write("Historical Data (first 5 rows):")
        st.write(hist_df.head())
        st.write("Historical Data (last 5 rows):")
        st.write(hist_df.tail())
        st.write("Forecast Data (first 5 rows):")
        st.write(fore_df.head())
        st.write("Forecast Data (last 5 rows):")
        st.write(fore_df.tail())

def train_ml_models(X_train, y_train):
    st.write("Training machine learning models...")
    
    # Adjust model complexity based on data size
    n_estimators = min(100, max(10, len(X_train) // 2))
    
    # Calculate max_depth ensuring it's at least 1
    max_depth_value = max(1, min(10, len(X_train) // 10))
    
    # Create more robust models with better hyperparameters
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth_value,  # Ensure this is at least 1
            min_samples_leaf=3,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,  # Fixed value that's always valid
            min_samples_leaf=3,
            random_state=42
        )
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def main():
    # Display current time information
    current_time = datetime.now()
    st.sidebar.write(f"**Current Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.write(f"**Forecasts will use data for:** {current_time.strftime('%H:00')} each day")
    
    # Add sidebar for input method selection
    input_method = st.sidebar.radio(
        "Select Input Method",
        ["Use Sample Data", "Use Weather API", "Upload Your Own Data"]
    )
    
    # Add sidebar for time period selection with white text
    st.sidebar.markdown("""
        <div style="margin: 20px 0 10px 0; padding: 0;">
            <h4 style="color: white !important; font-weight: 600 !important; margin: 0; padding: 0; font-size: 1.1rem;">
                Time Period Configuration
            </h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Use session state to track changes in forecast unit
    if 'prev_forecast_unit' not in st.session_state:
        st.session_state.prev_forecast_unit = "Days"
    
    forecast_unit = st.sidebar.selectbox(
        "Forecast Period Unit",
        ["Days", "Weeks", "Months", "Years"],
        index=0,  # Default to Days
        label_visibility="collapsed",  # Hide the default label
    )
    
    # Clear session state data when switching input methods
    if 'prev_input_method' not in st.session_state:
        st.session_state.prev_input_method = input_method
    
    # Check if input method has changed
    if st.session_state.prev_input_method != input_method:
        # Clear data when switching between input methods
        if 'data' in st.session_state:
            del st.session_state.data
        st.session_state.prev_input_method = input_method
    
    # Initialize data variable
    data = None
    
    # Load or generate data based on input method
    if input_method == "Use Sample Data":
        # Load data directly from the CSV file
        st.markdown("""
            <div style="background-color: #e8f4f8; color: #055160; padding: 16px; border-radius: 8px; border: 1px solid #b6effb; margin-bottom: 16px; font-weight: 500;">
                Loading rainfall data from CSV file.
            </div>
        """, unsafe_allow_html=True)
        
        data = load_sample_data('rainfall_data.csv')
        
        if data is not None:
            st.session_state.data = data  # Save to session state for future use
            st.markdown(f"""
                <div style="background-color: #d1e7dd; color: #0f5132; padding: 16px; border-radius: 8px; border: 1px solid #a3cfbb; margin-bottom: 16px; font-weight: 500;">
                    Data loaded successfully from rainfall_data.csv ({len(data)} records)
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Failed to load sample data. Please check if rainfall_data.csv exists.")
            return None
    
    elif input_method == "Use Weather API":
        # Weather API integration with enhanced features
        st.markdown(f"""
            <h2 style="color: #110361; margin-bottom: 16px;">Real-Time Weather Data</h2>
            <p style="font-size: 1.1rem;">Fetch historical and real-time weather data for any location worldwide.</p>
        """, unsafe_allow_html=True)

        # Create tabs for different API providers and forecast
        api_tabs = st.tabs(["WeatherAPI", "OpenWeather API", "Open-Meteo API", "WeatherBit API"])
        
        # WeatherAPI tab implementation
        with api_tabs[0]:
            st.write("### WeatherAPI")
            
            # Create tabs for historical and forecast data
            wa_tabs = st.tabs(["Historical Data", "Weather Forecast"])
            
            # Historical data tab
            with wa_tabs[0]:
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # Location input
                    location = st.text_input("Enter city name:", value="Bangalore", key="weatherapi_location")
                    
                    # Use the predefined API key directly
                    custom_api_key = WEATHERAPI_KEY
                    
                    # Add a note about using a predefined API key
                    st.info("Using predefined WeatherAPI key")
                    
                    # Add debug toggle
                    show_debug = st.checkbox("Show debug information", value=False, key="weatherapi_debug")
                
                with col2:
                    # Date range selection
                    today = datetime.now().date()
                    
                    start_date = st.date_input(
                        "Start Date:",
                        value=today - timedelta(days=7),
                        max_value=today,
                        key="weatherapi_start_date"
                    )
                    
                    end_date = st.date_input(
                        "End Date:",
                        value=today,
                        max_value=today,
                        key="weatherapi_end_date"
                    )
                
                # Button to fetch historical data
                if st.button("Fetch Historical Data", key="weatherapi_fetch_historical"):
                    if location:
                        with st.spinner(f"Fetching historical weather data for {location}..."):
                            # Fetch historical data using the modified function with debug parameter
                            weather_data, error_message = fetch_weather_data_from_api(
                                location, 
                                start_date, 
                                end_date, 
                                custom_api_key,
                                show_debug=show_debug  # Pass the debug toggle value
                            )
                            
                            if error_message:
                                st.warning(error_message)
                            
                            if weather_data is not None:
                                # Store in session state
                                st.session_state.data = weather_data
                                
                                # Display the data
                                st.success(f"Successfully fetched historical weather data for {location}")
                                st.dataframe(weather_data, use_container_width=True)
                                
                                # Create visualizations
                                st.subheader("Weather Visualizations")
                                
                                # Create tabs for different visualizations
                                viz_tabs = st.tabs(["Temperature", "Precipitation", "Wind & Humidity"])
                                
                                with viz_tabs[0]:
                                    # Temperature chart
                                    fig = px.line(
                                        weather_data, 
                                        x='date', 
                                        y='temperature',
                                        title=f"Temperature for {location}",
                                        labels={'temperature': 'Temperature (°C)'},
                                        color_discrete_sequence=['red']
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with viz_tabs[1]:
                                    # Precipitation chart
                                    fig2 = px.bar(
                                        weather_data, 
                                        x='date', 
                                        y='precipitation',
                                        title=f"Precipitation for {location}",
                                        labels={'precipitation': 'Precipitation (mm)'},
                                        color_discrete_sequence=['blue']
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                with viz_tabs[2]:
                                    # Wind and humidity chart
                                    fig3 = px.line(
                                        weather_data, 
                                        x='date', 
                                        y=['wind_speed', 'humidity'],
                                        title=f"Wind Speed and Humidity for {location}",
                                        labels={'value': 'Value', 'variable': 'Metric'},
                                        color_discrete_sequence=['orange', 'purple']
                                    )
                                    st.plotly_chart(fig3, use_container_width=True)
            
            # Forecast tab
            with wa_tabs[1]:
                # Create columns for better layout
                col1_fc, col2_fc = st.columns(2)
                
                with col1_fc:
                    # Location input
                    location_fc = st.text_input("Enter city name:", value="Bangalore", key="weatherapi_fc_location")
                    
                    # Use the predefined API key directly
                    api_key_fc = WEATHERAPI_KEY
                    
                    # Add a note about using a predefined API key
                    st.info("Using predefined WeatherAPI key")
                
                with col2_fc:
                    # Forecast options
                    forecast_days = st.slider("Forecast days:", 1, 14, 7, key="weatherapi_fc_days")
                
                # Button to trigger forecast
                if st.button("Generate Forecast", key="weatherapi_fc_button"):
                    if location_fc:
                        with st.spinner(f"Fetching {forecast_days}-day forecast for {location_fc}..."):
                            # Get current hour for consistent time-of-day forecasting
                            current_hour = datetime.now().hour
                            current_time = datetime.now().strftime('%H:%M')
                            
                            # Debug current time
                            st.write(f"Current hour: {current_hour}, Current time: {current_time}")
                            
                            # Fetch forecast data
                            try:
                                # Build the forecast API URL
                                forecast_url = (
                                    f"https://api.weatherapi.com/v1/forecast.json?key={api_key_fc}"
                                    f"&q={location_fc}&days={forecast_days}&aqi=yes&alerts=yes"
                                )
                                
                                st.info(f"Fetching forecast for {location_fc} at current time ({current_time})")
                                
                                response = requests.get(forecast_url)
                                if response.status_code == 200:
                                    forecast_data = response.json()
                                    
                                    # Extract forecast days
                                    forecast_days = forecast_data.get('forecast', {}).get('forecastday', [])
                                    
                                    if forecast_days:
                                        # Extract data for each day
                                        all_forecasts = []
                                        
                                        for day in forecast_days:
                                            date = day['date']
                                            day_data = day['day']
                                            
                                            # Extract hourly data for more detailed forecasts
                                            hour_data = day.get('hour', [])
                                            
                                            # Get data for current hour if available
                                            current_hour_data = None
                                            for hour in hour_data:
                                                hour_time = hour.get('time', '')
                                                if hour_time and hour_time.endswith(f"{current_hour:02d}:00"):
                                                    current_hour_data = hour
                                                    break
                                            
                                            # Use day data as fallback if hour data not available
                                            if current_hour_data:
                                                temp = current_hour_data.get('temp_c', day_data.get('avgtemp_c'))
                                                humidity = current_hour_data.get('humidity', day_data.get('avghumidity'))
                                                wind_speed = current_hour_data.get('wind_kph', day_data.get('maxwind_kph'))
                                                precip = current_hour_data.get('precip_mm', day_data.get('totalprecip_mm'))
                                            else:
                                                temp = day_data.get('avgtemp_c')
                                                humidity = day_data.get('avghumidity')
                                                wind_speed = day_data.get('maxwind_kph')
                                                precip = day_data.get('totalprecip_mm')
                                            
                                            all_forecasts.append({
                                                'date': pd.to_datetime(date),
                                                'temperature': temp,
                                                'max_temp': day_data.get('maxtemp_c'),
                                                'min_temp': day_data.get('mintemp_c'),
                                                'humidity': humidity,
                                                'pressure': 1013,  # WeatherAPI doesn't provide pressure directly
                                                'wind_speed': wind_speed,
                                                'precipitation': precip,
                                                'condition': day_data.get('condition', {}).get('text', '')
                                            })
                                        
                                        # Create DataFrame
                                        forecast_df = pd.DataFrame(all_forecasts)
                                        
                                        # Store the forecast data in session state
                                        st.session_state.forecast_data = forecast_df
                                        
                                        # Display the forecast data
                                        st.subheader(f"{forecast_days}-Day Forecast for {location_fc}")
                                        st.dataframe(forecast_df, use_container_width=True)
                                        
                                        # Create visualizations
                                        st.subheader("Forecast Visualizations")
                                        
                                        # Create tabs for different visualizations
                                        viz_tabs = st.tabs(["Temperature", "Precipitation", "Wind & Humidity"])
                                        
                                        with viz_tabs[0]:
                                            # Temperature chart
                                            fig_temp = px.line(
                                                forecast_df, 
                                                x='date', 
                                                y=['temperature', 'max_temp', 'min_temp'],
                                                title=f"Temperature Forecast for {location_fc}",
                                                labels={'value': 'Temperature (°C)', 'variable': 'Metric'},
                                                color_discrete_sequence=['green', 'red', 'blue']
                                            )
                                            st.plotly_chart(fig_temp, use_container_width=True)
                                        
                                        with viz_tabs[1]:
                                            # Precipitation chart
                                            fig_precip = px.bar(
                                                forecast_df, 
                                                x='date', 
                                                y='precipitation',
                                                title=f"Precipitation Forecast for {location_fc}",
                                                labels={'precipitation': 'Precipitation (mm)'},
                                                color_discrete_sequence=['blue']
                                            )
                                            st.plotly_chart(fig_precip, use_container_width=True)
                                        
                                        with viz_tabs[2]:
                                            # Wind and humidity chart
                                            fig_wind = px.line(
                                                forecast_df, 
                                                x='date', 
                                                y=['wind_speed', 'humidity'],
                                                title=f"Wind Speed and Humidity Forecast for {location_fc}",
                                                labels={'value': 'Value', 'variable': 'Metric'},
                                                color_discrete_sequence=['orange', 'purple']
                                            )
                                            st.plotly_chart(fig_wind, use_container_width=True)
                                    else:
                                        st.error("No forecast data found in the API response.")
                                else:
                                    st.error(f"Error fetching forecast: {response.status_code}")
                                    if response.text:
                                        st.error(f"Error details: {response.text}")
                            except Exception as e:
                                st.error(f"Error fetching forecast data: {str(e)}")
        
        # OpenWeather API tab implementation
        with api_tabs[1]:
            st.write("### OpenWeather API")
            
            # Create tabs for historical and forecast data
            ow_tabs = st.tabs(["Historical Data", "Weather Forecast"])
            
            # Historical data tab
            with ow_tabs[0]:
                # Create columns for better layout
                col1_ow, col2_ow = st.columns(2)
                
                with col1_ow:
                    # Location input
                    location_ow = st.text_input("Enter city name:", value="Bangalore", key="openweather_location")
                    
                    # Use the predefined API key directly
                    api_key_ow = OPENWEATHER_KEY
                    
                    # Add a note about using a predefined API key
                    st.info("Using predefined OpenWeather API key")
                
                with col2_ow:
                    # Date range selection
                    today_ow = datetime.now().date()
                    
                    start_date_ow = st.date_input(
                        "Start Date:",
                        value=today_ow - timedelta(days=7),
                        max_value=today_ow,
                        key="openweather_start_date"
                    )
                    
                    end_date_ow = st.date_input(
                        "End Date:",
                        value=today_ow,
                        max_value=today_ow,
                        key="openweather_end_date"
                    )
                
                # Button to fetch data
                if st.button("Fetch OpenWeather Data", key="openweather_fetch"):
                    if location_ow and api_key_ow:
                        with st.spinner(f"Fetching weather data for {location_ow}..."):
                            try:
                                # Build the API URL for current weather
                                current_url = f"https://api.openweathermap.org/data/2.5/weather?q={location_ow}&appid={api_key_ow}&units=metric"
                                
                                # Build the API URL for historical data (One Call API 3.0)
                                # First get coordinates from geocoding API
                                geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_ow}&limit=1&appid={api_key_ow}"
                                geo_response = requests.get(geo_url)
                                
                                if geo_response.status_code == 200 and geo_response.json():
                                    lat = geo_response.json()[0]['lat']
                                    lon = geo_response.json()[0]['lon']
                                    
                                    # Historical weather data API call
                                    hist_url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={int(start_date_ow.timestamp())}&appid={api_key_ow}&units=metric"
                                    hist_response = requests.get(hist_url)
                                    
                                    st.write("OpenWeather API response:", hist_response.status_code, hist_response.text)
                                    
                                    if hist_response.status_code == 200:
                                        data = hist_response.json()
                                        
                                        # Extract data
                                        hourly_data = data.get('hourly', [])
                                        
                                        records = []
                                        for hour in hourly_data:
                                            dt = datetime.fromtimestamp(hour['dt'])
                                            records.append({
                                                'date': dt.strftime('%Y-%m-%d %H:%M:%S'),
                                                'temperature': hour.get('temp', 25),
                                                'humidity': hour.get('humidity', 70),
                                                'pressure': hour.get('pressure', 1013),
                                                'wind_speed': hour.get('wind_speed', 5),
                                                'precipitation': hour.get('rain', {}).get('1h', 0) if 'rain' in hour else 0,
                                                'description': hour.get('weather', [{}])[0].get('description', 'Unknown')
                                            })
                                        
                                        # Create DataFrame
                                        weather_data_ow = pd.DataFrame(records)
                                        weather_data_ow['date'] = pd.to_datetime(weather_data_ow['date'])
                                        
                                        # Store in session state
                                        st.session_state.data = weather_data_ow
                                        
                                        # Display the data
                                        st.success(f"Successfully fetched weather data for {location_ow}")
                                        st.dataframe(weather_data_ow, use_container_width=True)
                                        
                                        # Create visualizations
                                        st.subheader("Weather Visualizations")
                                        
                                        # Create tabs for different visualizations
                                        viz_tabs_ow = st.tabs(["Temperature", "Precipitation", "Wind & Humidity"])
                                        
                                        with viz_tabs_ow[0]:
                                            # Temperature chart
                                            fig_ow = px.line(
                                                weather_data_ow, 
                                                x='date', 
                                                y='temperature',
                                                title=f"Temperature for {location_ow}",
                                                labels={'temperature': 'Temperature (°C)'},
                                                color_discrete_sequence=['red']
                                            )
                                            st.plotly_chart(fig_ow, use_container_width=True)
                                        
                                        with viz_tabs_ow[1]:
                                            # Precipitation chart
                                            fig_ow2 = px.bar(
                                                weather_data_ow, 
                                                y='precipitation',
                                                title=f"Precipitation for {location_ow}",
                                                labels={'precipitation': 'Precipitation (mm)'},
                                                color='precipitation',
                                                color_continuous_scale='Blues'
                                            )
                                            st.plotly_chart(fig_ow2, use_container_width=True)
                                        
                                        with viz_tabs_ow[2]:
                                            # Wind and humidity chart
                                            fig_ow3 = px.line(
                                                weather_data_ow, 
                                                x='date', 
                                                y=['wind_speed', 'humidity'],
                                                title=f"Wind Speed and Humidity for {location_ow}",
                                                labels={'value': 'Value', 'variable': 'Metric'},
                                                color_discrete_sequence=['orange', 'purple']
                                            )
                                            st.plotly_chart(fig_ow3, use_container_width=True)
                                    else:
                                        st.warning(f"Could not fetch data for {start_date_ow} to {end_date_ow}: {hist_response.status_code}")
                                        # Add placeholder data
                                        records.append({
                                            'date': start_date_ow.strftime('%Y-%m-%d'),
                                            'temperature': 25,
                                            'humidity': 70,
                                            'pressure': 1013,
                                            'wind_speed': 5,
                                            'precipitation': 0,
                                            'description': 'Data unavailable'
                                        })
                                
                                else:
                                    st.warning(f"Could not fetch data for {location_ow}: {response.status_code}")
                            except Exception as e:
                                st.error(f"Error fetching OpenWeather data: {str(e)}")
                                st.info("Using synthetic data as fallback.")
                                
                                # Generate synthetic data
                                date_range = pd.date_range(start=start_date_ow, end=end_date_ow, freq='D')
                                
                                # Create synthetic data
                                synthetic_data = pd.DataFrame({
                                    'date': date_range,
                                    'temperature': np.random.normal(25, 5, size=len(date_range)),
                                    'humidity': np.random.normal(70, 10, size=len(date_range)),
                                    'pressure': np.random.normal(1013, 5, size=len(date_range)),
                                    'wind_speed': np.random.normal(5, 2, size=len(date_range)),
                                    'precipitation': np.random.exponential(1, size=len(date_range)),
                                    'description': ['Partly cloudy'] * len(date_range)
                                })
                                
                                # Store in session state
                                st.session_state.data = synthetic_data
                                
                                # Display the synthetic data
                                st.subheader(f"Synthetic Weather Data for {location_ow}")
                                st.dataframe(synthetic_data, use_container_width=True)
        
        # Open-Meteo API tab implementation
        with api_tabs[2]:
            st.write("### Open-Meteo API")
            
            # Create tabs for historical and forecast data
            om_tabs = st.tabs(["Historical Data", "Weather Forecast"])
            
            # Historical data tab
            with om_tabs[0]:
                # Create columns for better layout
                col1_om, col2_om = st.columns(2)
                
                with col1_om:
                    # Location input
                    location_om = st.text_input("Enter city name:", value="Bangalore", key="openmeteo_location")
                    
                    # Debug toggle
                    show_debug_om = st.checkbox("Show debug info", value=False, key="openmeteo_debug")
                
                with col2_om:
                    # Date range selection
                    today_om = datetime.now().date()
                    
                    start_date_om = st.date_input(
                        "Start Date:",
                        value=today_om - timedelta(days=7),
                        max_value=today_om,
                        key="openmeteo_start_date"
                    )
                    
                    end_date_om = st.date_input(
                        "End Date:",
                        value=today_om,
                        max_value=today_om,
                        key="openmeteo_end_date"
                    )
                
                # Button to fetch historical data
                if st.button("Fetch Historical Data", key="openmeteo_fetch_historical"):
                    if location_om:
                        with st.spinner(f"Fetching historical weather data for {location_om}..."):
                            # Fetch historical data using the Open-Meteo function
                            weather_data, error_message = fetch_open_meteo_data(
                                location_om, 
                                start_date_om, 
                                end_date_om
                            )
                            
                            if error_message:
                                st.warning(error_message)
                            elif weather_data is not None:
                                # Display the data
                                st.success(f"Successfully retrieved weather data for {location_om}")
                                
                                # Show the data in a table
                                st.dataframe(weather_data)
                                
                                # Plot the data
                                fig = px.line(
                                    weather_data, 
                                    x='date', 
                                    y=['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation'],
                                    title=f"Weather Data for {location_om}"
                                )
                                st.plotly_chart(fig)
                            else:
                                st.error(f"Failed to retrieve weather data for {location_om}")
            
            # Forecast tab
            with om_tabs[1]:
                # Create columns for better layout
                col1_om_fc, col2_om_fc = st.columns(2)
                
                with col1_om_fc:
                    # Location input
                    location_om_fc = st.text_input("Enter city name:", value="Bangalore", key="openmeteo_fc_location")
                    
                    # Debug toggle
                    show_debug_om_fc = st.checkbox("Show debug info", value=False, key="openmeteo_fc_debug")
                
                with col2_om_fc:
                    # Forecast days
                    forecast_days_om = st.slider("Forecast Days:", min_value=1, max_value=16, value=7, key="openmeteo_fc_days")
                
                # Button to trigger forecast
                if st.button("Generate Open-Meteo Forecast", key="openmeteo_fc_button"):
                    if location_om_fc:
                        with st.spinner(f"Generating {forecast_days_om}-day forecast for {location_om_fc}..."):
                            try:
                                # Geocode the location (convert city name to lat/lon)
                                geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_om_fc}&count=1"
                                
                                if show_debug_om_fc:
                                    st.info(f"Geocoding URL: {geocoding_url}")
                                
                                geo_response = requests.get(geocoding_url)
                                
                                if show_debug_om_fc:
                                    st.write(f"Geocoding response status: {geo_response.status_code}")
                                
                                if geo_response.status_code == 200 and geo_response.json().get('results'):
                                    lat = geo_response.json()['results'][0]['latitude']
                                    lon = geo_response.json()['results'][0]['longitude']
                                    
                                    # Calculate date range for forecast
                                    today = datetime.now().date()
                                    end_date_fc = today + timedelta(days=forecast_days_om)
                                    
                                    # Build Open-Meteo Forecast API URL
                                    forecast_url = (
                                        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                                        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,"
                                        f"precipitation_hours,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,"
                                        f"shortwave_radiation_sum,et0_fao_evapotranspiration"
                                        f"&timezone=auto&forecast_days={forecast_days_om}"
                                    )
                                    
                                    if show_debug_om_fc:
                                        st.info(f"Forecast URL: {forecast_url}")
                                    
                                    forecast_response = requests.get(forecast_url)
                                    
                                    if show_debug_om_fc:
                                        st.write(f"Forecast response status: {forecast_response.status_code}")
                                    
                                    if forecast_response.status_code == 200:
                                        forecast_data = forecast_response.json()
                                        
                                        if show_debug_om_fc:
                                            st.write(f"Forecast data keys: {list(forecast_data.keys())}")
                                        
                                        # Extract daily data
                                        daily_data = {
                                            'date': pd.to_datetime(forecast_data['daily']['time']),
                                            'temp_max': forecast_data['daily']['temperature_2m_max'],
                                            'temp_min': forecast_data['daily']['temperature_2m_min'],
                                            'temp_mean': forecast_data['daily']['temperature_2m_mean'],
                                            'precipitation': forecast_data['daily']['precipitation_sum'],
                                            'wind_speed': forecast_data['daily']['windspeed_10m_max']
                                        }
                                        
                                        # Create DataFrame
                                        df_forecast = pd.DataFrame(daily_data)
                                        
                                        # Display the data
                                        st.success(f"Successfully retrieved {forecast_days_om}-day forecast for {location_om_fc}")
                                        
                                        # Show the data in a table
                                        st.dataframe(df_forecast)
                                        
                                        # Plot the temperature data
                                        fig_temp = px.line(
                                            df_forecast,
                                            x='date',
                                            y=['temp_max', 'temp_min', 'temp_mean'],
                                            title=f"Temperature Forecast for {location_om_fc}",
                                            labels={'value': 'Temperature (°C)', 'variable': 'Measurement'}
                                        )
                                        st.plotly_chart(fig_temp)
                                        
                                        # Plot precipitation data
                                        fig_precip = px.bar(
                                            df_forecast,
                                            x='date',
                                            y='precipitation',
                                            title=f"Precipitation Forecast for {location_om_fc}",
                                            labels={'precipitation': 'Precipitation (mm)'}
                                        )
                                        st.plotly_chart(fig_precip)
                                        
                                        # Plot wind speed data
                                        fig_wind = px.line(
                                            df_forecast,
                                            x='date',
                                            y='wind_speed',
                                            title=f"Wind Speed Forecast for {location_om_fc}",
                                            labels={'wind_speed': 'Wind Speed (km/h)'}
                                        )
                                        st.plotly_chart(fig_wind)
                                    else:
                                        st.error(f"Failed to retrieve forecast: {forecast_response.status_code}")
                                        if forecast_response.text:
                                            st.error(forecast_response.text)
                                else:
                                    st.error(f"Geocoding failed for location: {location_om_fc}")
                            except Exception as e:
                                st.exception(e)
                                st.error(f"Error generating forecast: {str(e)}")
        
        # WeatherBit API tab implementation
        with api_tabs[3]:
            st.write("### WeatherBit API")
            
            # Create tabs for historical and forecast data
            wb_tabs = st.tabs(["Historical Data", "Weather Forecast"])
            
            # Historical data tab
            with wb_tabs[0]:
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # Location input
                    location_wb = st.text_input("Enter city name:", value="Bangalore", key="weatherbit_location")
                    
                    # Country code input
                    country_code_wb = st.text_input("Country Code (optional):", value="IN", help="Two-letter country code (e.g., US, IN, UK)", key="weatherbit_country")
                    
                    # State code input (mainly for US)
                    state_code_wb = st.text_input("State Code (optional):", value="", help="Two-letter state code (mainly for US cities)", key="weatherbit_state")
                    
                    # Use the predefined API key directly
                    api_key_wb = WEATHERBIT_KEY
                    
                    # Add a note about using a predefined API key
                    st.info("Using predefined WeatherBit API key")
                
                with col2:
                    # Date range selection
                    today_wb = datetime.now().date()
                    
                    st.write("Note: WeatherBit API free tier has limitations on historical data.")
                    days_back_wb = st.slider("Days of historical data:", 1, 7, 5, key="weatherbit_days_hist")
                
                # Button to fetch data
                if st.button("Fetch WeatherBit Data", key="weatherbit_fetch"):
                    if location_wb and api_key_wb:
                        with st.spinner(f"Fetching weather data for {location_wb}..."):
                            # Build the location string
                            location_string = location_wb
                            if state_code_wb:
                                location_string += f",{state_code_wb}"
                            if country_code_wb:
                                location_string += f",{country_code_wb}"
                            
                            # Build the API URL
                            base_url = "https://api.weatherbit.io/v2.0/history/daily"
                            
                            # Calculate date range
                            end_date_wb = today_wb
                            start_date_wb = end_date_wb - timedelta(days=days_back_wb)
                            
                            params = {
                                "city": location_wb,
                                "key": api_key_wb,
                                "start_date": start_date_wb.strftime('%Y-%m-%d'),
                                "end_date": end_date_wb.strftime('%Y-%m-%d'),
                                "units": "M"  # Metric units
                            }
                            
                            # Add state and country if provided
                            if state_code_wb:
                                params["state"] = state_code_wb
                            if country_code_wb:
                                params["country"] = country_code_wb
                            
                            try:
                                response = requests.get(base_url, params=params)
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    
                                    if data and 'data' in data and len(data['data']) > 0:
                                        # Create a DataFrame from the data
                                        records = []
                                        for day in data['data']:
                                            records.append({
                                                'date': day['datetime'],
                                                'temperature': day.get('temp', 0),
                                                'max_temp': day.get('max_temp', 0),
                                                'min_temp': day.get('min_temp', 0),
                                                'humidity': day.get('rh', 0),
                                                'pressure': day.get('pres', 1013),
                                                'wind_speed': day.get('wind_spd', 0),
                                                'precipitation': day.get('precip', 0),
                                                'description': day.get('weather', {}).get('description', 'Unknown')
                                            })
                                        
                                        # Create DataFrame
                                        weather_data_wb = pd.DataFrame(records)
                                        weather_data_wb['date'] = pd.to_datetime(weather_data_wb['date'])
                                        
                                        # Store in session state
                                        st.session_state.data = weather_data_wb
                                        
                                        # Success message
                                        st.success(f"Successfully fetched weather data for {location_wb} ({len(weather_data_wb)} days)")
                                        
                                        # Display the data
                                        st.subheader(f"Weather Data for {location_wb}")
                                        st.dataframe(weather_data_wb, use_container_width=True)
                                        
                                        # Create visualizations
                                        st.subheader("Weather Visualizations")
                                        
                                        # Create tabs for different visualizations
                                        viz_tabs_wb = st.tabs(["Temperature", "Precipitation", "Wind & Humidity"])
                                        
                                        with viz_tabs_wb[0]:
                                            # Temperature chart
                                            fig_wb = px.line(
                                                weather_data_wb, 
                                                x='date', 
                                                y=['temperature', 'max_temp', 'min_temp'],
                                                title=f"Temperature for {location_wb}",
                                                labels={'value': 'Temperature (°C)', 'variable': 'Metric'},
                                                color_discrete_sequence=['green', 'red', 'blue']
                                            )
                                            st.plotly_chart(fig_wb, use_container_width=True)
                                        
                                        with viz_tabs_wb[1]:
                                            # Precipitation chart
                                            fig_wb2 = px.bar(
                                                weather_data_wb, 
                                                x='date', 
                                                y='precipitation',
                                                title=f"Precipitation for {location_wb}",
                                                labels={'precipitation': 'Precipitation (mm)'},
                                                color='precipitation',
                                                color_continuous_scale='Blues'
                                            )
                                            st.plotly_chart(fig_wb2, use_container_width=True)
                                        
                                        with viz_tabs_wb[2]:
                                            # Wind and humidity chart
                                            fig_wb3 = px.line(
                                                weather_data_wb, 
                                                x='date', 
                                                y=['wind_speed', 'humidity'],
                                                title=f"Wind Speed and Humidity for {location_wb}",
                                                labels={'value': 'Value', 'variable': 'Metric'},
                                                color_discrete_sequence=['orange', 'purple']
                                            )
                                            st.plotly_chart(fig_wb3, use_container_width=True)
                                    else:
                                        st.info("No weather data found for the specified location and date range. This could be due to API limitations or the location not being found.")
                                        
                                        # Create synthetic data as fallback
                                        st.info("Creating synthetic data for demonstration purposes.")
                                        
                                        # Generate dates
                                        dates = pd.date_range(start=start_date_wb, end=end_date_wb, freq='D')
                                        
                                        # Create synthetic data
                                        synthetic_data = pd.DataFrame({
                                            'date': dates,
                                            'temperature': np.random.normal(25, 5, size=len(dates)),
                                            'max_temp': np.random.normal(30, 3, size=len(dates)),
                                            'min_temp': np.random.normal(20, 3, size=len(dates)),
                                            'humidity': np.random.normal(70, 10, size=len(dates)),
                                            'pressure': np.random.normal(1013, 5, size=len(dates)),
                                            'wind_speed': np.random.normal(5, 2, size=len(dates)),
                                            'precipitation': np.random.exponential(1, size=len(dates)),
                                            'description': ['Partly cloudy'] * len(dates)
                                        })
                                        
                                        # Store in session state
                            except Exception as e:
                                st.error(f"Error fetching weather data: {str(e)}")
                                return
    elif input_method == "Upload Your Own Data":
        st.markdown("""
            <h2 style="color: #110361; margin-bottom: 16px;">Upload Your Weather Data</h2>
            <p style="font-size: 1.1rem;">Upload your own weather or rainfall data in CSV or Excel format.</p>
        """, unsafe_allow_html=True)
        
        # File uploader in the main area instead of sidebar for better visibility
        uploaded_file = st.file_uploader("Upload your weather data file", type=["csv", "xlsx", "xls", "json", "txt", "pdf"])
        
        if uploaded_file is not None:
            with st.spinner("Processing uploaded file..."):
                data, error_message = process_uploaded_data(uploaded_file)
                
                if error_message:
                    st.error(f"Error processing file: {error_message}")
                    return None
                
                if data is not None:
                    # Allow user to select target variable
                    st.subheader("Select Target Variable")
                    st.write("Choose which parameter you want to predict. The remaining parameters will be used as input features.")
                    
                    # Get numeric columns for potential target variables
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    
                    # Default to precipitation if available, otherwise first numeric column
                    default_target = 'precipitation' if 'precipitation' in numeric_cols else numeric_cols[0] if numeric_cols else None
                    
                    if default_target:
                        target_variable = st.selectbox(
                            "Select target variable to predict:",
                            options=numeric_cols,
                            index=numeric_cols.index(default_target),
                            help="This is the variable you want to predict based on other parameters"
                        )
                        
                        # Rename selected column to 'precipitation' for compatibility with existing code
                        if target_variable != 'precipitation':
                            st.info(f"Using '{target_variable}' as the target variable for prediction")
                            # Make a copy to avoid modifying the original
                            data_copy = data.copy()
                            # Store original target name for display purposes
                            if 'original_target_name' not in st.session_state:
                                st.session_state.original_target_name = target_variable
                            # If precipitation exists, temporarily store it
                            if 'precipitation' in data_copy.columns:
                                data_copy['original_precipitation'] = data_copy['precipitation']
                            # Rename target to precipitation for model compatibility
                            data_copy['precipitation'] = data_copy[target_variable]
                            # Store in session state
                            st.session_state.data = data_copy
                        else:
                            # Store in session state
                            st.session_state.data = data
                            if 'original_target_name' in st.session_state:
                                del st.session_state.original_target_name
                        
                        # Success message
                        st.success(f"Successfully processed uploaded file ({len(data)} records)")
                        
                        # Display the processed data
                        st.subheader("Processed Data")
                        display_data = data.copy()
                        if isinstance(display_data.index, pd.DatetimeIndex):
                            display_data = display_data.reset_index()
                        display_data = add_units_to_columns(display_data)
                        st.dataframe(display_data, use_container_width=True)
                        
                        # Create visualizations for the uploaded data
                        st.subheader("Data Visualizations")
                        
                        # Create tabs for different visualizations
                        viz_tabs = st.tabs(["Target Variable", "Input Features", "Correlation Analysis"])
                        
                        with viz_tabs[0]:
                            # Plot the target variable
                            fig = px.line(
                                data.reset_index(), 
                                x='date' if 'date' in data.reset_index().columns else data.reset_index().index,
                                y=target_variable,
                                title=f"{target_variable} Data",
                                labels={target_variable: f"{target_variable}"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with viz_tabs[1]:
                            # Select input features to display
                            input_cols = [col for col in numeric_cols if col != target_variable]
                            if input_cols:
                                selected_features = st.multiselect(
                                    "Select features to display:",
                                    options=input_cols,
                                    default=input_cols[:3] if len(input_cols) >= 3 else input_cols
                                )
                                
                                if selected_features:
                                    fig = px.line(
                                        data.reset_index(), 
                                        x='date' if 'date' in data.reset_index().columns else data.reset_index().index,
                                        y=selected_features,
                                        title="Input Features",
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Please select at least one feature to display.")
                            else:
                                st.info("No additional features found in the uploaded file.")
                        
                        with viz_tabs[2]:
                            # Show correlation matrix
                            st.write("#### Correlation Matrix")
                            st.write("This shows how different variables are related to each other.")
                            
                            # Calculate correlation matrix
                            corr_matrix = data.select_dtypes(include=['number']).corr()
                            
                            # Plot heatmap
                            fig = px.imshow(
                                corr_matrix,
                                text_auto=True,
                                color_continuous_scale='RdBu_r',
                                title="Correlation Between Variables"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Highlight correlations with target
                            if target_variable in corr_matrix.columns:
                                st.write(f"#### Correlation with {target_variable}")
                                target_corr = corr_matrix[target_variable].drop(target_variable).sort_values(ascending=False)
                                
                                # Plot bar chart of correlations
                                fig = px.bar(
                                    x=target_corr.index,
                                    y=target_corr.values,
                                    title=f"Correlation with {target_variable}",
                                    labels={'x': 'Feature', 'y': 'Correlation Coefficient'},
                                    color=target_corr.values,
                                    color_continuous_scale='RdBu_r'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Explain correlations
                                st.write("#### Interpretation:")
                                st.write("""
                                    - Values close to 1 indicate strong positive correlation (as one increases, the other increases)
                                    - Values close to -1 indicate strong negative correlation (as one increases, the other decreases)
                                    - Values close to 0 indicate little to no correlation
                                """)
                    else:
                        st.error("No numeric columns found for prediction")
                else:
                    st.error("Failed to process the uploaded file.")    
    # Common section for all input methods
    if 'data' in st.session_state:
        data = st.session_state.data
        
        st.write("### Processed Data:")
        display_data = data.copy()
        if isinstance(display_data.index, pd.DatetimeIndex):
            display_data = display_data.reset_index()
        display_data = add_units_to_columns(display_data)
        st.dataframe(display_data, use_container_width=True)  # Removed .head() to show all rows
        
        # Feature engineering (keep the computation but hide the display)
        data_features = create_features(data)
        
        # Preprocessing
        st.write("### Data Preprocessing:")
        
        # Display all parameters with their values
        st.write("#### Parameters Used in Model Training:")
        
        # Create a DataFrame to display parameters
        params_df = pd.DataFrame({
            'Parameter': ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Day of Year', 'Month', 'Season'],
            'Description': [
                'Air temperature in °C or °F',
                'Relative humidity in %',
                'Atmospheric pressure in hPa',
                'Wind speed in km/h or mph',
                'Day of the year (1-366)',
                'Month of the year (1-12)',
                'Season code (0: Winter, 1: Summer, 2: Monsoon, 3: Post-Monsoon)'
            ],
            'Importance': [
                'Higher temperatures can increase evaporation and capacity for precipitation',
                'Higher humidity indicates more moisture in the air, increasing precipitation potential',
                'Low pressure systems are often associated with rainfall',
                'Wind brings moisture from oceans/water bodies to land',
                'Seasonal patterns affect rainfall throughout the year',
                'Monthly patterns capture seasonal rainfall variations',
                'Different seasons have distinct rainfall patterns'
            ]
        })
        
        st.dataframe(params_df, use_container_width=True)
        
        # Process the data
        X_train, X_test, y_train, y_test, scaler, feature_cols, imputer = preprocess_data(data_features)
        
        # Train models
        ml_models = train_ml_models(X_train, y_train)
        
        # Train ARIMA model
        arima_model = train_arima_model(y_train)
        
        # Evaluate all models to find the best one
        model_metrics = evaluate_models(ml_models, X_test, y_test, arima_model)
        
        # Display model comparison
        st.write("#### Model Comparison:")
        st.dataframe(model_metrics, use_container_width=True)
        
        # Determine the best model based on lowest RMSE
        best_model_name = model_metrics['RMSE'].idxmin()
        best_model = ml_models.get(best_model_name, None)
        
        # Explain why this model is the best
        st.write(f"#### Best Model: {best_model_name}")
        
        model_explanations = {
            'Linear Regression': """
                **Linear Regression** performs best when there's a linear relationship between weather parameters and precipitation.
                It's simple, interpretable, and works well when the data doesn't have complex patterns.
                This model suggests that rainfall can be predicted as a weighted sum of the input features.
            """,
            'Random Forest': """
                **Random Forest** performs best when there are complex, non-linear relationships between weather parameters and precipitation.
                It's robust to outliers and can capture interactions between features that simpler models miss.
                This model suggests that rainfall patterns are complex and depend on multiple interacting factors.
            """,
            'Gradient Boosting': """
                **Gradient Boosting** performs best when there are subtle patterns in the data that other models miss.
                It builds models sequentially, with each new model correcting errors made by previous ones.
                This model suggests that rainfall prediction requires capturing both obvious and subtle weather patterns.
            """,
            'ARIMA': """
                **ARIMA** performs best when rainfall follows strong temporal patterns and is primarily influenced by its own past values.
                It's specialized for time series data and doesn't use weather parameters directly.
                This model suggests that future rainfall is best predicted by historical rainfall patterns rather than weather conditions.
            """
        }
        
        st.markdown(model_explanations.get(best_model_name, "No explanation available for this model."))
        
        # Evaluation
        st.write("### Model Evaluation:")
        if best_model_name != 'ARIMA' and best_model is not None:
            evaluate_model_and_show_importance(best_model, X_test, y_test, feature_cols, scaler)
        else:
            # If ARIMA is best, use Random Forest for feature importance
            st.write("Using Random Forest to show feature importance (ARIMA doesn't provide feature importance):")
            evaluate_model_and_show_importance(ml_models['Random Forest'], X_test, y_test, feature_cols, scaler)
        
        # Forecasting
        st.write("### Future Rainfall Forecast:")
        
        # Simple UI for forecast settings
        forecast_days = st.slider("Select number of days to forecast", 1, 365, 30)
        
        # Add unit selection for forecast period
        forecast_unit = st.selectbox(
            "Select unit for forecast period",
            ["Days", "Weeks", "Months", "Years"],
            index=0
        )
        
        # Button to trigger forecast
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                # Generate forecast
                best_model_name = "Random Forest"  # You can change this to use another model if desired
                time_granularity = "D"  # Daily granularity
                # Use the best model for forecasting
                if best_model_name == "ARIMA":
                    model_to_use = arima_model
                else:
                    model_to_use = ml_models.get(best_model_name, list(ml_models.values())[0])
                
                # Get historical data (last 30% of the original data)
                historical_size = int(len(data) * 0.3)
                historical_data = data.iloc[-historical_size:].copy()
                
                # Generate forecast
                future_forecast = generate_forecast(
                    data,
                    model_to_use,
                    periods=forecast_days,
                    time_granularity=time_granularity,
                    forecast_unit=forecast_unit,
                    forecast_days=forecast_days,
                    scaler=scaler,
                    feature_cols=feature_cols,
                    imputer=imputer
                )
                
                # Ensure forecast data has a date column
                if 'date' not in future_forecast.columns and isinstance(future_forecast.index, pd.DatetimeIndex):
                    future_forecast = future_forecast.reset_index()
                    future_forecast.rename(columns={'index': 'date'}, inplace=True)
                
                # Show forecast
                # Use original target name if it was renamed
                target_display_name = st.session_state.get('original_target_name', 'precipitation')
                
                st.write(f"### Future {target_display_name.title()} Forecast:")
                
                # Format the date column for display
                display_forecast = future_forecast.copy()
                
                # Check for duplicate columns and remove them
                if len(display_forecast.columns) != len(set(display_forecast.columns)):
                    display_forecast = display_forecast.loc[:, ~display_forecast.columns.duplicated()]
                
                # Create a copy for display with formatted dates
                display_df = display_forecast.copy()
                if 'date' in display_df.columns:
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                
                # Rename precipitation back to original target name for display
                if target_display_name != 'precipitation':
                    display_df = display_df.rename(columns={'precipitation': target_display_name})
                
                # Final check for duplicate columns
                if len(display_df.columns) != len(set(display_df.columns)):
                    display_df = display_df.loc[:, ~display_df.columns.duplicated()]
                
                st.dataframe(display_df)
                
                # Plot forecast using the new function
                st.write(f"### Historical Data and {target_display_name.title()} Forecast:")

                # Create tabs for different parameter visualizations
                forecast_tabs = st.tabs([f"{target_display_name.title()}", "Temperature", "Humidity", "Pressure", "Wind Speed"])
                
                with forecast_tabs[0]:
                    # Plot the target variable (precipitation or custom target)
                    plot_historical_and_forecast(
                        historical_data, 
                        future_forecast, 
                        'precipitation',  # Always use 'precipitation' as the internal column name
                        display_name=target_display_name,
                        title=f"{target_display_name.title()} - Historical vs Forecast"
                    )
                    
                    # Add a download button for the forecast data
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {target_display_name.title()} Forecast",
                        data=csv,
                        file_name=f"{target_display_name}_forecast.csv",
                        mime="text/csv"
                    )
                
                with forecast_tabs[1]:
                    # Plot temperature
                    plot_historical_and_forecast(
                        historical_data, 
                        future_forecast, 
                        'temperature',
                        display_name='Temperature (°C)',
                        title="Temperature - Historical vs Forecast"
                    )
                
                with forecast_tabs[2]:
                    # Plot humidity
                    plot_historical_and_forecast(
                        historical_data, 
                        future_forecast, 
                        'humidity',
                        display_name='Humidity (%)',
                        title="Humidity - Historical vs Forecast"
                    )
                
                with forecast_tabs[3]:
                    # Plot pressure
                    plot_historical_and_forecast(
                        historical_data, 
                        future_forecast, 
                        'pressure',
                        display_name='Pressure (hPa)',
                        title="Pressure - Historical vs Forecast"
                    )
                
                with forecast_tabs[4]:
                    # Plot wind speed
                    plot_historical_and_forecast(
                        historical_data, 
                        future_forecast, 
                        'wind_speed',
                        display_name='Wind Speed (m/s)',
                        title="Wind Speed - Historical vs Forecast"
                    )
    
    # Debugging tools
    if st.checkbox("Show debug information"):
        st.write("### Debug Information")
        
        # Show session state
        st.write("Session State:")
        st.write(st.session_state)

# Call the main function when the script is run
if __name__ == "__main__":
    main()





#precip_lag_1	precip_lag_2	precip_lag_3	precip_rolling_avg_7	precip_rolling_std_7

#remove all these columns from the forecast table and the importance graph and give forecast graph and imporatnce each parameters plays in predicting or having rainfall,consider these parameters only :
#temperature	humidity	pressure	wind_speed	precipitation

def preprocess_data(data):
    # Define features dynamically based on what's available
    potential_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 
                         'day_of_year', 'month', 'season_code']
    # Only use features that exist in the data
    feature_cols = [col for col in potential_features if col in data.columns]
    
    # If no features are available, add some basic ones
    if not feature_cols:
        data['day_of_year'] = data.index.dayofyear if isinstance(data.index, pd.DatetimeIndex) else 1
        data['month'] = data.index.month if isinstance(data.index, pd.DatetimeIndex) else 1
        feature_cols = ['day_of_year', 'month']
    
    # Make sure all feature columns are numeric
    for col in feature_cols:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        except:
            st.warning(f"Could not convert {col} to numeric. Using zeros.")
            data[col] = 0
    
    X = data[feature_cols]
    
    # Use precipitation as target if available, otherwise create a dummy target
    if 'precipitation' in data.columns:
        try:
            data['precipitation'] = pd.to_numeric(data['precipitation'], errors='coerce')
            y = data['precipitation']
        except:
            st.warning("Could not convert precipitation to numeric. Using random values.")
            y = pd.Series(np.random.normal(5, 2, len(data)), index=data.index)
            data['precipitation'] = y
    else:
        y = pd.Series(np.random.normal(5, 2, len(data)), index=data.index)
        data['precipitation'] = y
    
    # Handle missing values in X using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # Handle missing values in y
    y_imputed = y.fillna(y.mean() if not y.isna().all() else 0)
    
    # Check if there are still NaN values in y
    if y_imputed.isna().any():
        st.warning("Could not impute all missing values in precipitation data. Using zeros for remaining NaNs.")
        y_imputed = y_imputed.fillna(0)
    
    # Split data - use a smaller test set if data is limited
    test_size = min(0.2, max(1/len(X_imputed), 0.1)) if len(X_imputed) > 10 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, imputer

def process_uploaded_data(uploaded_file):
    """
    Process uploaded data file and add units to column names.
    
    Args:
        uploaded_file: File uploaded by the user
        
    Returns:
        Tuple of (processed_data, error_message)
    """
    try:
        # Determine file type from extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Read file based on extension
        if file_extension == 'csv':
            data = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            data = pd.read_json(uploaded_file)
        elif file_extension == 'txt':
            # Try to read as CSV first
            try:
                data = pd.read_csv(uploaded_file, sep=None, engine='python')
            except:
                return None, "Could not parse text file. Please ensure it's in a tabular format."
        elif file_extension == 'pdf':
            st.warning("PDF parsing is limited. Attempting to extract tabular data...")
            try:
                import tabula
                data = tabula.read_pdf(uploaded_file, pages='all')[0]
            except:
                return None, "Could not extract data from PDF. Please upload a CSV or Excel file instead."
        else:
            return None, f"Unsupported file format: {file_extension}"
        
        # Check if data was loaded successfully
        if data is None or len(data) == 0:
            return None, "No data found in the file"
        
        # Add units to column names
        data = add_units_to_columns(data)
        
        # Display raw data for debugging
        st.write("### Raw Uploaded Data:")
        st.dataframe(data, use_container_width=True)
        
        # Try to identify date column
        date_col = None
        for col in data.columns:
            if any(term in col.lower() for term in ['date', 'time', 'day', 'year', 'month']):
                date_col = col
                break
        
        if date_col is None:
            # If no date column found, use the first column if it looks like dates
            first_col = data.columns[0]
            try:
                pd.to_datetime(data[first_col], errors='raise')
                date_col = first_col
            except:
                # Create a date column if none exists
                st.warning("No date column found. Creating a synthetic date column.")
                data['date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
                date_col = 'date'
        
        # Convert date column to datetime
        try:
            data[date_col] = pd.to_datetime(data[date_col])
            st.success(f"Using '{date_col}' as date column")
        except:
            st.warning(f"Could not convert '{date_col}' to datetime. Creating a synthetic date column.")
            data['date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
            date_col = 'date'
        
        # Set date as index
        data = data.set_index(date_col)
        
        # Try to identify precipitation column
        precip_col = None
        precip_terms = ['precipitation', 'rainfall', 'rain', 'precip']
        
        for col in data.columns:
            if any(term in col.lower() for term in precip_terms):
                precip_col = col
                break
        
        if precip_col is None:
            # If no precipitation column found, look for numeric columns
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                # Don't automatically assign precipitation - let user choose target variable
                st.info("No precipitation column found. You'll be able to select a target variable.")
            else:
                return None, "No numeric columns found for analysis"
        else:
            # Rename the identified precipitation column
            data = data.rename(columns={precip_col: 'precipitation'})
            st.success(f"Using '{precip_col}' as precipitation data.")
        
        # Convert all remaining columns to numeric if possible
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except:
                pass
        
        # Check for other required columns
        required_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for col in required_cols:
            if not any(col in c.lower() for c in data.columns):
                # Generate synthetic data for missing columns
                data[col] = np.random.normal(
                    loc={'temperature': 25, 'humidity': 70, 'pressure': 1013, 'wind_speed': 10}[col],
                    scale={'temperature': 5, 'humidity': 10, 'pressure': 5, 'wind_speed': 3}[col],
                    size=len(data)
                )
                st.info(f"Generated synthetic '{col}' data.")
            else:
                matched_col = next(c for c in data.columns if col in c.lower())
                if matched_col != col:
                    data = data.rename(columns={matched_col: col})
                    st.success(f"Using '{matched_col}' as {col} data.")
        
        return data, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def generate_forecast(data, model, periods=30, time_granularity='D', forecast_unit='days', forecast_days=30, scaler=None, feature_cols=None, imputer=None):
    """
    Generate forecast for future periods using the trained model.
    """
    # Convert forecast_days to periods based on forecast_unit
    if forecast_unit.lower() == 'days':
        periods = forecast_days
    elif forecast_unit.lower() == 'weeks':
        periods = forecast_days * 7
    elif forecast_unit.lower() == 'months':
        periods = forecast_days * 30
    elif forecast_unit.lower() == 'years':
        periods = forecast_days * 365
    else:
        periods = forecast_days
    
    # Create future dates - ensure continuous dates from the last date in the data
    if isinstance(data.index, pd.DatetimeIndex):
        last_date = data.index[-1]
    else:
        # If index is not datetime, try to find a date column
        if 'date' in data.columns:
            last_date = pd.to_datetime(data['date']).max()
        else:
            last_date = pd.to_datetime('today')
    
    # Ensure last_date is a datetime object
    if not isinstance(last_date, (pd.Timestamp, datetime)):
        try:
            last_date = pd.to_datetime(last_date)
        except:
            last_date = pd.to_datetime('today')
    
    # Create continuous date range starting from the day after the last date
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), 
        periods=periods, 
        freq=time_granularity
    )
    
    # Create a DataFrame for future dates
    future_df = pd.DataFrame(index=future_dates)
    future_df['date'] = future_dates
    
    # Extract date features
    future_df['day_of_year'] = future_df['date'].dt.dayofyear
    future_df['month'] = future_df['date'].dt.month
    
    # Add season code
    def get_season_code(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Summer
        elif month in [6, 7, 8, 9]:
            return 2  # Monsoon
        else:
            return 3  # Post-Monsoon
    
    future_df['season_code'] = future_df['month'].apply(get_season_code)
    
    # Calculate average values from historical data
    avg_values = {}
    for col in ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']:
        if col in data.columns and not data[col].isna().all():
            avg_values[col] = data[col].mean()
        else:
            avg_values[col] = {'temperature': 25, 'humidity': 70, 'pressure': 1013, 'wind_speed': 10, 'precipitation': 5}[col]
    
    # Generate synthetic data for all weather parameters
    # This ensures we have complete data for visualization
    
    # Temperature varies by season
    month_temp_factors = {
        1: 0.7,   # January - cooler
        2: 0.8,   # February - cooler
        3: 0.9,   # March - warming
        4: 1.0,   # April - warming
        5: 1.1,   # May - warm
        6: 1.2,   # June - hot
        7: 1.1,   # July - hot
        8: 1.1,   # August - hot
        9: 1.0,   # September - cooling
        10: 0.9,  # October - cooling
        11: 0.8,  # November - cooler
        12: 0.7   # December - cooler
    }
    
    # Check if temperature already exists in the DataFrame
    if 'temperature' not in future_df.columns:
        future_df['temperature'] = future_df['month'].apply(
            lambda m: avg_values['temperature'] * month_temp_factors[m] * (1 + np.random.normal(0, 0.1))
        )
    
    # Humidity varies by season
    month_humidity_factors = {
        1: 0.8,   # January - drier
        2: 0.7,   # February - drier
        3: 0.7,   # March - drier
        4: 0.8,   # April - getting humid
        5: 0.9,   # May - getting humid
        6: 1.1,   # June - monsoon
        7: 1.2,   # July - monsoon
        8: 1.2,   # August - monsoon
        9: 1.1,   # September - monsoon ending
        10: 1.0,  # October - post-monsoon
        11: 0.9,  # November - drying
        12: 0.8   # December - drier
    }
    
    if 'humidity' not in future_df.columns:
        future_df['humidity'] = future_df['month'].apply(
            lambda m: min(95, max(30, avg_values['humidity'] * month_humidity_factors[m] * (1 + np.random.normal(0, 0.1))))
        )
    
    # Pressure varies less
    if 'pressure' not in future_df.columns:
        future_df['pressure'] = avg_values['pressure'] + np.random.normal(0, 5, size=len(future_df))
    
    # Wind speed varies somewhat randomly
    if 'wind_speed' not in future_df.columns:
        future_df['wind_speed'] = np.maximum(0, avg_values['wind_speed'] + np.random.normal(0, 2, size=len(future_df)))
    
    # Initialize precipitation column
    if 'precipitation' not in future_df.columns:
        future_df['precipitation'] = 0.0
    
    # Check if model is ARIMA
    if hasattr(model, 'forecast'):
        # For ARIMA, we need to forecast directly
        try:
            forecast_result = model.forecast(steps=periods)
            future_df['precipitation'] = forecast_result
        except Exception as e:
            st.warning(f"ARIMA forecast failed: {str(e)}. Using seasonal patterns instead.")
            # Use seasonal patterns for precipitation if ARIMA fails
            month_precip_factors = {
                1: 0.3,   # January - dry
                2: 0.2,   # February - dry
                3: 0.3,   # March - dry
                4: 0.4,   # April - some rain
                5: 0.6,   # May - pre-monsoon
                6: 1.2,   # June - monsoon
                7: 1.5,   # July - peak monsoon
                8: 1.4,   # August - monsoon
                9: 1.0,   # September - monsoon ending
                10: 0.7,  # October - post-monsoon
                11: 0.4,  # November - drying
                12: 0.3   # December - dry
            }
            
            future_df['precipitation'] = future_df['month'].apply(
                lambda m: max(0, avg_values['precipitation'] * month_precip_factors[m] * 
                             (1 + np.random.normal(0, 0.3)))  # Add some randomness
            )
    else:
        # For ML models, we need to prepare the features
        # Use only the feature columns that were used during training
        if feature_cols is None:
            # If feature_cols is not provided, try to infer from model
            if hasattr(model, 'feature_names_in_'):
                feature_cols = model.feature_names_in_
            else:
                # Default feature columns if we can't determine from model
                feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'day_of_year', 'month']
        
        # Ensure all required features are present in future_df
        for col in feature_cols:
            if col not in future_df.columns:
                if col in avg_values:
                    future_df[col] = avg_values[col]
                else:
                    # For unknown features, use zeros
                    future_df[col] = 0
        
        # Select only the features used during training
        X_future = future_df[feature_cols].copy()
        
        # Handle missing values if imputer is provided
        if imputer is not None:
            X_future = pd.DataFrame(imputer.transform(X_future), columns=X_future.columns, index=X_future.index)
        
        # Scale features if scaler is provided
        if scaler is not None:
            X_future_scaled = scaler.transform(X_future)
        else:
            X_future_scaled = X_future
        
        # Make predictions
        try:
            predictions = model.predict(X_future_scaled)
            
            # Apply post-processing to predictions to make them more realistic
            # 1. Ensure no negative values
            predictions = np.maximum(predictions, 0)
            
            # 2. Apply smoothing to avoid extreme jumps
            if len(predictions) > 1:
                smoothed_predictions = np.copy(predictions)
                for i in range(1, len(predictions)):
                    smoothed_predictions[i] = 0.7 * predictions[i] + 0.3 * smoothed_predictions[i-1]
                predictions = smoothed_predictions
            
            # 3. Scale predictions to be in a reasonable range
            if np.mean(predictions) > 3 * avg_values['precipitation']:
                predictions = predictions * (avg_values['precipitation'] / np.mean(predictions)) * 2
            
            future_df['precipitation'] = predictions
            
        except Exception as e:
            st.warning(f"Prediction failed: {str(e)}. Using seasonal patterns instead.")
            # Use seasonal patterns for precipitation if prediction fails
            month_precip_factors = {
                1: 0.3,   # January - dry
                2: 0.2,   # February - dry
                3: 0.3,   # March - dry
                4: 0.4,   # April - some rain
                5: 0.6,   # May - pre-monsoon
                6: 1.2,   # June - monsoon
                7: 1.5,   # July - peak monsoon
                8: 1.4,   # August - monsoon
                9: 1.0,   # September - monsoon ending
                10: 0.7,  # October - post-monsoon
                11: 0.4,  # November - drying
                12: 0.3   # December - dry
            }
            
            future_df['precipitation'] = future_df['month'].apply(
                lambda m: max(0, avg_values['precipitation'] * month_precip_factors[m] * 
                             (1 + np.random.normal(0, 0.3)))  # Add some randomness
            )
    
    # Ensure no negative precipitation
    future_df['precipitation'] = future_df['precipitation'].clip(lower=0)
    
    # Final check to ensure no None/NaN values in any column
    for col in future_df.columns:
        if future_df[col].isna().any():
            if col in avg_values:
                future_df[col] = future_df[col].fillna(avg_values[col])
            else:
                future_df[col] = future_df[col].fillna(0)
    
    # Round numeric columns to improve display
    numeric_cols = future_df.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        future_df[col] = future_df[col].round(2)
    
    # Ensure the date column is preserved
    if 'date' not in future_df.columns and isinstance(future_df.index, pd.DatetimeIndex):
        future_df['date'] = future_df.index
    
    # Check for duplicate columns and remove them
    if len(future_df.columns) != len(set(future_df.columns)):
        # Find duplicate columns
        cols_seen = set()
        duplicate_cols = []
        for col in future_df.columns:
            if col in cols_seen:
                duplicate_cols.append(col)
            else:
                cols_seen.add(col)
        
        # Keep only the first occurrence of each column
        future_df = future_df.loc[:, ~future_df.columns.duplicated()]
        st.warning(f"Removed duplicate columns: {duplicate_cols}")
    
    return future_df

def plot_historical_and_forecast(historical_data, forecast_data, column_name, display_name=None, title=None):
    """
    Plot historical data and forecast together with no gaps.
    """
    if display_name is None:
        display_name = column_name
    
    if title is None:
        title = f"{display_name} Historical Data and Forecast"
    
    # Create figure directly with plotly
    fig = go.Figure()
    
    # Prepare historical data for plotting
    hist_df = historical_data.copy()
    
    # Check for datetime column and rename it to date if found
    if 'datetime' in hist_df.columns and 'date' not in hist_df.columns:
        hist_df = hist_df.rename(columns={'datetime': 'date'})
    
    # CRITICAL: Ensure historical data has a date column
    if 'date' not in hist_df.columns:
        # Try to find a date-like column
        date_cols = [col for col in hist_df.columns if any(term in col.lower() for term in ['date', 'time', 'day'])]
        if date_cols:
            hist_df = hist_df.rename(columns={date_cols[0]: 'date'})
        else:
            # Create synthetic dates if no suitable column found
            hist_df['date'] = pd.date_range(
                end=pd.Timestamp.today() - pd.Timedelta(days=1),
                periods=len(hist_df),
                freq='D'
            )

    
    # Ensure date column is datetime type
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_dates = hist_df['date']

    
    # Prepare forecast data for plotting
    fore_df = forecast_data.copy()
    
    # CRITICAL: Ensure forecast data has a date column
    if 'date' not in fore_df.columns:
        if isinstance(fore_df.index, pd.DatetimeIndex):
            fore_df = fore_df.reset_index()
            fore_df.rename(columns={'index': 'date'}, inplace=True)
        else:
            return
    
    # Ensure date column is datetime type
    fore_df['date'] = pd.to_datetime(fore_df['date'])
    fore_dates = fore_df['date']
    
    # Get historical values
    if column_name in hist_df.columns:
        hist_values = hist_df[column_name]
    else:
        return
    
    # Get forecast values
    if column_name in fore_df.columns:
        fore_values = fore_df[column_name]
    else:
        return
    
    # Add historical trace
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_values,
        name='Historical Data',
        line=dict(color='blue'),
        mode='lines'
    ))
    
    # Add forecast trace
    fig.add_trace(go.Scatter(
        x=fore_dates,
        y=fore_values,
        name='Forecast',
        line=dict(color='red', dash='dash'),
        mode='lines'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=display_name,
        legend_title='Data Type',
        hovermode='x unified'
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Display the plot
    st.write("Forecast Data Sample:", fore_df[['date', column_name]].head(3))

