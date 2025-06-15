import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import base64
import matplotlib.pyplot as plt
from io import BytesIO

# --- Load model and feature info ---
@st.cache_resource
def load_model():
    with open('natural_gas_demand_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return model, features

model, features = load_model()

# --- Load last actuals for chart ---
def get_last_actuals(n=20):
    df = pd.read_excel('NaturalGasDemand_Input.xlsx')
    df['Month'] = pd.to_datetime(df['Month'])
    return df[['Month', 'India total Consumption of Natural Gas (in BCM)']].tail(n)

# --- UI Styling ---
st.set_page_config(page_title="Natural Gas Demand Forecast", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {background-color: #0066cc; color: white; font-weight: bold; border-radius: 8px;}
    .stDataFrame {background-color: #fff; border-radius: 8px;}
    .stTable {background-color: #fff; border-radius: 8px;}
    .stTextInput>div>div>input {border-radius: 8px;}
    .stFileUploader {border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Natural Gas Demand Forecast")
st.markdown("""
#### Upload or manually enter up to 10 future periods of independent variables to forecast India's total natural gas consumption.
""")

# --- Get independent variable names ---
all_columns = [col for col in features if not any(s in col for s in ['lag', 'rollmean', 'rollstd', 'sin', 'cos', 'quarter'])]

# --- Excel Upload or Manual Entry ---
with st.expander("1️⃣ Upload Excel or Enter Manually", expanded=True):
    upload_col, manual_col = st.columns([2,2])
    with upload_col:
        uploaded_file = st.file_uploader("Upload Excel with columns: Date (Month) + independent variables", type=["xlsx"])
        preview_btn = st.button("Preview Excel")
    with manual_col:
        st.write("Or enter values manually:")
        n_rows = st.number_input("Number of future periods (max 10)", min_value=1, max_value=10, value=3)
        today = datetime.date.today().replace(day=1)
        default_dates = [today + pd.DateOffset(months=i) for i in range(1, n_rows+1)]
        manual_df = pd.DataFrame({
            'Month': default_dates,
        })
        for col in all_columns:
            manual_df[col] = np.nan
        manual_input = st.data_editor(manual_df, num_rows="dynamic", key="manual_input")

# --- Data Preview/Edit ---
data_to_forecast = None
if uploaded_file and preview_btn:
    df = pd.read_excel(uploaded_file)
    st.subheader("Preview & Edit Uploaded Data")
    data_to_forecast = st.data_editor(df, num_rows="dynamic", key="excel_preview")
elif not uploaded_file:
    st.subheader("Manual Data Entry Preview")
    data_to_forecast = manual_input

# --- Forecast Button ---
forecast_btn = st.button("Forecast", type="primary")

# --- Forecast Logic ---
def feature_engineer(future_df, last_actuals, features):
    # Add cyclical features
    future_df['Month'] = pd.to_datetime(future_df['Month'])
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['Month'].dt.month / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['Month'].dt.month / 12)
    future_df['quarter'] = future_df['Month'].dt.quarter
    future_df['quarter_sin'] = np.sin(2 * np.pi * future_df['quarter'] / 4)
    future_df['quarter_cos'] = np.cos(2 * np.pi * future_df['quarter'] / 4)
    # Load last actuals for lags/rolling
    last_actuals = last_actuals.copy()
    history = pd.concat([last_actuals, future_df], ignore_index=True, sort=False)
    top_vars = [col for col in last_actuals.columns if col in features and col not in ['Month', 'India total Consumption of Natural Gas (in BCM)']]
    N_LAGS = 6
    for i in range(len(future_df)):
        idx = last_actuals.shape[0] + i
        for v in top_vars:
            for lag in [1, 3, 6]:
                history.at[idx, f'{v}_lag{lag}'] = history.at[idx-lag, v] if idx-lag >= 0 else np.nan
            for window in [3, 6, 12]:
                history.at[idx, f'{v}_rollmean{window}'] = history[v][max(0, idx-window):idx].mean() if idx >= window else np.nan
                history.at[idx, f'{v}_rollstd{window}'] = history[v][max(0, idx-window):idx].std() if idx >= window else np.nan
    return history.iloc[last_actuals.shape[0]:][features]

if forecast_btn and data_to_forecast is not None:
    # Load last actuals for lags/rolling
    last_actuals = pd.read_csv('last_actuals_full.csv')
    last_actuals['Month'] = pd.to_datetime(last_actuals['Month'])
    # Feature engineering
    X_future = feature_engineer(data_to_forecast.copy(), last_actuals, features)
    # Fill NA with 0 or forward fill for demo (production: handle more robustly)
    X_future = X_future.fillna(method='ffill').fillna(0)
    preds = model.predict(X_future)
    # Clamp to 5.6-6.5 as in your script
    preds = np.clip(preds, 5.6, 6.5)
    forecast_df = data_to_forecast[['Month']].copy()
    forecast_df['Forecasted_Natural_Gas_Consumption'] = preds
    st.subheader("📈 Forecast Table")
    st.dataframe(forecast_df, use_container_width=True)
    # --- Chart ---
    st.subheader(":bar_chart: Actual vs Forecast Chart")
    last_actuals = get_last_actuals(20)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(last_actuals['Month'], last_actuals['India total Consumption of Natural Gas (in BCM)'], label='Actual', marker='o')
    ax.plot(forecast_df['Month'], forecast_df['Forecasted_Natural_Gas_Consumption'], label='Forecast', marker='o', color='orange')
    ax.set_xlabel('Month')
    ax.set_ylabel('Natural Gas Consumption (BCM)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    # Download option
    towrite = BytesIO()
    forecast_df.to_excel(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="forecast_results.xlsx"><button style="background-color:#0066cc;color:white;padding:8px 16px;border:none;border-radius:8px;font-weight:bold;">Download Forecast as Excel</button></a>', unsafe_allow_html=True)

st.markdown("""
---
<small>Made with ❤️ using Streamlit | © 2025 Natural Gas Demand Forecast</small>
""")
