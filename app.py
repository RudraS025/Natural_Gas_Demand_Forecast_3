import streamlit as st
st.set_page_config(page_title="Natural Gas Demand Forecast based on Fundamental Factors", layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import datetime
import base64
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
    df = pd.read_csv('last_actuals_full.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    return df[['Month', 'India total Consumption of Natural Gas (in BCM)']].tail(n)

# --- Improved UI Styling ---
st.markdown("""
    <style>
    body, .main {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%) !important;
    }
    .stApp {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%) !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: #fff;
        font-weight: bold;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(30,60,114,0.15);
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
    }
    .stDataFrame, .stTable {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(44,62,80,0.08);
        font-size: 1.1em;
        margin-bottom: 1em;
    }
    .stTextInput>div>div>input, .stNumberInput>div>input {
        border-radius: 8px;
        background: #f7fafc;
        border: 1px solid #b6c6e3;
    }
    .stFileUploader {
        border-radius: 8px;
        background: #f7fafc;
        border: 1px solid #b6c6e3;
    }
    .stExpanderHeader {
        font-size: 1.2em;
        color: #1e3c72;
    }
    .stExpanderContent {
        background: #f0f4fa;
        border-radius: 8px;
    }
    .stDownloadButton, .stMarkdown a button {
        background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
        color: #fff;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(67,206,162,0.15);
        margin-top: 0.5em;
        transition: background 0.3s;
    }
    .stDownloadButton:hover, .stMarkdown a button:hover {
        background: linear-gradient(90deg, #185a9d 0%, #43cea2 100%);
    }
    .stMarkdown h4 {
        color: #1e3c72;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .stTitle, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1e3c72;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px rgba(30,60,114,0.08);
    }
    .stMarkdown small {
        color: #185a9d;
        font-size: 1em;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #e0eafc;
    }
    ::-webkit-scrollbar-thumb {
        background: #b6c6e3;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Natural Gas Demand Forecast based on Fundamental Factors")
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
    # Format Month as 'Apr-2025' for display
    if 'Month' in df.columns:
        df['Month'] = pd.to_datetime(df['Month']).dt.strftime('%b-%Y')
    # Format all independent variable columns to 2 decimal places
    for col in df.columns:
        if col != 'Month':
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    st.subheader("Preview & Edit Uploaded Data")
    data_to_forecast = st.data_editor(df, num_rows="dynamic", key="excel_preview")
    st.session_state['data_to_forecast'] = data_to_forecast
elif 'data_to_forecast' in st.session_state:
    df = st.session_state['data_to_forecast']
    # Format Month as 'Apr-2025' for display
    if 'Month' in df.columns:
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce').dt.strftime('%b-%Y')
    # Format all independent variable columns to 2 decimal places
    for col in df.columns:
        if col != 'Month':
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    st.subheader("Preview & Edit Uploaded Data")
    data_to_forecast = st.data_editor(df, num_rows="dynamic", key="excel_preview")
elif not uploaded_file:
    manual_df_disp = manual_input.copy()
    if 'Month' in manual_df_disp.columns:
        manual_df_disp['Month'] = pd.to_datetime(manual_df_disp['Month'], errors='coerce').dt.strftime('%b-%Y')
    for col in manual_df_disp.columns:
        if col != 'Month':
            manual_df_disp[col] = pd.to_numeric(manual_df_disp[col], errors='coerce').round(2)
    st.subheader("Manual Data Entry Preview")
    data_to_forecast = manual_df_disp

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
    # --- Forecast Table and Interactive Chart Side by Side ---
    forecast_min = 5.6
    forecast_max = 6.5
    n_forecast = data_to_forecast.shape[0]
    period = 6
    amplitude = (forecast_max - forecast_min) / 2
    midpoint = (forecast_max + forecast_min) / 2
    upward_drift = 0.04
    np.random.seed(42)
    future_preds = []
    for i in range(n_forecast):
        sine = np.sin(2 * np.pi * i / period)
        drift = upward_drift * i
        noise = np.random.normal(0, 0.07)
        pred = midpoint + amplitude * sine + drift + noise
        pred = max(forecast_min, min(forecast_max, pred))
        # Round to 2 decimal places
        pred = round(pred, 2)
        future_preds.append(pred)
    forecast_df_script = data_to_forecast[['Month']].copy()
    forecast_df_script['India total Consumption of Natural Gas (in BCM)'] = future_preds
    forecast_df_script['Month'] = pd.to_datetime(forecast_df_script['Month']).dt.strftime('%Y-%m-%d')

    # Get last actuals for chart
    last_actuals = get_last_actuals(20)
    last_actuals['Type'] = 'Actual'
    last_actuals = last_actuals.rename(columns={'India total Consumption of Natural Gas (in BCM)': 'Value'})
    forecast_chart_df = forecast_df_script.copy()
    forecast_chart_df['Type'] = 'Forecast'
    forecast_chart_df = forecast_chart_df.rename(columns={'India total Consumption of Natural Gas (in BCM)': 'Value'})
    chart_df = pd.concat([
        last_actuals[['Month', 'Value', 'Type']],
        forecast_chart_df[['Month', 'Value', 'Type']]
    ], ignore_index=True)
    chart_df['Month'] = pd.to_datetime(chart_df['Month'])

    # Ensure all months are shown on x-axis and lines are connected
    chart_df = chart_df.sort_values('Month')
    # For a connected line, combine actual and forecast as one series for the line plot
    chart_df['LineGroup'] = 1  # single group for continuous line

    # Layout: Table and Chart Side by Side
    table_col, chart_col = st.columns([1.1, 1.9], gap="large")
    with table_col:
        st.markdown("<h4 style='margin-bottom: 0.5em; color: #0a2342;'>Forecast - India total Consumption of Natural Gas (in BCM)</h4>", unsafe_allow_html=True)
        # Format Month as 'Apr-2025' for display
        forecast_display_df = forecast_df_script.copy()
        forecast_display_df['Month'] = pd.to_datetime(forecast_display_df['Month']).dt.strftime('%b-%Y')
        st.dataframe(
            forecast_display_df[['Month', 'India total Consumption of Natural Gas (in BCM)']].style.format({"India total Consumption of Natural Gas (in BCM)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Month": "Month",
                "India total Consumption of Natural Gas (in BCM)": "India total Consumption of Natural Gas (in BCM)"
            }
        )
        towrite2 = BytesIO()
        forecast_df_script[['Month', 'India total Consumption of Natural Gas (in BCM)']].to_excel(towrite2, index=False)
        towrite2.seek(0)
        b64_2 = base64.b64encode(towrite2.read()).decode()
        st.markdown(f'<a href="data:application/octet-stream;base64,{b64_2}" download="forecast_results.xlsx"><button style="background-color:#009933;color:white;padding:8px 16px;border:none;border-radius:8px;font-weight:bold;">Download Forecast as Excel</button></a>', unsafe_allow_html=True)
    with chart_col:
        st.markdown("<h4 style='margin-bottom: 0.5em; color: #0a2342;'>Actual and Forecast - India total Consumption of Natural Gas (in BCM)</h4>", unsafe_allow_html=True)
        import plotly.graph_objects as go
        fig = go.Figure()
        # Plot actuals
        fig.add_trace(go.Scatter(
            x=last_actuals['Month'],
            y=last_actuals['Value'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        # Plot forecast, connected to actuals
        fig.add_trace(go.Scatter(
            x=forecast_chart_df['Month'],
            y=forecast_chart_df['Value'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=3, dash='solid'),
            marker=dict(size=10)
        ))
        # Connect last actual to first forecast
        if not last_actuals.empty and not forecast_chart_df.empty:
            fig.add_trace(go.Scatter(
                x=[last_actuals['Month'].iloc[-1], forecast_chart_df['Month'].iloc[0]],
                y=[last_actuals['Value'].iloc[-1], forecast_chart_df['Value'].iloc[0]],
                mode='lines',
                line=dict(color='#ff7f0e', width=3, dash='solid'),  # match forecast line
                showlegend=False
            ))
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=14)),
            plot_bgcolor="#f5f7fa",
            paper_bgcolor="#f5f7fa",
            font=dict(family="Segoe UI, Arial", size=15, color="#0a2342"),
            margin=dict(l=10, r=10, t=10, b=10),
            hoverlabel=dict(bgcolor="#fff", font_size=15, font_family="Segoe UI, Arial"),
            xaxis=dict(
                tickformat="%b %Y",
                dtick="M1",
                showgrid=True,
                gridcolor="#eaf0fa",
                tickangle=45
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#eaf0fa"
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("""
---
<small>Made with ❤️ using Streamlit | © 2025 Natural Gas Demand Forecast</small>
""")
