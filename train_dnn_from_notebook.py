import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ELU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# File paths
DATA_PATH = 'NaturalGasDemand_Input.xlsx'
FUTURE_PATH = 'NaturalGasDemand_Future_Input.xlsx'
MODEL_PATH = 'natural_gas_demand_dnn_model.pkl'
SCALER_PATH = 'scaler.save'
FEATURE_NAMES_PATH = 'feature_names.txt'
RESULTS_EXCEL_PATH = 'test_vs_prediction.xlsx'
FUTURE_FORECAST_PATH = 'future_forecast.xlsx'

# Load data
print('Loading data...')
df = pd.read_excel(DATA_PATH)
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month')

target_col = 'India total Consumption of Natural Gas (in BCM)'
exog_vars = [col for col in df.columns if col not in ['Month', target_col]]

# Feature engineering: lags, rolling, cyclical, interaction, polynomial
corrs = df[exog_vars + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
top_vars = corrs.index[1:4].tolist()  # top 3 correlated exog_vars
for v in top_vars:
    for lag in [1, 3, 6]:
        df[f'{v}_lag{lag}'] = df[v].shift(lag)
    for window in [3, 6, 12]:
        df[f'{v}_rollmean{window}'] = df[v].rolling(window).mean()
        df[f'{v}_rollstd{window}'] = df[v].rolling(window).std()
# Cyclical features
# Month
    
df['month_sin'] = np.sin(2 * np.pi * df['Month'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month'].dt.month / 12)
df['quarter'] = df['Month'].dt.quarter
df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
# Interaction and polynomial features
interaction_features = [f'{v1}_x_{v2}' for i, v1 in enumerate(top_vars) for v2 in top_vars[i+1:]]
poly_features = [f'{v}_squared' for v in top_vars]
for i, v1 in enumerate(top_vars):
    for v2 in top_vars[i+1:]:
        df[f'{v1}_x_{v2}'] = df[v1] * df[v2]
for v in top_vars:
    df[f'{v}_squared'] = df[v] ** 2

# Features list
features = exog_vars + \
    [f'{v}_lag{lag}' for v in top_vars for lag in [1, 3, 6]] + \
    [f'{v}_rollmean{window}' for v in top_vars for window in [3, 6, 12]] + \
    [f'{v}_rollstd{window}' for v in top_vars for window in [3, 6, 12]] + \
    ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos'] + \
    interaction_features + poly_features

# Drop NA from lag/rolling
log_target = False  # Notebook does not use log1p
df = df.dropna(subset=features + [target_col])

X = df[features]
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DNN model (use best params from notebook or grid search)
def build_dnn(input_dim, hidden_units=128, n_layers=48, dropout=0.2, lr=0.001):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=input_dim))
    model.add(ELU())
    model.add(BatchNormalization())
    for _ in range(n_layers-1):
        model.add(Dense(hidden_units))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

print('Training DNN (ELU, 48 layers, batch norm)...')
model = build_dnn(X_train_scaled.shape[1], hidden_units=128, n_layers=48, dropout=0.2, lr=0.001)
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model.fit(X_train_scaled, y_train, epochs=1000, batch_size=8, verbose=1, validation_split=0.1, callbacks=[es])

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score

def get_metrics(model, X, y, n_features):
    y_pred = model.predict(X)
    y_true = y
    y_pred_inv = y_pred.flatten()
    mse = mean_squared_error(y_true, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred_inv)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return mse, rmse, r2, adj_r2, y_pred_inv

metrics = get_metrics(model, X_test_scaled, y_test, X_test.shape[1])
print(f'DNN: MSE={metrics[0]:.4f}, RMSE={metrics[1]:.4f}, R2={metrics[2]:.4f}, AdjR2={metrics[3]:.4f}')

# Save model, scaler, features
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
with open(FEATURE_NAMES_PATH, 'w') as f:
    f.write('\n'.join(features))

# Save test vs prediction
best_pred = model.predict(X_test_scaled).flatten()
results_df = pd.DataFrame({'Actual': y_test.values, 'Prediction': best_pred}, index=y_test.index)
results_df.to_excel(RESULTS_EXCEL_PATH)

print('DNN training complete. All files saved.')

# === FUTURE FORECASTING SECTION ===
print('\nForecasting future values using DNN and future independent variables...')

future_df = pd.read_excel(FUTURE_PATH)
future_df['Month'] = pd.to_datetime(future_df['Month'])
future_df = future_df.sort_values('Month')

# Feature engineering (must match training)
future_df['month_sin'] = np.sin(2 * np.pi * future_df['Month'].dt.month / 12)
future_df['month_cos'] = np.cos(2 * np.pi * future_df['Month'].dt.month / 12)
future_df['quarter'] = future_df['Month'].dt.quarter
future_df['quarter_sin'] = np.sin(2 * np.pi * future_df['quarter'] / 4)
future_df['quarter_cos'] = np.cos(2 * np.pi * future_df['quarter'] / 4)
for v in top_vars:
    for lag in [1, 3, 6]:
        col = f'{v}_lag{lag}'
        if col not in future_df.columns:
            future_df[col] = 0
    for window in [3, 6, 12]:
        for stat in ['rollmean', 'rollstd']:
            col = f'{v}_{stat}{window}'
            if col not in future_df.columns:
                future_df[col] = 0
for i, v1 in enumerate(top_vars):
    for v2 in top_vars[i+1:]:
        col = f'{v1}_x_{v2}'
        if col not in future_df.columns:
            future_df[col] = future_df[v1] * future_df[v2]
for v in top_vars:
    col = f'{v}_squared'
    if col not in future_df.columns:
        future_df[col] = future_df[v] ** 2

with open(FEATURE_NAMES_PATH, 'r') as f:
    feature_list = [line.strip() for line in f.readlines()]
for col in feature_list:
    if col not in future_df.columns:
        future_df[col] = 0
X_future = future_df[feature_list]
X_future_scaled = scaler.transform(X_future)
future_pred = model.predict(X_future_scaled).flatten()
future_results = future_df[['Month']].copy()
future_results['Forecasted_Natural_Gas_Consumption'] = future_pred
future_results.to_excel(FUTURE_FORECAST_PATH, index=False)
print('Future forecast complete. Results saved to future_forecast.xlsx')

# Print min/max of each feature for future input
describe_future = X_future.describe().T[['min', 'max']]
print('\nMin/Max of each feature in future input:')
print(describe_future)
describe_future.to_excel('future_features_minmax.xlsx')
