# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================================================================
# 1. LOAD & CLEAN DATA
# ==============================================================================
# Read raw CSV and fix column whitespace
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()

# 1.1 Chronological Sorting: Ensure time flows correctly
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# 1.2 Interpolation: Fill gaps (up to 10 rows) to keep the timeline continuous
cols_to_fill = ['tem', 'pm2_5', 'tsr', 'hum', 'pressure', 'wind_speed', 'illumination', 'co2']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit=10)

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
# 2.1 Cyclical Time Features: Convert hour into circle coordinates (Sin/Cos)
df['hour'] = df['time'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 2.2 Lag Features: Give the model 'Memory' of what happened 1 and 2 steps ago
targets = ['tem', 'pm2_5', 'tsr']
for t in targets:
    df[f'{t}_lag1'] = df[t].shift(1)
    df[f'{t}_lag2'] = df[t].shift(2)

# 2.3 Cleanup: Drop rows where lags are NaN and save processed data
processed_df = df.dropna(subset=['tem_lag2', 'pm2_5_lag2', 'tsr_lag2', 'hum', 'pressure']).reset_index(drop=True)
processed_df.to_csv("combined_processed_weather_data.csv", index=False)

# Define the final high-performance feature set
features = [
    'hum', 'pressure', 'wind_speed', 'illumination', 'co2',
    'hour_sin', 'hour_cos', 'tem_lag1', 'pm2_5_lag1', 'tsr_lag1'
]
# Print summary of feature engineering
print("--- Feature Engineering Summary ---")
print(f"Original rows: {len(df)}")
print(f"Processed rows (after lags/cleaning): {len(processed_df)}")
print(f"Final high-performance feature set:",features)
# ==============================================================================
# 3. TRAIN-TEST SPLIT (Chronological)
# ==============================================================================
# Take the first 80% for training and the final 20% for testing (no shuffling)
split_idx = int(len(processed_df) * 0.8)
train, test = processed_df.iloc[:split_idx], processed_df.iloc[split_idx:]
X_train, X_test = train[features], test[features]
```

## Output:
<img width="1080" height="1402" alt="image" src="https://github.com/user-attachments/assets/e3408627-0564-4e5d-87c9-a06e5fbac613" />



## Result:
