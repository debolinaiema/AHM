import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURE INPUT FILE PATHS
# -----------------------------
actual_csv_path = r"D:\Projects(Debolina)\AHM_multivariate\data\Duo_006_6-8-25_to_13-8-25.csv"
forecast_csv_path = r"D:\Projects(Debolina)\AHM_multivariate\data\forecast_output.csv"


# -----------------------------
# LOAD DATA
# -----------------------------
df_actual = pd.read_csv(actual_csv_path)
df_forecast = pd.read_csv(forecast_csv_path)

# Convert timestamps to datetime
df_actual['timestamp'] = pd.to_datetime(df_actual['timestamp'])
df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])

# -----------------------------
# MERGE ACTUAL & FORECAST
# -----------------------------
merged = pd.merge(
    df_actual[['timestamp', 'vibration_x', 'vibration_y', 'vibration_z']],
    df_forecast[['timestamp', 'vibration_x', 'vibration_y', 'vibration_z']],
    on='timestamp',
    suffixes=('_actual', '_forecast')
)

# -----------------------------
# CALCULATE DIFFERENCE
# -----------------------------
merged['diff_x'] = merged['vibration_x_actual'] - merged['vibration_x_forecast']
merged['diff_y'] = merged['vibration_y_actual'] - merged['vibration_y_forecast']
merged['diff_z'] = merged['vibration_z_actual'] - merged['vibration_z_forecast']

# -----------------------------
# PLOT DIFFERENCES
# -----------------------------
plt.figure(figsize=(14, 7))

plt.plot(merged['timestamp'], merged['diff_x'], label='Difference X (Actual - Forecast)', color='blue')
plt.plot(merged['timestamp'], merged['diff_y'], label='Difference Y (Actual - Forecast)', color='green')
plt.plot(merged['timestamp'], merged['diff_z'], label='Difference Z (Actual - Forecast)', color='red')

plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.title("Vibration Difference Plot (Actual - Forecast)")
plt.xlabel("Timestamp")
plt.ylabel("Difference Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("✅ Difference plot generated successfully!")
