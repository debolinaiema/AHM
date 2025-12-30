import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# CONFIG
NORMAL_FILE = r"C:\Users\user\Downloads\AHM_multivariate\forecast_next_day.csv"
FED_FILE    = r"C:\Users\user\Downloads\AHM_multivariate\federated_forecast_nextday.csv"
SAVE_METRICS = r"C:\Users\user\Downloads\AHM_multivariate\comparison_metrics.csv"
SAVE_PLOT    = r"C:\Users\user\Downloads\AHM_multivariate\comparison_overlay.png"
SAVE_SUMMARY = r"C:\Users\user\Downloads\AHM_multivariate\comparison_summary.txt"

# HELPERS
def load_forecast(path):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).reset_index(drop=True)

def compute_metrics(y_true, y_pred):
    if len(y_true) == 0: return np.nan, np.nan, np.nan, np.nan
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # R² safe
    if np.var(y_true) == 0:
        r2 = "N/A (zero variance)"
    else:
        r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, r2

# LOAD
print("Loading forecasts …")
normal_df = load_forecast(NORMAL_FILE)
fed_df    = load_forecast(FED_FILE)

print(f"   Centralized rows : {len(normal_df)}")
print(f"   Federated rows   : {len(fed_df)}")

# MERGE
merged = normal_df.merge(fed_df, on="timestamp", how="inner", suffixes=("_central", "_fed"))

if merged.empty:
    raise RuntimeError("No common timestamp!")

print(f"   Overlap rows     : {len(merged)}")

# TARGETS
target_cols = ["temperature_one", "temperature_two", "vibration_x", "vibration_y", "vibration_z"]

# METRICS + ACTUAL VALUES
metrics = []
print("\n" + "="*80)
print("            ACTUAL PREDICTIONS (Denormalized Values)")
print("="*80)
for col in target_cols:
    c_col = f"{col}_central"
    f_col = f"{col}_fed"
    if c_col not in merged.columns or f_col not in merged.columns:
        continue
    c_val = merged[c_col].iloc[0]
    f_val = merged[f_col].iloc[0]
    print(f"{col:15} | Centralized: {c_val:8.4f} | Federated: {f_val:8.4f} | Diff: {abs(c_val - f_val):.6f}")

    mse, mae, rmse, r2 = compute_metrics(merged[c_col].values, merged[f_col].values)
    metrics.append({
        "Feature": col,
        "Centralized": f"{c_val:.4f}",
        "Federated": f"{f_val:.4f}",
        "Diff": f"{abs(c_val - f_val):.6f}",
        "MSE": f"{mse:.6f}",
        "MAE": f"{mae:.6f}",
        "RMSE": f"{rmse:.6f}",
        "R2": r2
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(SAVE_METRICS, index=False)
print("\n" + "="*80)
print("METRICS TABLE")
print(metrics_df[["Feature", "Centralized", "Federated", "Diff", "MSE", "MAE", "RMSE", "R2"]].to_string(index=False))
print(f"\nMetrics saved → {SAVE_METRICS}")

# PLOT
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(target_cols))
width = 0.35

central_vals = [float(metrics_df[metrics_df["Feature"] == col]["Centralized"].iloc[0]) for col in target_cols]
fed_vals     = [float(metrics_df[metrics_df["Feature"] == col]["Federated"].iloc[0]) for col in target_cols]

plt.bar(x_pos - width/2, central_vals, width, label="Centralized", color="#1f77b4")
plt.bar(x_pos + width/2, fed_vals,     width, label="Federated",   color="#ff7f0e")

plt.xticks(x_pos, target_cols, rotation=45)
plt.ylabel("Predicted Value")
plt.title("Next-Day Forecast: Centralized vs Federated", fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SAVE_PLOT, dpi=200)
plt.close()
print(f"Bar plot saved → {SAVE_PLOT}")

# SUMMARY
summary = [
    "COMPARISON SUMMARY", "="*60,
    f"Overlap: {len(merged)} row(s)",
    f"Timestamp: {merged['timestamp'].iloc[0]}",
    "", "PREDICTIONS", "-"*60
]
for _, r in metrics_df.iterrows():
    summary.append(f"{r['Feature']:<15} Central: {r['Centralized']:<8} Fed: {r['Federated']:<8} Diff: {r['Diff']}")
summary.append("\n" + "="*60)

with open(SAVE_SUMMARY, "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print(f"Summary saved → {SAVE_SUMMARY}")
print("\nComparison finished! Check the bar plot for visual difference.")