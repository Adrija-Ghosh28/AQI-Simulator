# =========================================
# STEP 1: IMPORTS
# =========================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# =========================================
# STEP 2: FEATURE ENGINEERING (FINAL)
# =========================================

# Normalize variance (important for noise handling)
df["var_norm"] = df["variance"] / (df["avg"] + 1)

# Light non-linearity without heavy ops
df["avg_sqrt"] = np.sqrt(df["avg"])

features = ["avg", "avg_sqrt", "var_norm", "humidity", "temp"]

X = df[features]
y = df["pm2_5"]


# =========================================
# STEP 3: TRAIN MODEL
# =========================================
model = lr()
model.fit(X, y)

coef = model.coef_
intercept = model.intercept_

print("Model trained!")


# =========================================
# STEP 4: PREDICTION
# =========================================
pred = model.predict(X)

# clamp negatives (important for embedded)
pred = np.maximum(pred, 0)


# =========================================
# STEP 5: EVALUATION
# =========================================
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))

print("\nEvaluation:")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))


# =========================================
# STEP 6: VISUAL CHECK
# =========================================
plt.figure()
plt.scatter(y, pred, alpha=0.5)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted")
plt.show()


# =========================================
# STEP 7: FINAL EQUATION
# =========================================
print("\nFinal Model Equation:\n")

print(f"""
PM2.5 =
{coef[0]:.6f} * avg
+ {coef[1]:.6f} * sqrt(avg)
+ {coef[2]:.6f} * (variance / avg)
+ {coef[3]:.6f} * humidity
+ {coef[4]:.6f} * temp
+ {intercept:.6f}
""")


# =========================================
# STEP 8: ESP32 C CODE (FINAL)
# =========================================
print("\nESP32 C CODE:\n")

print(f"""
#include <math.h>

float predict_pm25(float avg, float variance, float temp, float humidity) {{

    float avg_sqrt = sqrtf(avg);
    float var_norm = variance / (avg + 1.0f);

    float pm =
        ({coef[0]:.6f}f * avg)
      + ({coef[1]:.6f}f * avg_sqrt)
      + ({coef[2]:.6f}f * var_norm)
      + ({coef[3]:.6f}f * humidity)
      + ({coef[4]:.6f}f * temp)
      + ({intercept:.6f}f);

    if (pm < 0) pm = 0;

    return pm;
}}
""")


# =========================================
# STEP 9: AQI CLASSIFICATION
# =========================================
def get_aqi(pm):
    if pm <= 30:
        return "Good"
    elif pm <= 60:
        return "Satisfactory"
    elif pm <= 90:
        return "Moderate"
    elif pm <= 120:
        return "Poor"
    elif pm <= 250:
        return "Very Poor"
    else:
        return "Severe"

df["pred_pm"] = pred
df["pred_aqi"] = df["pred_pm"].apply(get_aqi)

print("\nAQI Sample:")
print(df[["pred_pm", "pred_aqi"]].head())


# =========================================
# STEP 10: OPTIONAL SMOOTHING SIMULATION
# =========================================
def smooth_series(values, alpha=0.2):
    smoothed = []
    prev = values[0]
    for v in values:
        prev = alpha * v + (1 - alpha) * prev
        smoothed.append(prev)
    return smoothed

df["smoothed_pm"] = smooth_series(df["pred_pm"].values)

print("\nSmoothed Output Sample:")
print(df[["pred_pm", "smoothed_pm"]].head())
