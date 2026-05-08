# =========================================
# AQI SENSOR DATA GENERATOR (FINAL VERSION)
# =========================================

import pandas as pd
import numpy as np
import random

# ----------------------------
# CONFIG
# ----------------------------
N = 1500   # >= 1000 (you can increase)

data = []
drift = 0

for i in range(N):

    # ----------------------------
    # ENVIRONMENT
    # ----------------------------
    temp = random.uniform(22, 35)          # °C
    humidity = random.uniform(40, 85)      # %

    # ----------------------------
    # TRUE PM (hidden ground truth)
    # ----------------------------
    true_pm25 = random.uniform(5, 200)

    # ----------------------------
    # OPTICAL SENSOR MODEL
    # ----------------------------

    # Non-linear optical response
    avg = (true_pm25 ** 1.15) * 140

    # Humidity increases scattering
    avg *= (1 + humidity * 0.004)

    # Noise (random)
    avg += random.uniform(-200, 200)

    # Drift (slow change over time)
    drift += random.uniform(0, 0.3)
    avg += drift

    # Prevent unrealistic values
    avg = max(500, avg)

    # ----------------------------
    # VARIANCE (signal instability)
    # ----------------------------
    variance = avg * random.uniform(0.01, 0.04)

    # Humidity adds noise
    variance += humidity * random.uniform(1, 2)

    # ----------------------------
    # SENSOR OUTPUT (IMPERFECT PM)
    # ----------------------------
    pm2_5 = true_pm25 + random.uniform(-4, 4)
    pm10 = pm2_5 * random.uniform(1.2, 1.5)

    # ----------------------------
    # AQI CATEGORY
    # ----------------------------
    if true_pm25 <= 30:
        aqi = "good"
    elif true_pm25 <= 60:
        aqi = "satisfactory"
    elif true_pm25 <= 90:
        aqi = "moderate"
    elif true_pm25 <= 120:
        aqi = "poor"
    elif true_pm25 <= 250:
        aqi = "very poor"
    else:
        aqi = "severe"

    # ----------------------------
    # STORE RECORD
    # ----------------------------
    data.append({
        "avg": avg,
        "variance": variance,
        "temp": temp,
        "humidity": humidity,
        "pm2_5": pm2_5,
        "pm10": pm10,
        "aqi": aqi
    })


# ----------------------------
# CREATE DATAFRAME
# ----------------------------
df = pd.DataFrame(data)

print("Sample Data:")
print(df.head())

print("\nTotal Rows:", len(df))


# ----------------------------
# SAVE DATASET
# ----------------------------
df.to_csv("aqi_training_data.csv", index=False)

print("\nDataset saved as aqi_training_data.csv")
