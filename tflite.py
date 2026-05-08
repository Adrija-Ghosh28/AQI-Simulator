# =========================================
# STEP 1: IMPORTS
# =========================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# =========================================
# STEP 2: FEATURE ENGINEERING
# =========================================

# Normalize variance
df["var_norm"] = df["variance"] / (df["avg"] + 1)

# Light non-linearity
df["avg_sqrt"] = np.sqrt(df["avg"])

features = ["avg", "avg_sqrt", "var_norm", "humidity", "temp"]

X = df[features].values.astype(np.float32)
y = df["pm2_5"].values.astype(np.float32)


# =========================================
# STEP 3: BUILD MODEL
# =========================================

# Linear model equivalent to LinearRegression
model = keras.Sequential([
    keras.layers.Dense(
        1,
        input_shape=(5,),
        activation="linear"
    )
])

model.compile(
    optimizer="adam",
    loss="mse"
)


# =========================================
# STEP 4: TRAIN MODEL
# =========================================
model.fit(
    X,
    y,
    epochs=200,
    verbose=1
)

print("Model trained!")


# =========================================
# STEP 5: PREDICTION
# =========================================
pred = model.predict(X).flatten()

# clamp negatives
pred = np.maximum(pred, 0)


# =========================================
# STEP 6: EVALUATION
# =========================================
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))

print("\nEvaluation:")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))


# =========================================
# STEP 7: VISUAL CHECK
# =========================================
plt.figure(figsize=(6,6))
plt.scatter(y, pred, alpha=0.5)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()


# =========================================
# STEP 8: CONVERT TO TFLITE
# =========================================

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save model
with open("aqi_model.tflite", "wb") as f:
    f.write(tflite_model)

print("\nTFLite model saved as aqi_model.tflite")


# =========================================
# STEP 9: TEST TFLITE MODEL
# =========================================

# Load interpreter
interpreter = tf.lite.Interpreter(model_path="aqi_model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example input
sample = X[0:1]

interpreter.set_tensor(
    input_details[0]['index'],
    sample
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]['index']
)

print("\nTFLite Prediction:")
print(output)


# =========================================
# STEP 10: AQI CLASSIFICATION
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
