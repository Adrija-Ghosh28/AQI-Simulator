# TinyML Based AQI Prediction System using ESP32 and TensorFlow Lite

## Overview

This project is a TinyML-based Air Quality Index (AQI) prediction system designed for edge and embedded devices such as ESP32.

The system simulates optical particulate sensor behavior, generates synthetic AQI datasets, trains machine learning models, and deploys lightweight inference models using TensorFlow Lite.

The project demonstrates:

- Synthetic sensor data generation
- Optical sensor simulation
- Feature engineering
- Linear Regression modeling
- TensorFlow Lite conversion
- TinyML deployment workflow
- Embedded AI prediction pipeline
- AQI classification system

---

# Features

- Synthetic AQI dataset generator
- Realistic optical sensor simulation
- Humidity scattering simulation
- Sensor drift simulation
- Feature engineered ML pipeline
- Linear Regression baseline model
- TensorFlow Lite model generation
- ESP32-compatible prediction equation
- AQI classification system
- Prediction smoothing
- TinyML-ready deployment workflow

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| NumPy | Numerical operations |
| Pandas | Dataset handling |
| Scikit-learn | Machine learning |
| TensorFlow | Neural network training |
| TensorFlow Lite | TinyML deployment |
| Matplotlib | Data visualization |
| ESP32 | Embedded deployment |

---

# Installation

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow
```

---

# Project Workflow

```text
Synthetic Sensor Data
        ↓
Feature Engineering
        ↓
Machine Learning Model
        ↓
PM2.5 Prediction
        ↓
AQI Classification
        ↓
TensorFlow Lite Conversion
        ↓
ESP32 Deployment
```

---

# Dataset Generator

The dataset generator simulates realistic optical particulate sensor behavior.

## Simulated Parameters

- Temperature
- Humidity
- Optical scattering
- Sensor noise
- Sensor drift
- Signal variance
- PM2.5
- PM10

---

# Optical Sensor Model

The optical sensor response is modeled as:

```text
avg = (true_pm25 ^ 1.12) * 135
```

Additional environmental effects include:

- Humidity scattering
- Random optical noise
- Sensor aging drift
- Signal instability

---

# Generated Dataset Columns

| Column | Description |
|---|---|
| avg | Optical sensor average signal |
| variance | Signal fluctuation |
| temp | Temperature |
| humidity | Relative humidity |
| pm2_5 | PM2.5 concentration |
| pm10 | PM10 concentration |
| aqi | AQI category |

---

# Feature Engineering

Two lightweight embedded-friendly engineered features are used.

## Variance Normalization

```python
var_norm = variance / (avg + 1)
```

## Square Root Feature

```python
avg_sqrt = sqrt(avg)
```

These improve:

- Non-linearity handling
- Noise robustness
- Embedded inference efficiency

---

# Machine Learning Models

## 1. Linear Regression Model

A lightweight regression model is trained using:

- avg
- avg_sqrt
- var_norm
- humidity
- temp

This model is computationally inexpensive and suitable for embedded deployment.

---

## 2. TensorFlow Lite Model

A TinyML-compatible TensorFlow Lite model is generated for deployment on embedded systems.

Benefits:

- Lightweight inference
- Low memory usage
- Fast execution
- Edge AI compatibility
- ESP32 deployment readiness

---

# Evaluation Metrics

The following evaluation metrics are used:

## MAE

Mean Absolute Error

## RMSE

Root Mean Squared Error

These metrics measure prediction accuracy between actual and predicted PM2.5 values.

---

# AQI Classification

| PM2.5 Range | AQI Category |
|---|---|
| 0–30 | Good |
| 31–60 | Satisfactory |
| 61–90 | Moderate |
| 91–120 | Poor |
| 121–250 | Very Poor |
| >250 | Severe |

---

# TensorFlow Lite Conversion

The trained model is converted using:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

Generated model file:

```text
aqi_model.tflite
```

---

# ESP32 Deployment

The `.tflite` model can be converted into a C header file:

```bash
xxd -i aqi_model.tflite > aqi_model.h
```

The generated header can be embedded directly into ESP32 firmware.

---

# ESP32 Prediction Pipeline

```text
Sensor Readings
      ↓
Feature Engineering
      ↓
TFLite Inference
      ↓
PM2.5 Prediction
      ↓
AQI Classification
```

---

# Prediction Smoothing

Exponential smoothing is applied to stabilize prediction output.

```python
prev = alpha * v + (1 - alpha) * prev
```

Benefits:

- Reduced sensor fluctuation
- Stable display values
- Better real-time behavior

---

# Visualization

The project visualizes:

- Actual vs Predicted PM2.5
- Regression performance
- Prediction behavior

This helps evaluate model quality and prediction consistency.

---

# Applications

- Smart AQI monitors
- Indoor air quality systems
- IoT environmental monitoring
- TinyML edge devices
- Embedded pollution tracking
- Smart city systems

---

# Advantages

- Lightweight computation
- Embedded-friendly architecture
- TinyML ready
- Real-time capable
- Low memory usage
- Portable deployment pipeline

---

# Limitations

- Synthetic dataset
- No real sensor calibration
- Limited environmental parameters
- Basic neural architecture

---

# Future Improvements

- Real sensor integration
- OLED display dashboard
- Cloud connectivity
- Mobile application support
- Quantized TinyML models
- Real-time calibration
- Multi-sensor fusion

---

# Conclusion

This project demonstrates a complete TinyML workflow for AQI prediction using synthetic sensor modeling, machine learning, TensorFlow Lite conversion, and embedded deployment readiness.

The system is optimized for lightweight edge devices such as ESP32 while maintaining efficient real-time inference capability.

---

# Authors

- Pritam Mondal
- Adrija Ghosh

---

# Project Type

TinyML and Embedded AI based AQI Prediction System

This project demonstrates:

- Synthetic sensor data generation
- Machine learning based AQI prediction
- TensorFlow Lite conversion
- Embedded deployment workflow for ESP32
- Lightweight edge inference techniques
