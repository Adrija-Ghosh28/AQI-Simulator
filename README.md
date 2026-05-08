# TinyML Based AQI Prediction System using ESP32 and TensorFlow Lite

## Overview

This project is a TinyML-based Air Quality Index (AQI) prediction system designed for embedded and edge AI devices such as ESP32.

The system generates synthetic AQI sensor data, trains machine learning models, and converts the trained model into TensorFlow Lite format for lightweight embedded deployment.

The project demonstrates:

- Synthetic AQI data generation
- Feature engineering
- Machine learning based PM2.5 prediction
- AQI classification
- TensorFlow Lite conversion
- TinyML deployment workflow

---

# Repository Structure

```text
.
├── datagenerator.py
├── model.py
├── tflite.py
└── README.md
```

---

# File Descriptions

## datagenerator.py

This file generates a synthetic AQI dataset containing approximately 1500–2000 rows of environmental and optical sensor data.

### Generated Parameters

- Temperature
- Humidity
- Optical sensor average
- Signal variance
- PM2.5
- PM10
- AQI category

### Features Simulated

- Optical scattering
- Humidity effect
- Sensor noise
- Sensor drift
- Signal instability

### Output

```text
aqi_training_data.csv
```

---

## model.py

This file contains the machine learning pipeline for AQI prediction.

### Operations Performed

- Feature engineering
- Model training
- PM2.5 prediction
- AQI classification
- Performance evaluation
- Visualization

### Machine Learning Algorithm

- Linear Regression

### Engineered Features

```python
avg_sqrt
var_norm
```

### Evaluation Metrics

- MAE
- RMSE

### Visualization

- Actual vs Predicted PM2.5 graph

---

## tflite.py

This file is used for TensorFlow Lite model creation.

### Purpose

- Train TensorFlow model
- Convert trained model to `.tflite`
- Generate TinyML compatible inference model

### Output

```text
aqi_model.tflite
```

### Usage

The generated `.tflite` model can later be deployed on:

- ESP32
- Raspberry Pi
- Edge AI systems
- Embedded TinyML devices

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Development |
| NumPy | Numerical operations |
| Pandas | Dataset processing |
| Scikit-learn | Machine learning |
| TensorFlow | Neural network training |
| TensorFlow Lite | TinyML deployment |
| Matplotlib | Visualization |
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

# Dataset Features

| Column | Description |
|---|---|
| avg | Optical sensor average |
| variance | Signal fluctuation |
| temp | Temperature |
| humidity | Relative humidity |
| pm2_5 | PM2.5 concentration |
| pm10 | PM10 concentration |
| aqi | AQI category |

---

# Feature Engineering

The project uses lightweight feature engineering for embedded efficiency.

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
- Prediction quality

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

# TinyML Deployment

The TensorFlow Lite model can later be converted into a C header file for ESP32 deployment.

```bash
xxd -i aqi_model.tflite > aqi_model.h
```

This enables lightweight embedded inference on edge devices.

---

# Applications

- Smart AQI monitors
- IoT pollution monitoring
- Indoor air quality systems
- TinyML embedded systems
- Smart city applications

---

# Advantages

- Lightweight
- Embedded-friendly
- TinyML ready
- Low computational cost
- Real-time capable

---

# Future Improvements

- Real sensor integration
- OLED display support
- Cloud dashboard
- WiFi connectivity
- Quantized TinyML models
- Real-time calibration

---

# Authors

- Adrija Ghosh
- Pritam Mondal

---

# Project Type

TinyML and Embedded AI based AQI Prediction System


readme.md
