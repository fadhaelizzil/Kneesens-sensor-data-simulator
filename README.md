# IoT-Data-Simulation-Algorithms

Production-grade simulation of wearable IoT sensor data for healthcare and embedded machine learning, starting with **3-axis accelerometer fall-detection signals** for **Kneesens**.

## Overview

This repository contains structured simulation scripts for generating realistic raw sensor data for healthcare-focused embedded systems and wearable medical device projects.

The long-term goal is to build a high-quality collection of simulation-based datasets and algorithm prototypes for:

- wearable sensing
- fall detection
- embedded AI / TinyML
- biomedical signal processing
- healthcare IoT systems

The repository simulates raw **3-axis accelerometer** data for an elderly fall-detection wearable named **Kneesens**, positioned near the knee.

## Project Motivation

Collecting large volumes of labeled hardware data is expensive and time-consuming during early-stage embedded system development. This repository addresses that gap by generating realistic synthetic sensor data that can be used for:

- rapid algorithm prototyping
- threshold logic testing
- feature engineering
- anomaly detection experiments
- ML pipeline development
- validating data ingestion and preprocessing workflows before large-scale real-world acquisition

## Current Module

### Kneesens Accelerometer Simulator

**Script:** `scripts/kneesens_simulator.py`

This module simulates raw 3-axis accelerometer measurements for two scenarios:

1. **Normal Walking**
2. **Sudden Fall**

## What The Script Models
**1. Normal Walking**

Normal walking is represented as a low-amplitude periodic gait signal. Each axis is modeled using a dominant sinusoidal gait component plus a weaker harmonic term:

<img width="394" height="40" alt="image" src="https://github.com/user-attachments/assets/f585cf2a-59b4-4699-b468-954d468b2c9b" />

This captures the fact that human gait is periodic but not perfectly sinusoidal. The Z-axis is centered near 1 g to account for gravitational acceleration measured by the accelerometer.

**2. Sudden Fall**

The fall sequence is modeled in three phases:
* **Pre-fall gait:**
  Regular walking motion.
* **Impact spike:**
  A short, high-amplitude impulse is added using a Gaussian pulse:

  <img width="248" height="70" alt="image" src="https://github.com/user-attachments/assets/76aa3b66-43e8-4a0b-bdb4-49cd0d152e8f" />

This approximates the abrupt acceleration/deceleration that occurs during impact.
* **Post-fall inactivity:**
After the impact, the wearable shows near-zero dynamic motion on X and Y, while Z remains close to 1 g, representing a mostly static body lying on the floor.

**3. Sensor Noise**

To reflect low-cost embedded hardware constraints, the script injects white Gaussian noise into each axis. This makes the dataset more realistic for downstream filtering, anomaly detection, and ML prototyping.

## Why This is Useful

This script provides a clean starting point for building a fall-detection dataset pipeline before large-scale hardware capture is available. It is useful for:

1. embedded algorithm prototyping,
2. threshold-based fall logic,
3. digital filtering experiments,
4. feature engineering,
5. machine learning baseline development,
6. validating data ingestion pipelines for future real sensor logs.
