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

The repository starts with **Day 1**, which simulates raw **3-axis accelerometer** data for an elderly fall-detection wearable named **Kneesens**, positioned near the knee.

## Project Motivation

Collecting large volumes of labeled hardware data is expensive and time-consuming during early-stage embedded system development. This repository addresses that gap by generating realistic synthetic sensor data that can be used for:

- rapid algorithm prototyping
- threshold logic testing
- feature engineering
- anomaly detection experiments
- ML pipeline development
- validating data ingestion and preprocessing workflows before large-scale real-world acquisition

## Current Module

### Day 1 — Kneesens Accelerometer Simulator

**Script:** `scripts/day01_kneesens_simulator.py`

This module simulates raw 3-axis accelerometer measurements for two scenarios:

1. **Normal Walking**
2. **Sudden Fall**

The generated output is exported to:

```bash
kneesens_fallback_data.csv
