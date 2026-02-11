# Cryptocurrency Price Analysis and Forecasting Framework
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

This repository contains a Python-based research framework designed to analyze historical cryptocurrency price action and generate future predictions. The project implements a multi-model approach, allowing for a direct comparison between traditional statistical forecasting and modern machine learning techniques.

## Overview

The framework evaluates four different methodologies to determine their efficacy in predicting volatile financial assets:

* **Facebook Prophet**: An additive modeling tool designed for handling seasonality and trend shifts.
* **ARIMA**: A standard statistical approach used as a baseline for non-stationary time series data.
* **SARIMA**: An extension of ARIMA that incorporates seasonal cycles, configured here for monthly patterns.
* **LSTM (Long Short-Term Memory)**: A Deep Learning architecture (RNN) trained to identify complex, non-linear dependencies in historical price sequences.

---

## Technical Setup

### Prerequisites
The environment requires Python 3.8 or higher. You will need to install the following dependencies:

```bash
pip install pandas numpy matplotlib prophet statsmodels scikit-learn keras tensorflow
```

## Data Preparation

The script is designed to work with historical data files in the root directory. Ensure your datasets are named as follows:

* `Bitcoin Historical Data.csv`
* `Ethereum Historical Data.csv`
* `Litecoin Historical Data.csv`

The internal logic automatically cleans the data, handles currency formatting (removing commas), and converts percentage strings into float values for analysis.

## Execution

To run the analysis, execute the main Python file:

```bash
python main.py
```
Upon execution, the program will prompt you for two inputs:

1. **Target Asset**: Select from BTC, ETH, or LTC.
2. **Forecast Window**: Specify the number of days you wish to project into the future.

## Visualizations

The script generates a series of Matplotlib windows displaying:

* Historical price trends and daily volatility.
* 7-day and 30-day Moving Averages to identify momentum.
* Individual forecast plots for each model, including confidence intervals for the statistical models and training/testing splits for the LSTM.

## Model Parameters

The models are configured with the following specifications:

* **LSTM Architecture**: Two stacked LSTM layers with 100 units each, utilizing 20% Dropout layers to prevent overfitting. It uses a 60-day look-back window and the Adam optimizer.
* **Prophet Configuration**: Set to multiplicative seasonality with a custom monthly period (30.5 days) and a 0.05 changepoint prior scale.
* **ARIMA/SARIMA**: Configured with an order of (5, 1, 1) to account for recent lags and differencing.

## Disclaimer

This project was developed for research and academic purposes. Cryptocurrency markets are influenced by high volatility and external factors that mathematical models may not fully capture. The outputs of this script should not be considered financial advice.
