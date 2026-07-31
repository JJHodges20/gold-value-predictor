# Gold Value Predictor
## Project Context & Development Guide

Last Updated: July 2026

---

# Project Vision

The Gold Value Predictor is a long-term Python software engineering project intended to become a polished financial analytics application rather than simply a forecasting script.

The primary goal is educational while maintaining professional software architecture. Every module should be reusable, testable, and separated by responsibility.

The project is designed so that future interfaces (CLI, desktop GUI, web application, or API) can all consume the same analytics layer without modification.

The application intentionally avoids making unrealistic claims about predicting gold prices. Forecasting is presented as hypothetical scenario analysis based on historical growth assumptions rather than guaranteed future performance.

---

# Design Philosophy

The project follows a layered architecture.

```
Data
        ↓
Analytics
        ↓
Visualizations
        ↓
CLI / Dashboard / Future UI
```

Each layer has one responsibility.

## Data Layer

Responsible for:

- Loading datasets
- Cleaning datasets
- Merging datasets
- Inflation data
- Gold price history

This layer should never perform analytics.

---

## Analytics Layer

Responsible for all calculations.

Analytics modules include:

- loader.py
- returns.py
- rolling.py
- inflation.py
- forecasting.py

Analytics should never produce charts.

Analytics should only return:

- Series
- DataFrames
- Dictionaries
- Primitive values

Visualization code should never contain financial calculations.

---

## Visualization Layer

Responsible only for rendering charts.

Charts should receive prepared data from analytics functions.

Charts should never duplicate calculations.

Every chart should:

- Return Figure
- Return Axes

Never call plt.show() internally.

Never save files automatically.

---

# Current Project Status

## Completed

### Historical Charts

✔ Historical price

✔ Recent price

---

### Inflation Charts

✔ Inflation-adjusted prices

✔ Nominal vs inflation-adjusted comparison

---

### Return Charts

✔ Monthly returns

✔ Cumulative returns

✔ Annual returns

✔ Return distribution

---

### Rolling Analysis Charts

✔ Rolling average

✔ Rolling volatility

✔ Rolling return

✔ Rolling drawdown

✔ Rolling high / low

---

### Analytics

Completed modules:

✔ loader.py

✔ returns.py

✔ rolling.py

✔ inflation.py

Forecasting module currently under development.

---

# Remaining Roadmap

## Section 7 — Forecasting

Current status:

Validation complete.

Remaining work:

### 7.2

Core forecast calculations

- project_future_value()
- generate_forecast_series()

### 7.3

Scenario forecasting

- Conservative
- Expected
- Optimistic

### 7.4

Forecast summaries

- build_forecast_summary()
- generate_forecast_summary()
- print_forecast_summary()

### 7.5

Forecast visualizations

- Historical + Forecast chart
- Forecast Scenario chart

---

## Section 8 — Export Utilities

Create a dedicated export module.

Goals:

- Save charts
- Batch export every visualization
- Export PNG
- Export high-DPI images
- Optional PDF support

Visualization code should never manage export directories.

---

## Section 9 — Polish

Improve overall appearance.

Includes:

Consistent styling

Professional titles

Better legends

Better annotations

Unified spacing

Improved typography

Optional dark theme

Optional project-wide color palette

The objective is to make every chart publication quality.

---

## Section 10 — Dashboard Integration

Long-term objective.

Instead of:

plt.show()

the project should eventually support:

generate_dashboard()

which creates a complete analytics dashboard from one function call.

Possible future interfaces:

CLI

Tkinter

PySide

Streamlit

Flask

FastAPI

The dashboard should consume existing chart functions without rewriting analytics.

---

# Coding Standards

Every module follows the same structure.

Imports

Constants

Validation

Core calculations

Summary builders

Formatting helpers

Console output

main()

Public functions require:

Complete docstrings

Type hints

Validation

Clear naming

No duplicated calculations

No hidden side effects

---

# Architecture Rules

Analytics modules may depend on:

loader.py

returns.py

rolling.py

inflation.py

Visualization modules may depend on analytics.

Analytics should never depend on visualization.

No circular imports.

---

# Forecasting Philosophy

Forecasting is intentionally deterministic.

The project does NOT claim to predict future gold prices.

Forecasts represent hypothetical outcomes assuming a chosen annual growth rate.

Preferred visualizations:

Historical + Forecast

Scenario Forecasts

Conservative

Expected

Optimistic

Statistical confidence intervals are intentionally omitted until genuine probabilistic forecasting models are implemented.

Future forecasting engines may include:

Monte Carlo

ARIMA

Prophet

Machine Learning

Bayesian models

The chart API should be flexible enough to support those models later.

---

# Testing Philosophy

Every completed section should satisfy:

Module compiles.

Preview charts render.

Validation functions tested.

No duplicated calculations.

Charts use analytics rather than implementing their own math.

---

# Current Goal

Complete Forecasting (Section 7).

After forecasting is complete, continue with export utilities, project polish, and finally dashboard integration.

The finished application should resemble a professional financial analytics library rather than a classroom exercise.