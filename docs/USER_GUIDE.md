# Gold Value Predictor
# User Guide

---

# Overview

Welcome to the Gold Value Predictor.

This application provides professional-quality financial analytics for historical gold prices, inflation-adjusted values, returns, risk metrics, rolling statistics, hypothetical forecasting, and a consolidated analytics dashboard.

Version 1.0 has been designed with a layered architecture emphasizing:

- Reusable analytics
- Professional visualizations
- Comprehensive automated testing
- Modular design
- Future extensibility

---

# System Requirements

Recommended

- Python 3.14+
- Windows 10/11
- VS Code
- Git

The project also runs on Linux and macOS with minor command adjustments.

---

# Initial Installation

Clone the repository.

Create a virtual environment.

```powershell
python -m venv venv
```

Activate it.

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies.

```powershell
pip install -r requirements.txt
```

---

# Project Structure

```
gold-value-predictor/

analytics/
Business logic and financial calculations

data/
Processed datasets

data_sources/
Downloaders, mergers, update pipeline

visualizations/
Charts, styling, dashboard

tests/
Automated tests

exports/
Generated PNG and SVG figures

dashboard_preview.py
Launch dashboard preview

chart_preview.py
Preview individual charts

export_preview.py
Generate exported figures

visual_review.py
Generate every visualization for inspection
```

---

# Typical Development Workflow

The recommended workflow for updating and validating the project is:

```
Update Data
      ↓
Run Tests
      ↓
Preview Charts
      ↓
Preview Dashboard
      ↓
Export Figures
      ↓
Commit Changes
```

Following this order helps catch issues before exporting or publishing results.

---

# Updating the Dataset

Version 1.0 does **not** automatically refresh market data when the application starts.

To download the newest available data, run:

```powershell
python -m data_sources.updater
```

This process will:

- Download the latest available historical gold prices.
- Download the latest available FRED economic indicators.
- Validate downloaded datasets.
- Merge all datasets.
- Rebuild the processed master dataset.

The primary dataset used throughout the application is:

```
data/processed/master_dataset.csv
```

The dashboard and analytics always read from this processed dataset.

---

# Viewing the Dashboard

Launch the dashboard preview:

```powershell
python dashboard_preview.py
```

The dashboard currently contains:

- Summary KPIs
- Historical Gold Price
- Recent Gold Price
- Nominal vs. Inflation-Adjusted Price
- Monthly Returns
- Annual Returns
- Rolling Volatility
- Rolling Drawdown
- Hypothetical Forecast

---

# Previewing Individual Charts

To preview the project's individual visualizations:

```powershell
python chart_preview.py
```

Use this when developing or modifying chart layouts.

---

# Visual Review

To generate every visualization for manual inspection:

```powershell
python visual_review.py
```

This is recommended after making visualization changes.

Review:

- titles
- spacing
- colors
- annotations
- legends
- axis formatting

before exporting.

---

# Exporting Figures

Generate all exportable figures.

```powershell
python export_preview.py
```

Current export formats include:

- PNG
- SVG

Exports are written to:

```
exports/
```

---

# Running Tests

Run the complete automated test suite:

```powershell
python -m pytest
```

Run dashboard tests only:

```powershell
python -m pytest tests/test_dashboard.py
```

Run visualization tests:

```powershell
python -m pytest tests/test_charts.py
```

Run forecasting tests:

```powershell
python -m pytest tests/test_forecasting.py
```

Run a specific test:

```powershell
python -m pytest tests/test_dashboard.py -k "forecast"
```

Always ensure all tests pass before committing changes.

---

# Dashboard Overview

The dashboard is intended to tell a complete analytical story.

## Summary

Displays headline metrics:

- Latest Gold Price
- Inflation-Adjusted Price
- CAGR
- Annualized Volatility
- Maximum Drawdown
- Projected Value

---

## Historical Price

Displays the complete historical price series.

---

## Recent Price

Shows recent price movement with a shorter time horizon.

---

## Inflation Comparison

Compares nominal prices with inflation-adjusted prices.

---

## Monthly Returns

Shows month-over-month percentage changes.

---

## Annual Returns

Displays yearly returns using semantic gain/loss coloring.

---

## Rolling Volatility

Displays annualized rolling volatility.

---

## Rolling Drawdown

Shows the decline from rolling highs over time.

---

## Forecast

Displays a hypothetical compound-growth scenario.

Forecasts are **not predictions**.

They represent deterministic projections using user-defined assumptions.

---

# Configuration

Many dashboard behaviors are controlled through `DashboardConfig`.

Examples include:

- Rolling window
- Forecast years
- Annual growth assumption
- History length
- Annotation visibility
- Forecast disclaimer visibility

Whenever possible, modify configuration rather than hard-coding values.

---

# Data Refresh Policy

Version 1.0 uses a manual refresh process.

Recommended workflow:

```
Run updater
↓

Run tests

↓

Preview dashboard

↓

Export figures
```

This prevents dashboard failures caused by temporary network or API issues.

Automatic refresh is planned for Version 2.0.

---

# Troubleshooting

## Missing Module

Example:

```
ModuleNotFoundError
```

Solution:

- Activate the virtual environment.
- Install dependencies.

```powershell
pip install -r requirements.txt
```

---

## Tests Fail

Run the failing test individually.

```powershell
python -m pytest tests/test_dashboard.py -v
```

Read the first error carefully before attempting additional fixes.

---

## Dashboard Appears Empty

Confirm:

- master_dataset.csv exists.
- Data updater completed successfully.
- Dashboard preview is loading the processed dataset.
- Tests pass.

---

## Charts Look Incorrect

Run:

```powershell
python visual_review.py
```

Compare all charts against previous exports.

---

## Export Missing

Run:

```powershell
python export_preview.py
```

Verify the export directory exists.

---

# Version 1.0 Limitations

Version 1.0 intentionally does not include:

- Live market feeds
- Automatic refresh
- Interactive forecasting
- Machine learning forecasts
- Monte Carlo simulations
- Portfolio comparisons
- Desktop GUI
- Web application

These features are planned for future releases.

---

# Planned Version 2.0

Current roadmap includes:

## Forecasting

- Interactive forecasting
- User-defined assumptions
- Multiple scenarios
- Monte Carlo simulation
- ARIMA
- Prophet
- Machine learning

---

## Dashboard

- Interactive controls
- Theme support
- Additional KPI cards
- Dynamic filtering

---

## Data

- Automatic refresh
- Refresh status
- Scheduled updates

---

## Interfaces

- Desktop application
- Web dashboard
- REST API

---

# Recommended Maintenance Routine

Whenever new market data becomes available:

```
1. Activate virtual environment

2. Run updater

3. Run complete test suite

4. Review charts

5. Review dashboard

6. Export figures

7. Commit changes
```

Following this workflow ensures analytics, visualizations, and exported reports remain synchronized.

---

# Documentation

This repository includes:

- Development_Log.md
- VERSION_HISTORY.md
- TECHNICAL_ARCHITECTURE.md
- USER_GUIDE.md

Together these documents describe:

- Project evolution
- Release history
- Software architecture
- Daily usage and maintenance

---

# Final Notes

Gold Value Predictor Version 1.0 was designed as both a financial analytics platform and a software engineering project.

The application emphasizes:

- clean architecture
- maintainability
- reusable analytics
- professional visualization
- comprehensive testing

Future versions are expected to extend functionality while preserving the modular architecture established in Version 1.0.

---

**End of User Guide**