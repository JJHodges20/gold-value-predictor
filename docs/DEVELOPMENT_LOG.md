# Gold Value Predictor
## Development Log

---

# Project Information

Project Name:
Gold Value Predictor

Purpose:
Develop a professional-quality financial analytics application for analyzing historical gold prices, inflation-adjusted values, rolling statistics, returns, and hypothetical future growth scenarios.

Primary Language:
Python

Architecture:
Layered (Data → Analytics → Visualization → Interface)

---

# Development Timeline

---

## Session 1
### Initial Project Planning

Goals

- Define project vision.
- Establish layered architecture.
- Separate analytics from visualization.
- Build reusable modules instead of one large script.

Major Decisions

✔ Analytics performs calculations.

✔ Visualization only plots.

✔ CLI should consume analytics.

✔ Future GUI should reuse analytics without modification.

Result

Project architecture established.

---

## Session 2
### Historical Price Analysis

Completed

✔ Historical price chart

✔ Recent price chart

Highlights

- Standardized chart helpers.
- Shared formatting functions.
- Currency axis formatting.
- Consistent figure sizing.

Files

visualizations/charts.py

Status

Complete

---

## Session 3
### Inflation Analysis

Completed

✔ Inflation-adjusted prices

✔ Nominal vs. inflation-adjusted comparison

Major Improvement

Refactored charts to use

calculate_real_price()

instead of duplicating inflation calculations.

Bug Fixed

NameError

calculate_real_price

resolved by importing the analytics function.

Files

analytics/inflation.py

visualizations/charts.py

Status

Complete

---

## Session 4
### Return Analysis

Completed

✔ Monthly Returns

✔ Cumulative Returns

✔ Annual Returns

✔ Return Distribution

Major Improvement

Charts now consume

analytics/returns.py

instead of recalculating returns.

Bug Fixed

Incorrect helper name

format_percent_axis()

changed to

format_percentage_axis()

Additional Improvement

Extended

format_percentage_axis()

to support both X and Y axes.

Status

Complete

---

## Session 5
### Rolling Analysis

Completed

✔ Rolling Average

✔ Rolling Volatility

✔ Rolling Return

✔ Rolling Drawdown

✔ Rolling High / Low

Architecture Improvements

Rolling calculations remain entirely inside

analytics/rolling.py

Charts only visualize returned Series.

Lessons Learned

dropna()

is used only for visualization.

Analytics continues returning the complete Series.

Status

Complete

---

## Session 6
### Forecasting Module

Current Progress

Section 7.1 completed.

Completed

✔ Module imports

✔ Constants

✔ Validation helpers

Functions Added

validate_years()

validate_growth_rate()

validate_months_per_year()

Validation Tests

Passed

Module compiles successfully.

Current Status

Ready for Section 7.2

---

# Current Architecture

```
Data
        ↓
Analytics
        ↓
Visualizations
        ↓
CLI / Dashboard
```

---

# Current Module Status

## Data

✔ Complete

---

## Analytics

loader.py

✔ Complete

returns.py

✔ Complete

rolling.py

✔ Complete

inflation.py

✔ Complete

forecasting.py

🚧 In Progress

Current completion:

Approximately 15%

---

## Visualizations

Historical Charts

✔ Complete

Inflation Charts

✔ Complete

Return Charts

✔ Complete

Rolling Charts

✔ Complete

Forecast Charts

Not started

---

# Known Improvements

None currently.

---

# Lessons Learned

## Analytics owns the math.

Visualization owns presentation.

Never duplicate calculations.

---

## Return Series

Charts should plot

dropna()

instead of modifying analytics.

---

## Forecast Philosophy

Forecasts are deterministic.

They represent hypothetical growth assumptions.

They are NOT predictions.

Avoid statistical confidence intervals until probabilistic forecasting models exist.

---

# Remaining Roadmap

## Section 7

Forecast Analytics

Current Step

➡ Section 7.2

Remaining

project_future_value()

generate_forecast_series()

Scenario Forecasts

Forecast Summary

Forecast Charts

---

## Section 8

Export Utilities

PNG export

Batch export

High-resolution output

---

## Section 9

Chart Polish

Unified styling

Improved annotations

Professional typography

Optional dark theme

---

## Section 10

Dashboard

Generate complete dashboard

Single function call

Reusable for:

CLI

Desktop

Web

API

---

# Future Ideas

Potential additions after Version 1.0

Monte Carlo simulations

ARIMA forecasting

Prophet

Machine Learning forecasting

Bayesian confidence intervals

Portfolio comparison

Inflation comparison dashboard

Live gold price integration

Economic indicator overlays

---

# End of Current Session

Last Completed Section

Forecasting 7.1

Next Objective

Forecasting 7.2

Estimated Overall Completion

≈ 75%

The core architecture is complete.

Remaining work focuses primarily on extending functionality, improving usability, and polishing presentation rather than redesigning the system.

## Forecasting and Forecast Charts

Completed:

- Core compound-growth projections
- Monthly forecast Series
- Default and custom scenario forecasts
- Forecast summary generation
- Forecast console reporting
- Historical-plus-forecast chart
- Scenario comparison chart
- Forecasting test coverage
- Forecast-chart test coverage

Validation:

- Forecast module compiles
- Forecast previews render
- Forecasting tests pass
- Chart tests pass
- Full test suite passes

Next:

- Section 8.1 — Export configuration and validation