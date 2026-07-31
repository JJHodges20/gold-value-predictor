# Gold Value Predictor
# Technical Architecture

---

# Overview

Gold Value Predictor is a layered financial analytics application designed around the principle of separation of responsibilities.

Each package performs a single role.

```
                    Data Sources
                          │
                          ▼
                  Data Processing
                          │
                          ▼
                    Analytics Layer
                          │
                          ▼
                 Visualization Layer
                          │
                          ▼
               Dashboard / Exports / CLI
```

The goal is to ensure that calculations, visualization, user interfaces, and data acquisition remain independent.

---

# Architectural Principles

The project follows several core principles.

## Single Responsibility

Each module should have one clearly defined responsibility.

Examples:

- Data loading
- Inflation calculations
- Forecast generation
- Chart rendering
- Dashboard composition

No module should perform multiple unrelated tasks.

---

## Reusable Analytics

All mathematical calculations belong inside the Analytics package.

Charts never calculate statistics.

Instead they consume reusable analytics.

Example:

```
Dashboard

↓

calculate_monthly_returns()

↓

Visualization
```

This prevents duplicated calculations throughout the project.

---

## Centralized Styling

Every chart shares:

- typography
- colors
- spacing
- formatting
- annotations
- legends

through reusable styling helpers.

Changing one styling helper updates the entire application.

---

## Extensive Testing

Every major feature is accompanied by automated tests whenever practical.

Testing is considered part of feature development rather than an optional step.

---

# Package Structure

```
gold-value-predictor/

│
├── analytics/
│
├── data/
│
├── data_sources/
│
├── visualizations/
│
├── tests/
│
├── exports/
│
├── dashboard_preview.py
│
├── chart_preview.py
│
├── export_preview.py
│
└── visual_review.py
```

---

# Analytics Layer

Purpose

Perform all calculations used throughout the application.

Nothing in this package produces charts.

Major modules include:

```
analytics/

inflation.py

returns.py

risk.py

rolling.py

forecasting.py

summary.py
```

Responsibilities include:

- historical calculations
- return calculations
- inflation adjustments
- rolling statistics
- forecasting
- summary statistics

---

# Visualization Layer

Purpose

Transform analytics into publication-quality figures.

Charts never calculate statistics directly.

Instead they receive prepared data from Analytics.

Major modules:

```
visualizations/

charts.py

dashboard.py

export.py

styling/
```

---

# Styling System

The styling package centralizes every visual decision.

```
styling/

colors.py

typography.py

formatting.py

annotations.py

constants.py
```

Benefits:

- consistent appearance
- minimal duplicated code
- simple future redesigns

---

# Dashboard System

The dashboard is built in stages.

```
Dashboard Figure

↓

Grid Layout

↓

Axes Creation

↓

Summary Panel

↓

Price Panels

↓

Return Panels

↓

Rolling Panels

↓

Forecast Panel
```

Each panel is populated independently.

This keeps the dashboard modular and easy to extend.

---

# Data Flow

```
CSV Files

↓

Loader

↓

Validation

↓

Analytics

↓

Charts

↓

Dashboard

↓

Exports
```

Every layer receives validated data from the previous layer.

---

# Data Sources

Current data includes:

- Historical Gold Prices
- Consumer Price Index (CPI)
- Federal Funds Rate
- 10-Year Treasury Yield
- Unemployment Rate
- Recession Indicator
- WTI Oil Prices
- S&P 500 Index

Processed datasets are merged into:

```
master_dataset.csv
```

which becomes the primary source used by the analytics layer.

---

# Dashboard Panels

The Version 1.0 dashboard contains nine panels.

```
Summary

Historical Price

Recent Price

Nominal vs Inflation

Monthly Returns

Annual Returns

Rolling Volatility

Rolling Drawdown

Forecast
```

Each panel owns only its visualization.

Calculations remain inside Analytics.

---

# Export System

Exports reuse the same visualization functions.

No export-specific charts exist.

Current export formats:

- PNG
- SVG

Future export formats may include:

- PDF
- HTML
- PowerPoint

---

# Testing Strategy

Testing follows multiple layers.

```
Unit Tests

↓

Visualization Tests

↓

Dashboard Tests

↓

Export Tests

↓

Integration Tests
```

The goal is to catch regressions before they reach users.

---

# Design Decisions

Several important architectural decisions were made during development.

## Analytics First

Calculations should exist once.

Charts should consume calculations rather than recreate them.

---

## Dashboard as Consumer

The dashboard is not a separate analytics engine.

It simply coordinates existing components.

---

## Configuration Driven

Behavior is controlled through configuration objects rather than hard-coded values.

Examples include:

- forecast years
- rolling windows
- annual growth assumptions
- history length

---

## Modular Growth

New analytics should be added by extending existing packages rather than modifying unrelated code.

Future features should require minimal changes to the existing architecture.

---

# Planned Evolution

Version 2.0 is expected to add:

- Interactive forecasting
- Automatic data refresh
- Multiple forecasting models
- Desktop application
- Web interface
- REST API

These additions should integrate without requiring major architectural redesign thanks to the modular structure established in Version 1.0.

---

# Architectural Summary

Gold Value Predictor follows a layered, modular architecture emphasizing:

- Separation of responsibilities
- Reusable analytics
- Centralized styling
- Automated testing
- Extensibility
- Maintainability

The result is a codebase designed to support future interfaces and advanced analytical capabilities without restructuring the underlying application.

---

**End of Technical Architecture**