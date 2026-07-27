# Gold Value Predictor Architecture

## Project Purpose

The Gold Value Predictor is an educational Python project that analyzes
historical gold prices alongside economic indicators.

The project does not provide financial advice or guarantee future gold prices.
Its estimates and comparisons are hypothetical and based on historical data.

## Data Pipeline

The project follows an Extract, Transform, Load workflow:

1. Extract data from external sources.
2. Save the original response in `data/raw/`.
3. Normalize dates, values, and column names.
4. Validate the cleaned datasets.
5. Save cleaned files in `data/processed/`.
6. Merge all datasets by month.
7. Calculate derived economic and gold metrics.
8. Load the final dataset into the analyzer.

## Project Structure

```text
gold-value-predictor/
│
├── data/
│   ├── raw/
│   │   ├── fred/
│   │   └── gold/
│   └── processed/
│
├── data_sources/
│   ├── __init__.py
│   ├── config.py
│   ├── fred.py
│   ├── gold.py
│   ├── merger.py
│   ├── updater.py
│   └── utils.py
│
├── docs/
│   └── architecture.md
│
├── tests/
│   ├── test_config.py
│   ├── test_fred.py
│   ├── test_gold.py
│   └── test_merger.py
│
├── .env
├── .env.example
├── .gitignore
├── gold_analyzer.py
├── README.md
└── requirements.txt