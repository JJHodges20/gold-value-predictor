import matplotlib.pyplot as plt

from analytics.loader import load_master_data
from visualizations.charts import (
    plot_forecast,
    plot_forecast_scenarios,
    plot_historical_price,
    plot_recent_price,
)


data = load_master_data()

source = "Source: Gold Value Predictor master dataset"
brand = "Gold Value Predictor"


plot_historical_price(
    data=data,
    source_note=source,
    watermark=brand,
)

plot_recent_price(
    data=data,
    years=5,
    source_note=source,
    watermark=brand,
)

plot_forecast(
    data=data,
    forecast_years=5,
    source_note=source,
    watermark=brand,
)

plot_forecast_scenarios(
    data=data,
    forecast_years=5,
    source_note=source,
    watermark=brand,
)

plt.show()