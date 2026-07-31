import matplotlib.pyplot as plt

from analytics.loader import load_master_data
from visualizations.charts import (
    plot_rolling_average,
    plot_rolling_drawdown,
    plot_rolling_high_low,
    plot_rolling_return,
    plot_rolling_volatility,
)

from visualizations.charts import (
    plot_historical_price,
    plot_rolling_average,
    plot_rolling_drawdown,
    plot_rolling_high_low,
    plot_rolling_return,
    plot_rolling_volatility,
)

data = load_master_data()

figure, axes = plot_historical_price(
    data
)

plt.show()

from visualizations.charts import plot_forecast

data = load_master_data()


figure, axes = plot_rolling_average(data)

plt.show()


figure, axes = plot_rolling_volatility(data)

plt.show()


figure, axes = plot_rolling_return(data)

plt.show()


figure, axes = plot_rolling_drawdown(data)

plt.show()


figure, axes = plot_rolling_high_low(data)

plt.show()

figure, axes = plot_forecast(
    data,
    annual_growth_rate=0.05,
    forecast_years=5,
    history_years=10,
)

plt.show()

from visualizations.charts import (
    plot_forecast_scenarios,
)


figure, axes = plot_forecast_scenarios(
    data,
    forecast_years=5,
    history_years=10,
)

plt.show()