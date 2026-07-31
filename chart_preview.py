import matplotlib.pyplot as plt

from analytics.loader import load_master_data
from visualizations.charts import (
    plot_rolling_average,
    plot_rolling_drawdown,
    plot_rolling_high_low,
    plot_rolling_return,
    plot_rolling_volatility,
)


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