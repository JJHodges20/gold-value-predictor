import matplotlib.pyplot as plt

from analytics.loader import load_master_data
from visualizations.charts import (
    plot_rolling_high_low,
)


data = load_master_data()

figure, axes = plot_rolling_high_low(
    data=data,
    column="Gold Price",
    window=12,
)

plt.show()