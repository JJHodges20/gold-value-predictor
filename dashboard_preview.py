"""
Preview the Gold Value Predictor dashboard layout.
"""

import matplotlib.pyplot as plt

from analytics.loader import (
    load_master_data,
)

from visualizations.dashboard import (
    build_dashboard_layout,
)


data = load_master_data()

dashboard = build_dashboard_layout(
    data
)

plt.show()