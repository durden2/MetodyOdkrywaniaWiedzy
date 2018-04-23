import plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

import numpy as np
import pandas as pd
from scipy.stats import shapiro

plotly.tools.set_credentials_file(username='gandi45', api_key='1pMLdgB9it4xug1TWwUc')

#x = [145, 180, 187, 195, 204, 205, 210, 211, 218, 232, 237, 240, 256, 261, 264, 310, 378]
x = [378, 346, 245, 285, 365, 245, 208, 306, 296, 224, 292]

shapiro_results = shapiro(x)

matrix_sw = [
    ['', 'DF', 'Test Statistic', 'p-value'],
    ['Sample Data', len(x) - 1, shapiro_results[0], shapiro_results[1]]
]

shapiro_table = FF.create_table(matrix_sw, index=True)
plotly.plotly.iplot(shapiro_table, filename='shapiro-table')