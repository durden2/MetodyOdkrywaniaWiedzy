import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


df = pd.read_csv("new.csv")
print(df.describe())
df['PM'].hist(bins=50)
df['TEMP'].hist(bins=100)
df['DEW'].hist(bins=50)
#df.boxplot(column='DEW')
plt.show()
