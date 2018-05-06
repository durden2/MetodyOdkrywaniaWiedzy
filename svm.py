import pandas as pd
import numpy as np
from scipy import sparse
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
df = pd.read_csv("new.csv")

from sklearn.tree import DecisionTreeClassifier, export_graphviz
style.use('ggplot')

from sklearn import svm

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model

df.dropna(inplace=True)

X = np.array(df.drop(['PM', 'DATE', 'LS', 'LWS', 'LR'], 1))
y = np.array(df['PM'])
windowSize = 36

#X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)

#create test date variables
X_test = X[1:len(X):36]
y_test = y[1:len(y):36]

#remove training data from training sets
np.delete(X, np.arange(1, len(X), 36))
np.delete(X, np.arange(1, len(X), 36))


for z in range(1, 20):

    clf = svm.SVC(kernel='linear', gamma=0.001)
    i = 2;
    while i < len(X) - windowSize:
        idx = list(range(i, i + windowSize))
        i = i + windowSize;
        trainTemp = X[idx]
        ytemp = y[idx]

        clf.fit(trainTemp, ytemp)

    confidence = clf.score(X_test, y_test)
    print("Gamma: ", 10 / z, " C: ", 10 / z)
    print("Confidence: ", confidence)

d = clf.predict(X[3:6]);
print(d)

for z in range(1, 20):

    clf = svm.SVC(kernel='rbf', gamma=0.01, C=100)
    i = 2;
    while i < len(X) - windowSize:
        idx = list(range(i, i + windowSize))
        i = i + windowSize;
        trainTemp = X[idx]
        ytemp = y[idx]

        clf.fit(trainTemp, ytemp)

    confidence = clf.score(X_test, y_test)
    print("Gamma: ", z/1000, " C: ", z / 100)
    print("Confidence: ", confidence)

d = clf.predict(X[3:6]);
print(d)

forecast_col = 'PM'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['PM'] = df[forecast_col].shift(-forecast_out)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
forecast_set = clf.predict(X_lately)

plt.plot(forecast_set)
plt.legend(loc=4)
plt.xlabel('Data')
plt.ylabel('PM')
plt.show()

print(predictions)