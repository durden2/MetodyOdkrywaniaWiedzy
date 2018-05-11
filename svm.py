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

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model

df.dropna(inplace=True)

X = np.array(df.drop(['PM', 'DATE', 'NE', 'NW', 'SE'], 1))
y = np.array(df['PM'])

windowSize = 72
window_test_size = 24

# X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)
# clf = svm.NuSVR(kernel='rbf', gamma=0.03, nu=0.05)
#
# clf.fit(X_train2, y_train2)
# confidence = clf.score(X_test2, y_test2)
# print(confidence)

i = windowSize

X_test = np.array([[0,0,0,0,0,0]])
y_test = np.array([0])

while i < len(df) - windowSize:
    #create test date variables
    idx = list(range(i, i + window_test_size))
    X_test = np.append(X_test, X[idx], axis=0)
    y_test = np.append(y_test, y[idx])
    i = i + windowSize + window_test_size

i = 0

newX = np.array([[0,0,0,0,0,0]])
newY = np.array([0])

while i < len(df) - windowSize:
    #create test date variables
    idx = list(range(i, i + windowSize))
    newX = np.append(newX, X[idx], axis=0)
    newY = np.append(newY, y[idx])
    i = i + windowSize + window_test_size


X = newX
y = newY

dane = np.array([0, 0, 0], ndmin=2)

for z in range(50, 51):
    for j in range(50, 51):

        clf = svm.NuSVR(kernel='rbf', gamma=(z / 100), nu=(j / 100))
        i = 0;
        clf.fit(X, y)

        confidence = clf.score(X_test, y_test)
        #print("Gamma: ", z / 100, " NU: ", j / 100)
        dane = np.append(dane, [z / 100, j / 100, confidence]);
        print("Confidence: ", confidence)

print(dane)
# for z in range(1, 20):
#
#     clf = svm.SVC(kernel='rbf')
#     i = 0;
#     while i < len(X) - windowSize:
#         idx = list(range(i, i + windowSize))
#         i = i + windowSize;
#         trainTemp = X[idx]
#         ytemp = y[idx]
#
#         clf.fit(trainTemp, ytemp)
#
#     confidence = clf.score(X_test, y_test)
#     print("Gamma: ", z/1000, " C: ", z / 100)
#     print("Confidence: ", confidence)


forecast_col = 'PM'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['PM'] = df[forecast_col].shift(-forecast_out)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
forecast_set = clf.predict(X[1:200])

print(forecast_set)

plt.plot(forecast_set)
plt.legend(loc=4)
plt.xlabel('Data')
plt.ylabel('PM')
plt.show()
