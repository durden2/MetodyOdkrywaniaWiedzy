import pandas as pd
import numpy as np
from scipy import sparse
df = pd.read_csv("new.csv")

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

y = df['PM']
X = df.drop(['PM'], axis = 1)
X = df.drop(['DATE'], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

lab_enc = preprocessing.LabelEncoder()

X_test = lab_enc.fit_transform(X_test)
X_train = lab_enc.fit_transform(X_train)
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)


# print("\nX_train:\n")
# print(X_train.head())
# print(X_train.shape)
# print ("\nX_test:\n")
# print(X_test.head())
# print (X_test.shape)

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object
model = svm.SVC(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X_train, y_train)
model.score(X_train, y_train)
#Predict Output
predicted= model.predict(X_test)

print(predicted)