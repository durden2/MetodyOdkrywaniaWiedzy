import pandas as pd
import numpy as np
from scipy import sparse
df = pd.read_csv("new.csv")

# Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold  # For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE


# Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    # Fit the model:
    #print(data[predictors])
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors], data[outcome])

outcome_var = 'PM'
model = DecisionTreeClassifier()
predictor_var = ['TEMP', 'PRESS', 'DEW']
classification_model(model, df,predictor_var,outcome_var)

# data_final_vars=df.columns.values.tolist()
# y=['PM']
# X = sparse.coo_matrix((df['TEMP'],(df['DEW'],df['PRESS'])),shape=(4,4)).tocsr()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4, random_state=0)
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)

# create the tuple data
#

print(df['PM'].__len__())

