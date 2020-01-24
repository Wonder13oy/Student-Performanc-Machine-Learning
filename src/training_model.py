import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv('data/student-mat.csv', sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best = 0
for _ in range(30):
    # SEPARATING DATA INTO TRAINING AND TESTING DATA(10% OF THE DATA)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # CREATING THE MODEL --> Y = MX + B OR Y = B + M1X1 ... + MnXn
    lm = linear_model.LinearRegression()
    lm.fit(x_train, y_train) # TRAINING THE MODEL
    acc = lm.score(x_test, y_test) # TESTING THE MODEL --> GETTING ACCURACY SCORE
    print(acc)

    if acc > best:
        best = acc
        # SAVING THE MODEL
        with open("models/Student_Model.pickle", "wb") as f:
            pickle.dump(lm, f)