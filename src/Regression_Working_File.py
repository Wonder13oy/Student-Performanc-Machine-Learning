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

# SEPARATING DATA INTO TRAINING AND TESTING DATA(10% OF THE DATA)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# LOADING THE MODEL
pickle_in = open("models/Student_Model.pickle", "rb")
lm = pickle.load(pickle_in)

# DISPLAYING THE COEFFICIENTS(M GRADIENTS) AND Y-INTERCEPTS
print(f'Coefficient: {lm.coef_}')
print(f'Intercept: {lm.intercept_}')

# USED TO PREDICT GRADES
predictions = lm.predict(x_test)

for x in range(len(predictions)):
    print(f'Predictions: {predictions[x]}\nInput Data: {x_test[x]}\nActual Value: {y_test[x]}')

# PLOTTING THE DATA
p='G1'
style.use('ggplot')
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel('Final grade')
plt.show()