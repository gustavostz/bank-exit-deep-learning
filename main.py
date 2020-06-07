# Part 1 - Data Preprocessing

# Importing the libraries
# %%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Importing the dataset
dataset = pd.read_csv(r'.\data\Churn_Modelling.csv')

# %%
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# %%

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# %%
# Dummy variables
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
# Avoid Dummy variable trap
X = X[:, 1:]

# %%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# %%
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
# Part 2 - Now let's make the ANN!

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the frist hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# %%
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=25)

# %%
# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred) * 100
print("The accuracy was {}%".format(accuracy))

# %%
