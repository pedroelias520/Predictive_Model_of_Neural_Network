#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#MNIST DATASET

@author: rangelnunes
"""

# carregando o dataset MNIST
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd


titanic_train = pd.read_csv("train.csv")
print(titanic_train[0])

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

y = y.astype(np.float64)


# exibindo um digito qualquer
some_digit = X[44000]

import matplotlib.pyplot as plt
plt.imshow(some_digit.reshape((28, 28)), cmap='gray')
plt.show()

# verificando o rotulo
y[44000]

# dividindo a base entre treinamento e teste
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# embaralhando o conjunto de treinamento

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# treinando um classificador binário
y_train_0 = (y_train == 0)

y_test_0 = (y_test == 0)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)

# testando o classificador
sgd_clf.predict([some_digit])

# imprimindo algumuas previsoes
for index in range(1000, 70000, 1000):
    digit = X[index]
    predicted = sgd_clf.predict([digit])
    if predicted:
        print("{} == {}".format(predicted, y[index]), end=",\n")


# validacao cruzada
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy")

# matriz de confusao

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3)
confusion_matrix(y_train_0, y_train_pred)
# Lembre-se que o sgd_clf é um classificador binário (0 e não-0)  
# entao, ele retornará um array 2x2

# uma matriz de confusao ideal tera os elementos não-zero apenas na diagonal
confusion_matrix(y_train_0, y_train_0)

# precisao e recall

from sklearn.metrics import precision_score, recall_score
precision_score(y_train_0, y_train_pred)


recall_score(y_train_0, y_train_pred)
