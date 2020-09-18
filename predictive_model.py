import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


titanic_dataset_train = pd.DataFrame(pd.read_csv("train.csv"))
titanic_dataset_test = pd.DataFrame(pd.read_csv("test.csv"))
   

#treinamento de classificadores 
#Onde o segmento de treino achará os passageiros sobreviventes
titanic_dataset_train_Column = titanic_dataset_train[["Survived"]]
titanic_train_segment = (titanic_dataset_train[["Survived"]]==1)
segment = np.array(titanic_train_segment).ravel()

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(titanic_dataset_train_Column, segment)
some_person = titanic_dataset_train_Column.iloc[[1]]

#teste de classificador
sgd_clf.predict(some_person)

#validação cruzada 
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, titanic_dataset_train_Column, segment, cv=3, scoring="accuracy")

#matriz de consfusão
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix 

y_train_pred = cross_val_predict(sgd_clf, titanic_dataset_train_Column,segment, cv=3)
confusion_matrix(segment,y_train_pred)

#controle de precisão
from sklearn.metrics import precision_score, recall_score
precision_score(segment, y_train_pred)