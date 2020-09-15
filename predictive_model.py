import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


titanic_dataset_train = pd.DataFrame(pd.read_csv("train.csv"))
titanic_dataset_test = pd.DataFrame(pd.read_csv("test.csv"))
    
titanic_train = (titanic_dataset_train[["Survived"]]>1)
titanic_train_Pclass = (titanic_dataset_train[["Sex"]])


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(titanic_train, titanic_train_Pclass)
