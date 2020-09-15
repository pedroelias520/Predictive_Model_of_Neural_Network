import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


titanic_dataset_train = pd.DataFrame(pd.read_csv("train.csv"))
titanic_dataset_test = pd.DataFrame(pd.read_csv("test.csv"))
    
titannic_train = (titanic_dataset_train[["Survived"]]>1)
titanic_train_Pclass = (titanic_dataset_test[["Sex"]] == "Male")


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

