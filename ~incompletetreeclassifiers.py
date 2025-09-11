import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def trainmodel():
    
    return model

def gridsearch(model, param_dict):
    # input is the ml model and dictionary with parameters as keys and a list of values to search within as values
    return best_params

def metrics(model):
    #returns metrics and needed values to plot and compare models
    return



if __name__ == "__main__":
    df = pd.read_csv("TreeModelDataset.csv")
    X = df[df.columns[:-1]]
    y = df["quality"]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=34)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=34) #not really required if we do 5-fold crossvalidation?

    X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), X_test.to_numpy()
    y_train, y_valid, y_test = y_train.to_numpy(), y_valid.to_numpy(), y_test.to_numpy()

    hyperparams = dict()