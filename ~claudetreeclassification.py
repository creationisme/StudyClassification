import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time


def trainmodel(model_type, X_train, y_train, **kwargs):
    if model_type == 'dt':
        model = DecisionTreeClassifier(random_state=42, **kwargs)
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, **kwargs)
    elif model_type == 'xgb':
        model = XGBClassifier(random_state=42, eval_metric='logloss', **kwargs)
    else:
        raise ValueError("model_type must be 'dt', 'rf', or 'xgb'")
    
    model.fit(X_train, y_train)
    return model

def gridsearch(model, param_dict, X_train, y_train, cv_folds=5):
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_dict,
        cv=kfold,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_

def metrics(model, X_train, y_train, X_test, y_test, model_name="Model"):
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    y_pred_test = model.predict(X_test)
    class_report = classification_report(y_test, y_pred_test, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    
    results = {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'predictions': y_pred_test,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }
    
    return results

def compare_models(results_list):
    print("MODEL COMPARISON SUMMARY")
    print(f"{'Model':<15} {'Train Acc':<10} {'Test Acc':<10}")
    print("-"*40)
    
    for result in results_list:
        print(f"{result['model_name']:<15} {result['train_accuracy']:<10.4f} {result['test_accuracy']:<10.4f}")
    
    best_model = max(results_list, key=lambda x: x['test_accuracy'])
    print(f"\nBest performing model: {best_model['model_name']} (Test Accuracy: {best_model['test_accuracy']:.4f})")

if __name__ == "__main__":
    print("Loading and preparing data...")
    df = pd.read_csv("TreeModelDataset.csv")
    X = df[df.columns[:-1]]
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=518, shuffle=True)

    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    hyperparams = {
        'dt': {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'xgb': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    }

    results = []
    
    models_to_train = [
        ('dt', 'Decision Tree'),
        ('rf', 'Random Forest'),
        ('xgb', 'XGBoost')
    ]
    
    for model_type, model_name in models_to_train:
        print(f"\nTRAINING {model_name.upper()}")
        
        base_model = trainmodel(model_type, X_train, y_train)
        
        print(f"Performing grid search for {model_name}...")
        start_time = time.time()
        
        best_params = gridsearch(base_model, hyperparams[model_type], X_train, y_train)
        
        print(f"Grid search completed in {time.time() - start_time:.2f} seconds")
        print(f"Best parameters for {model_name}: {best_params}")
        
        print(f"Training final {model_name} with best parameters...")
        final_model = trainmodel(model_type, X_train, y_train, **best_params)
        
        model_results = metrics(final_model, X_train, y_train, X_test, y_test, model_name)
        results.append(model_results)
        
        print(f"\n{model_name} Results:")
        print(f"Train Accuracy: {model_results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {model_results['test_accuracy']:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, model_results['predictions']))
    
    print(f"\n")
    compare_models(results)
    
    print(f"\nTraining completed! All models have been trained, tuned, and evaluated.")