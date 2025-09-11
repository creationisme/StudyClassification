import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

def train_model(model_type, X_train, y_train, **kwargs):
    """
    Trains a specified model with given hyperparameters.
    """
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


def random_search(model, param_dict, X_train, y_train, n_iter=50, cv_folds=5):
    """
    Performs Randomized Search for hyperparameter tuning.
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dict,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    return random_search.best_params_


def metrics(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Calculates and returns model performance metrics.
    """
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    y_pred_test = model.predict(X_test)
    #class_report = classification_report(y_test, y_pred_test, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    # Check for OOB score for Random Forest
    oob_score = None
    if isinstance(model, RandomForestClassifier) and model.oob_score:
        oob_score = model.oob_score_
    
    results = {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'oob_score': oob_score,
        'predictions': y_pred_test,
        'confusion_matrix': conf_matrix
    }
    
    return results


def compare_models(results_list):
    """
    Prints a summary of model performance.
    """
    print("\nMODEL COMPARISON SUMMARY")
    print(f"{'Model':<15} {'Train Acc':<10} {'Test Acc':<10} {'OOB Score':<12}")
    print("-" * 50)
    
    for result in results_list:
        oob_str = f"{result['oob_score']:.4f}" if result['oob_score'] is not None else "N/A"
        print(f"{result['model_name']:<15} {result['train_accuracy']:<10.4f} {result['test_accuracy']:<10.4f} {oob_str:<12}")
    
    best_model = max(results_list, key=lambda x: x['test_accuracy'])
    print(f"\nBest performing model: {best_model['model_name']} (Test Accuracy: {best_model['test_accuracy']:.4f})")


def interpret_features(model, model_name, feature_names):
    """
    Interprets feature importance for a trained model and plots it.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"\nFeature importance not available for {model_name}.")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n--- Feature Importance for {model_name} ---")
    print(feature_importance_df.head(10))
    
    # Plotting feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
    plt.title(f'Top 10 Feature Importances for {model_name}', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.show()


if __name__ == "__main__":
    print("Loading and preparing data...")
    df = pd.read_csv("TreeModelDataset.csv")
    X = df[df.columns[:-1]]
    y = df["quality"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=518, shuffle=True)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    hyperparams = {
        'dt': {
            'max_depth': [3, 5, 8, 12, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        },
        'rf': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgb': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
    }

    results = []
    trained_models = {}
    
    models_to_train = [
        ('dt', 'Decision Tree'),
        ('rf', 'Random Forest'),
        ('xgb', 'XGBoost')
    ]
    
    for model_type, model_name in models_to_train:
        print(f"\nTRAINING {model_name.upper()}")
        
        base_model = train_model(model_type, X_train, y_train)

        print(f"Performing randomized search for {model_name}...")
        start_time = time.time()
        
        best_params = random_search(base_model, hyperparams[model_type], X_train, y_train, n_iter=50)
        
        print(f"Randomized search completed in {time.time() - start_time:.2f} seconds")
        print(f"Best parameters for {model_name}: {best_params}")
        
        print(f"Training final {model_name} with best parameters...")

        final_model = train_model(model_type, X_train, y_train, **best_params)

        model_results = metrics(final_model, X_train, y_train, X_test, y_test, model_name)
        results.append(model_results)
        trained_models[model_name] = final_model
        
        print(f"\n{model_name} Results:")
        print(f"Train Accuracy: {model_results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {model_results['test_accuracy']:.4f}")
        if model_results['oob_score'] is not None:
            print(f"OOB Score: {model_results['oob_score']:.4f}")
        print(f"Confusion matrix:\n {model_results['confusion_matrix']}")
        
        #print("\nDetailed Classification Report:")
        #print(classification_report(y_test, model_results['predictions']))
    
    print(f"\n")
    compare_models(results)

    print("\n\n--- FEATURE SELECTION & INTERPRETATION ---")
    for model_name, model in trained_models.items():
        interpret_features(model, model_name, feature_names)

    print(f"\nTraining and analysis completed! All models have been trained, tuned, and evaluated.")