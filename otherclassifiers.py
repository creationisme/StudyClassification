import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.\n")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None

def prepare_data(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3480
    )
    return X_train, X_test, y_train, y_test, X.columns

def train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, model_name, dataset_name):
    print("-" * 60)
    print(f"Training {model_name} on {dataset_name}")
    print("-" * 60)
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print(f"Best Parameters found: {best_params}")
    
    best_model = grid_search.best_estimator_
    
    y_pred_train = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    
    y_pred_test = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_accuracy:.4f}\n")
    
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(conf_matrix)
    '''
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name} ({dataset_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()'''
    
    return best_model, best_params, test_accuracy

def interpret_logistic_regression_parameters(model, feature_names):
    print("\n" + "-" * 60)
    print("Logistic Regression Model Parameter Interpretation")
    print("-" * 60)
    
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
        
        print("The coefficients indicate the influence of each feature on the prediction.")
        print("A positive coefficient suggests that as the feature value increases, the likelihood of 'good' quality wine increases.")
        print("A negative coefficient suggests the opposite.\n")
        print(coef_df.to_string(index=False))
    else:
        print("Could not retrieve coefficients from the model.")
    print("-" * 60 + "\n")


def generate_report(results):
    print("\n\n" + "=" * 70)
    print(" " * 20 + "COMPREHENSIVE FINAL REPORT")
    print("=" * 70)

    report_df = pd.DataFrame(results)
    print(report_df.to_string(index=False))

    print("\n\n--- Analysis and Comparison ---\n")
    
    best_model_info = report_df.loc[report_df['Test Accuracy'].idxmax()]
    
    print(f"1. Overall Best Performance:")
    print(f"   The model with the highest test accuracy is the {best_model_info['Model']} trained on the '{best_model_info['Dataset']}' dataset, achieving an accuracy of {best_model_info['Test Accuracy']:.4f}.")
    print(f"   The optimal hyperparameters were: {best_model_info['Best Parameters']}.")

    print("\n2. Impact of PCA:")
    accuracy_no_pca = report_df[report_df['Dataset'] == 'Without PCA']['Test Accuracy'].mean()
    accuracy_with_pca = report_df[report_df['Dataset'] == 'With PCA']['Test Accuracy'].mean()
    
    print(f"   On average, models trained without PCA had an accuracy of {accuracy_no_pca:.4f}, while models trained with PCA had an accuracy of {accuracy_with_pca:.4f}.")
    if accuracy_no_pca > accuracy_with_pca:
        print("   This suggests that for this specific problem, performing PCA led to a slight decrease in predictive performance. The dimensionality reduction may have discarded some information that was valuable for the classification task.")
    else:
        print("   This suggests that performing PCA improved the overall predictive performance, likely by reducing noise and multicollinearity in the data.")
    
    print("\n" + "=" * 70)

def main():
    df_no_pca = load_data("NNDataset.csv")
    df_pca = load_data("PCADataset.csv")
    
    if df_no_pca is None or df_pca is None:
        print("Execution halted due to missing data files.")
        return

    X_train_no_pca, X_test_no_pca, y_train_no_pca, y_test_no_pca, feature_names = prepare_data(df_no_pca)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca, _ = prepare_data(df_pca)
    
    models_to_run = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), {
            'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']
        }),
        "SVM": (SVC(random_state=42), {
            'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']
        }),
        "KNN": (KNeighborsClassifier(), {
            'n_neighbors': list(range(5, 26, 2)), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']
        })
    }
    
    results = []
    
    for model_name, (model, params) in models_to_run.items():
        # Without PCA
        best_model_no_pca, best_params_no_pca, acc_no_pca = train_and_evaluate_model(
            model, params, X_train_no_pca, y_train_no_pca, X_test_no_pca, y_test_no_pca, model_name, "Without PCA"
        )
        results.append({
            "Model": model_name, "Dataset": "Without PCA", "Test Accuracy": acc_no_pca, "Best Parameters": best_params_no_pca
        })
        
        if model_name == "Logistic Regression":
            interpret_logistic_regression_parameters(best_model_no_pca, feature_names)

        # With PCA
        _, best_params_pca, acc_pca = train_and_evaluate_model(
            model, params, X_train_pca, y_train_pca, X_test_pca, y_test_pca, model_name, "With PCA"
        )
        results.append({
            "Model": model_name, "Dataset": "With PCA", "Test Accuracy": acc_pca, "Best Parameters": best_params_pca
        })
        
    generate_report(results)

if __name__ == "__main__":
    main()
