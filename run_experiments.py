import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Load sample data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

def train_and_log_model(model, model_name):
    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Log parameters and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Save and log classification reports
        train_report = classification_report(y_train, train_predictions, output_dict=True)
        test_report = classification_report(y_test, test_predictions, output_dict=True)
        
        train_report_df = pd.DataFrame(train_report).transpose()
        test_report_df = pd.DataFrame(test_report).transpose()
        
        train_report_df.to_csv("results/train_classification_report.csv", index=False)
        test_report_df.to_csv("results/test_classification_report.csv", index=False)
        
        mlflow.log_artifact("results/train_classification_report.csv")
        mlflow.log_artifact("results/test_classification_report.csv")
        
        # Log feature importances if applicable
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            feature_importances_df = pd.DataFrame({
                'feature': data.feature_names,
                'importance': feature_importances
            }).sort_values(by='importance', ascending=False)
            
            # Save feature importances as a CSV
            feature_importances_df.to_csv(f"results/{model_name}_feature_importances.csv", index=False)
            
            # Plot and save feature importances as a PNG
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importances_df['feature'], feature_importances_df['importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.title(f'{model_name} Feature Importances')
            plt.gca().invert_yaxis()  # Highest importance at the top
            plt.savefig(f"results/{model_name}_feature_importances.png")
            plt.close()
            
            # Log feature importance CSV and PNG
            mlflow.log_artifact(f"results/{model_name}_feature_importances.csv")
            mlflow.log_artifact(f"results/{model_name}_feature_importances.png")
        
        # Save and log predictions
        train_predictions_df = pd.DataFrame(train_predictions, columns=["Train_Predictions"])
        test_predictions_df = pd.DataFrame(test_predictions, columns=["Test_Predictions"])
        
        train_predictions_df.to_csv("results/train_predictions.csv", index=False)
        test_predictions_df.to_csv("results/test_predictions.csv", index=False)
        
        mlflow.log_artifact("results/train_predictions.csv")
        mlflow.log_artifact("results/test_predictions.csv")

# Train and log Decision Tree model
decision_tree = DecisionTreeClassifier()
train_and_log_model(decision_tree, "Decision_Tree")

# Train and log Random Forest model
random_forest = RandomForestClassifier()
train_and_log_model(random_forest, "Random_Forest")
