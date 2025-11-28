import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.utils import get_logger

logger = get_logger("ModelEvaluation")

def evaluate_models(models, X_test, y_test):
    """
    Evaluates trained models and returns the best one.
    """
    logger.info("Starting model evaluation...")
    
    results = []
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": auc
        })
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f"plots/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()

    # Create Results DataFrame
    results_df = pd.DataFrame(results)
    print("\n--- Model Performance ---")
    print(results_df)
    results_df.to_csv("models/evaluation_results.csv", index=False)
    
    # Identify Best Model (based on F1 Score)
    best_model_name = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]
    logger.info(f"Best Model: {best_model_name}")
    
    # Feature Importance (for Tree-based models)
    if hasattr(best_model, "feature_importances_"):
        logger.info("Generating feature importance plot...")
        importances = best_model.feature_importances_
        feature_names = X_test.columns
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_df)
        plt.title(f'Top 10 Features - {best_model_name}')
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        plt.close()
        
    return best_model, best_model_name

if __name__ == "__main__":
    pass
