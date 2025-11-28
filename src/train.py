import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.utils import get_logger

logger = get_logger("ModelTraining")

def train_models(df):
    """
    Splits data, handles imbalance, trains models, and performs tuning.
    Returns a dictionary of trained models and the test sets.
    """
    logger.info("Starting model training pipeline...")
    
    # 1. Split Data
    if 'booking_status' not in df.columns:
        raise ValueError("Target column 'booking_status' not found!")
        
    X = df.drop('booking_status', axis=1)
    y = df['booking_status']
    
    # Stratified split to maintain class ratio in test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Handle Class Imbalance (SMOTE) - ONLY on Training Data
    logger.info("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE applied. New Train shape: {X_train_resampled.shape}")

    # 3. Model Initialization
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    trained_models = {}

    # 4. Train Models
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        trained_models[name] = model
        logger.info(f"{name} trained.")

    # 5. Hyperparameter Tuning (Task 8)
    # We'll tune Random Forest as requested
    logger.info("Starting Hyperparameter Tuning for Random Forest...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train_resampled, y_train_resampled)
    
    logger.info(f"Best RF Params: {rf_grid.best_params_}")
    trained_models["Random Forest Tuned"] = rf_grid.best_estimator_
    
    return trained_models, X_test, y_test

if __name__ == "__main__":
    # Test run logic would go here
    pass
