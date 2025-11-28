from src.data_loader import load_data
from src.eda import perform_eda
from src.clean_data import clean_data
from src.feature_engineering import engineer_features
from src.outlier_treatment import treat_outliers
from src.encoding import encode_data
from src.train import train_models
from src.evaluate import evaluate_models
from src.utils import get_logger
import joblib
import os

logger = get_logger("Main")

def main():
    data_path = "data/Hotel Reservations.csv"
    
    try:
        # Step 1: Load Data
        df = load_data(data_path)
        
        # Step 2: EDA
        perform_eda(df)
        
        # Step 3: Data Cleaning
        df = clean_data(df)
        
        # Step 4: Feature Engineering
        df = engineer_features(df)
        
        # Step 5: Outlier Treatment
        df = treat_outliers(df)
        
        # Step 5b: Encoding (Task 5)
        df = encode_data(df)
        
        logger.info("Preprocessing completed. Starting Model Training...")
        
        # Step 6, 7, 8: Train Models (includes SMOTE & Tuning)
        trained_models, X_test, y_test = train_models(df)
        
        # Step 9: Evaluate Models
        best_model, best_model_name = evaluate_models(trained_models, X_test, y_test)
        
        # Step 12: Save Best Model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/best_model_{best_model_name.replace(' ', '_')}.joblib"
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
