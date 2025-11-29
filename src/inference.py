import pandas as pd
import joblib
import os
from src.data_loader import load_data
from src.clean_data import clean_data
from src.feature_engineering import engineer_features
from src.outlier_treatment import treat_outliers
from src.encoding import encode_data
from src.utils import get_logger

logger = get_logger("Inference")

def make_predictions(data_path, model_path):
    """
    Loads model and data, runs preprocessing, and generates predictions.
    """
    logger.info(f"Starting inference using model: {model_path}")
    
    try:
        # 1. Load Data
        df = load_data(data_path)
        original_df = df.copy() # Keep for display
        
        # 2. Preprocessing (Must match training pipeline)
        logger.info("Preprocessing data...")
        df = clean_data(df)
        df = engineer_features(df)
        df = treat_outliers(df)
        df = encode_data(df)
        
        # 3. Load Model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
        
        # 4. Predict
        # Ensure columns match (drop target if present)
        if 'booking_status' in df.columns:
            df = df.drop('booking_status', axis=1)
            
        # Align columns with model (simple check)
        # In a real prod system, we'd save the feature list. 
        # Here we assume the pipeline produces consistent features.
        
        predictions = model.predict(df)
        probs = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else [0]*len(predictions)
        
        # 5. Create Results
        results = pd.DataFrame({
            'Booking_ID': original_df['Booking_ID'],
            'Predicted_Status': predictions,
            'Cancellation_Probability': probs
        })
        
        # Map back to labels if needed (1=Canceled, 0=Not_Canceled)
        results['Predicted_Label'] = results['Predicted_Status'].map({1: 'Canceled', 0: 'Not_Canceled'})
        
        print("\n--- Predictions Preview ---")
        print(results.head(10))
        
        output_path = "data/predictions.csv"
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")

if __name__ == "__main__":
    # Use sample data for demonstration
    sample_data = "tests/sample_data.csv"
    
    # Find the best model (take the first one found in models/)
    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    if model_files:
        best_model_path = os.path.join(model_dir, model_files[0])
        make_predictions(sample_data, best_model_path)
    else:
        logger.error("No trained model found in models/ directory. Run main.py first.")
