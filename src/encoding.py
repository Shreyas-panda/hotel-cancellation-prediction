import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils import get_logger

logger = get_logger("Encoding")

def encode_data(df):
    """
    Encodes categorical variables using Label Encoding.
    Handles the target variable 'booking_status' specifically.
    """
    logger.info("Starting categorical encoding...")
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle Target Variable 'booking_status'
    # 'Canceled' = 1, 'Not_Canceled' = 0
    if 'booking_status' in df.columns:
        logger.info("Encoding target variable 'booking_status'...")
        # Check unique values to be sure
        unique_vals = df['booking_status'].unique()
        logger.info(f"Unique values in 'booking_status': {unique_vals}")
        
        # Map explicitly if possible for safety, otherwise LabelEncode
        if 'Canceled' in unique_vals and 'Not_Canceled' in unique_vals:
             df['booking_status'] = df['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})
        else:
            le = LabelEncoder()
            df['booking_status'] = le.fit_transform(df['booking_status'])
            logger.info(f"LabelEncoded 'booking_status'. Classes: {le.classes_}")
            
        # Remove from cat_cols list so we don't double encode
        if 'booking_status' in cat_cols:
            cat_cols.remove('booking_status')

    # Encode other categorical features
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        logger.info(f"Encoded column: {col}")

    logger.info("Encoding completed.")
    return df

if __name__ == "__main__":
    from src.data_loader import load_data
    # Test run
    try:
        df = load_data("data/processed_data_step4.csv") # Assuming this exists from previous run
        df = encode_data(df)
        print(df.head())
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
