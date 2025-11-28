import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger("OutlierTreatment")

def treat_outliers(df, columns=['lead_time', 'adr']):
    """
    Detects and treats outliers using the IQR method (Capping).
    """
    logger.info("Starting outlier treatment...")
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            logger.info(f"Column '{col}': Found {outliers_count} outliers.")
            
            # Capping
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            
            logger.info(f"Column '{col}': Outliers capped at {lower_bound:.2f} and {upper_bound:.2f}.")
        else:
            logger.warning(f"Column '{col}' not found for outlier treatment.")
            
    logger.info("Outlier treatment completed.")
    return df

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.clean_data import clean_data
    from src.feature_engineering import engineer_features
    
    data_path = "data/Hotel Reservations.csv"
    try:
        df = load_data(data_path)
        df = clean_data(df)
        df = engineer_features(df)
        df = treat_outliers(df)
        print(df.describe())
    except Exception as e:
        logger.error(f"Outlier treatment failed: {e}")
