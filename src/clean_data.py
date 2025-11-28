import pandas as pd
from src.utils import get_logger

logger = get_logger("DataCleaning")

def clean_data(df):
    """
    Cleans the dataset by handling duplicates and missing values.
    """
    logger.info("Starting data cleaning...")
    
    # 1. Remove Duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    logger.info(f"Removed duplicates. Rows dropped: {initial_shape[0] - df.shape[0]}")

    # 2. Handle Missing Values
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        logger.info(f"Found {missing_values} missing values. Handling them...")
        # Strategy: Impute numerical with median, categorical with mode
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                else:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
        logger.info("Missing values handled.")
    else:
        logger.info("No missing values found.")

    # 3. Type Conversion (if needed)
    # Ensure categorical columns are properly typed if necessary
    
    logger.info(f"Data cleaning completed. Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    from src.data_loader import load_data
    data_path = "data/Hotel Reservations.csv"
    try:
        df = load_data(data_path)
        df_cleaned = clean_data(df)
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
