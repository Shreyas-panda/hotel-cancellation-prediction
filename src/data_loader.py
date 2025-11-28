import pandas as pd
import os
from src.utils import get_logger

logger = get_logger("DataLoader")

def load_data(file_path):
    """
    Loads the dataset from the specified file path.
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found at {file_path}")
            raise FileNotFoundError(f"File not found at {file_path}")

        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

if __name__ == "__main__":
    # Example usage
    data_path = "data/Hotel Reservations.csv"
    try:
        df = load_data(data_path)
        print(df.head())
    except Exception as e:
        print(e)
