import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger("FeatureEngineering")

def engineer_features(df):
    """
    Creates new features from existing columns.
    """
    logger.info("Starting feature engineering...")
    
    # 1. Total stay nights
    if 'no_of_weekend_nights' in df.columns and 'no_of_week_nights' in df.columns:
         df['total_stay_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    elif 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
        df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    else:
        logger.warning("Columns for stay nights not found. Skipping 'total_stay_nights'.")

    # 2. Total guests
    # Check for column variations (adults/children/babies)
    try:
        # Feature 1: Total stay nights (Redundant check but ensures consistency if block above failed silently)
        if 'stays_in_weekend_nights' in df.columns:
            df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        elif 'no_of_weekend_nights' in df.columns:
             df['total_stay_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        
        # Feature 2: Total guests
        if 'adults' in df.columns:
            df['total_guests'] = df['adults'] + df['children'] + df['babies']
        elif 'no_of_adults' in df.columns:
            df['total_guests'] = df['no_of_adults'] + df['no_of_children'] 

        # Feature 3: Booking lead time category
        if 'lead_time' in df.columns:
            df['booking_lead_time_category'] = pd.cut(df['lead_time'], 
                                                      bins=[-1, 7, 30, 9999], 
                                                      labels=['short', 'medium', 'long'])
        
        # Feature 4: Average ADR per person
        if 'adr' in df.columns and 'total_guests' in df.columns:
            df['avg_adr_per_person'] = df['adr'] / df['total_guests'].replace(0, 1) 
            
        # Feature 5: Weekend booking flag
        if 'stays_in_weekend_nights' in df.columns:
            df['weekend_booking_flag'] = (df['stays_in_weekend_nights'] > 0).astype(int)
        elif 'no_of_weekend_nights' in df.columns:
            df['weekend_booking_flag'] = (df['no_of_weekend_nights'] > 0).astype(int)

        logger.info(f"Feature engineering completed. New columns: {df.columns[-5:].tolist()}")
        return df

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise e

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.clean_data import clean_data
    
    data_path = "data/Hotel Reservations.csv"
    try:
        df = load_data(data_path)
        df = clean_data(df)
        df = engineer_features(df)
        print(df.head())
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
