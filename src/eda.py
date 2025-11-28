import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils import get_logger
from src.data_loader import load_data

logger = get_logger("EDA")

def perform_eda(df, output_dir="plots"):
    """
    Performs Exploratory Data Analysis on the dataset.
    """
    logger.info("Starting EDA...")
    
    # 1. Dataset Summary
    logger.info("Generating dataset summary...")
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Duplicates ---")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\n--- Statistical Summary ---")
    print(df.describe())

    # Create output directory for plots
    os.makedirs(output_dir, exist_ok=True)

    # 2. Visualizations
    logger.info(f"Saving visualizations to {output_dir}...")
    
    # Numerical columns distribution
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f"{output_dir}/dist_{col}.png")
        plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()

    # Categorical columns count
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Count of {col}')
        plt.savefig(f"{output_dir}/count_{col}.png")
        plt.close()

    logger.info("EDA completed.")

if __name__ == "__main__":
    data_path = "data/Hotel Reservations.csv"
    try:
        df = load_data(data_path)
        perform_eda(df)
    except Exception as e:
        logger.error(f"EDA failed: {e}")
