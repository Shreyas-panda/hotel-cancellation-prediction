# Hotel Booking Cancellation Prediction

This project implements a machine learning pipeline to predict hotel booking cancellations. It includes data analysis, preprocessing, feature engineering, model training, and evaluation.

## Project Structure

```
├── data/                   # Dataset folder (not included in repo)
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks for EDA
├── plots/                  # Generated plots and visualizations
├── src/                    # Source code modules
│   ├── data_loader.py      # Data loading
│   ├── eda.py              # Exploratory Data Analysis
│   ├── clean_data.py       # Data cleaning
│   ├── feature_engineering.py # Feature creation
│   ├── outlier_treatment.py # Outlier handling
│   ├── encoding.py         # Categorical encoding
│   ├── train.py            # Model training & tuning
│   ├── evaluate.py         # Model evaluation
│   └── utils.py            # Logging utility
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data:**
    *   Place the `Hotel Reservations.csv` file inside the `data/` directory.

## Usage

Run the full pipeline using the main script:

```bash
python main.py
```

This will:
1.  Load the data.
2.  Perform EDA and save plots to `plots/`.
3.  Clean the data and engineer features.
4.  Train Logistic Regression, Random Forest, and XGBoost models.
5.  Evaluate them and save the best model to `models/`.

## Pipeline Details

*   **EDA:** Generates distribution plots and correlation matrices.
*   **Preprocessing:** Handles missing values, duplicates, and outliers (IQR method).
*   **Feature Engineering:** Creates features like `total_stay_nights`, `total_guests`, etc.
*   **Modeling:** Uses SMOTE for class imbalance and GridSearchCV for hyperparameter tuning.
*   **Evaluation:** Metrics include Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
