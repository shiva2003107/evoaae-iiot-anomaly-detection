import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# WADI Training Data Preprocessing
# ============================================================
# Final preprocessing used for the EvoAAE training pipeline.
#
# Steps:
# 1. Load WADI normal-operation training data
# 2. Remove non-numeric / metadata columns
# 3. Convert values to numeric
# 4. Handle missing values
# 5. Apply feature-wise Min-Max normalization
# 6. Create overlapping sliding windows
# 7. Save the processed data as a NumPy array
#
# Final representation:
#     (number_of_windows, 100, 123)
#
# The final EvoAAE experiment uses the RAW + Min-Max
# representation and does not use the earlier Spectral
# Residual representation.
# ============================================================


WINDOW_SIZE = 100
STRIDE = 10


def load_training_data(file_path):
    """
    Load and clean the WADI normal-operation training data.
    """

    df = pd.read_csv(
        file_path,
        sep=",",
        header=1,
        low_memory=False
    )

    # Remove surrounding spaces from column names.
    df.columns = [str(col).strip() for col in df.columns]

    # Remove common metadata columns if present.
    columns_to_drop = [
        "Row",
        "Date",
        "Time"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # Keep numeric columns only.
    features = df.select_dtypes(
        include=np.number
    ).copy()

    # Convert all selected columns to numeric.
    for column in features.columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )

    # Replace missing values with zero.
    features = features.fillna(0)

    return features


def create_sliding_windows(
    normalized_data,
    window_size=WINDOW_SIZE,
    stride=STRIDE
):
    """
    Convert normalized time-series data into
    overlapping sliding windows.
    """

    samples = []

    total_rows = normalized_data.shape[0]

    for start in range(
        0,
        total_rows - window_size + 1,
        stride
    ):
        window = normalized_data[
            start:start + window_size
        ]

        samples.append(window)

    return np.asarray(samples)


def preprocess_training_data(
    file_path,
    output_file="processed_training_data_raw.npy",
    scaler_file="saliency_scaler.pkl"
):
    """
    Complete training-data preprocessing pipeline.
    """

    print("Loading WADI training data...")

    features = load_training_data(file_path)

    print(
        "Original numeric data shape:",
        features.shape
    )

    # --------------------------------------------------------
    # Feature-wise Min-Max normalization
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    normalized_data = scaler.fit_transform(
        features
    )

    print(
        "Normalized data shape:",
        normalized_data.shape
    )

    # Save training scaler so the same statistics can
    # be reused when preprocessing test data.
    joblib.dump(
        scaler,
        scaler_file
    )

    print(
        f"Scaler saved to: {scaler_file}"
    )

    # --------------------------------------------------------
    # Sliding-window sampling
    # --------------------------------------------------------

    processed_data = create_sliding_windows(
        normalized_data,
        window_size=WINDOW_SIZE,
        stride=STRIDE
    )

    print(
        "Processed training data shape:",
        processed_data.shape
    )

    # --------------------------------------------------------
    # Save processed training data
    # --------------------------------------------------------

    np.save(
        output_file,
        processed_data
    )

    print(
        f"Processed data saved to: {output_file}"
    )

    return processed_data


if __name__ == "__main__":

    # Change this path to the WADI normal-operation
    # training CSV when running the preprocessing script.
    TRAINING_FILE = "WADI_14days_new.csv"

    if not os.path.exists(TRAINING_FILE):

        print(
            f"Training file not found: {TRAINING_FILE}"
        )

        print(
            "Place the WADI training CSV in the project "
            "directory or update TRAINING_FILE."
        )

    else:

        processed_data = preprocess_training_data(
            TRAINING_FILE
        )

        print("\nPreprocessing completed successfully.")

        print(
            "Final dataset shape:",
            processed_data.shape
        )
