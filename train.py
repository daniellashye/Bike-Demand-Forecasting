#!/usr/bin/env python3
"""
Training script for the bike-demand submission.

Run from this folder:

    cd submissions/YOUR_TEAM_NAME
    python train.py

Expected dataset:

    ../../dataset/train_set.csv

Output:

    weights.joblib

The evaluator will later load weights.joblib through predict.py.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression

DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "train_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"


def create_features_and_target(df: pd.DataFrame):

    df['target_hour_start'] = pd.to_datetime(
        df['started_at'], errors='coerce').dt.floor('h')

    hourly_data = df.groupby(
        ['start_station_id', 'target_hour_start']).size().reset_index(name='demand')

    hourly_data['hour'] = hourly_data['target_hour_start'].dt.hour
    hourly_data['weekday'] = hourly_data['target_hour_start'].dt.weekday
    hourly_data['is_weekend'] = (hourly_data['weekday'] >= 5).astype(int)

    features = ['hour', 'weekday', 'is_weekend']
    X = hourly_data[features]
    y = hourly_data['demand']

    return X, y


def main() -> None:
    train = pd.read_csv(TRAIN_CSV, low_memory=False)

    # TODO: Create your training features.
    X_train, y_train = create_features_and_target(train)

    # TODO: Train your model.
    # Example:
    model = LinearRegression()
    model.fit(X_train, y_train)

    # TODO: Save every object needed later during prediction.
    # This can include:
    #   - trained model
    #   - feature column names
    #   - scalers / encoders
    #   - lookup tables
    #   - medians / fallback values

    artifacts = {
        "model": model,
        "feature_columns": X_train.columns.tolist(),
    }

    joblib.dump(artifacts, OUTPUT_WEIGHTS)

    print(f"Saved {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
