import numpy as np
import pandas as pd


class BikeDemandModel:
    """
    Put your actual model logic here.

    This file should contain:
        - feature creation used during prediction
        - model-specific preprocessing
        - prediction logic

    Do NOT load weights.joblib here.
    The weights are loaded in predict.py and passed into this class.
    """

    def __init__(self):
        self.artifacts = None

    def clean_and_impute_rows(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Step 5.1: Cleans rows, validates constraints, and handles missing values.
        """
        df_clean = df.copy()

        # --- Part A: Handle missing key columns ---
        key_columns = ["start_station_id", "date", "hour_ts", "city"]

        # Verify which key columns actually exist in the DataFrame
        existing_keys = [col for col in key_columns if col in df_clean.columns]

        if is_train:
            # During training: Drop rows where any key column is missing
            df_clean = df_clean.dropna(subset=existing_keys)
        else:
            # During inference: Dropping rows is forbidden! Fill with fallback defaults to prevent crashes
            for col in existing_keys:
                if col == "city":
                    df_clean[col] = df_clean[col].fillna("city1")
                elif col == "hour_ts":
                    df_clean[col] = df_clean[col].fillna(12)
                else:
                    df_clean[col] = df_clean[col].fillna(0)

        # --- Part B: Impute missing values and validate ranges for existing columns ---

        # 1. working_day (Must be 0 or 1, default: 1)
        if "working_day" in df_clean.columns:
            df_clean["working_day"] = df_clean["working_day"].fillna(1)
            valid_mask = df_clean["working_day"].isin([0, 1])
            df_clean.loc[~valid_mask, "working_day"] = 1

        # 2. weekday (Must be between 1 and 7, default: random integer between 1 and 7)
        if "weekday" in df_clean.columns:
            invalid_mask = df_clean["weekday"].isna() | (
                ~df_clean["weekday"].between(1, 7))
            num_invalid = invalid_mask.sum()
            if num_invalid > 0:
                random_days = np.random.randint(1, 8, size=num_invalid)
                df_clean.loc[invalid_mask, "weekday"] = random_days

        # 3. precipitation (Must be between 0 and 300, default: 0)
        if "precipitation" in df_clean.columns:
            invalid_mask = df_clean["precipitation"].isna() | (
                ~df_clean["precipitation"].between(0, 300))
            df_clean.loc[invalid_mask, "precipitation"] = 0.0

        # 4. temperature_2m (Must be between -50 and 50, default: 20)
        if "temperature_2m" in df_clean.columns:
            invalid_mask = df_clean["temperature_2m"].isna() | (
                ~df_clean["temperature_2m"].between(-50, 50))
            df_clean.loc[invalid_mask, "temperature_2m"] = 20.0

        return df_clean

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5.2: Drops columns that are irrelevant or unavailable during testing.
        """
        columns_to_drop = [
            "ended_at",
            "end_station_id",
            "user_type",
            "usage_time_minutes",
            "distance_meters"
        ]

        # Only drop columns that actually exist in the input DataFrame to prevent KeyError
        existing_cols_to_drop = [
            col for col in columns_to_drop if col in df.columns]

        return df.drop(columns=existing_cols_to_drop)

    def create_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Main pipeline function that orchestrates data cleaning and feature processing.
        """
        # 1. Clean rows and handle missing values / out-of-bound constraints
        processed_df = self.clean_and_impute_rows(df, is_train=is_train)

        # 2. Drop prohibited or unnecessary features
        processed_df = self.drop_unnecessary_columns(processed_df)

        return processed_df

    def load_artifacts(self, artifacts: dict) -> None:
        """
        Store all objects created by train.py.

        Examples:
            artifacts["model"]
            artifacts["feature_columns"]
            artifacts["scaler"]
        """
        self.artifacts = artifacts

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict bike demand for each row in test_df.

        Parameters
        ----------
        test_df:
            Hidden station-hour test features provided by the evaluator.
            It does NOT contain the demand column.

        Returns
        -------
        np.ndarray:
            One numeric prediction per row in test_df.
        """
        if self.artifacts is None:
            raise RuntimeError(
                "Model is not loaded. Call load_artifacts() first.")

        # Build the same features used during training (Inference mode: is_train=False)
        X = self.create_features(test_df, is_train=False)

        # Ensure feature columns exactly match the training configuration
        if "feature_columns" in self.artifacts and self.artifacts["feature_columns"]:
            feature_cols = self.artifacts["feature_columns"]
            # Filter down to the exact trained features if they exist in X
            existing_cols = [col for col in feature_cols if col in X.columns]
            X = X[existing_cols]

        # Load and use your trained model from self.artifacts
        model = self.artifacts.get("model")
        if model is not None:
            preds = model.predict(X)
        else:
            # Fallback placeholder if no model is found in artifacts yet
            preds = np.zeros(len(test_df), dtype=float)

        # Bike demand cannot be negative.
        return np.maximum(0.0, preds)
