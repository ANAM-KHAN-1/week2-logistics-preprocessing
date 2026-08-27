"""
Week 2 Task - Data Collection, Cleaning, and Preprocessing for Logistics Analysis
Author: Anam Khan
Reference dataset: DataCo Smart Supply Chain for Big Data Analysis (Kaggle)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess_logistics_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)

    # 1. Missing values
    for col in ["Shipping Cost", "Order Item Quantity", "Order Item Total"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["Shipping Mode", "Customer Segment"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna(subset=["shipping date (DateOrders)"])

    # 2. Duplicates
    df = df.drop_duplicates(subset=["Order Id", "Product Card Id", "Order Item Id"])

    # 3. Type / format standardization
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df["shipping date (DateOrders)"] = pd.to_datetime(df["shipping date (DateOrders)"], errors="coerce")
    df["Shipping Mode"] = df["Shipping Mode"].str.strip().str.title()

    # 4. Outliers (logical bound check on shipping duration)
    df["Actual Shipping Days"] = (
        df["shipping date (DateOrders)"] - df["order date (DateOrders)"]
    ).dt.days
    invalid = (df["Actual Shipping Days"] < 0) | (df["Actual Shipping Days"] > 60)
    df.loc[invalid, "Actual Shipping Days"] = np.nan
    df["Actual Shipping Days"] = df["Actual Shipping Days"].fillna(
        df["Actual Shipping Days"].median()
    )

    # 5. Statistical outlier flag (IQR) - flagged, not deleted
    Q1, Q3 = df["Sales"].quantile(0.25), df["Sales"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df["Sales_Outlier_Flag"] = ~df["Sales"].between(lower, upper)

    return df.reset_index(drop=True)


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler


if __name__ == "__main__":
    clean_df = preprocess_logistics_data("data/staging/logistics_orders_raw.csv")

    feature_cols = ["Sales", "Order Item Quantity", "Shipping Cost", "Actual Shipping Days"]
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42)
    train_scaled, test_scaled, _ = scale_features(train_df, test_df, feature_cols)

    clean_df.to_csv("data/processed/logistics_orders_clean.csv", index=False)
    print(f"Pipeline complete: {clean_df.shape[0]} clean rows, {clean_df.shape[1]} columns")
