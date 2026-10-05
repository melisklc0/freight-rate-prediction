import pandas as pd

EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]
START_DATE = pd.Timestamp("2025-01-01")

FEATURES = [
    "distance",
    "equipment",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "day_of_week",
]


def build_features(df, features=FEATURES):
    X = df[
        ["distance", "weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    ].copy()
    X["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT_TYPES)
    X["day_of_week"] = df["date"].dt.dayofweek
    X["days_since_start"] = (df["date"] - START_DATE).dt.days
    return X[features]


def rate_per_mile(df):
    return df["posted_rate"] / df["distance"]
