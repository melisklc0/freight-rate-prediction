import numpy as np
import pandas as pd

from src import config


def load(path):
    return pd.read_csv(path, parse_dates=["date"])


def clean(df):
    """Fixes applied to every file. Never drops a row."""
    df = df.copy()
    df.loc[df["weight"] < 0, "weight"] = np.nan
    return df


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def remove_noise(train):
    """Drops loads with a wrong distance or an extreme rate per mile. Training data only."""
    straight = haversine_miles(
        train["pickup_lat"], train["pickup_lon"], train["delivery_lat"], train["delivery_lon"]
    )
    wrong_distance = train["distance"] / straight > config.MAX_DISTANCE_RATIO

    rate_per_mile = train["posted_rate"] / train["distance"]
    extreme_rate = (rate_per_mile < config.MIN_RATE_PER_MILE) | (
        rate_per_mile > config.MAX_RATE_PER_MILE
    )

    return train[~wrong_distance & ~extreme_rate].copy()
