"""Train on January-August, score on September-October."""

import numpy as np
import pandas as pd

from src import config
from src.data import clean, load, remove_noise
from src.features import FEATURES, rate_per_mile
from src.model import DEFAULT_TARGET, TARGETS, feature_importance, predict_rate, train_model

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)


def metrics(actual, predicted):
    error = np.abs(actual - predicted)
    return {
        "MAE": error.mean(),
        "MAPE %": 100 * (error / actual).mean(),
        "RMSE": np.sqrt((error**2).mean()),
    }


def main():
    data = clean(load(config.TRAIN_FILE))
    is_holdout = data["date"] >= config.HOLDOUT_START
    train = remove_noise(data[~is_holdout])
    holdout = data[is_holdout]

    equipment_rate = rate_per_mile(train).groupby(train["equipment"]).median()
    baseline = holdout["equipment"].map(equipment_rate) * holdout["distance"]

    model = train_model(train)

    results = pd.DataFrame(
        {
            "Baseline: median $/mile by equipment": metrics(holdout["posted_rate"], baseline),
            "LightGBM": metrics(holdout["posted_rate"], predict_rate(model, holdout)),
        }
    ).T.round(2)
    print(results)

    print("\nWhere does LightGBM miss? (MAE by group)")
    print(error_breakdown(train, holdout, predict_rate(model, holdout)).round(2))

    print("\nWhat the model leans on (% of total gain)")
    print(feature_importance(model).round(1))

    print("\nTarget experiment (MAE): what should the model predict?")
    print(target_experiment(data).round(2))

    print("\nUnseen city experiment (MAE): do coordinates carry a city the model never saw?")
    print(unseen_city_experiment(data).round(2))

    print("\nDate experiment (MAE): does a time trend help?")
    print(date_experiment(data).round(2))


def error_breakdown(train, holdout, predicted):
    error = np.abs(holdout["posted_rate"] - predicted)
    known_cities = set(train["pickup"]) | set(train["delivery"])
    new_city = ~holdout["pickup"].isin(known_cities) | ~holdout["delivery"].isin(known_cities)
    noise = ~holdout.index.isin(remove_noise(holdout).index)

    groups = {
        "equipment": holdout["equipment"],
        "distance": pd.cut(holdout["distance"], [0, 250, 500, 1000, 2000, np.inf]),
        "city seen in training": np.where(new_city, "new city", "seen"),
        "load type": np.where(noise, "noise (would be dropped from training)", "clean"),
    }
    tables = [
        error.groupby(values, observed=True)
        .agg(["mean", "size"])
        .set_axis(["MAE", "loads"], axis=1)
        for values in groups.values()
    ]
    return pd.concat(tables, keys=groups.keys())


def uses_any(df, cities):
    return df["pickup"].isin(cities) | df["delivery"].isin(cities)


def window_mae(model, window, features=FEATURES, target=DEFAULT_TARGET):
    return metrics(window["posted_rate"], predict_rate(model, window, features, target))["MAE"]


def target_experiment(data):
    """Same features and model, three ways of writing down what it should predict."""
    rows = {}
    for start, end in config.TIME_FOLDS:
        train = remove_noise(data[data["date"] < start])
        window = data[(data["date"] >= start) & (data["date"] < end)]
        rows[f"{start[:7]} to {end[:7]}"] = {
            name: window_mae(train_model(train, target=name), window, target=name)
            for name in TARGETS
        }
    return pd.DataFrame(rows).T


def unseen_city_experiment(data):
    """Hides a few cities from training, then scores the holdout loads that use them.

    The real validation has 8 cities the training months never saw, but September-October has
    none, so the holdout cannot check whether coordinates carry a new city. This rebuilds the
    case and compares three models on it: every city seen, the city hidden with coordinates
    kept, and the city hidden with no location at all.
    """
    is_holdout = data["date"] >= config.HOLDOUT_START
    train_all, holdout = data[~is_holdout], data[is_holdout]
    cities = np.sort(pd.unique(train_all[["pickup", "delivery"]].to_numpy().ravel()))
    without_coordinates = [f for f in FEATURES if not f.endswith(("_lat", "_lon"))]

    reference = train_model(remove_noise(train_all))
    rng = np.random.default_rng(config.SEED)
    rows = {}
    for round_number in range(config.UNSEEN_CITY_ROUNDS):
        hidden = set(rng.choice(cities, config.UNSEEN_CITIES, replace=False))
        train = remove_noise(train_all[~uses_any(train_all, hidden)])
        window = holdout[uses_any(holdout, hidden)]
        rows[f"round {round_number + 1}"] = {
            "loads": len(window),
            "city seen (reference)": window_mae(reference, window, FEATURES),
            "new city, coordinates": window_mae(train_model(train), window, FEATURES),
            "new city, no location": window_mae(
                train_model(train, without_coordinates), window, without_coordinates
            ),
        }
    table = pd.DataFrame(rows).T
    return pd.concat([table, table.mean().to_frame("mean").T])


def date_experiment(data):
    """Each window is predicted by a model trained on everything before it."""
    variants = {
        "day of week only": FEATURES,
        "day of week + trend": FEATURES + ["days_since_start"],
    }
    rows = {}
    for start, end in config.TIME_FOLDS:
        train = remove_noise(data[data["date"] < start])
        window = data[(data["date"] >= start) & (data["date"] < end)]
        rows[f"{start[:7]} to {end[:7]}"] = {
            name: window_mae(train_model(train, f), window, f) for name, f in variants.items()
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    main()
