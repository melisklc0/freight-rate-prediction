"""Train the final model on January-October and fill both prediction files."""

import pandas as pd

from src import config
from src.data import clean, load, remove_noise
from src.model import predict_rate, train_model

PREDICTIONS_FILE = config.ROOT / "validation_predictions.csv"


def city_coordinates(train):
    """Every city has one fixed coordinate, so a lookup from the training data is enough."""
    pickup = train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    delivery = train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(
        ["city", "lat", "lon"], axis=1
    )
    return pd.concat([pickup, delivery]).drop_duplicates("city").set_index("city")


def add_coordinates(df, coords):
    df = df.copy()
    for side in ["pickup", "delivery"]:
        df[f"{side}_lat"] = df[side].map(coords["lat"])
        df[f"{side}_lon"] = df[side].map(coords["lon"])
    return df


def main():
    data = clean(load(config.TRAIN_FILE))
    model = train_model(remove_noise(data))

    validation = clean(load(config.VALIDATION_FILE))
    predictions = pd.DataFrame(
        {
            "load_id": validation["load_id"],
            "predicted_rate": predict_rate(model, validation).round(2),
        }
    )
    template = pd.read_csv(config.TEMPLATE_FILE)[["load_id"]]
    template.merge(predictions, on="load_id", how="left").to_csv(PREDICTIONS_FILE, index=False)

    december = pd.read_csv(config.DECEMBER_FILE)
    inputs = add_coordinates(
        december.assign(date=pd.to_datetime(december["date"])), city_coordinates(data)
    )
    december["predicted_rate"] = predict_rate(model, inputs).round(2)
    december.to_csv(config.DECEMBER_FILE, index=False)

    print(f"Wrote {len(predictions):,} predictions to {PREDICTIONS_FILE.name}")
    print(december[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
