import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.features import FEATURES, build_features, rate_per_mile

# what the model fits, and the way back to dollars
TARGETS = {
    "rate per mile": (rate_per_mile, lambda fitted, df: fitted * df["distance"].to_numpy()),
    "log rate": (lambda df: np.log(df["posted_rate"]), lambda fitted, df: np.exp(fitted)),
    "dollars": (lambda df: df["posted_rate"], lambda fitted, df: fitted),
}
DEFAULT_TARGET = "rate per mile"


def train_model(train, features=FEATURES, target=DEFAULT_TARGET):
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    return model.fit(build_features(train, features), TARGETS[target][0](train))


def predict_rate(model, df, features=FEATURES, target=DEFAULT_TARGET):
    """Turns the fitted target back into dollars."""
    return TARGETS[target][1](model.predict(build_features(df, features)), df)


def feature_importance(model):
    """Share of the total gain, so the columns are ranked by how much they cut the error."""
    gain = model.booster_.feature_importance(importance_type="gain")
    return pd.Series(
        100 * gain / gain.sum(), index=model.feature_name_, name="% of gain"
    ).sort_values(ascending=False)
