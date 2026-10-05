# Decision Log

## Constraints

Every validation load needs a positive prediction, so no row can be dropped.
The December file has no coordinates, `market_index` or `quote_signal`. Only the date changes in the chart.

**Decision:** The model only uses features that exist in the December file too, or that can be rebuilt for it.

## Split

Train covers January to October and validation covers November to December, with no overlap.
The task is predicting the future. A random split would leak the same days into training.

**Decision:** Time-based split. Train on January to August, test on September to October.

## Target

Distance explains most of the rate. Rate per mile stays in a narrow band around $2.
Equipment changes the price a bit; Reefer is the most expensive.

**Decision:** Predict rate per mile, then multiply by distance. Equipment is a feature.

I compared three ways of writing the target on the same features and model, over the three two-month windows (MAE):

| Window | Rate per mile | Log rate | Dollars |
|---|---|---|---|
| May-Jun | **193.25** | 195.07 | 193.97 |
| Jul-Aug | **103.85** | 105.16 | 105.48 |
| Sep-Oct | **97.97** | 99.72 | 99.80 |

Rate per mile wins all three, but by 1-2%. The honest reading is that a gradient-boosted model finds distance on its own, so the target shape matters less than it would for a linear model. Rate per mile stays because it wins everywhere, it is the unit the industry quotes in, and it keeps the December chart readable as $/mile.

## Cities and Lanes

Some cities and lanes exist only in validation. Cities close to each other have similar rates.
A model that memorizes city names can't say anything about new cities.

**Decision:** No city or lane IDs on their own. Location comes from coordinates, which also work for new cities.

The September-October holdout cannot check this: every city in it was already seen in training. The real validation has 8 new cities behind 1,447 loads, 12% of the score. So I built the situation on purpose, hiding 8 random cities from training over 5 rounds and scoring only the holdout loads that use them.

| Average MAE over 5 rounds | |
|---|---|
| City seen in training (reference) | 97.21 |
| New city, with coordinates | 100.00 |
| New city, no location at all | 113.54 |

A new city costs $2.79 (2.9%) with coordinates and $16.33 (17%) without them. Coordinates recover 83% of the gap, so the decision holds and the 12% of the validation behind new cities is not a blind spot.

## Coordinates

Every city has one fixed coordinate, and nothing is missing.
Distance is slightly longer than the straight line for almost every load, so coordinates and distance agree.
The coordinates are shifted from the real map.
A few loads have a distance much longer than their straight line, so their distance is wrong.

**Decision:** Coordinates are used as relative positions, not with real map data. December gets them from the city name. Loads with a wrong distance are removed from training only.

## Date

The price level rises until June, drops in July and recovers slightly in October.
Day of week has a small effect; midweek is slightly more expensive.
November and December were never seen, so the latest months are the best hint for them.

**Decision:** Add date features (day of week and a time trend), and check what the December chart shows.

## Market Index and Quote Signal

`market_index` has no relation to the price. `quote_signal` flips direction from month to month.
Both are missing in the December file.

**Decision:** Drop both.

## Broken Data

Some loads have no weight, and some have a negative weight. A negative weight is impossible, and the real value can't be verified.
Very cheap and very expensive loads are spread evenly over all months, with no pattern behind them.

**Decision:** Negative weight is treated as missing, since weight barely affects the price. Very cheap and very expensive loads are removed from training only. Every validation load still gets a prediction.

## Metric

No evaluation metric was given. The predictions are used for pricing, so an error costs money in both directions: too high loses the load, too low loses margin.
MAE reads directly as dollars per quote and is not dominated by the random extreme loads. RMSE is.

**Decision:** MAE is the main metric, and every model and feature decision is made with it. MAPE and RMSE are reported next to it, but no decision is based on them.

## What the Model Leans On

Share of total gain in the final model: distance 68.9%, equipment 20.7%, weight 4.7%, the four coordinates 5.3% together, day of week 0.5%.

This matches the decisions above: distance carries the price, equipment is the real modifier, and the date is a small correction. Two things are worth reading carefully. Weight earns 4.7% even though its correlation with the rate is 0.03, which is what a tree model is for: it finds the effect inside a distance and equipment group, where a single correlation cannot. And the coordinates' 5.3% is an average over all loads; on loads with a city the model never saw, dropping location costs 17% of MAE. A low overall share is not the same as unimportant.

## Date Features

A time trend lets the model carry the latest price level forward. I tested it on three two-month windows, each predicted by a model trained on everything before it.
The trend helps when the price keeps moving the same way (May to June), and hurts badly when it turns (July to August). It loses in two of the three windows.
Nobody knows which way November and December will go.

**Decision:** No trend. Day of week is the only date feature, so the model predicts the average price level of the training period plus the weekly pattern.
