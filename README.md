# Freight Rate Prediction

Predicts the posted rate of a freight load from its lane, distance, equipment, weight and date.

LightGBM on 48,000 labeled loads (January to October 2025), validated on a time-based holdout.
Against a median-$/mile-by-equipment baseline, MAE drops from **$228.74 to $97.97** (4.3% MAPE).

Every decision behind the model (what to clean, what to drop, what to predict) is written up
with the measurement that justifies it in **[DECISION_LOG.md](DECISION_LOG.md)**.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The two source files are not in this repository. Drop the provided `train_test.csv` and
`validation.csv` into `data/`, next to the template and the December chart inputs already there.

## Run

```bash
# Time-based holdout: train on January-August, score on September-October.
# Prints the baseline comparison, the error breakdown, feature importance,
# and the three experiments behind the design decisions.
uv run python -m src.evaluate

# Train the final model on all ten months and write both prediction files.
uv run python -m src.predict
```

`src.predict` writes:

- `validation_predictions.csv`: 12,000 rows, `load_id,predicted_rate`
- `data/december_chart_inputs.csv`: fills the `predicted_rate` column in place

## Score

`score.py` is the scorer provided with the assessment, and `requirements.txt` holds **its**
dependencies only. The model itself is installed through `uv sync` above.

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

Writes `scorer_results/candidate_december.png`, the chart used in the report.

## Layout

```
src/config.py      paths, seed, cleaning thresholds, holdout dates, model params
src/data.py        loading, cleaning, noise removal (training data only)
src/features.py    the single feature list shared by training and prediction
src/model.py       LightGBM wrapper and the three target transforms
src/evaluate.py    holdout scoring and the experiments behind each decision
src/predict.py     final model, writes both prediction files
notebooks/eda.ipynb  exploration the decisions came out of
DECISION_LOG.md    every decision with the number that justifies it
reports/report.pdf the submission report
```

## The short version

**Split.** The task is predicting the future, so a random split would leak the same days into
training. Train on January to August, score on September and October, with no overlap.

**Target.** The model predicts rate per mile and multiplies back by distance. Compared against
log-rate and raw dollars across three separate two-month windows; rate per mile won all three.

**Features.** Only columns that also exist in the December chart file, or that can be rebuilt for
it. No city or lane IDs: location comes from coordinates, so a city the model has never seen
still gets a sensible rate. `market_index` and `quote_signal` are dropped: no relation to price,
and the signal flips direction month to month.

**Data quality.** Negative weights are treated as missing. Loads whose distance is far longer than
the straight line between their cities, and loads with an extreme rate per mile, are removed from
training only, 1.6% of rows. Nothing is ever removed from validation, and all 12,000 loads get a
prediction.

**The holdout's blind spot.** Every city in September and October was already in training, but the
real validation has 8 unseen cities behind 1,447 loads, 12% of the score. That case was rebuilt on
purpose: hiding 8 random cities over 5 rounds costs $2.79 in MAE with coordinates and $16.33
without them.
