# Travelers UMC 2026: Quote Conversion

This repository is an entry for the Travelers University Modeling Competition 2026. The task is to predict the probability that an auto insurance quote from the fictitious BlueShell Insurance converts to a bound policy. Submissions are scored on AUC, with precision in the top decile of scores as the tie break.

## Result

The final submission is `blend_v8`. It is 90 percent of a readable logistic formula and 10 percent of a tuned CatBoost model. On 15-fold cross validation (5 folds repeated with 3 seeds) it scores an AUC of 0.7257. The untuned gradient boosting baseline scored 0.690.

## Key findings

- The data generator behaves differently in different periods. From 1 January 2017 onward, price and credit matter much less, while driver education and ZIP population density matter much more.
- Within each period, conversion is close to additive, with sharp thresholds:
  - More drivers than vehicles converts at about 5 percent, against 11 to 14 percent otherwise.
  - Households with a mean driver age between 35 and 50 convert more.
  - From 2017, conversion jumps when more than half of the drivers have higher education.
  - Discount and home policy matter only in 2016.
- Density matches 2010 Census population divided by land area. There is also a regional effect by 3-digit ZIP from 2017 onward.
- Channel, multi-quote, days to follow-up, prior claims and most vehicle attributes have little or no effect.
- Many variables, such as mileage, make and model, and marital status, are random draws. Others, such as credit, price and catastrophe zone, follow synthetic state groupings rather than real-world patterns.
- The model is limited by the amount of labeled data rather than by missing features. Exhaustive scans for further rules, interactions and non-linear effects found nothing beyond chance.

## Repository layout

- `PROCESSING`: data preparation notebooks, run in order.
  - `01_preprocessing`: cleans the raw tables and writes them as parquet files.
  - `02_external_census`: downloads Census ACS features by ZIP.
  - `03_build_model_table`: builds the modeling table with one row per policy.
- `EDA`: exploratory analysis notebook and presentation charts.
- `MODELS`: modeling notebooks numbered 01 to 19, plus shared code.
  - `common.py`: the experiment harness, with fixed cross validation folds and paired comparisons.
  - `features.py`: engineered features.
  - `formula.py` and `formula_steps.py`: the logistic formula and its development path.
  - Notebooks 01 to 08: baselines, feature selection, feature engineering, tuning and ensembles.
  - Notebooks 09 to 19: the study of how the data was generated, which led to the final formula.
- `SUBMISSIONS`: submission files (not tracked).
- `DATA`: raw and processed data (not tracked).

## Setup

The project uses Python 3.12 in a local virtual environment.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name travelers-umc --display-name "Python (travelers-umc)"
```

Place the competition zip files in `DATA/raw`. The Census notebook needs a `CENSUS_API_KEY` in a `.env` file at the project root.

To run a notebook headless:

```
.venv/bin/jupyter nbconvert --to notebook --execute --inplace PROCESSING/01_preprocessing.ipynb
```
