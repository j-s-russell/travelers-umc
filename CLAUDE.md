# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Travelers UMC 2026 modeling competition (fictitious "BlueShell Insurance"): predict the probability that an auto insurance quote converts to a bound policy. Full brief is in `descriptions/overview.md` (`descriptions/data.md` is empty). Not a git repo. `processing/` holds data-prep notebooks, `models/` modeling code.

## Pipeline

- `processing/01_preprocessing.ipynb`: raw zips → `data/processed/{policies,drivers,vehicles,call_notes}.parquet`. All splits are stacked into one table per entity; `policies.SPLIT` ∈ train/valid/holdout and holdout has `CONVERT_IND = NaN`. It drops `CALL_NOTE`, `SPLIT_FLAG` and the duplicate columns listed below, adds `*_MISSING` flags and a `"Missing"` level for categoricals, zero-pads `ZIP_CD`, adds `ZIP3`, splits `MAKE_MODEL` into `MAKE`/`MODEL`, and caps invalid driver age (>100) and safety rating (<0). Driver/vehicle tables are **not** aggregated yet.
- Run headless: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace processing/01_preprocessing.ipynb` (notebooks use paths relative to their own folder, `../data`).

Deliverables:
- A submission CSV in the format of `sample_submission.csv`: columns `POLICY_ID,CONVERT_IND_PRED` (probability), one row per **holdout** policy. Max 3 submissions/day.
- A business-partner presentation covering what was done and found, key conversion drivers (price competitiveness, channel, shopping behavior), and recommendations. It must report AUC and top-decile precision on holdout, a decile lift chart, and a calibration plot (predicted vs. observed by score band).

Scoring: **AUC-ROC** is primary; tie-break is **precision in the top decile** of scores. The benchmark is a GBM.

## Data

Raw data sits zipped in `data/` (training, validation, holdout). Each split has three tables keyed on `POLICY_ID`:
- `policies_*` — one row per quote (target `CONVERT_IND`, ~11% positive in training).
- `drivers_*` — many rows per policy (one per driver).
- `vehicles_*` — many rows per policy (one per car, `CAR_NO`).

Driver and vehicle tables must be aggregated to policy level before joining. Splits:
- **training**: `policies_train.csv` includes `CONVERT_IND` (~22k policies).
- **validation**: features plus a separate `policies_validate_target.csv` with labels (~7.4k). Use it for local evaluation, or fold it into training for the final model.
- **holdout**: features only (~7.4k). This is the set to predict for submission.

All splits cover the same 2015–2018 `QUOTE_DT` range (random split, not temporal). `SPLIT_FLAG` is `"Train"` everywhere and carries no information.

Gotchas:
- **`CALL_NOTE` exists only in training** and leaks the target. It is free text like "Why quote: … Why not bound: …". It cannot be a model feature because validation/holdout lack it. It is useful for EDA and for presentation insights (e.g. reasons quotes were lost).
- Several columns are deterministic copies of others: `MONTHLY_PREMIUM` ≈ `QUOTED_AMT/12`, `CREDIT_SCORE_SCALED` = `CREDIT_SCORE/100`, `DRIVER_AGE_MONTHS` = `DRIVER_AGE*12`, `VEHICLE_AGE_MONTHS` = `VEHICLE_AGE*12`, `SAFETY_SCORE_ADJ` = `SAFETY_RATING*1.25`, `DEPRECIATION_FACTOR` = `min(VEHICLE_AGE*0.08, 0.9)`.
- The policy-level vehicle counts (`TOTAL_NUM_VEH`, `NUM_OWNED_VEH`, etc.) match the vehicle-table row counts only ~20% of the time; they look independently generated. `NUM_DRIVERS` does match the driver table.
- Missing values: `PRIOR_CARRIER_GRP` ~10%; small amounts in `ZIP_CD`, `QUOTED_AMT`, `CREDIT_SCORE`, `COV_PKG_TYPE`, `CAT_ZONE`.
- `CALL_NOTE` contains commas, so always parse with a real CSV reader. `wc -l` row counts are unreliable.

## Environment

Use the project venv at `.venv/` (Python 3.12, created from Anaconda's interpreter but isolated from it). **Don't use the base Anaconda Python**: its pyarrow/numexpr/bottleneck are built for NumPy 1.x and break against its NumPy 2.5.

- Set up from scratch: `/opt/anaconda3/bin/python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m ipykernel install --user --name travelers-umc --display-name "Python (travelers-umc)"`
- Notebooks use the `Python (travelers-umc)` kernel. Run scripts with `.venv/bin/python`, or `source .venv/bin/activate` first.
- `requirements.txt` pins the top-level packages: pandas, numpy, pyarrow, scikit-learn, lightgbm, catboost, xgboost, optuna, shap, matplotlib, seaborn, jupyter. Add new dependencies there with exact versions.
- CatBoost writes a `catboost_info/` folder to the working directory. Pass `train_dir` or `allow_writing_files=False` to keep it out of the repo. Data can be read straight from the zips, e.g. `pd.read_csv(zipfile.ZipFile('data/2026-travelers-umc-training.zip').open('training/policies_train.csv'))`, or extracted under `data/`.
