Dataset Description
Files
The competition dataset is organized as three relational tables linked by POLICY_ID. You are provided with the following files:

policies_train.csv - the training set, policy-level features and the response variable CONVERT_IND (22,122 rows)
drivers_train.csv - the training set, one row per driver; join to policies on POLICY_ID (47,689 rows)
vehicles_train.csv - the training set, one row per vehicle; join to policies on POLICY_ID (76,200 rows)
policies_validate_features.csv - the validation set, policy-level features (7,374 rows)
policies_validate_target.csv - the validation set, policy-level response variable CONVERT_IND (7,374 rows)
drivers_validate.csv - the validation set, one row per driver; join to policies_validate on POLICY_ID (15,941 rows)
vehicles_validate.csv - the validation set, one row per vehicle; join to policies_validate on POLICY_ID (25,277 rows)
policies_holdout_features.csv - the holdout set on which you will make your predictions, policy-level features only; CONVERT_IND is withheld (7,375 rows)
drivers_holdout.csv - the holdout set, one row per driver; join to policies_holdout on POLICY_ID (15,985 rows)
vehicles_holdout.csv - the holdout set, one row per vehicle; join to policies_holdout on POLICY_ID (25,445 rows)
sample_submission.csv - a sample submission file in the correct format: POLICY_ID and CONVERT_IND_PRED (predicted probability), one row per holdout policy
Dataset Statistics
Average drivers per policy: 2.16
Average vehicles per policy: 3.44
Quote date range: 2015–2018
States covered: AL, CT, FL, GA, MN, NJ, NY, WI
Training conversion rate: 11.2%
The response variable CONVERT_IND (1 = bound, 0 = not bound) is provided in the training and validation files. Your task is to predict the conversion probability for each policy in the holdout set and submit predictions in the format shown in sample_submission.csv.
Variable Descriptions
Policy-Level Table
POLICY_ID: Unique identifier for the quote/policy record; foreign key for joining to drivers and vehicles
QUOTE_DT: Date the quote was submitted (YYYY-MM-DD)
QUOTED_AMT: Annual quoted premium amount in USD
MONTHLY_PREMIUM: Monthly installment equivalent of the annual quoted premium
COV_PKG_TYPE: Coverage package level (High / Medium / Low)
DISCOUNT_IND: Whether a discount was applied to the quote (Yes / No)
PRIOR_CARRIER_GRP: Prior insurance carrier group
CREDIT_SCORE: Credit score of the primary policyholder (range: 369–850)
CREDIT_SCORE_SCALED: Credit score on a normalized scale
CAT_ZONE: Catastrophe risk zone score (1–5)
HOME_POLICY_IND: Whether the customer has an existing home policy (Y = Yes, N = No)
ZIP_CD: US ZIP code of the policyholder
STATE_CD: State of the policyholder (AL, CT, FL, GA, MN, NJ, NY, WI)
NUM_DRIVERS: Number of drivers on the policy
NUM_OWNED_VEH: Number of owned vehicles
NUM_LOANED_VEH: Number of vehicles with a loan/lien
NUM_LEASED_VEH: Number of leased vehicles
TOTAL_NUM_VEH: Total number of vehicles on the policy
PRIMARY_PARKING: Where vehicles are primarily parked (home/driveway, parking garage, street, unknown)
QUOTE_CHANNEL: Channel through which the quote was generated (online / agent / call_center / aggregator)
PRIOR_CLAIMS_CNT: Number of at-fault claims in the prior 3 years (0–5)
MULTI_QUOTE_IND: Whether the customer obtained quotes from multiple carriers (1 = yes, 0 = no)
DAYS_TO_FOLLOWUP: Days from quote date to first agent follow-up contact (1–60)
CALL_NOTE: Free-text agent call note summarizing the customer's shopping reason and non-bind objection (available in training set only)
CONVERT_IND: Response variable (1 = the quote converted to a bound policy, 0 = otherwise); training set only
Driver-Level Table
One row per driver; multiple rows per policy. Join to the policy tables on POLICY_ID.

POLICY_ID: Foreign key to the policy tables
GENDER: Driver gender (M = Male, F = Female)
LIVING_STATUS: Housing status (own / rent / dependent)
DRIVER_AGE: Driver age in years
DRIVER_AGE_MONTHS: Driver age in months
SAFETY_RATING: Driver safety rating index
SAFETY_SCORE_ADJ: Driver safety rating on an adjusted scale
HIGH_EDU_IND: Whether the driver attained higher education (1 = yes, 0 = no)
MARITAL_STATUS: Marital status (married / single / divorced / widowed)
COMMUTE_MILES_DAILY: Daily commute distance in miles
TELEMATICS_OPT_IN: Whether the driver enrolled in a telematics/usage-based program (1 = yes, 0 = no)
ACCIDENT_FREE_YRS: Years since the driver's last at-fault accident
Vehicle-Level Table
One row per vehicle; multiple rows per policy. Join to the policy tables on POLICY_ID.

POLICY_ID: Foreign key to the policy tables
CAR_NO: Vehicle sequence number within the policy (1 = primary vehicle)
OWNERSHIP_TYPE: How the vehicle is held (owned / loaned / leased)
COLOR: Vehicle color
VEHICLE_AGE: Vehicle age in years
VEHICLE_AGE_MONTHS: Vehicle age in months
DEPRECIATION_FACTOR: Estimated depreciation fraction based on vehicle age
MAKE_MODEL: Vehicle make and model (e.g., BMW : 750LI)
ANNUAL_MILEAGE: Estimated annual miles driven
ANTI_THEFT_IND: Whether an anti-theft device is installed (1 = yes, 0 = no)
SAFETY_FEATURE_SCORE: Composite vehicle safety equipment score (ABS, airbags, ADAS)
GARAGE_IND: Whether the vehicle is garaged (1 = yes, 0 = no)
Submission Format
The sample_submission.csv file shows the correct format for your predictions:

POLICY_ID: Unique policy identifier; must match exactly the POLICY_ID values in policies_holdout_features.csv
CONVERT_IND_PRED: Your predicted conversion probability for that policy (a value between 0 and 1)
Your submission must contain exactly 7,375 rows (one per holdout policy) plus a header row. The file must include only these two columns, in this order.