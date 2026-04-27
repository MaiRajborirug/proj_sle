# proj_sle — SLE Screening Tool

Predict Systemic Lupus Erythematosus (SLE) from 15 clinical features derived from the 2019 EULAR/ACR criteria (immunological markers excluded). Includes a benchmark evaluation script and a Streamlit web app for clinical screening.

---

## Files


| File                                     | Purpose                                                                                                                                                                                                                                                                                      |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SLE_NotSLE.csv`                         | Dataset — 402 patients, binary SLE/non-SLE label, 24 raw features                                                                                                                                                                                                                            |
| `Dataset5.py`                            | Benchmark: 10-fold cross-validation of 6 models (EULAR rule-based, Decision Tree, Random Forest, KNN, SVM, XGBoost) with bootstrapped 95% CI                                                                                                                                                 |
| `train_model.py`                         | Train the final calibrated Random Forest on the full dataset and save to `model.joblib` — **run this once before launching the app****The `CalibratedClassifierCV` use 5-fold CV on the dataset, so that we get the statistical probability (70% mean ~ 70% patient have SLE 402 patient)** |
| `criteria.json`                          | Metadata for the 15 EULAR/ACR input features: English name, Thai name, domain, score weight, and bilingual description                                                                                                                                                                       |
| `app.py`                                 | Streamlit screening app — checkbox form per criterion, info popover, SLE probability + 95% CI from tree ensemble                                                                                                                                                                             |
| `Dataset 5 for training tuned 402.ipynb` | Original exploratory notebook — hyperparameter tuning and initial model selection                                                                                                                                                                                                            |
| `anvil_works_webapp.txt`                 | Archived Anvil Works app (previous version, kept for reference)                                                                                                                                                                                                                              |
| `md.kmitl.sle.yaml`                      | Anvil app configuration schema (archive)                                                                                                                                                                                                                                                     |


```python
CalibratedClassifierCV(estimator, method=, cv=5) # probability classification
```

Background: classidier return model 'confidence' not actual proabilities

CalibratedClassifierCV is used to map these raw scores into calibrated probabilities

- 'Sigmoid' Platt Scalling: assume S-shape relationship between the model output and true prob $P(y=1|f) = \frac{1}{1+\exp(Af+B)}$, good with SVM
- 
- Isotonic Regression
  - Assumption

---

## Quickstart

```bash
pip install streamlit scikit-learn joblib numpy pandas xgboost

# 1. Train and save model (once)
python train_model.py

# 2. Launch app
streamlit run app.py
```

---

## Deploy

### Snowflake (Streamlit in Snowflake) — primary

Run the following SQL in a Snowflake Worksheet:

```sql
-- 1. Connect GitHub repo
CREATE OR REPLACE API INTEGRATION github_integration
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com/MaiRajborirug/')
  ENABLED = TRUE;

CREATE OR REPLACE GIT REPOSITORY proj_sle_repo
  API_INTEGRATION = github_integration
  ORIGIN = 'https://github.com/MaiRajborirug/proj_sle';

ALTER GIT REPOSITORY proj_sle_repo FETCH;

-- 2. Create Streamlit app
CREATE OR REPLACE STREAMLIT sle_screening
  FROM @proj_sle_repo/branches/master/
  MAIN_FILE = '/app.py'
  QUERY_WAREHOUSE = COMPUTE_WH;
```

To sync after a new push:
```sql
ALTER GIT REPOSITORY proj_sle_repo FETCH;
```

### Streamlit Community Cloud — alternative (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo → `app.py`

---

## Model

Random Forest (`n_estimators=300, max_depth=10`) with isotonic calibration (`CalibratedClassifierCV`). Probability intervals come from per-tree variance across the 300 estimators (±1.96 SD).

**Disclaimer:** Screening aid only. This model excludes immunological markers (ANA, anti-dsDNA, complement) required by the full EULAR/ACR 2019 criteria. Not a diagnostic tool.