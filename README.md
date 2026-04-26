# proj_sle — SLE Screening Tool

Predict Systemic Lupus Erythematosus (SLE) from 15 clinical features derived from the 2019 EULAR/ACR criteria (immunological markers excluded). Includes a benchmark evaluation script and a Streamlit web app for clinical screening.

---

## Files

| File | Purpose |
|---|---|
| `SLE_NotSLE.csv` | Dataset — 402 patients, binary SLE/non-SLE label, 24 raw features |
| `Dataset5.py` | Benchmark: 10-fold cross-validation of 6 models (EULAR rule-based, Decision Tree, Random Forest, KNN, SVM, XGBoost) with bootstrapped 95% CI |
| `train_model.py` | Train the final calibrated Random Forest on the full dataset and save to `model.joblib` — **run this once before launching the app** |
| `criteria.json` | Metadata for the 15 EULAR/ACR input features: English name, Thai name, domain, score weight, and bilingual description |
| `app.py` | Streamlit screening app — checkbox form per criterion, info popover, SLE probability + 95% CI from tree ensemble |
| `Dataset 5 for training tuned 402.ipynb` | Original exploratory notebook — hyperparameter tuning and initial model selection |
| `anvil_works_webapp.txt` | Archived Anvil Works app (previous version, kept for reference) |
| `md.kmitl.sle.yaml` | Anvil app configuration schema (archive) |

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

## Deploy (Streamlit Community Cloud — free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo → `app.py`
3. Set up [UptimeRobot](https://uptimerobot.com) (free) to ping the URL every 5 min to prevent cold-start sleep

> **Note:** `model.joblib` is in `.gitignore`. Streamlit Cloud will run `train_model.py` on first deploy if you add it to a startup script, or you can commit the file directly if the dataset is not sensitive.

---

## Model

Random Forest (`n_estimators=300, max_depth=10`) with isotonic calibration (`CalibratedClassifierCV`). Probability intervals come from per-tree variance across the 300 estimators (±1.96 SD).

**Disclaimer:** Screening aid only. This model excludes immunological markers (ANA, anti-dsDNA, complement) required by the full EULAR/ACR 2019 criteria. Not a diagnostic tool.
