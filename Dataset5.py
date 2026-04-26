# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix,
)

xgb.set_config(verbosity=0)

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('SLE_NotSLE.csv')
df = df.drop(columns=[
    'Sex', 'Age',
    'Anti-dsDNA or Anti-Sm', 'Antiphospholipid',
    'Low C3 or C4', 'Low C4 and C3', 'ANA',
    'Renal class II or V LN', 'Renal class III or IV LN',
])

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# %%
# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    'Fever', 'ACL', 'SCL or DL', 'Oral Ulcer', 'Alopecia',
    'Joint involvement', 'Acute pericarditis',
    'Pleural or pericardial effusion', 'Proteinuria',
    'Delirium', 'Psychosis', 'Seizure',
    'Leukopenia', 'Thrombocytopenia', 'AIHA',
]

EULAR_DOMAINS = {
    'Constitutional':   {'Fever': 2},
    'Mucocutaneous':    {'ACL': 6, 'SCL or DL': 4, 'Oral Ulcer': 2, 'Alopecia': 2},
    'Musculoskeletal':  {'Joint involvement': 6},
    'Serosal':          {'Acute pericarditis': 6, 'Pleural or pericardial effusion': 5},
    'Renal':            {'Proteinuria': 4},
    'Neuropsychiatric': {'Delirium': 2, 'Psychosis': 3, 'Seizure': 5},
    'Hematologic':      {'Leukopenia': 3, 'Thrombocytopenia': 4, 'AIHA': 4},
}
MAX_SCORE = 29

kf = KFold(n_splits=10, shuffle=True, random_state=42)


# ── Helpers ───────────────────────────────────────────────────────────────────
def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    boot_means = [
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return np.mean(data), lo, hi


def fold_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'acc':  accuracy_score(y_true, y_pred),
        'sens': recall_score(y_true, y_pred),
        'spec': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'prec': precision_score(y_true, y_pred),
        'f1':   f1_score(y_true, y_pred),
        'auc':  roc_auc_score(y_true, y_prob),
    }


def print_metrics(label, m):
    print(f"\n=== {label} ===")
    for name, key in [('Accuracy', 'acc'), ('Sensitivity', 'sens'),
                      ('Specificity', 'spec'), ('Precision', 'prec'),
                      ('F1-Score', 'f1'), ('ROC-AUC', 'auc')]:
        mean, lo, hi = bootstrap_ci(m[key])
        print(f"  {name:<12}: {mean:.3f}  (95% CI: {lo:.3f}–{hi:.3f})")


def evaluate_sklearn(label, model):
    m = {k: [] for k in ['acc', 'sens', 'spec', 'prec', 'f1', 'auc']}
    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_prob = model.predict_proba(X[test_idx])[:, 1]
        for k, v in fold_metrics(y[test_idx], y_pred, y_prob).items():
            m[k].append(v)
    print_metrics(label, m)


# ── EULAR/ACR 2019 rule-based baseline ───────────────────────────────────────
def eular_score(row):
    data = {FEATURE_ORDER[i]: bool(row[i]) for i in range(len(FEATURE_ORDER))}
    score = sum(
        max((pts for feat, pts in items.items() if data.get(feat)), default=0)
        for items in EULAR_DOMAINS.values()
    )
    return score / MAX_SCORE, int(score >= 10)


m = {k: [] for k in ['acc', 'sens', 'spec', 'prec', 'f1', 'auc']}
for _, test_idx in kf.split(X):
    probs, preds = zip(*[eular_score(row) for row in X[test_idx]])
    for k, v in fold_metrics(y[test_idx], list(preds), list(probs)).items():
        m[k].append(v)
print_metrics("EULAR/ACR 2019 (rule-based)", m)

# ── ML models ─────────────────────────────────────────────────────────────────
evaluate_sklearn("Decision Tree", DecisionTreeClassifier())
evaluate_sklearn("Random Forest", RandomForestClassifier(
    bootstrap=True, max_depth=10, min_samples_leaf=1,
    min_samples_split=2, n_estimators=300,
))
evaluate_sklearn("KNN", KNeighborsClassifier())
evaluate_sklearn("SVM", SVC(C=10, gamma=1, kernel='rbf', probability=True))

# XGBoost uses DMatrix, handled separately
m = {k: [] for k in ['acc', 'sens', 'spec', 'prec', 'f1', 'auc']}
for train_idx, test_idx in kf.split(X):
    bst = xgb.train(
        {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.9},
        xgb.DMatrix(X[train_idx], label=y[train_idx]),
        num_boost_round=100,
    )
    y_prob = bst.predict(xgb.DMatrix(X[test_idx]))
    y_pred = (y_prob > 0.5).astype(int)
    for k, v in fold_metrics(y[test_idx], y_pred, y_prob).items():
        m[k].append(v)
print_metrics("XGBoost", m)

print(f"\nFeatures ({len(df.columns) - 1}): {list(df.columns[:-1])}")
