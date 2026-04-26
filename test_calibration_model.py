#%%
"""
test_calibration_model.py
Compare calibration methods (none, sigmoid, isotonic, temperature) across 4 classifiers.
Outputs: reliability diagram (calibration_reliability.png), metrics table, top-3 summary.
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import f1_score, recall_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

# ── Data ──────────────────────────────────────────────────────────────────────
DATASET = 6
print(f"Running for dataset {DATASET}")
if DATASET == 5:
    DROP_COLS = [
        'Sex', 'Age',
        'Anti-dsDNA or Anti-Sm', 'Antiphospholipid',
        'Low C3 or C4', 'Low C4 and C3', 'ANA',
        'Renal class II or V LN', 'Renal class III or IV LN',
    ]
elif DATASET == 6:
    DROP_COLS = [
        'Sex', 'Age',
        'Anti-dsDNA or Anti-Sm', 'Antiphospholipid',
        'Low C3 or C4', 'Low C4 and C3', 'ANA',
        'Renal class II or V LN', 'Renal class III or IV LN',
        'Leukopenia', 'Thrombocytopenia', 'AIHA'
    ]
elif DATASET == 7:
    DROP_COLS = [
        'Sex', 'Age',
        'Anti-dsDNA or Anti-Sm', 'Antiphospholipid',
        'Low C3 or C4', 'Low C4 and C3', 'ANA',
        'Renal class II or V LN', 'Renal class III or IV LN',
        'Leukopenia', 'Thrombocytopenia', 'AIHA',
        'Acute pericarditis', 'Pleural or pericardial effusion',
    ]
elif DATASET == 8:
    DROP_COLS = [
        'Sex', 'Age',
        'Anti-dsDNA or Anti-Sm', 'Antiphospholipid',
        'Low C3 or C4', 'Low C4 and C3', 'ANA',
        'Renal class II or V LN', 'Renal class III or IV LN',
        'Leukopenia', 'Thrombocytopenia', 'AIHA',
        'Acute pericarditis', 'Pleural or pericardial effusion',
        'Proteinuria'
    ]
elif DATASET == 9:
    DROP_COLS = [
        'Sex', 'Age',
        'Anti-dsDNA or Anti-Sm', 'Antiphospholipid',
        'Low C3 or C4', 'Low C4 and C3', 'ANA',
        'Renal class II or V LN', 'Renal class III or IV LN',
        'Leukopenia', 'Thrombocytopenia', 'AIHA',
        'Acute pericarditis', 'Pleural or pericardial effusion',
        'Delirium', 'Psychosis', 'Seizure', 
    ]

elif DATASET == 10:
    DROP_COLS = [
        'Sex', 'Age',
        'Anti-dsDNA or Anti-Sm', 'Antiphospholipid',
        'Low C3 or C4', 'Low C4 and C3', 'ANA',
        'Renal class II or V LN', 'Renal class III or IV LN',
        'Leukopenia', 'Thrombocytopenia', 'AIHA',
        'Acute pericarditis', 'Pleural or pericardial effusion',
        'Delirium', 'Psychosis', 'Seizure', 'Proteinuria'
    ]
df = pd.read_csv('SLE_NotSLE.csv').drop(columns=DROP_COLS)
print("n_positive:", df[df['Diagnosis'] == 1].shape[0])
print("n_negative:", df[df['Diagnosis'] == 0].shape[0])
print("used features:\n",df.columns)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ── Temperature scaling (not in sklearn — implemented manually) ───────────────
class TemperatureScaling:
    """Post-hoc calibration via a single temperature T that scales log-odds."""
    def __init__(self):
        self.T = 1.0

    def fit(self, probs, y_true):
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        def nll(T):
            p = expit(logit(probs) / T)
            return -np.mean(y_true * np.log(p + 1e-10) + (1 - y_true) * np.log(1 - p + 1e-10))
        self.T = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded').x

    def predict(self, probs):
        return expit(logit(np.clip(probs, 1e-6, 1 - 1e-6)) / self.T)


# ── Helpers ───────────────────────────────────────────────────────────────────
def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def ece_score(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        total += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return total / len(y_true)


def get_probs(base_model, method, X_tr, y_tr, X_te):
    """Train a model with the given calibration method and return test probabilities."""
    if method == 'none':
        m = copy.deepcopy(base_model)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]

    if method in ('sigmoid', 'isotonic'):
        m = CalibratedClassifierCV(copy.deepcopy(base_model), method=method, cv=3)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]

    if method == 'temperature':
        X_sub, X_cal, y_sub, y_cal = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42
        )
        m = copy.deepcopy(base_model)
        m.fit(X_sub, y_sub)
        ts = TemperatureScaling()
        ts.fit(m.predict_proba(X_cal)[:, 1], y_cal)
        return ts.predict(m.predict_proba(X_te)[:, 1])


# ── Models & methods ──────────────────────────────────────────────────────────
MODELS = {
    'Random Forest': RandomForestClassifier(
        bootstrap=True, max_depth=10, min_samples_leaf=1,
        min_samples_split=2, n_estimators=300, random_state=42,
    ),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN':           KNeighborsClassifier(),
    'SVM':           SVC(C=10, gamma=1, kernel='rbf', probability=True, random_state=42),
}
METHODS = ['none', 'sigmoid', 'isotonic', 'temperature']

# ── Cross-validated metrics ───────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
records = []

for model_name, base_model in MODELS.items():
    for method in METHODS:
        f1s, senss, specs, briers, eces = [], [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            prob = get_probs(base_model, method, X[train_idx], y[train_idx], X[test_idx])
            y_pred = (prob >= 0.5).astype(int)
            f1s.append(f1_score(y[test_idx], y_pred))
            senss.append(recall_score(y[test_idx], y_pred))
            specs.append(specificity_score(y[test_idx], y_pred))
            briers.append(brier_score_loss(y[test_idx], prob))
            eces.append(ece_score(y[test_idx], prob))

        records.append({
            'Model': model_name, 'Calibration': method,
            'F1': np.mean(f1s), 'Sensitivity': np.mean(senss),
            'Specificity': np.mean(specs),
            'Brier': np.mean(briers), 'ECE': np.mean(eces),
        })

results = pd.DataFrame(records).sort_values('F1', ascending=False)

print("\n=== All results (sorted by F1) ===")
print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n=== Top 3 model × calibration combos ===")
for _, row in results.head(3).iterrows():
    print(f"\n{row['Model']} + {row['Calibration']}")
    print(f"  F1:          {row['F1']:.3f}")
    print(f"  Sensitivity: {row['Sensitivity']:.3f}")
    print(f"  Specificity: {row['Specificity']:.3f}")
    print(f"  Brier score: {row['Brier']:.3f}  (lower = better)")
    print(f"  ECE:         {row['ECE']:.3f}  (lower = better calibration)")

# ── Reliability diagrams ──────────────────────────────────────────────────────
# One subplot per model; all calibration methods overlaid on each.
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

COLORS = {'none': '#aaaaaa', 'sigmoid': '#2196F3', 'isotonic': '#E91E63', 'temperature': '#FF9800'}

fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4.5), sharey=True)
fig.suptitle("Reliability Diagrams by Model", fontsize=13)

for ax, (model_name, base_model) in zip(axes, MODELS.items()):
    for method in METHODS:
        prob = get_probs(base_model, method, X_tr_p, y_tr_p, X_te_p)
        frac_pos, mean_pred = calibration_curve(y_te_p, prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker='o', color=COLORS[method],
                label=method, linewidth=1.8, markersize=4)

    ax.plot([0, 1], [0, 1], '--', color='black', linewidth=1, label='perfect')
    ax.set_title(model_name, fontsize=10)
    ax.set_xlabel("Mean predicted probability")
    if ax is axes[0]:
        ax.set_ylabel("Fraction of positives")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('calibration_reliability.png', dpi=150, bbox_inches='tight')
print("\nSaved: calibration_reliability.png")
plt.show()

# %%
