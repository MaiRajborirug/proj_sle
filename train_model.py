"""
Run once to train and save the calibrated Random Forest used by app.py.
Output: model.joblib
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

DROP_COLS = [
    'Sex', 'Age',
    'Anti-dsDNA or Anti-Sm', 'Antiphospholipid',
    'Low C3 or C4', 'Low C4 and C3', 'ANA',
    'Renal class II or V LN', 'Renal class III or IV LN',
]

df = pd.read_csv('SLE_NotSLE.csv').drop(columns=DROP_COLS)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_cal, y_train, y_cal = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(
    bootstrap=True, max_depth=10, min_samples_leaf=1,
    min_samples_split=2, n_estimators=300, random_state=42,
)
rf.fit(X_train, y_train)

calibrated = CalibratedClassifierCV(rf, method='isotonic', cv='prefit')
calibrated.fit(X_cal, y_cal)

joblib.dump(calibrated, 'model.joblib')
print("Saved model.joblib")
print(f"Features: {list(df.columns[:-1])}")
