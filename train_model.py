# %%
"""
Run once to train and save both calibrated SVMs used by app.py.
Outputs: model_d5.joblib  (15 features, SVM + sigmoid, sens 0.955)
         model_d9.joblib  (7 features,  SVM + isotonic, sens 0.905)
"""
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

BASE_DROP = [
    'Sex', 'Age',
    'Anti-dsDNA or Anti-Sm', 'Antiphospholipid',
    'Low C3 or C4', 'Low C4 and C3', 'ANA',
    'Renal class II or V LN', 'Renal class III or IV LN',
]
D9_EXTRA_DROP = [
    'Acute pericarditis', 'Pleural or pericardial effusion',
    'Delirium', 'Psychosis', 'Seizure',
    'Leukopenia', 'Thrombocytopenia', 'AIHA',
]

raw = pd.read_csv('SLE_NotSLE.csv')


def train_and_save(df, method, filename):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    svm = SVC(C=10, gamma=1, kernel='rbf', probability=False, random_state=42)
    model = CalibratedClassifierCV(svm, method=method, cv=5)
    model.fit(X, y)
    joblib.dump(model, filename)
    print(f"Saved {filename}  |  features ({len(df.columns)-1}): {list(df.columns[:-1])}")


df_d5 = raw.drop(columns=BASE_DROP)
df_d9 = raw.drop(columns=BASE_DROP + D9_EXTRA_DROP)

train_and_save(df_d5, method='sigmoid',  filename='model_d5.joblib')
train_and_save(df_d9, method='isotonic', filename='model_d9.joblib')
