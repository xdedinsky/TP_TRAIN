# binary_model.py
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt # Pridaný import
import seaborn as sns # Pridaný import
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

# Načítanie a príprava dát
data = pd.read_csv("preprocessed_data.csv").drop_duplicates()
data["userid"] = LabelEncoder().fit_transform(data["userid"])

# BINARIZÁCIA: Vyberte jedného užívateľa ako pozitívnu triedu
target_user = 0  # Zmeňte podľa potreby
data["is_target"] = np.where(data["userid"] == target_user, 1, 0)
X = data.drop(["userid", "is_target"], axis=1)
y = data["is_target"]

# Rozdelenie dát
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Pipeline pre binárnu klasifikáciu
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=np.sqrt(len(y_train) / y_train.sum())  # Vyváženie tried
    ))
])

# Hyperparametre
param_dist = {
    'classifier__n_estimators': randint(200, 2000),
    'classifier__max_depth': randint(3, 15),
    'classifier__learning_rate': uniform(0.005, 0.3),
    'classifier__subsample': uniform(0.5, 0.5),
    'classifier__gamma': uniform(0, 2),
}

# Hľadanie optimálnych parametrov
random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=100, cv=5,
    scoring='roc_auc', n_jobs=-1, random_state=42
)

# Trénovanie
start_time = time.time()
random_search.fit(X_train, y_train)
print(f"Čas trénovania: {(time.time()-start_time)/60:.2f} min")
print(f"Najlepšie parametre: {random_search.best_params_}")

# Vyhodnotenie
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(f"\nPresnosť: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Vykreslenie Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Pridaný kód pre vizualizáciu
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Target', 'Target'], yticklabels=['Not Target', 'Target'])
plt.title('Confusion Matrix - Binary Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Uloženie modelu
joblib.dump(best_model, "binary_model.pkl")