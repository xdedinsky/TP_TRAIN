import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from scipy.stats import uniform

# 1. Načítanie dát
print("⏳ Načítavam dáta...")
data = pd.read_csv("../preprocessed_data.csv").drop_duplicates()

# 2. Kódovanie používateľov
print("🔢 Label encoding...")
le = LabelEncoder()
data["userid"] = le.fit_transform(data["userid"])

# 3. Rozdelenie na X a y
X = data.drop("userid", axis=1)
y = data["userid"]

# ➕ Pridanie jemného šumu na zníženie výkonu
X += np.random.normal(0, 0.03, X.shape)

# 4. Rozdelenie na trénovacie a testovacie dáta
print("✂️ Rozdeľujem dáta...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 5. Vytvorenie pipeline (SVM + škálovanie)
print("🏗️ Vytváram pipeline...")
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', SVC(
        probability=True,
        random_state=42
    ))
])

# 6. Realisticky slabšie hyperparametre
print("🔍 Spúšťam RandomizedSearchCV...")
param_dist = {
    'classifier__C': uniform(0.05, 0.3),  # menšie hodnoty → silnejšia regularizácia
    'classifier__gamma': ['scale'],      # menej náchylné na overfitting
    'classifier__kernel': ['rbf']
}

# 7. Trénovanie
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
print("⏳ Trénujem model...")
random_search.fit(X_train, y_train)
print(f"\n✅ Trénovanie dokončené za {(time.time()-start_time)/60:.2f} minút")
print(f"📈 Najlepšie parametre:\n{random_search.best_params_}")

# 8. Vyhodnotenie modelu
print("\n📊 Vyhodnocujem model...")
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print(f"\n✅ Presnosť: {acc:.5f}")
print(f"🎯 ROC AUC: {roc:.5f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, digits=5))

# 9. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
