# multiclass_model.py
import pandas as pd
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
le = LabelEncoder() # Uchováme si LabelEncoder pre neskoršie použitie pri popiskoch grafu
data["userid"] = le.fit_transform(data["userid"])
X = data.drop("userid", axis=1)
y = data["userid"]

# Rozdelenie dát
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Pipeline pre multi-triednu klasifikáciu
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', XGBClassifier(
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    ))
])

# Hyperparametre
param_dist = {
    'classifier__n_estimators': randint(200, 2000),
    'classifier__max_depth': randint(3, 15),
    'classifier__learning_rate': uniform(0.005, 0.3),
    'classifier__subsample': uniform(0.5, 0.5),
    'classifier__colsample_bytree': uniform(0.5, 0.5),
}

# Hľadanie optimálnych parametrov
random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=100, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42
)

# Trénovanie
start_time = time.time()
random_search.fit(X_train, y_train)
print(f"Čas trénovania: {(time.time()-start_time)/60:.2f} min")
print(f"Najlepšie parametre: {random_search.best_params_}")

# Vyhodnotenie
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

print(f"\nPresnosť: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC (OvR): {roc_auc_score(y_test, y_proba, multi_class='ovr'):.4f}")
print("\nClassification Report:")
# Získanie názvov tried z LabelEncoderu pre krajší report
target_names = le.inverse_transform(sorted(y.unique())) # Získanie pôvodných názvov tried
print(classification_report(y_test, y_pred, target_names=[str(name) for name in target_names])) # Použitie názvov

# Vykreslenie Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Pridaný kód pre vizualizáciu
plt.figure(figsize=(10, 8)) # Môže byť potrebné upraviť veľkosť podľa počtu tried
# Získanie názvov tried pre osi grafu
class_names = [str(name) for name in target_names]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Multiclass Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right') # Otočenie popisiek pre lepšiu čitateľnosť pri viacerých triedach
plt.yticks(rotation=0)
plt.tight_layout() # Zabezpečí, že sa popisky neprekrývajú
plt.show()

# Uloženie modelu
joblib.dump(best_model, "multiclass_model.pkl")