import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

# 1. Načítanie dát
print("⏳ Načítavam dáta...")
data = pd.read_csv("preprocessed_data.csv").drop_duplicates()

# 2. Kódovanie používateľov
print("🔢 Label encoding...")
le = LabelEncoder()
data["userid"] = le.fit_transform(data["userid"])

# 3. Rozdelenie na X a y
X = data.drop("userid", axis=1)
y = data["userid"]

# 4. Rozdelenie na trénovacie a testovacie dáta
print("✂️ Rozdeľujem dáta...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# 5. Vytvorenie pipeline
print("🏗️ Vytváram pipeline...")

base_model = XGBClassifier(
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()),
    ('feature_selection', SelectFromModel(base_model)),
    ('classifier', base_model)
])

# 6. Hyperparameter tuning
print("🔍 Spúšťam GridSearchCV...")
param_grid = {
    'classifier__n_estimators': [100, 300],
    'classifier__max_depth': [6, 8],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
start_time = time.time()
grid.fit(X_train, y_train)
print(f"✅ Trénovanie dokončené za {time.time() - start_time:.2f}s")
print(f"📈 Najlepšie parametre: {grid.best_params_}")

# 7. Vyhodnotenie
print("\n📊 Vyhodnocujem model...")
best_pipeline = grid.best_estimator_
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)

print(f"\n✅ Presnosť: {accuracy_score(y_test, y_pred):.5f}")
print(f"🎯 ROC AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr')}")
print("\n📋 Classification report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8. Uloženie modelu
print("💾 Ukladám model a ďalšie súbory...")
joblib.dump(best_pipeline, "xgb_best_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "original_feature_columns.pkl")
print("✅ Model uložený ako 'xgb_best_model.pkl'")

# 9. Funkcia na autentifikáciu
def authenticate_user(sample_data, threshold=0.8):
    """Autentifikácia používateľa na základe senzorových dát"""
    try:
        sample_df = pd.DataFrame([sample_data], columns=X.columns)
        sample_prepared = best_pipeline.named_steps['imputer'].transform(sample_df)
        sample_scaled = best_pipeline.named_steps['scaler'].transform(sample_prepared)
        selected = best_pipeline.named_steps['feature_selection'].transform(sample_scaled)
        probas = best_pipeline.named_steps['classifier'].predict_proba(selected)[0]
        max_prob = np.max(probas)
        predicted_user = le.inverse_transform([np.argmax(probas)])[0]

        if max_prob >= threshold:
            return {
                'status': 'authenticated',
                'user': predicted_user,
                'confidence': float(max_prob),
                'message': 'Authentication successful'
            }
        else:
            return {
                'status': 'rejected',
                'user': predicted_user,
                'confidence': float(max_prob),
                'message': 'Low confidence score'
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

# 10. Test autentifikácie
print("\n🧪 Test autentifikácie:")
if X_test.shape[0] > 0:
    test_sample = X_test.iloc[0].values
    result = authenticate_user(test_sample)
    print(result)
else:
    print("❌ Nie sú dostupné testovacie dáta.")