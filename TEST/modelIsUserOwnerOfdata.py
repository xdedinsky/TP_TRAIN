import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    stratify=y,
    random_state=42
)

# 5. Vytvorenie pipeline
print("🏗️ Vytváram pipeline...")
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', XGBClassifier(
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        enable_categorical=False
    ))
])

# 6. Nastavenie hyperparametrov
print("🔍 Spúšťam RandomizedSearchCV...")
param_dist = {
    'classifier__n_estimators': randint(200, 2000),
    'classifier__max_depth': randint(3, 15),
    'classifier__learning_rate': uniform(0.005, 0.3),
    'classifier__subsample': uniform(0.5, 0.5),
    'classifier__colsample_bytree': uniform(0.5, 0.5),
    'classifier__gamma': uniform(0, 2),
    'classifier__reg_alpha': uniform(0, 2),
    'classifier__reg_lambda': uniform(0, 2),
    'classifier__min_child_weight': randint(1, 10),
}

# 7. Trénovanie
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=300,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    error_score='raise'
)

start_time = time.time()
print("⏳ Trénovanie môže trvať dlhšie...")
random_search.fit(X_train, y_train)
print(f"\n✅ Trénovanie dokončené za {(time.time()-start_time)/60:.2f} minút")
print(f"📈 Najlepšie parametre:\n{random_search.best_params_}")

# 8. Vyhodnotenie modelu
print("\n📊 Vyhodnocujem model...")
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

print(f"\n✅ Presnosť: {accuracy_score(y_test, y_pred):.5f}")
print(f"🎯 ROC AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr'):.5f}")
print("\n📋 Detailný classification report:")
print(classification_report(y_test, y_pred, digits=5))

# 9. Vizualizácia
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Confusion Matrix")

importances = best_model.named_steps['classifier'].feature_importances_
features = X.columns
sns.barplot(x=importances, y=features, ax=ax[1], palette="viridis")
ax[1].set_title("Dôležitosť príznakov")
plt.tight_layout()
plt.show()

# 10. Uloženie modelu
print("💾 Ukladám model...")
joblib.dump({
    'model': best_model,
    'encoder': le,
    'features': X.columns.tolist(),
    'metadata': {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr'),
        'training_time': (time.time()-start_time)/60
    }
}, "optimized_verification_model.pkl")
print("✅ Model uložený ako 'optimized_verification_model_vzor3.pkl'")

# 11. Verifikačná funkcia
def verify_identity(sample, claimed_userid, threshold=0.85):
    try:
        df = pd.DataFrame([sample], columns=X.columns)
        prepared = best_model.named_steps['imputer'].transform(df)
        probas = best_model.named_steps['classifier'].predict_proba(prepared)[0]

        claimed_index = le.transform([claimed_userid])[0]
        claimed_prob = probas[claimed_index]

        return claimed_prob >= threshold
    except Exception as e:
        print(f"⚠️ Chyba pri overení: {e}")
        return False

# 12. Testovanie verifikácie
print("\n🧪 Testovanie verifikácie identity:")
sample_count = 5
test_samples = X_test.sample(sample_count, random_state=42)
true_ids = y_test.loc[test_samples.index]

for i in range(sample_count):
    sample = test_samples.iloc[i].values
    true_id = le.inverse_transform([true_ids.iloc[i]])[0]
    result = verify_identity(sample, claimed_userid=true_id)
    print(f"Test {i+1}: Skutočný používateľ: {true_id} ➜ Overenie: {result}")
