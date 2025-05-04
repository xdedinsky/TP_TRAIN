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
#ZMENIT
data = pd.read_csv("preprocessed_data.csv").drop_duplicates()
#data = pd.read_csv("preprocessed_data_vzor1.csv").drop_duplicates()
#data = pd.read_csv("preprocessed_data_vzor2.csv").drop_duplicates()
#data = pd.read_csv("preprocessed_data_vzor3.csv").drop_duplicates()

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

# 5. Vytvorenie optimalizovanej pipeline
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

# 6. Rozšírený priestor hyperparametrov
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

# 7. Optimalizácia s rozšírenými parametrami
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

# 8. Spustenie trénovania
start_time = time.time()
print("⏳ Trénovanie môže trvať dlhšie...")
random_search.fit(X_train, y_train)
print(f"\n✅ Trénovanie dokončené za {(time.time()-start_time)/60:.2f} minút")
print(f"📈 Najlepšie parametre:\n{random_search.best_params_}")

# 9. Komplexné vyhodnotenie modelu
print("\n📊 Vyhodnocujem model...")
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

print(f"\n✅ Presnosť: {accuracy_score(y_test, y_pred):.5f}")
print(f"🎯 ROC AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr'):.5f}")
print("\n📋 Detailný classification report:")
print(classification_report(y_test, y_pred, digits=5))

# 10. Vizualizácia výsledkov
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Confusion Matrix")

# Feature importance
importances = best_model.named_steps['classifier'].feature_importances_
features = X.columns
sns.barplot(x=importances, y=features, ax=ax[1], palette="viridis")
ax[1].set_title("Dôležitosť príznakov")
plt.tight_layout()
plt.show()

# 11. Uloženie modelu a metadát
print("💾 Ukladám kompletný model...")
joblib.dump({
    'model': best_model,
    'encoder': le,
    'features': X.columns.tolist(),
    'metadata': {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr'),
        'training_time': (time.time()-start_time)/60
    }
}, "optimized_xgb_model_all.pkl")

#ZMENIT toto nad tym na:
#optimized_xgb_model_all.pkl
#optimized_xgb_model_vzor1.pkl
#optimized_xgb_model_vzor2.pkl
#optimized_xgb_model_vzor3.pkl

print("✅ Model uložený ako 'optimized_xgb_model_all.pkl'")

# 12. Vylepšená autentifikačná funkcia
def authenticate(sample, threshold=0.85, fallback_threshold=0.5):
    try:
        df = pd.DataFrame([sample], columns=X.columns)
        prepared = best_model.named_steps['imputer'].transform(df)
        probas = best_model.named_steps['classifier'].predict_proba(prepared)[0]
        
        top_idx = np.argmax(probas)
        top_prob = probas[top_idx]
        user = le.inverse_transform([top_idx])[0]
        
        result = {
            'user': user,
            'confidence': float(top_prob),
            'top_3_probs': dict(zip(le.classes_, [round(p, 4) for p in probas]))
        }
        
        if top_prob >= threshold:
            result['status'] = 'authenticated'
            result['message'] = 'High confidence authentication'
        elif top_prob >= fallback_threshold:
            result['status'] = 'partial_authenticated'
            result['message'] = 'Requires 2FA verification'
        else:
            result['status'] = 'rejected'
            result['message'] = 'Low confidence score'
            
        return result
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# 13. Rozšírený test autentifikácie
print("\n🧪 Rozšírený test autentifikácie:")
test_sample = X_test.sample(3, random_state=42)
for i in range(len(test_sample)):
    print(f"\nTest {i+1}:")
    result = authenticate(test_sample.iloc[i].values)
    print(pd.DataFrame(result).to_string(index=False))