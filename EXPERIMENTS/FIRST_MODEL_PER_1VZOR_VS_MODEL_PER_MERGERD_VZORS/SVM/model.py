import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from scipy.stats import uniform

# === Nastavenia ===
datasets = {
    "vzor1": "../../preprocessed_data_vzor1.csv",
    "vzor2": "../../preprocessed_data_vzor2.csv",
    "vzor3": "../../preprocessed_data_vzor3.csv",
    "all": "../../preprocessed_data.csv"
}

results = []

# === Trénovanie modelu pre každý dataset ===
for label, file_path in datasets.items():
    print(f"\n=== 🔄 Spracúvam dataset: {label} ===")
    
    try:
        # 1. Načítanie dát
        data = pd.read_csv(file_path).drop_duplicates()
        le = LabelEncoder()
        data["userid"] = le.fit_transform(data["userid"])

        X = data.drop("userid", axis=1)
        y = data["userid"]

        # ➕ Pridanie šumu do dát
        X += np.random.normal(0, 0.05, X.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        # 2. Pipeline – SVM + škálovanie
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                probability=True,
                random_state=42
            ))
        ])

        # 3. Hyperparametre (obmedzené a slabšie)
        param_dist = {
            'classifier__C': uniform(0.01, 0.5),  # slabšia regularizácia
            'classifier__gamma': ['scale'],
            'classifier__kernel': ['rbf']
        }

        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, n_iter=5,
            cv=3, scoring='accuracy', n_jobs=-1, random_state=42
        )

        # 4. Trénovanie
        start_time = time.time()
        print("⏳ Trénujem model...")
        random_search.fit(X_train, y_train)
        training_time = (time.time() - start_time) / 60

        y_pred = random_search.predict(X_test)
        y_proba = random_search.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        results.append({
            "dataset": label,
            "users": len(np.unique(y)),
            "samples": len(data),
            "accuracy": acc,
            "roc_auc": roc,
            "training_time_min": training_time
        })

        # 5. Uloženie modelu
        # joblib.dump(random_search.best_estimator_, f"optimized_svm_model_{label}.pkl")
        # print(f"✅ Model pre '{label}' uložený ako 'optimized_svm_model_{label}.pkl'")

        # 6. Konfúzna matica
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {label}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Chyba pri spracovaní datasetu {label}: {e}")

# === Výpis výsledkov ===
results_df = pd.DataFrame(results)
print("\n📊 Porovnanie výsledkov modelov:\n")
print(results_df.to_string(index=False))

# === Grafy ===
plt.figure(figsize=(8, 5))
sns.barplot(x="dataset", y="accuracy", data=results_df)
plt.title("Presnosť modelov podľa datasetu")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="dataset", y="roc_auc", data=results_df)
plt.title("ROC AUC skóre modelov podľa datasetu")
plt.ylim(0, 1)
plt.ylabel("ROC AUC")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="dataset", y="training_time_min", data=results_df)
plt.title("Čas trénovania modelov podľa datasetu")
plt.ylabel("Training Time (min)")
plt.show()
