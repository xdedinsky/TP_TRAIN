Rozdiely a analýza:

Cieľová premenná
- Binárna: y je 0/1 (konkrétny používateľ vs. ostatní)
- Multi-triedna: y obsahuje všetky možné userid hodnoty

## 1. Binárny Model (`binary_model.py`)

Tento model rieši úlohu binárnej klasifikácie: rozlíšiť jedného konkrétneho ("cieľového") používateľa od všetkých ostatných.

* **Vstup (Príznaky / Features):**
    * Ako vstup slúžia **všetky stĺpce zo súboru `preprocessed_data.csv` OKREM stĺpcov `userid` a `is_target`**. Stĺpec `is_target` sa vytvára interne a slúži ako cieľová premenná (y), teda to, čo sa model snaží predpovedať (1 pre cieľového používateľa, 0 pre ostatných).
* **Vyhodnotenie:**
    * Výkon modelu na testovacej sade sa meria pomocou nasledujúcich metrík:
        * **Presnosť (Accuracy):** Celkové percento správnych predikcií.
        * **ROC AUC:** Plocha pod ROC krivkou, meria schopnosť modelu rozlišovať medzi triedami.
        * **Classification Report:** Poskytuje detailné metriky pre každú triedu (Target vs. Not Target):
            * Precision (Presnosť)
            * Recall (Návratnosť / Senzitivita)
            * F1-score (Harmonický priemer Precision a Recall)
        * **Confusion Matrix:** Tabuľka zobrazujúca počty správnych a nesprávnych predikcií (True Positives, True Negatives, False Positives, False Negatives). Matica sa aj vizualizuje pomocou `seaborn.heatmap`.
    * Pri optimalizácii hyperparametrov (`RandomizedSearchCV`) sa ako hlavné kritérium (`scoring`) používa `roc_auc`.

## 2. Multi-class Model (`multiple_model.py`)

Tento model rieši úlohu klasifikácie do viacerých tried: identifikovať konkrétneho používateľa spomedzi všetkých používateľov v dátach.

* **Vstup (Príznaky / Features):**
    * Ako vstup slúžia **všetky stĺpce zo súboru `preprocessed_data.csv` OKREM stĺpca `userid`**. Zakódovaný stĺpec `userid` slúži priamo ako cieľová premenná (y), ktorú sa model snaží predpovedať.
* **Vyhodnotenie:**
    * Výkon modelu na testovacej sade sa meria pomocou nasledujúcich metrík:
        * **Presnosť (Accuracy):** Celkové percento správnych predikcií naprieč všetkými triedami (používateľmi).
        * **ROC AUC (OvR - One-vs-Rest):** Priemer ROC AUC skóre vypočítaných pre každú triedu oproti všetkým ostatným.
        * **Classification Report:** Poskytuje detailné metriky (Precision, Recall, F1-score) pre každú triedu (používateľa) zvlášť.
        * **Confusion Matrix:** Tabuľka zobrazujúca, ako často boli prípady jednej triedy klasifikované ako iná trieda. Matica sa aj vizualizuje pomocou `seaborn.heatmap`.
    * Pri optimalizácii hyperparametrov (`RandomizedSearchCV`) sa ako hlavné kritérium (`scoring`) používa `accuracy`.

---

Oba modely používajú `Pipeline`, ktorý zahŕňa `SimpleImputer` na doplnenie chýbajúcich hodnôt (stratégia `median`) a `XGBClassifier` ako samotný klasifikačný algoritmus. Najlepšie nájdené modely sa ukladajú do súborov `binary_model.pkl` a `multiclass_model.pkl`.