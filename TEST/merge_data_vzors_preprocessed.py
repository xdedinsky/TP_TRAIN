import pandas as pd

# Zoznam súborov na zmergovanie
subory = [
    'preprocessed_data_vzor1.csv',
    'preprocessed_data_vzor2.csv',
    'preprocessed_data_vzor3.csv'
]

# Načítanie a spojenie všetkých súborov do jedného DataFrame
spojeny_df = pd.concat([pd.read_csv(subor) for subor in subory], ignore_index=True)

# Uloženie do nového CSV súboru
spojeny_df.to_csv('preprocessed_data.csv', index=False)

print("Hotovo! Súbor 'preprocessed_data.csv' bol vytvorený.")
