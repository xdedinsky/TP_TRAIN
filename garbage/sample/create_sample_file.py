import os
import pandas as pd


def merge_files(directory):
    """ Funkcia, ktorá spojí všetky CSV súbory v danom adresári do jedného """
    # Zoznam všetkých CSV súborov v adresári
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    # Načítať každý CSV a pridať ho do zoznamu
    all_data = []
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        # Pridanie do zoznamu
        all_data.append(df)

    # Spojiť všetky dáta do jedného DataFrame
    merged_df = pd.concat(all_data, ignore_index=True)

    # Uložíme výsledok do "merged_data.csv"
    merged_df.to_csv("merged_data.csv", index=False)
    print("Všetky CSV súbory boli úspešne spojené do 'merged_data.csv'.")


def edit():
    """ Funkcia, ktorá upraví poradie stĺpcov v 'merged_data.csv', upraví hodnoty v stĺpci 'orientation' a 'touch_event_type' """
    # Načítať existujúci merged_data.csv
    merged_df = pd.read_csv("merged_data.csv")

    # Požadované poradie stĺpcov
    desired_columns = [
        "userid", "timestamp", "orientation", "touch_event_type",
        "touch_x", "touch_y", "touch_pressure", "touch_size",
        "accelerometer_x", "accelerometer_y", "accelerometer_z",
        "gyroscope_x", "gyroscope_y", "gyroscope_z"
    ]

    # Usporiadame stĺpce podľa požiadaviek
    merged_df = merged_df[desired_columns]

    # Upravíme hodnoty v stĺpci 'orientation'
    merged_df["orientation"] = merged_df["orientation"].apply(lambda x: "portrait" if x == 0 else "vertical")

    # Upravíme hodnoty v stĺpci 'touch_event_type'
    merged_df["touch_event_type"] = merged_df["touch_event_type"].apply(
        lambda x: "down" if x == 0 else "move" if x == 2 else "up" if x == 1 else x
    )

    # Uložíme výsledok späť do "merged_data.csv"
    merged_df.to_csv("merged_data.csv", index=False)
    print("Poradie stĺpcov a hodnoty v stĺpcoch 'orientation' a 'touch_event_type' boli upravené v 'merged_data.csv'.")


# Použitie funkcie
directory = os.path.dirname(os.path.abspath(__file__))  # Aktuálny adresár
merge_files(directory)
edit()
