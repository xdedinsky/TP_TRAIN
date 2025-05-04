import os
import pandas as pd

userid = "201848"
temp_directory = "temp/201848"
session_dir = os.path.join(temp_directory, "201848_session_2")  # Použijeme len tento podadresár
merged_data_path = "user_5.csv"

# Inicializácia merged_df, ak neexistuje
if os.path.exists(merged_data_path) and os.stat(merged_data_path).st_size > 0:
    merged_df = pd.read_csv(merged_data_path)
else:
    merged_df = pd.DataFrame(columns=[
        "timestamp", "userid", "touch_event_type",
        "touch_x", "touch_y", "touch_pressure",
        "touch_size", "orientation"
    ])

def add_touch_data():
    """ Pridá údaje z TouchEvent.csv do merged_df """
    global merged_df
    touch_event_path = os.path.join(session_dir, "TouchEvent.csv")

    if os.path.isfile(touch_event_path):
        touch_df = pd.read_csv(touch_event_path, header=None)

        temp_df = pd.DataFrame({
            "timestamp": touch_df.iloc[:, 0],
            "userid": userid,
            "touch_event_type": touch_df.iloc[:, 5],
            "touch_x": touch_df.iloc[:, 6],
            "touch_y": touch_df.iloc[:, 7],
            "touch_pressure": touch_df.iloc[:, 8],
            "touch_size": touch_df.iloc[:, 9],
            "orientation": touch_df.iloc[:, 10]
        })

        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

def add_accelerometer_data():
    """ Pridá údaje z Accelerometer.csv bez ohľadu na timestamp a podľa správnych stĺpcov """
    global merged_df
    accel_path = os.path.join(session_dir, "Accelerometer.csv")

    if os.path.isfile(accel_path):
        accel_df = pd.read_csv(accel_path, header=None)

        # Získať hodnoty x, y, z (4., 5., 6. stĺpec)
        accelerometer_data = accel_df.iloc[:, 3:6].values

        # Prispôsobenie počtu riadkov v merged_df
        num_rows = len(merged_df)
        repeated_values = accelerometer_data[:num_rows] if len(accelerometer_data) >= num_rows else \
                          pd.DataFrame(accelerometer_data).sample(n=num_rows, replace=True).values

        # Pridať hodnoty do merged_df
        merged_df[["accelerometer_x", "accelerometer_y", "accelerometer_z"]] = repeated_values

def add_gyroscope_data():
    global merged_df
    gyro_path = os.path.join(session_dir, "Gyroscope.csv")

    if os.path.isfile(gyro_path):
        gyro_df = pd.read_csv(gyro_path, header=None)

        # Získať hodnoty x, y, z (4., 5., 6. stĺpec)
        gyroscope_data = gyro_df.iloc[:, 3:6].values

        # Prispôsobenie počtu riadkov v merged_df
        num_rows = len(merged_df)
        repeated_values = gyroscope_data[:num_rows] if len(gyroscope_data) >= num_rows else \
                          pd.DataFrame(gyroscope_data).sample(n=num_rows, replace=True).values

        # Pridať hodnoty do merged_df
        merged_df[["gyroscope_x", "gyroscope_y", "gyroscope_z"]] = repeated_values


def reduce_file():
    global merged_df

    if len(merged_df) > 1000:
        merged_df = merged_df.head(1000)

    while len(merged_df) > 0 and merged_df.iloc[-1]["touch_event_type"] != 1:
        merged_df = merged_df.drop(merged_df.index[-1])

# clear file
# open("merged_data.csv", "w").close()

add_touch_data()
add_accelerometer_data()
add_gyroscope_data()
reduce_file()

merged_df.to_csv(merged_data_path, index=False)

