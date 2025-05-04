Vysvetlenie jednotlivych suborov:
----------------------------------------

from_database.csv - data ziskane z databazy

sample_data.csv - manualne vytvorene data pripravene na predspracovanie (-> preprocessing.py)

preprocessed_data.csv - predspracovane data (vystup z preprocessing.py)

----------------------------------------


Vysvetlenie features:
----------------------------------------

ATMS - priemerna rychlost touch pohybu (priemer vsetkych TMS)

max_TMS - maximalna rychlost touch pohybu 

min_TMS - maximalna rychlost touch pohybu 

length - celková vzdialenosť tocuh pohybu (sucet vsetkych ciastkovych pohybov)

accel_? - priemerne zrychlenie na osi "?"
    (? - oznacenie osi (x,y,z))

max_accel_? - maximalne zrychlenie v jednom tiku na osi "?"

min_accel_? - minimalne zrychlenie v jednom tiku na osi "?"

total_accel - priemerne celkove zrychlenie na vsetkych troch osiach (priemerna magnituda)

gyro_? - priemerne naklonenie na osi "?"

max_gyrol_? - maximalne naklonenie v jednom tiku na osi "?"

min_gyro_? - minimalne naklonenie v jednom tiku na osi "?"

total_gyro - priemerne celkove naklonenie na vsetkych troch osiach (priemerna magnituda)

----------------------------------------