import csv

# Názvy výstupných súborov
output_files = {
    '1': 'vzor_1.csv',
    '2': 'vzor_2.csv',
    '3': 'vzor_3.csv'
}

# Otvorenie vstupného súboru
with open('from_database_data.csv', 'r') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Načítanie hlavičky
    
    # Inicializácia výstupných writerov
    writers = {}
    file_handles = {}
    for vzor_id, filename in output_files.items():
        f = open(filename, 'w', newline='')
        file_handles[vzor_id] = f
        writer = csv.writer(f)
        writer.writerow(header)
        writers[vzor_id] = writer

    # Spracovanie riadkov
    for row in reader:
        vzor_id = row[1].strip()  # Stĺpec s vzor_id
        
        if vzor_id in writers:
            writers[vzor_id].writerow(row)
        else:
            print(f"Varovanie: Neznámy vzor_id {vzor_id} v riadku {reader.line_num}")

# Zatvorenie všetkých výstupných súborov
for f in file_handles.values():
    f.close()





# Spracovanie CSV súborov pre každý vzor_id
def process_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Prvá fáza - doplnenie senzorických údajov
    for i in range(len(rows)):
        current_row = rows[i]
        
        if current_row[4].strip():  # touch_event_type nie je prázdny
            last_accel_data = None
            last_gyro_data = None
            
            # Hľadáme smerom nahor pre senzorické dáta
            for j in range(i-1, -1, -1):
                prev_row = rows[j]
                
                # Zastaviť ak nájdeme ďalší touch event
                if prev_row[4].strip():
                    break
                
                # Akcelerometer (stĺpce 9-11)
                accel_fields = prev_row[9:12]
                if any(field.strip() for field in accel_fields):
                    last_accel_data = accel_fields
                
                # Gyroskop (stĺpce 12-14)
                gyro_fields = prev_row[12:15]
                if any(field.strip() for field in gyro_fields):
                    last_gyro_data = gyro_fields
                
                # Ak máme oba typy dát, môžeme prestať hľadať
                if last_accel_data and last_gyro_data:
                    break

            # Aktualizovať aktuálny riadok
            if last_accel_data:
                current_row[9:12] = last_accel_data
            if last_gyro_data:
                current_row[12:15] = last_gyro_data

    # Druhá fáza - odstránenie nepotrebných riadkov
    filtered_rows = []
    for row in rows:
        # Kontrola či má touch_event alebo senzorické dáta
        has_touch_event = bool(row[4].strip())
        has_accel_data = any(field.strip() for field in row[9:12])
        has_gyro_data = any(field.strip() for field in row[12:15])
        
        # Ponechať riadok ak má touch_event alebo nejaké senzorické dáta
        if has_touch_event and has_accel_data and has_gyro_data:
            filtered_rows.append(row)

    # Uloženie výsledného súboru
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filtered_rows)

# Spracovanie pre každý vzor_id
for vzor_id in ['1', '2', '3']:
    input_filename = f'vzor_{vzor_id}.csv'
    output_filename = f'vzor_{vzor_id}.csv'
    #if want another file name uncomment this line and comment above
    #output_filename = f'vzor_{vzor_id}_processed.csv'
    process_csv(input_filename, output_filename)