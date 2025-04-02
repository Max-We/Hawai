import uuid
import pandas as pd

def add_uuid_to_personas(input_file, output_file):
    # CSV-Datei einlesen
    personas_df = pd.read_csv(input_file)
    
    # Eine UUID für jede Zeile hinzufügen
    personas_df['uuid'] = [str(uuid.uuid4()) for _ in range(len(personas_df))]
    
    # CSV-Datei mit UUID speichern
    personas_df.to_csv(output_file, index=False)
    print(f"UUIDs added and saved to {output_file}")

# Beispielaufruf der Methode

add_uuid_to_personas("personasSimp.csv", "personas_Simp_uuid.csv")