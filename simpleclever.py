import random
import pandas as pd
import openai  # OpenAI direkt importieren

# OpenAI API-Schlüssel setzen
openai.api_key = "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"  # Ersetze mit deinem API-Schlüssel

def generate_and_save_personas(num_personas, output_file):
    # Ernährungspräferenzen mit Wahrscheinlichkeiten
    DIET_OPTIONS = {
        "Omnivore": 35,
        "Vegan": 20,
        "Vegetarian": 20,
        "Lactose Intolerant": 20,
        "Nut Allergy": 5
    }
    
    # Mögliche Werte für Namen, Länder und Geschlechter
    names = [
        "Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Hannah", "Isaac", "Jack",
        "Katherine", "Liam", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rachel", "Samuel", "Tina"
    ]
    countries = ["USA", "Germany", "France", "Japan", "Brazil"]
    genders = ["Male", "Female", "Non-binary"]
    
    personas = []
    for _ in range(num_personas):
        persona = {
            "name": random.choice(names),
            "age": random.randint(18, 65),
            "gender": random.choice(genders),
            "country": random.choice(countries),
            "diet": random.choices(list(DIET_OPTIONS.keys()), weights=DIET_OPTIONS.values())[0]
        }
        
        # Beschreibung generieren mit GPT-3.5
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that creates character descriptions."},
                {"role": "user", "content": f"Describe a person with these attributes be aware on the specific diet: {persona}"}
            ],
            max_tokens=150
        )
        
        persona["description"] = response["choices"][0]["message"]["content"].strip()
        personas.append(persona)
    
    # Daten in eine CSV-Datei speichern
    pd.DataFrame(personas).to_csv(output_file, index=False)
    print(f"Saved {num_personas} personas to {output_file}")

if __name__ == "__main__":
    generate_and_save_personas(25, "personasSimp.csv")