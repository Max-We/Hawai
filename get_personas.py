import random
import uuid

from randomuser import RandomUser
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from config import PERSONAS_FILE, NUM_PERSONAS, TEMPERATURE_PERSONAS, TOKENS_PERSONAS, MODEL_PERSONAS

# Load OPENAI_API_KEY from .env file
load_dotenv()

# Diet: Probability
DIET_OPTIONS = [
    ("Omnivore (none)", 60),
    ("Vegan", 5),
    ("Vegetarian", 10),
    ("Lactose Intolerant", 15),
    ("Nut Allergy", 10)
]
assert sum(weight for _, weight in DIET_OPTIONS) == 100, "Diet options probabilities should sum to 100"

def sample_dietary_preference():
    """
    Randomly assign a diet based on predefined probabilities.
    Returns a single dietary preference keyword.
    """
    import random

    choices, weights = zip(*DIET_OPTIONS)
    return random.choices(choices, weights=weights, k=1)[0]


def format_dietary_preference(selected_preference):
    """
    Format a dietary preference as a string with yes/no values.

    Args:
        selected_preference (str): The selected dietary preference

    Returns:
        str: A formatted string with the selected preference as 'yes' and others as 'no'
    """
    choices = [option for option, _ in DIET_OPTIONS]

    result = ""
    for option in choices:
        value = "yes" if option == selected_preference else "no"
        result += f"{option}: {value}, "

    # Remove trailing comma and space
    return result.rstrip(", ")


def generate_random_personas(num_personas):
    """
    Generate a list of random person dictionaries with basic information.

    Args:
        num_personas (int): Number of personas to generate

    Returns:
        list: List of dictionaries containing person information
    """
    person_list = RandomUser.generate_users(num_personas)
    personas_dicts = [
        {
            "uuid": uuid.uuid4(),
            "name": person.get_full_name(),
            "age": person.get_age(),
            "gender": person.get_gender(),
            "country": person.get_country(),
            "city": person.get_city(),
            "dietary_preference": sample_dietary_preference()
        } for person in person_list
    ]
    return personas_dicts


def generate_person_description(keywords_dict):
    """
    Generate a detailed description of a person based on keywords

    Args:
        keywords_dict (dict): Dictionary containing person attributes
        Example: {"name": "John", "age": "25", "country": "Canada",
                 "city": "Toronto", "gender": "male"}

    Returns:
        str: Detailed description of the person
    """
    template = """Given the following basic information about a person, create a detailed description of their unique food preferences and eating habits. Include information about:

    - Favorite cuisines or dishes they enjoy regularly
    - Specific foods they particularly love or crave
    - Foods they strongly dislike or avoid
    - Any seasonal or time-of-day preferences for certain foods
    - Special preparation methods they prefer (e.g., spice level, cooking styles)

    The goal is to create a unique food profile that captures their individual tastes and relationship with food beyond general categories like 'vegan' or 'keto':
    
    Name: {name}
    Age: {age}
    Country: {country}
    City: {city}
    Gender: {gender}
    Dietary Preference: {dietary_preference_string}

    Detailed description:"""

    prompt = PromptTemplate(
        input_variables=["name", "age", "country", "city", "gender", "dietary_preference_string"],
        template=template
    )
    keywords_dict["dietary_preference_string"] = format_dietary_preference(keywords_dict["dietary_preference"])

    llm = OpenAI(temperature=TEMPERATURE_PERSONAS, model_name=MODEL_PERSONAS, max_tokens=TOKENS_PERSONAS)
    chain = prompt | llm
    result = chain.invoke(keywords_dict)

    return result.strip()


def generate_personas(num_personas, output_file):
    """
    Generate random personas, their descriptions, and export to CSV

    Args:
        num_personas (int): Number of personas to generate
        output_file (str): Path to the output CSV file
    """
    # 1. Generate random person keywords
    print("Generating random persona keywords")
    personas = generate_random_personas(num_personas)

    # 2. Generate descriptions for each person based on these keywords
    for person in tqdm(personas, total=len(personas), desc="Generating persona descriptions"):
        try:
            description = generate_person_description(person)
            person['description'] = description
        except Exception as e:
            print(f"Error generating description for {person['name']}: {str(e)}")
            person['description'] = "Error generating description"

    # Convert to DataFrame and export to CSV
    df = pd.DataFrame(personas)
    df.to_csv(output_file, index=False)
    print(f"Generated {num_personas} person descriptions and saved to {output_file}\n")

    return personas


def print_persona_keywords(person_dict):
    """
    Print person keywords in a nicely formatted table

    Args:
        person_dict (dict): Dictionary containing person attributes
    """
    table_data = [[key.capitalize(), value] for key, value in person_dict.items() if key != "description"]
    print(tabulate(table_data,
                   headers=['Attribute', 'Value'],
                   tablefmt='orgtbl',
                   colalign=('left', 'left')) + "\n")


if __name__ == "__main__":
    try:
        # Generate personas and export to CSV
        output_file = PERSONAS_FILE

        print(f"Generating {NUM_PERSONAS} random personas")
        personas = generate_personas(NUM_PERSONAS, output_file)

        # Example result
        print_persona_keywords(personas[0])
        print("-" * 25)
        print("Generated Story")
        print("-" * 25)
        print(personas[0]['description'])
        print("-" * 25)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
