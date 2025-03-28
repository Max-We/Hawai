import uuid

import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from randomuser import RandomUser
from tabulate import tabulate
from tqdm import tqdm
import concurrent.futures

from config import PERSONAS_FILE, NUM_PERSONAS, TEMPERATURE_PERSONAS, TOKENS_PERSONAS, MODEL_PERSONAS, \
    CONCURRENT_WORKERS

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
    - Special preparation methods they prefer (e.g., spice level, cooking styles)

    The goal is to create a unique food profile that captures their individual tastes and relationship with food beyond general categories like 'vegan' or 'keto':
    
    Name: {name}
    Age: {age}
    Country: {country}
    City: {city}
    Gender: {gender}
    Dietary Preference: {dietary_preference_string}

    Detailed description (not formatted, plain text):"""

    prompt = PromptTemplate(
        input_variables=["name", "age", "country", "city", "gender", "dietary_preference_string"],
        template=template
    )
    keywords_dict["dietary_preference_string"] = format_dietary_preference(keywords_dict["dietary_preference"])

    llm = ChatOpenAI(temperature=TEMPERATURE_PERSONAS, model_name=MODEL_PERSONAS, max_tokens=TOKENS_PERSONAS)
    chain = prompt | llm
    result = chain.invoke(keywords_dict)

    return result.content.strip()


def process_single_persona(persona):
    """
    Process a single persona by generating its description.
    This function is designed to be used with concurrent processing.

    Args:
        persona (dict): Dictionary containing person attributes

    Returns:
        dict: Updated persona dictionary with description
    """
    try:
        description = generate_person_description(persona)
        persona['description'] = description
        return persona
    except Exception as e:
        print(f"Error generating description for {persona['name']}: {str(e)}")
        persona['description'] = f"Error generating description: {str(e)}"
        return persona


def generate_personas_concurrent(num_personas, output_file):
    """
    Generate random personas with concurrent processing, their descriptions, and export to CSV

    Args:
        num_personas (int): Number of personas to generate
        output_file (str): Path to the output CSV file
        max_workers (int, optional): Maximum number of workers for concurrent processing.
                                    If None, it will use the default value based on system.

    Returns:
        list: List of generated personas
    """
    # 1. Generate random person keywords
    print("Generating random persona keywords")
    personas = generate_random_personas(num_personas)

    # 2. Generate descriptions concurrently
    print(f"Generating {num_personas} persona descriptions...")

    # Create a progress bar that will be updated by the main thread
    progress_bar = tqdm(total=num_personas, desc="Processing personas")
    processed_count = 0

    # Process personas concurrently using ThreadPoolExecutor
    # (ThreadPoolExecutor is better for I/O-bound tasks like API calls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        # Submit all persona generation tasks
        future_to_persona = {executor.submit(process_single_persona, persona): i
                             for i, persona in enumerate(personas)}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_persona):
            idx = future_to_persona[future]
            try:
                # Get the result and update the personas list
                result = future.result()
                personas[idx] = result

                # Update progress
                processed_count += 1
                progress_bar.update(1)

            except Exception as exc:
                print(f"Persona {idx} generated an exception: {exc}")
                personas[idx]['description'] = f"Error: {str(exc)}"
                progress_bar.update(1)

    progress_bar.close()

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
        personas = generate_personas_concurrent(NUM_PERSONAS, output_file)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
