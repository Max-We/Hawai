from randomuser import RandomUser
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate

# Generation settings
NUM_PERSONAS = 1
PERSONAS_FILE = "data/personas.csv"
# LLM settings for generating person descriptions
MODEL = "gpt-3.5-turbo-instruct"
TEMPERATURE = 0.7
DESCRIPTION_TOKENS = 250

# Load OPENAI_API_KEY from .env file
load_dotenv()

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
            "name": person.get_full_name(),
            "age": person.get_age(),
            "gender": person.get_gender(),
            "country": person.get_country(),
            "city": person.get_city()
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
    template = """Given the following basic information about a person, create a detailed description including additional personality traits, hobbies, and life details:

    Name: {name}
    Age: {age}
    Country: {country}
    City: {city}
    Gender: {gender}

    Detailed description:"""

    prompt = PromptTemplate(
        input_variables=["name", "age", "country", "city", "gender"],
        template=template
    )

    llm = OpenAI(temperature=TEMPERATURE, model_name=MODEL, max_tokens=DESCRIPTION_TOKENS)
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
    personas = generate_random_personas(num_personas)

    # 2. Generate descriptions for each person based on these keywords
    for person in personas:
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
