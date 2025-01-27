import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd
from dotenv import load_dotenv

from get_personas import PERSONAS_FILE
from questionnaire import FoodAndActivityPreferences

PREFERENCES_FILE = "data/preferences.csv"
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.2

# Load environment variables
load_dotenv()


def create_preferences_prompt(instructions: str, persona: str, query: str) -> str:
    """
    Create a prompt template for food and activity preferences.

    Args:
        instructions (str): Specific instructions for the model
        persona (str): Description of the persona to adopt

    Returns:
        ChatPromptTemplate: Formatted prompt template
    """
    prompt = f"""
    Instructions: {instructions}

    Persona: {persona}

    Query: {query}
    """

    return prompt

def get_structured_preferences(
        instructions: str,
        persona: str,
        query: str,
) -> FoodAndActivityPreferences:
    """
    Get structured food and activity preferences using LangChain.

    Args:
        instructions (str): Specific instructions for the model
        persona (str): Description of the persona to adopt
        query (str): The specific query or context
        model_name (str): OpenAI model to use
        temperature (float): Model temperature (0.0 = deterministic)
        use_json_mode (bool): Whether to use JSON mode instead of function calling

    Returns:
        FoodAndActivityPreferences: Structured preferences
    """
    model = ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL)

    structured_llm = model.with_structured_output(FoodAndActivityPreferences)

    prompt = create_preferences_prompt(instructions, persona, query)

    print(instructions)
    print(persona)
    print(query)

    return structured_llm.invoke(prompt)

def remove_breaks(text):
    # Remove empty lines and join all lines
    return ' '.join(line.strip() for line in text.splitlines() if line.strip())


def preferences_to_dataframe(personas_df: pd.DataFrame,
                             preferences_list: List[FoodAndActivityPreferences]) -> pd.DataFrame:
    # Convert each preferences object to a dictionary
    preferences_dicts = [prefs.model_dump() for prefs in preferences_list]

    # Create DataFrame from preferences
    preferences_df = pd.DataFrame(preferences_dicts)

    # Add persona information
    results_df = pd.concat([
        personas_df.reset_index(drop=True),
        preferences_df
    ], axis=1)

    return results_df


# Example usage
if __name__ == "__main__":
    # Example instructions and persona
    instructions = """
    On a scale from 1 (extremely dislike) to 9 (extremely like), please rate how much the persona LIKES each 
    presented item. The more the persona likes the item the higher you should rate it. The less the persona likes the 
    item, the lower you should rate it. If you are unfamiliar with any of the foods or the persona hasn't tried 
    any of the activities please answer 10 (never tried). If the persona prefers not to answer an item please 
    select 11 (prefer not to answer). You will notice that some of the items are 
    not food related. It is very important to us that you respond to these items using the same 
    parameters as you use for foods. Please remember that we would like you to report how much you 
    like each food or activity NOT how many time you eat each food or undertake each activity. 
    The answers to this questionnaire should fit to the persona specified in the persona section.
    use the description part and use the full skale.
    """

    query = "What would be this person's likely food and activity preferences?"

    personas_df = pd.read_csv(PERSONAS_FILE)

    preferences_results = []
    for _, person in personas_df.iterrows():
        description = person['description']
        preferences = get_structured_preferences(
            instructions=remove_breaks(instructions),
            persona=remove_breaks(description),
            query=query
        )
        preferences_results.append(preferences)
        print(preferences)

    results_df = preferences_to_dataframe(personas_df, preferences_results)
    results_df.to_csv(PREFERENCES_FILE, index=False)
