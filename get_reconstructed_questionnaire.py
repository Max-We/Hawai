import warnings
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import QUESTIONNAIRES_FILE, TEMPERATURE_QUESTIONNAIRE, MODEL_QUESTIONNAIRES, RATINGS_FILE, \
    QUESTIONNAIRES_RECONSTRUCTED_FILE
from get_personas import PERSONAS_FILE
from structs.questionnaire import FoodAndActivityQuestionnaire


# Load environment variables
load_dotenv()


def create_questionnaire_prompt(instructions: str, query: str, data_string: str) -> str:
    """Create a prompt template for food and activity questionnaires."""
    prompt = f"""
    Instructions: {instructions}

    Recipe ratings: {data_string}

    Query: {query}
    """

    return prompt

def get_structured_questionnaire(
        instructions: str,
        query: str,
        data_string: str
) -> FoodAndActivityQuestionnaire:
    """Get structured food and activity questionnaires using LangChain."""
    with warnings.catch_warnings():
        # Ignore warning about model when using 3.5 turbo
        warnings.filterwarnings("ignore", category=UserWarning)

        model = ChatOpenAI(temperature=TEMPERATURE_QUESTIONNAIRE, model_name=MODEL_QUESTIONNAIRES)

        structured_llm = model.with_structured_output(FoodAndActivityQuestionnaire)

        prompt = create_questionnaire_prompt(instructions, query, data_string)

        return structured_llm.invoke(prompt)

def remove_breaks(text):
    # Remove empty lines and join all lines
    return ' '.join(line.strip() for line in text.splitlines() if line.strip())


def questionnaires_to_dataframe(uuids: list,
                                questionnaires_list: List[FoodAndActivityQuestionnaire]) -> pd.DataFrame:
    # Convert each questionnaires object to a dictionary
    questionnaires_dicts = [prefs.model_dump() for prefs in questionnaires_list]

    # Create DataFrame from questionnaires
    questionnaires_df = pd.DataFrame(questionnaires_dicts)

    # Add uuid from personas_df to questionnaires_df
    questionnaires_df['uuid'] = uuids

    # Reorder columns so that 'uuid' is the first column
    cols = ['uuid'] + [col for col in questionnaires_df.columns if col != 'uuid']
    questionnaires_df = questionnaires_df[cols]

    return questionnaires_df


# Example usage
if __name__ == "__main__":
    # Example instructions and persona
    instructions = """
    You are now acting as the user described in the data below. Based on the user's recipe ratings, ingredients preferences, and cooking history, you will fill out a food and activity preference questionnaire.

    Rate each item on a scale from 1 (extremely dislike) to 9 (extremely like). Your ratings should authentically reflect how much this specific user would like each item based on their recipe history.

    Follow these guidelines:
    - Use 1 (extremely dislike) to 9 (extremely like) for expressing preferences
    - Use 10 (never tried) if the user likely hasn't encountered this item
    - Use 11 (prefer not to answer) only when appropriate privacy concerns exist

    For non-food activities, infer preferences based on the user's food choices and what they might suggest about lifestyle and preferences. For example, someone who rates outdoor grilling recipes highly might enjoy camping.

    Respond to each item as if you were this user, not as an AI assistant analyzing the user.
    """

    query = "Based on the recipe ratings and ingredients data provided, please complete the following food and activity preference questionnaire as if you were this user. Rate how much you like each item, not how frequently you consume or do it."

    ratings_df = pd.read_csv(RATINGS_FILE)

    uuids = ratings_df['uuid'].unique()
    questionnaires_reconstructed_results = []
    for uuid in tqdm(uuids, total=len(uuids), desc="Generating reconstructed questionnaires"):
        item_titles = ratings_df[ratings_df['uuid'] == uuid]['item_title']
        item_ingredients = ratings_df[ratings_df['uuid'] == uuid]['item_ingredients']
        ratings = ratings_df[ratings_df['uuid'] == uuid]['rating']
        # Transform rating range from -2,2 to 1,9
        ratings = (ratings*2) + 5 # 5 is avg rating of 1-9

        data_string = ""
        for i in range(len(item_titles)):
            data_string += "{"
            data_string += f"title: '{item_titles.iloc[i]}', "
            data_string += f"ingredients: '{item_ingredients.iloc[i]}', "
            data_string += f"rating: '{ratings.iloc[i]}'"
            data_string += "}, "

        questionnaire = get_structured_questionnaire(
            instructions=remove_breaks(instructions),
            query=query,
            data_string=data_string
        )
        questionnaires_reconstructed_results.append(questionnaire)

    results_df = questionnaires_to_dataframe(uuids, questionnaires_reconstructed_results)
    results_df.to_csv(QUESTIONNAIRES_RECONSTRUCTED_FILE, index=False)
