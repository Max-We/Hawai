import warnings
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import QUESTIONNAIRES_FILE, TEMPERATURE_QUESTIONNAIRE, MODEL_QUESTIONNAIRES
from get_personas import PERSONAS_FILE
from structs.questionnaire import FoodAndActivityQuestionnaire


# Load environment variables
load_dotenv()


def create_questionnaire_prompt(instructions: str, persona: str, query: str) -> str:
    """
    Create a prompt template for food and activity questionnaires.

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

def get_structured_questionnaire(
        instructions: str,
        persona: str,
        query: str,
) -> FoodAndActivityQuestionnaire:
    """
    Get structured food and activity questionnaires using LangChain.

    Args:
        instructions (str): Specific instructions for the model
        persona (str): Description of the persona to adopt
        query (str): The specific query or context
        model_name (str): OpenAI model to use
        temperature (float): Model temperature (0.0 = deterministic)
        use_json_mode (bool): Whether to use JSON mode instead of function calling

    Returns:
        FoodAndActivityQuestionnaire: Structured questionnaires
    """
    with warnings.catch_warnings():
        # Ignore warning about model when using 3.5 turbo
        warnings.filterwarnings("ignore", category=UserWarning)

        model = ChatOpenAI(temperature=TEMPERATURE_QUESTIONNAIRE, model_name=MODEL_QUESTIONNAIRES)

        structured_llm = model.with_structured_output(FoodAndActivityQuestionnaire)

        prompt = create_questionnaire_prompt(instructions, persona, query)

        return structured_llm.invoke(prompt)

def remove_breaks(text):
    # Remove empty lines and join all lines
    return ' '.join(line.strip() for line in text.splitlines() if line.strip())


def questionnaires_to_dataframe(personas_df: pd.DataFrame,
                                questionnaires_list: List[FoodAndActivityQuestionnaire]) -> pd.DataFrame:
    # Convert each questionnaires object to a dictionary
    questionnaires_dicts = [prefs.model_dump() for prefs in questionnaires_list]

    # Create DataFrame from questionnaires
    questionnaires_df = pd.DataFrame(questionnaires_dicts)

    # Add uuid from personas_df to questionnaires_df
    questionnaires_df['uuid'] = personas_df['uuid']

    # Reorder columns so that 'uuid' is the first column
    cols = ['uuid'] + [col for col in questionnaires_df.columns if col != 'uuid']
    questionnaires_df = questionnaires_df[cols]

    return questionnaires_df


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

    questionnaires_results = []
    for _, person in tqdm(personas_df.iterrows(), total=len(personas_df), desc="Generating questionnaires"):
        description = person['description']
        questionnaire = get_structured_questionnaire(
            instructions=remove_breaks(instructions),
            persona=remove_breaks(description),
            query=query
        )
        questionnaires_results.append(questionnaire)

    results_df = questionnaires_to_dataframe(personas_df, questionnaires_results)
    results_df.to_csv(QUESTIONNAIRES_FILE, index=False)
