import time
import warnings
from functools import partial
from typing import List

from langchain_openai import ChatOpenAI
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures

from config import QUESTIONNAIRES_FILE, TEMPERATURE_QUESTIONNAIRE, MODEL_QUESTIONNAIRES, CONCURRENT_WORKERS, \
    REQUEST_TIMEOUT
from get_personas import PERSONAS_FILE
from structs.questionnaire import FoodAndActivityQuestionnaire, FoodAndActivityQuestionnairePart1, \
    FoodAndActivityQuestionnairePart2, FoodAndActivityQuestionnairePart3, FoodAndActivityQuestionnairePart4

# Load environment variables
load_dotenv()


def create_questionnaire_prompt(instructions: str, persona: str, query: str, part_num: int) -> str:
    """
    Create a prompt template for a specific part of the food and activity questionnaire.

    Args:
        instructions (str): Specific instructions for the model
        persona (str): Description of the persona to adopt
        query (str): The specific query or context
        part_num (int): The part number (1, 2, 3, or 4)

    Returns:
        str: Formatted prompt text
    """
    prompt = f"""
    Persona: <{persona}>

    Instructions: {instructions}

    You are completing PART {part_num} of 4 for this questionnaire. Each part contains a different set of items.

    Query: {query}
    """

    return prompt


def get_structured_questionnaire_part(
        instructions: str,
        persona: str,
        query: str,
        part_num: int
):
    """
    Get a structured part of the food and activity questionnaire using LangChain.

    Args:
        instructions (str): Specific instructions for the model
        persona (str): Description of the persona to adopt
        query (str): The specific query or context
        part_num (int): The part number (1, 2, 3, or 4)

    Returns:
        BaseModel: Structured questionnaire part
    """
    with warnings.catch_warnings():
        # Ignore warning about model when using 3.5 turbo
        warnings.filterwarnings("ignore", category=UserWarning)

        model = ChatOpenAI(temperature=TEMPERATURE_QUESTIONNAIRE, model_name=MODEL_QUESTIONNAIRES)

        # Choose the appropriate questionnaire part class
        if part_num == 1:
            questionnaire_class = FoodAndActivityQuestionnairePart1
        elif part_num == 2:
            questionnaire_class = FoodAndActivityQuestionnairePart2
        elif part_num == 3:
            questionnaire_class = FoodAndActivityQuestionnairePart3
        else:
            questionnaire_class = FoodAndActivityQuestionnairePart4

        structured_llm = model.with_structured_output(questionnaire_class)

        prompt = create_questionnaire_prompt(instructions, persona, query, part_num)

        n_trials = 3
        for i in range(n_trials):
            try:
                return structured_llm.invoke(prompt)
            except:
                print(f"Retrying {i}/3")
                time.sleep(REQUEST_TIMEOUT)


def merge_questionnaire_parts(part1, part2, part3, part4) -> FoodAndActivityQuestionnaire:
    """
    Merge the four questionnaire parts into a complete questionnaire.

    Args:
        part1 (FoodAndActivityQuestionnairePart1): Part 1 of the questionnaire
        part2 (FoodAndActivityQuestionnairePart2): Part 2 of the questionnaire
        part3 (FoodAndActivityQuestionnairePart3): Part 3 of the questionnaire
        part4 (FoodAndActivityQuestionnairePart4): Part 4 of the questionnaire

    Returns:
        FoodAndActivityQuestionnaire: A complete merged questionnaire
    """
    # Combine all parts into a single dictionary
    merged_dict = {}

    # Add fields from each part
    merged_dict.update(part1.model_dump())
    merged_dict.update(part2.model_dump())
    merged_dict.update(part3.model_dump())
    merged_dict.update(part4.model_dump())

    # Create and return a complete questionnaire
    return FoodAndActivityQuestionnaire(**merged_dict)


def get_complete_questionnaire(
        instructions: str,
        persona: str,
        query: str
) -> FoodAndActivityQuestionnaire:
    """
    Get a complete food and activity questionnaire by generating and merging all four parts.

    Args:
        instructions (str): Specific instructions for the model
        persona (str): Description of the persona to adopt
        query (str): The specific query or context

    Returns:
        FoodAndActivityQuestionnaire: Complete structured questionnaire
    """
    # Generate each part of the questionnaire
    part1 = get_structured_questionnaire_part(instructions, persona, query, 1)
    part2 = get_structured_questionnaire_part(instructions, persona, query, 2)
    part3 = get_structured_questionnaire_part(instructions, persona, query, 3)
    part4 = get_structured_questionnaire_part(instructions, persona, query, 4)

    # Merge all parts into a complete questionnaire
    return merge_questionnaire_parts(part1, part2, part3, part4)


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


def process_persona_to_questionnaire(persona_row, instructions, query):
    """
    Process a single persona to generate a complete questionnaire.
    This function is designed to be used with concurrent.futures.

    Args:
        persona_row: A row from the personas DataFrame
        instructions: Instructions for the questionnaire
        query: Query for the questionnaire

    Returns:
        tuple: (persona_index, questionnaire)
    """
    idx = persona_row[0]
    person = persona_row[1]
    description = person['description']

    try:
        # Generate complete questionnaire by combining all four parts
        questionnaire = get_complete_questionnaire(
            instructions=remove_breaks(instructions),
            persona=remove_breaks(description),
            query=query
        )
        return idx, questionnaire
    except Exception as e:
        print(f"Error processing persona {idx}: {e}")
        time.sleep(3)
        return idx, None


# Example usage
if __name__ == "__main__":
    # Example instructions and persona
    instructions = f"""
    On a scale from 1 (extremely dislike) to 9 (extremely like), please rate how much the persona LIKES or DISLIKES each 
    presented item. The more the persona likes the item the higher you should rate it. The less the persona likes the 
    item, the lower you should rate it. If you are unfamiliar with any of the foods or the persona hasn't tried 
    any of the activities please answer 10 (never tried). If the persona prefers not to answer an item please 
    select 11 (prefer not to answer). You will notice that some of the items are 
    not food related. It is very important to us that you respond to these items using the same 
    parameters as you use for foods. Please remember that we would like you to report how much the person 
    likes each food or activity NOT how many time he/she eats each food or undertakes each activity. 
    The answers to this questionnaire should fit to the persona specified in the persona section.
    """

    query = "What would be this person's likely food and activity preferences? Use the description and use the full scale (1-9)."

    personas_df = pd.read_csv(PERSONAS_FILE)

    # Create a list to store results
    questionnaires_results = [None] * len(personas_df)

    # Process personas concurrently
    print(f"Generating questionnaires...")

    # Create a partial function with fixed arguments
    process_func = partial(process_persona_to_questionnaire, instructions=remove_breaks(instructions), query=query)

    # Use ThreadPoolExecutor for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        # Submit all tasks and create a map of futures
        future_to_idx = {
            executor.submit(process_func, (idx, person)): idx
            for idx, person in personas_df.iterrows()
        }

        # Use tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                           total=len(personas_df),
                           desc="Generating questionnaires"):
            idx, questionnaire = future.result()
            if questionnaire:
                questionnaires_results[idx] = questionnaire

    # Filter out any None values (failed processing)
    questionnaires_results = [q for q in questionnaires_results if q is not None]

    # Create DataFrame from results and save to CSV
    results_df = questionnaires_to_dataframe(
        personas_df,
        questionnaires_results)
    results_df.to_csv(QUESTIONNAIRES_FILE, index=False)

    print(f"Completed processing {len(questionnaires_results)} personas out of {len(personas_df)}.")
