import concurrent.futures
import os
import random
import re
import time
import warnings
from functools import partial
from typing import List

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from config import TEMPERATURE_QUESTIONNAIRE, MODEL_QUESTIONNAIRES, RATINGS_FILE, \
    QUESTIONNAIRES_RECONSTRUCTED_FILE, CONCURRENT_WORKERS, REQUEST_TIMEOUT
from structs.questionnaire import FoodAndActivityQuestionnaire, FoodAndActivityQuestionnairePart1, \
    FoodAndActivityQuestionnairePart2, FoodAndActivityQuestionnairePart3, FoodAndActivityQuestionnairePart4

# Load environment variables
load_dotenv()


def create_questionnaire_prompt(instructions: str, query: str, ratings_string: str, recommendations_string: str, part_num: int) -> str:
    """
    Create a prompt template for a specific part of the food and activity questionnaire.

    Args:
        instructions (str): Specific instructions for the model
        query (str): The specific query or context
        ratings_string (str): Recipe ratings and ingredient data
        part_num (int): The part number (1, 2, 3, or 4)

    Returns:
        str: Formatted prompt text
    """
    prompt = f"""
    Instructions: {instructions}

    Recipe ratings of this user: {ratings_string if ratings_string else '[None]'}
    
    Recommended recipes for this user (he/she might like): {recommendations_string if recommendations_string else '[None]'}

    You are completing PART {part_num} of 4 for this questionnaire. Each part contains a different set of items.

    Query: {query}
    """

    return prompt


def get_structured_questionnaire_part(
        instructions: str,
        query: str,
        ratings_string: str,
        recommendations_string: str,
        part_num: int
):
    """
    Get a structured part of the food and activity questionnaire using LangChain.

    Args:
        instructions (str): Specific instructions for the model
        query (str): The specific query or context
        ratings_string (str): Recipe ratings and ingredient data
        recommendations_string (str): Recommendations for the user
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

        prompt = create_questionnaire_prompt(instructions, query, ratings_string, recommendations_string, part_num)

        n_trials = 3
        for i in range(n_trials):
            try:
                return structured_llm.invoke(prompt)
            except Exception as e:
                print(f"Retrying {i}/3...")
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
        query: str,
        ratings_string: str,
        recommendations_string: str
) -> FoodAndActivityQuestionnaire:
    """
    Get a complete food and activity questionnaire by generating and merging all four parts.

    Args:
        instructions (str): Specific instructions for the model
        query (str): The specific query or context
        ratings_string (str): Recipe ratings and ingredient data
        recommendations_string (str): Recommendations for the user

    Returns:
        FoodAndActivityQuestionnaire: Complete structured questionnaire
    """
    # Generate each part of the questionnaire
    part1 = get_structured_questionnaire_part(instructions, query, ratings_string, recommendations_string, 1)
    part2 = get_structured_questionnaire_part(instructions, query, ratings_string, recommendations_string, 2)
    part3 = get_structured_questionnaire_part(instructions, query, ratings_string, recommendations_string, 3)
    part4 = get_structured_questionnaire_part(instructions, query, ratings_string, recommendations_string, 4)

    # Merge all parts into a complete questionnaire
    return merge_questionnaire_parts(part1, part2, part3, part4)


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


def process_user_questionnaire(uuid, ratings_df, recommendations_df, instructions, query):
    """Process a single user's questionnaire and return the result"""

    # Resulting format: "{title: 'a', ingredients: 'b', rating: 'c'}, {title: 'd', ...}, ..."
    ratings_string = ""
    if ratings_df is not None:
        rated_item_titles = ratings_df[ratings_df['uuid'] == uuid]['item_title']
        rated_item_ingredients = ratings_df[ratings_df['uuid'] == uuid]['item_ingredients']
        ratings = ratings_df[ratings_df['uuid'] == uuid]['rating']
        # Transform rating range from -2,2 to 1,9
        ratings = (ratings * 2) + 5  # 5 is avg rating of 1-9

        for i in range(len(rated_item_titles)):
            ratings_string += "{"
            ratings_string += f"title: '{rated_item_titles.iloc[i]}', "
            ratings_string += f"ingredients: '{rated_item_ingredients.iloc[i]}', "
            ratings_string += f"rating: '{ratings.iloc[i]}'"
            ratings_string += "}, "

    recommendations_string = ""
    if recommendations_df is not None:
        recommended_item_titles = recommendations_df[recommendations_df['uuid'] == uuid]['item_title']
        recommended_item_ingredients = recommendations_df[recommendations_df['uuid'] == uuid]['item_ingredients']

        for i in range(len(recommended_item_titles)):
            recommendations_string += "{"
            recommendations_string += f"title: '{recommended_item_titles.iloc[i]}', "
            recommendations_string += f"ingredients: '{recommended_item_ingredients.iloc[i]}'"
            recommendations_string += "}, "

    # Generate complete questionnaire by combining all four parts
    questionnaire = get_complete_questionnaire(
        instructions=remove_breaks(instructions),
        query=query,
        ratings_string=ratings_string,
        recommendations_string=recommendations_string
    )

    return uuid, questionnaire


def find_recommendation_files(directory='data'):
    """
    Find all CSV files in the specified directory that match the pattern 'recommendations_*.csv'

    Args:
        directory (str): Directory to search in, defaults to current directory

    Returns:
        list: List of matching filenames with algorithm names and their full paths
    """
    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter for CSV files that start with 'recommendations_'
    recommendation_files = []
    pattern = re.compile(r'recommendations_([^.]+)\.csv')

    for filename in all_files:
        match = pattern.match(filename)
        if match:
            algorithm_name = match.group(1)
            file_path = os.path.join(directory, filename)
            recommendation_files.append((algorithm_name, file_path))

    return recommendation_files

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

    recommendations_dfs = [
        ("none", ""),       # no ratings or recommendations
        ("ratings", ""),    # ratings only, no recommendations
        ("random", "")      # fully random questionnaire (serves as baseline)
    ]
    recommendations_dfs += find_recommendation_files("data") # ratings + recommendations
    # recommendations_dfs = recommendations_dfs[3:] # continue  from a specific recommender on

    print(f"Processing {len(recommendations_dfs)} types of reconstructed questionnaires...")

    # Create a partial function with the common parameters
    for recommender_algorithm, recommender_df_path in recommendations_dfs:
        recommendations_df = pd.read_csv(recommender_df_path) if recommender_df_path else None

        if recommender_algorithm == "random":
            questionnaires_reconstructed_results = []

            results = {}
            for uuid in tqdm(uuids, total=len(uuids), desc=f"Generating reconstructed questionnaires: {recommender_algorithm}"):
                # Get all field names from the model
                field_names = list(FoodAndActivityQuestionnaire.__annotations__.keys())
                # Create a dictionary with random values 1-9 for each field
                random_values = {field: random.randint(1, 11) for field in field_names}
                # Instantiate the model with these random values
                results[uuid] = FoodAndActivityQuestionnaire(**random_values)
            questionnaires_reconstructed_results = [results[uuid] for uuid in uuids]
        else:
            process_func = partial(
                process_user_questionnaire,
                ratings_df=None if recommender_algorithm == "none" else ratings_df,
                recommendations_df=recommendations_df,
                instructions=remove_breaks(instructions),
                query=query
            )

            # Container for results
            results = {}

            # Info: Use max of 5 workers to avoid OpenAI API rate limits (o4-mini)
            # Info: for o4 (non-mini) use only 1 worker (has strict rate limits)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, CONCURRENT_WORKERS//3)) as executor:
                # Submit all users as separate tasks
                future_to_uuid = {
                    executor.submit(process_func, uuid): uuid
                    for uuid in uuids
                }

                # Process results as they complete
                for future in tqdm(
                        concurrent.futures.as_completed(future_to_uuid),
                        total=len(uuids),
                        desc=f"Generating reconstructed questionnaires: {recommender_algorithm}"
                ):
                    try:
                        # Get result from this task
                        uuid, questionnaire = future.result()
                        results[uuid] = questionnaire
                    except Exception as exc:
                        uuid = future_to_uuid[future]
                        print(f"UUID {uuid} generated an exception: {exc}")

            # Ensure same order as input UUIDs
            questionnaires_reconstructed_results = [results[uuid] for uuid in uuids]

        # Create final dataframe and save results
        results_df = questionnaires_to_dataframe(uuids, questionnaires_reconstructed_results)
        results_df.to_csv(QUESTIONNAIRES_RECONSTRUCTED_FILE+recommender_algorithm+".csv", index=False)
