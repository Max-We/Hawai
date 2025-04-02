import concurrent
import math
import os
import random
import time
import warnings
from functools import partial

import kagglehub
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import config
from config import QUESTIONNAIRES_FILE, TEMPERATURE_RATING, MODEL_RATING, RATINGS_FILE
from get_questionnaires import remove_breaks
from structs.questionnaire import FoodAndActivityQuestionnaire
from structs.rating import RecipeRating

# Load environment variables
load_dotenv()


def load_items_information():
    path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")

    recipes_path = os.path.join(path, "RAW_recipes.csv")
    recipes = pd.read_csv(recipes_path)

    return recipes

def load_user_item_matrix():
    path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")

    def updateLabels(interactions_data):
        interactions_data["label"] = interactions_data["label"].apply(lambda x: int(x))
        return interactions_data

    def rename_and_drop_columns(interactions_data):
        interactions_data.rename(
            columns={"user_id": "user", "recipe_id": "item", "rating": "label"},
            inplace=True
        )
        for column in interactions_data.columns:
            if column != "user" and column != "item" and column != "label":
                interactions_data.drop(columns=column, inplace=True)

        updateLabels(interactions_data)

        return interactions_data

    # 2) Vorhandene Interactions-Dateien kombinieren, weil ansonsten ein out of bounds Fehler auftritt
    eval_data_path = os.path.join(path, "interactions_validation.csv")
    eval_data = pd.read_csv(eval_data_path)

    train_data_path = os.path.join(path, "interactions_train.csv")
    train_data = pd.read_csv(train_data_path)

    test_data_path = os.path.join(path, "interactions_test.csv")
    test_data = pd.read_csv(test_data_path)

    # Data muss zusammengefügt werden, damit sie gefiltert und im gleichen Verhältnis wieder aufgeteilt werden kann
    data = pd.concat([train_data, eval_data, test_data], ignore_index=True)
    data = rename_and_drop_columns(data)

    threshold = config.MIN_INTERACTIONS

    # 1) Items filtern, die mindestens * Interaktionen haben:
    min_item_interactions = threshold
    item_counts = data["item"].value_counts()
    items_to_keep = item_counts[item_counts >= min_item_interactions].index
    data_filtered = data[data["item"].isin(items_to_keep)]

    # 2) User filtern, die mindestens * Interaktionen haben:
    min_user_interactions = threshold
    user_counts = data_filtered["user"].value_counts()
    users_to_keep = user_counts[user_counts >= min_user_interactions].index
    data_filtered = data_filtered[data_filtered["user"].isin(users_to_keep)]

    user_item_matrix = data_filtered.pivot(index='user', columns='item', values='label')
    user_item_matrix_norm = user_item_matrix.fillna(3).to_numpy() - 3  # center around 0, replace NaN with 0

    unique_items = data_filtered['item'].unique()
    position_to_item = dict(enumerate(unique_items))
    # item_to_position = {item: pos for pos, item in position_to_item.items()}

    return user_item_matrix_norm, position_to_item

def project(data):
    # project from 1,9 to -2,2
    # [1,2]: -2, [3,4]: -1, [5]: 0, [6,7]: 1, [8,9]: 2
    #return math.ceil(abs((data-5)/2)) * (-1 if data < 5 else 1)
    valid_values=[2,1,-1,-2]
    return np.where(np.isin(data, list(valid_values)), data, 0)

    
def get_rating(item_idx, items_information, questionnaire):
    instructions = """the input is a questionair of a persons food and activity preferences on a scale from 1 to 9, 1 means a person would never eat it and strongly disklikes it, 9 means the persons loves 10 means the person has never tryed it and 11 means the person prefers not to awnser. you should act like the person would act and rate the following recipe based on the input."""
    query = "How would this person rate this recipe, on a scale from -2(strongly dislikes),-1(not enjoying),1(enjoying)2(strongly likes)?"

    item_name = items_information.loc[item_idx]["name"]
    item_ingredients = items_information.loc[item_idx]["ingredients"]

    rating_response = get_llm_recipe_rating(instructions, questionnaire, item_name, item_ingredients, query)
    print(rating_response)
    if rating_response is None or rating_response.rating is None:
        print(f"Fehler: Kein gültiges Rating für {item_name} erhalten!")
        return None, item_name, item_ingredients  # Oder ein Default-Wert

    rating = project(rating_response.rating)

    # print(f"Item: {item_name}, Rating (1-9) {rating_response.rating}, Projected: {rating}")

    return rating, item_name, item_ingredients


def random_learning_loop(user_item_matrix, items_information, idx_lookup_dict, questionnaire, n_iterations):
   
    n_items = user_item_matrix.shape[1]

    # Iterative random selection process
    summary = []
    user_vector = np.zeros(n_items)
    not_rated_mask = np.ones(n_items, dtype=bool)

    for _ in range(n_iterations):
        # Find indices of unrated items
        unrated_indices = np.where(not_rated_mask==1)[0]

        selected_item_id, item_idx = None, None
        while item_idx not in items_information.index or item_idx is None:
            # Try until a valid index is selected
            selected_item_id = np.random.choice(unrated_indices)
            item_idx = idx_lookup_dict[selected_item_id]

        # Ask oracle to rate the new item
        time.sleep(2)
        rating, title, ingredients = get_rating(item_idx, items_information, questionnaire)

        # Update user vector and mask
        user_vector[selected_item_id] = rating
        not_rated_mask[selected_item_id] = False

        summary.append((item_idx, rating, title, ingredients))

    return summary


def active_learning_loop(user_item_matrix, items_information, idx_lookup_dict, questionnaire, n_iterations, n_factors, k_neighbors, latent_neighbor):
    n_users, n_items = user_item_matrix.shape

    # SVD
    U, sigma, Vt = svds(user_item_matrix, k=n_factors)

    # KNN
    if not latent_neighbor:
        knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        knn_model.fit(user_item_matrix)
    else:
        knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        knn_model.fit(U)

    # Iterative active learning process
    summary = []
    user_vector, user_latent = np.zeros(n_items), np.mean(U, axis=0)
    not_rated_mask = np.ones(n_items)
    
    for _ in range(n_iterations):
        # Find similar users in original vector space
        print(user_latent,user_vector,user_item_matrix)
        print("Anzahl NaNs in user_latent:", np.sum(np.isnan(user_latent)))
        print("Anzahl NaNs in user_vector:", np.sum(np.isnan(user_vector)))
        print("Anzahl NaNs in user_item_matrix:", np.sum(np.isnan(user_item_matrix)))   

        user_latent = np.nan_to_num(user_latent, nan=0.0)
        user_vector = np.nan_to_num(user_vector, nan=0.0)
        user_item_matrix = np.nan_to_num(user_item_matrix, nan=0.0)
        if not latent_neighbor:
            _, similar_users_indices = knn_model.kneighbors([user_vector], n_neighbors=min(k_neighbors, len(user_item_matrix)))
        else:
            _, similar_users_indices = knn_model.kneighbors([user_latent], n_neighbors=min(k_neighbors, len(U)))

        # Get the latent vectors for these similar users for prediction
        similar_users_latents = U[similar_users_indices[0]]

        # Active learning by uncertainty sampling (= maximizing information gain)
        # 1. Calculate projected ratings
        rating_projections = np.dot(similar_users_latents, np.dot(np.diag(sigma), Vt))
        # 2. Calculate variance of ratings across selected users
        rating_variances = np.var(rating_projections, axis=0) * not_rated_mask
        # 3. Uncertainty sampling: select the item with the highest variance
        selected_item_id = np.argmax(rating_variances)

        # Ask oracle to rate the new item
        item_idx = idx_lookup_dict[selected_item_id] # convert user-item matrix id to original (sparse df) id
        rating, title, ingredients = get_rating(item_idx, items_information, questionnaire)
        # Update user vector & latent representation
        user_vector[selected_item_id] = rating
        not_rated_mask[selected_item_id] = 0
        user_latent = np.dot(user_vector, np.dot(Vt.T, np.diag(1.0 / sigma)))
        # Add to summary
        summary.append((item_idx, rating, title, ingredients))

    return summary


def create_ratings_prompt(instructions: str, food_questionnaire: FoodAndActivityQuestionnaire, recipe_title: str,
                          recipe_ingredients, query: str) -> str:
    prompt = f"""
    Person food preferences: <{food_questionnaire}>

    Instructions: {instructions}

    Recipe title: <{recipe_title}>

    Recipe ingredients: {recipe_ingredients}

    Important:  -2 for vervy diskliked food the person does not eat,  -1 for food the person does not like to eat but is abel to, use 1 for food the person enjoys eating and 2 for food the preson really likes a lot.
                

    Query: {query}
    """
    return prompt

def get_llm_recipe_rating(
        instructions: str,
        user_questionnaire: FoodAndActivityQuestionnaire,
        recipe_title: str,
        recipe_ingredients: str,
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

        model = ChatOpenAI(temperature=TEMPERATURE_RATING, model_name=MODEL_RATING)

        structured_llm = model.with_structured_output(RecipeRating)

        prompt = create_ratings_prompt(instructions, user_questionnaire, recipe_title, recipe_ingredients, query)

        for i in range(3):
            try:
                return structured_llm.invoke(prompt)
            except:
                print(f"Retrying {i}/3...")
                time.sleep(config.REQUEST_TIMEOUT)

def df_row_to_questionnaire(df_row: pd.Series) -> FoodAndActivityQuestionnaire:
    """
    Convert a DataFrame row into a FoodAndActivityPreferences object.

    Args:
        df_row (Union[pd.Series, pd.DataFrame]): A single row from the questionnaires DataFrame,
                                                either as a Series or single-row DataFrame

    Returns:
        FoodAndActivityQuestionnaire: Structured questionnaires object
    """
    # If input is a DataFrame, ensure it's a single row and convert to Series
    if not isinstance(df_row, pd.Series):
        raise ValueError("Please provide a single row from the DataFrame!")

    # Filter out persona-related columns (assuming they exist in the DataFrame)
    questionnaires_dict = {
        col: df_row[col]
        for col in df_row.index
        if col in FoodAndActivityQuestionnaire.model_fields.keys()
    }

    # Convert numeric values to integers where needed
    for key, value in questionnaires_dict.items():
        if pd.api.types.is_numeric_dtype(type(value)):
            questionnaires_dict[key] = int(value)

    # Create and return the FoodAndActivityPreferences object
    return FoodAndActivityQuestionnaire(**questionnaires_dict)

def process_questionnaire(questionnaire_row, user_item_matrix, items_information, idx_lookup_dict):
    """Process a single questionnaire row and return the resulting ratings"""
    # convert to questionnaire object
    questionnaire = df_row_to_questionnaire(questionnaire_row)

    item_ratings = active_learning_loop(
        user_item_matrix=user_item_matrix,
        items_information=items_information,
        idx_lookup_dict=idx_lookup_dict,
        questionnaire=questionnaire,
        n_iterations=20,
        n_factors=20,
        k_neighbors=50,
        latent_neighbor=False
    )

    # item_ratings = random_learning_loop(
    #     user_item_matrix=user_item_matrix,
    #     items_information=items_information,
    #     idx_lookup_dict=idx_lookup_dict,
    #     questionnaire=questionnaire,
    #     n_iterations=20
    # )

    # Create rows for this questionnaire
    questionnaire_rows = []
    for item_id, rating, title, ingredients in item_ratings:
        questionnaire_rows.append({
            "uuid": questionnaire_row["uuid"],
            "item_id": item_id,
            "rating": rating,
            "item_title": title,
            "item_ingredients": ingredients
        })

    return questionnaire_rows


if __name__ == "__main__":
    print("Loading questionnaires")
    questionnaires_df = pd.read_csv("/Users/felixipfling/Documents/GitHub/Hawai/questionairsMOD.csv")
    print("Loading user-item matrix")
    user_item_matrix, idx_lookup_dict = load_user_item_matrix()
    print("Loading item information")
    items_information = load_items_information()

    # Create a partial function with the common parameters
    process_func = partial(
        process_questionnaire,
        user_item_matrix=user_item_matrix,
        items_information=items_information,
        idx_lookup_dict=idx_lookup_dict
    )

    # Container for all result rows
    all_rows = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.CONCURRENT_WORKERS) as executor:
        # Submit all questionnaires as separate tasks
        future_to_questionnaire = {
            executor.submit(process_func, row): i
            for i, row in questionnaires_df.iterrows()
        }

        # Process results as they complete
        for future in tqdm(
                concurrent.futures.as_completed(future_to_questionnaire),
                total=len(questionnaires_df),
                desc="Rating items for each user"
        ):
            # Get results from this task
            questionnaire_rows = future.result()
            all_rows.extend(questionnaire_rows)

    # Create df with user id, item id, rating
    ratings_df = pd.DataFrame(all_rows)
    ratings_df.to_csv(RATINGS_FILE, index=False)
