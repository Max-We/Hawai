import random

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors

from get_responses import PREFERENCES_FILE, remove_breaks
from questionnaire import FoodAndActivityPreferences
from rating import RecipeRating

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.2

# Load environment variables
load_dotenv()


def get_rating(item_id, default_value=None):
    if default_value:
        return default_value

    return random.randint(-2, 2)

def active_learning_recommendation(user_item_matrix, n_iterations=10, n_factors=20, k_neighbors=10, latent_neighbor=True, default_rating=None):
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
    for _ in range(n_iterations):
        # Find similar users in original vector space
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
        rating_variances = np.var(rating_projections, axis=0) * (user_vector == 0)
        # 3. Uncertainty sampling: select the item with the highest variance
        selected_item = np.argmax(rating_variances)

        # Ask oracle to rate the new item
        rating = get_rating(selected_item, default_rating)
        # Update user vector & latent representation
        user_vector[selected_item] = rating
        user_latent = np.dot(user_vector, np.dot(Vt.T, np.diag(1.0 / sigma)))
        # Add to summary
        summary.append((selected_item, rating))

    return summary


def create_ratings_prompt(instructions: str, food_preferences: FoodAndActivityPreferences, recipe_title: str, recipe_ingredients, query: str) -> str:
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

    Person food preferences: {food_preferences}
    
    Recipe title: {recipe_title}
    
    Recipe ingredients: {recipe_ingredients}

    Query: {query}
    """

    return prompt

def get_recipe_rating(
        instructions: str,
        food_perferences: FoodAndActivityPreferences,
        recipe_title: str,
        recipe_ingredients: str,
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

    structured_llm = model.with_structured_output(RecipeRating)

    prompt = create_ratings_prompt(instructions, food_perferences, recipe_title, recipe_ingredients, query)

    return structured_llm.invoke(prompt)

def dataframe_to_preferences(df_row: pd.Series) -> FoodAndActivityPreferences:
    """
    Convert a DataFrame row into a FoodAndActivityPreferences object.

    Args:
        df_row (Union[pd.Series, pd.DataFrame]): A single row from the preferences DataFrame,
                                                either as a Series or single-row DataFrame

    Returns:
        FoodAndActivityPreferences: Structured preferences object
    """
    # If input is a DataFrame, ensure it's a single row and convert to Series
    if not isinstance(df_row, pd.Series):
        raise ValueError("Please provide a single row from the DataFrame!")

    # Filter out persona-related columns (assuming they exist in the DataFrame)
    preferences_dict = {
        col: df_row[col]
        for col in df_row.index
        if col in FoodAndActivityPreferences.model_fields.keys()
    }

    # Convert numeric values to integers where needed
    for key, value in preferences_dict.items():
        if pd.api.types.is_numeric_dtype(type(value)):
            preferences_dict[key] = int(value)

    # Create and return the FoodAndActivityPreferences object
    return FoodAndActivityPreferences(**preferences_dict)

# Example usage
if __name__ == "__main__":
    instructions = """Given a person with the following food and activity preferences, rate how he / she would rate the following recipe:"""

    query = "How would this person rate this recipe?"

    preferences_df = pd.read_csv(PREFERENCES_FILE)

    ratings_result = []
    for _, person in preferences_df.iterrows():
        # Todo: get recipe title and ingredients from DataSet
        recipe_title = "Chef John's American Goulash"
        recipe_ingredients = """
        1 tablespoon olive oil, 
        2 pounds ground beef, 
        1 large onion, diced, 
        4 cloves garlic, minced, 
        2 large bay leaves, 
        2 tablespoons Hungarian paprika, 
        2 teaspoons Italian seasoning, 
        2 teaspoons kosher salt, 
        ½ teaspoon ground black pepper, 
        1 pinch cayenne pepper, or to taste, 
        1 quart chicken broth or water, 
        1 (24 ounce) jar marinara sauce, 
        1 (15 ounce) can diced tomatoes, 
        1 cup water, 
        2 tablespoons soy sauce, 
        2 cups elbow macaroni, 
        ¼ cup chopped Italian parsley, 
        1 cup shredded white Cheddar cheese (Optional), 
        """
        food_preferences = dataframe_to_preferences(person)
        rating = get_recipe_rating(
            instructions=instructions,
            food_perferences=food_preferences,
            recipe_title=recipe_title,
            recipe_ingredients=remove_breaks(recipe_ingredients),
            query=query
        )
        ratings_result.append(rating)

        # Todo: save the ratings to csv / into user-item matrix
        print(f"{person["name"]}, rating {rating}")
