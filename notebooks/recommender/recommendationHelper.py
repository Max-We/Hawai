import kagglehub
import os
import pandas as pd



class RHelper:
    def load_items_information():
        path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
        recipes_path = os.path.join(path, "RAW_recipes.csv")
        recipes = pd.read_csv(recipes_path)
        return recipes

    def get_user_interactions(user_id,data):
        recipes = []
        df = data[data['user'] == user_id]
        for _, row in df.iterrows():
            recipes.append((row['item'],row['label']))
        return recipes
    
    def get_name_df(data):
        return data[["name", "id"]]
    
    def get_recipe_name(recipe_id,name_df):
        """Helper to get recipe name from ID"""
        name = name_df.loc[name_df['id'] == recipe_id, 'name']
        return name.values[0] if not name.empty else "Unknown Recipe"
    
    def get_name_df(data):  
        return data[["name", "id"]]

    def get_recipe_ingredients(recipe_id,data):
        ingredients = data.loc[data["id"] == recipe_id, 'ingredients']
        return ingredients if not ingredients.empty else "Unknown Recipe"