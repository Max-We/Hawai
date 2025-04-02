import kagglehub
import os
import pandas as pd 

def load_items_information():
        path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
        recipes_path = os.path.join(path, "RAW_recipes.csv")
        recipes = pd.read_csv(recipes_path)
        return recipes

df=load_items_information()

df.to_csv("rezepte.csv", index=True)