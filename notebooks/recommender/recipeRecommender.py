import json
import os
import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import BPR, UserCF, ItemCF, SVD, SVDpp, ALS
from libreco.evaluation import evaluate
import kagglehub
import tensorflow as tf
import random
import numpy as np
from recipeRecommenderCore import RecipeRecommenderCore

class RecipeRecommender:
       
    def __init__(self):
        self.recommender = RecipeRecommenderCore()
        self.data_information = self.__load_items_information()
        self.name_df = self.__get_item_df(self.data_information)
        
        
    def load_and_preprocess(self,min_interactions):
        self.recommender.load_and_preprocess(min_interactions)
        
    def import_ratings_csv(self,path):
        self.recommender.import_ratings_csv(path)    
        
    def set_model(self,model):
        self.recommender.set_model(model)
    
    def train(self):
        self.recommender.train()
        
    def evaluate(self):
        return evaluate(
            model=self.recommender.model,
            data=self.recommender.test_data,
            neg_sampling=True,
            metrics=["loss", "roc_auc", "precision", "recall", "ndcg"]
        )
        
    def import_csv(self,path):
        self.recommender.import_ratings_csv(path)
        
    def get_data(self):
        return self.recommender.get_data()
    
    def get_userid_map(self):
        return self.recommender.get_userid_map()
        
    def save_recommendations_as_csv(self,amount_of_recs, path):
      df = self.get_recommendations_UUID(amount_of_recs)
      df.to_csv(path, index=False)
      return df
  
    def get_recommendations_UUID(self, n_rec):
        dfs = []
        for user_identifier in self.recommender.get_userid_map():
            recommendations = self.get_recommendation_UUID(user_identifier, n_rec)
            
            # Convert recommendations to DataFrame
            for recommendation in recommendations:
                df = self.map_recipe_to_df(recommendation,user_identifier)
                dfs.append(df)
        
        # Concatenate all DataFrames
        final_df = pd.concat(dfs, ignore_index=True)
        return final_df
    
    def map_recipe_to_df(self, recommendation, user_identifier):
        return pd.DataFrame({
            'uuid': [user_identifier],
            'item_id': [recommendation], 
            'item_title': [self.get_recipe_name(recommendation)],
            'item_ingredients': [self.get_recipe_ingredients(recommendation)]
        })

    def get_recommendation_UUID(self, user_identifier, n_rec):
        # Modell-Check
        if not self.recommender.model:
            raise ValueError("Model not trained. Call train() first.")

        # User-ID Mapping
        user_id = user_identifier
        if isinstance(user_identifier, str):
            if user_identifier not in self.recommender.get_userid_map():
                raise ValueError(f"User UUID '{user_identifier}' not found")
            user_id = self.recommender.get_userid_map()[user_identifier]

        recommendations = self.recommender.model.recommend_user(
            user=user_id, n_rec=n_rec, filter_consumed=True
        )
        return recommendations[user_id]
    

    
    def get_recommendation(self, user_id, n_rec):
        # Modell-Check
        if not self.recommender.model:
            raise ValueError("Model not trained. Call train() first.")

        recommendations = self.recommender.model.recommend_user(
            user=user_id, n_rec=n_rec, filter_consumed=True, cold_start="average"
        )
        return recommendations
    
    def __load_items_information(self):
        path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
        recipes_path = os.path.join(path, "RAW_recipes.csv")
        recipes = pd.read_csv(recipes_path)
        return recipes
    
    def __get_item_df(self,data):
        return data[["name", "id","ingredients"]]
    
    def get_recipe_name(self,recipe_id):
        """Helper to get recipe name from ID"""
        name = self.name_df.loc[self.name_df['id'] == recipe_id, 'name']
        return name.values[0] if not name.empty else "Unknown Recipe"
    
    def get_recipe_ingredients(self,recipe_id):
        """Helper to get recipe name from ID"""
        ingredients = self.name_df.loc[self.name_df['id'] == recipe_id, 'ingredients']
        return ingredients.values[0] if not ingredients.empty else "Unknown Recipe"