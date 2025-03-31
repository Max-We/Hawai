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
        
    def get_data(self):
        return self.recommender.get_data()
        
    def save_recommendations_as_csv(self,items_information,amount_of_recs, path):
      df = self.get_recommendations_UUID(items_information,amount_of_recs)
      df.to_csv(path, index=False)
      return df
  
    def get_recommendation(self, user_id, n_rec):
        # Modell-Check
        if not self.recommender.model:
            raise ValueError("Model not trained. Call train() first.")

        recommendations = self.recommender.model.recommend_user(
            user=user_id, n_rec=n_rec, filter_consumed=True
        )
        return recommendations
  
  
    def get_recommendations_UUID(self, items_information, n_rec):
        dfs = []
        for user_identifier in self.user_id_map:
            df = self.get_recommendation_UUID(user_identifier, n_rec, items_information)
            dfs.append(df)
        # Alle einzelnen DataFrames zusammenfügen
        final_df = pd.concat(dfs, ignore_index=True)
        return final_df

    def get_recommendation_UUID(self, user_identifier, n_rec, items_information):
        # Modell-Check
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")

        # User-ID Mapping
        user_id = user_identifier
        if isinstance(user_identifier, str):
            if user_identifier not in self.user_id_map:
                raise ValueError(f"User UUID '{user_identifier}' not found")
            user_id = self.user_id_map[user_identifier]

        recommendations = self.recommender.model.recommend_user(
            user=user_id, n_rec=n_rec, filter_consumed=True
        )
        
        return recommendations