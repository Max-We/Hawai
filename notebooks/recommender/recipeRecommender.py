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
        self.recommender.evaluate()