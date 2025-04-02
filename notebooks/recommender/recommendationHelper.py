import kagglehub
import os
import pandas as pd



class RHelper:
    def __init__(self,recommender):
        self.data_information = self.__load_items_information()
        self.name_df = self.__get_name_df(self.data_information)
        self.recommender = recommender
        self.recommender_data = recommender.get_data()
        
    def __load_items_information(self):
        path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
        recipes_path = os.path.join(path, "RAW_recipes.csv")
        recipes = pd.read_csv(recipes_path)
        return recipes
    
    def __get_name_df(self,data):
        return data[["name", "id"]]
    
    def get_recipe_name(self,recipe_id):
        """Helper to get recipe name from ID"""
        name = self.name_df.loc[self.name_df['id'] == recipe_id, 'name']
        return name.values[0] if not name.empty else "Unknown Recipe"
    
    def __get_recipe_ingredients(self,recipe_id):
        ingredients = self.data_information.loc[self.data_information["id"] == recipe_id, 'ingredients']
        return ingredients if not ingredients.empty else "Unknown Recipe"
    
    def get_formatted_recommendations(self, user_id, n_rec):
        if user_id not in self.recommender_data["user"]:
            user_id = self.recommender.recommender.get_userid_map()[user_id]
            
        recommendations = self.recommender.get_recommendation(user_id, n_rec)
        recs = recommendations[user_id]
        
        formatted_recs = []
        for rec in recs:
            recipe_name = self.get_recipe_name(rec)  # Nur die Rezept-ID übergeben
            formatted_recs.append({"recipe_id": rec, "recipe_name": recipe_name})
        return pd.DataFrame(formatted_recs)
            
    def get_user_interactions(self,user_id):
        if user_id not in self.recommender_data["user"]:
            user_id = self.recommender.recommender.get_userid_map()[user_id]
            
        recipes = []
        df = self.recommender_data[self.recommender_data['user'] == user_id]
        for _, row in df.iterrows():
            item = row['item']
            rating = row['label']
            item_name = self.get_recipe_name(row['item'])
            recipes.append({"recipe_id": item, "rating":rating, "recipe_name": item_name})
        return pd.DataFrame(recipes)