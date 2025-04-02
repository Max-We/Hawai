import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import BPR, UserCF, ItemCF, SVD, SVDpp, ALS
import kagglehub


class RecipeRecommenderCore:
    def __init__(self, data_path="shuyangli94/food-com-recipes-and-user-interactions"):
        self.embed_size = 64     # Für Embedding-basierte Modelle
        self.n_epochs = 10        # Für trainierbare Modelle
        self.lr = 1e-4            # Lernrate
        self.batch_size = 256    # Batch-Größe
        self.num_neg = 10 # Negative Samples
        self.reg=1e-5
        self.num_threads=4
        self.k_sim = 50           # Für CF-Modelle
        self.sim_type = "cosine"  # Ähnlichkeitsmaß

        # Gemeinsame Objekte
        self.model = None
        self.data_info = None
        self.name_df = None
        self.data_filtered = None
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        self.user_id_map = {}
        self.item_popularity = None

        self.data_path = data_path


    def set_model(self,model_name):
        """Zentrale Methode zur Modellauswahl mit einheitlichen Parametern"""
        tf.compat.v1.reset_default_graph()
        self.__prepare_data()  
        
        model_dict = {
        "SVD": self.set_model_SVD,
        "SVDpp": self.set_model_SVDpp,
        "BPR": self.set_model_BPR,
        "ALS": self.set_model_ALS,
        "UserCF": self.set_model_UserCF,
        "ItemCF": self.set_model_ItemCF
    }

        if model_name in model_dict:
            self.model = model_dict[model_name]()  
        else:
            raise ValueError(f"Unbekanntes Modell: {model_name}")
        
    
    def set_model_SVD(self):
        return SVD(
                    "ranking",
                    data_info=self.data_info,
                    loss_type="cross_entropy",
                    embed_size=64,
                    n_epochs=10,
                    lr=3e-4,
                    reg=1e-5,
                    batch_size=256,
                    num_neg=10,
                    )
        
    def set_model_SVDpp(self):
        return SVDpp(
                    "ranking",
                    data_info=self.data_info,
                    loss_type="cross_entropy",
                    embed_size=64,
                    n_epochs=10,
                    lr=3e-4,
                    reg=1e-5,
                    batch_size=256,
                    num_neg=10,
                    )
    def set_model_BPR(self):
        return BPR(
                    "ranking",
                    data_info=self.data_info,
                    loss_type="bpr",
                    embed_size=64,
                    n_epochs=10,
                    lr=3e-4,
                    reg=1e-5,
                    batch_size=256,
                    num_neg=10,
                    )
    def set_model_ALS(self):
        return ALS(
                    "ranking",
                    data_info=self.data_info,
                    embed_size=64,
                    n_epochs=10,
                    reg=1e-5,
                    )
    def set_model_UserCF(self):
        return UserCF(
                        "ranking",
                        data_info=self.data_info,
                        sim_type="cosine",  
                        k=50,
                    )
    def set_model_ItemCF(self):
        return ItemCF(
                        "ranking",
                        data_info=self.data_info,
                        sim_type="cosine",  
                        k=50,
                    )

    def train(self):
        if not self.model:
            raise ValueError("Model not trained. Call set_model() first.")
        
        if isinstance(self.model, (UserCF, ItemCF)):  # Tuple für kürzere Syntax
            self.model.fit(
                train_data=self.train_data,
                neg_sampling=True,
                verbose=2,
                eval_data=self.eval_data,
                metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
            )
        else:
            self.model.fit(
                train_data=self.train_data,
                neg_sampling=True,
                shuffle=True,
                verbose=2,
                eval_data=self.eval_data,
                metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
            )

    def load_and_preprocess(self, min_interactions):
        """Load and preprocess interaction data"""
        # Download and load dataset
        path = kagglehub.dataset_download(self.data_path)

        # Load and combine interaction data
        train = pd.read_csv(os.path.join(path, "interactions_train.csv"))
        eval = pd.read_csv(os.path.join(path, "interactions_validation.csv"))
        test = pd.read_csv(os.path.join(path, "interactions_test.csv"))

        combined = pd.concat([train, eval, test], ignore_index=True)
        combined = self._rename_and_filter_data(combined)
        
        combined = combined[["user", "item", "label"]]
        # Filter items
        item_counts = combined["item"].value_counts()
        items_to_keep = item_counts[item_counts >= min_interactions].index
        filtered = combined[combined["item"].isin(items_to_keep)]

        # Filter users
        user_counts = filtered["user"].value_counts()
        users_to_keep = user_counts[user_counts >= min_interactions].index
        self.data_filtered = filtered[filtered["user"].isin(users_to_keep)]
        

        print(f"Finale Daten: {len(self.data_filtered)} Interaktionen ")

    def __prepare_data(self):
        # Binarize ratings: 0-2 → 0, 3-5 → 1
        self.data_filtered['label'] = self.data_filtered['label'].apply(
            lambda x: 0 if x <= 2 else 1
        )
        # Split data
        self.train_data, self.eval_data, self.test_data = random_split(
            self.data_filtered,
            multi_ratios=[0.8, 0.1, 0.1]
        )

        # Build datasets
        self.train_data, self.data_info = DatasetPure.build_trainset(self.train_data)
        self.eval_data = DatasetPure.build_evalset(self.eval_data)
        self.test_data = DatasetPure.build_testset(self.test_data)



    def _rename_and_filter_data(self, interactions_data):
      # Erzeuge explizite Kopie des DataFrames
      df = interactions_data.copy()

      # Spalten umbenennen (ohne inplace)
      df = df.rename(columns={
          "user_id": "user",
          "recipe_id": "item",
          "rating": "label"
      })

      # Typkonvertierung mit .loc
      df.loc[:, "label"] = df["label"].astype(int)
      return df



    def import_ratings_csv(self, file_path):
      """Import ratings from CSV and map UUIDs to numeric IDs"""
      try:
          # Load CSV
          df = pd.read_csv(file_path)
          df = df.drop(columns=['item_title', 'item_ingredients'], errors='ignore')

          # Check required columns
          required = {"uuid", "item_id", "rating"}
          if not required.issubset(df.columns):
              missing = required - set(df.columns)
              raise ValueError(f"Fehlende Spalten: {missing}")

          # Process and map UUIDs
          processed_df = self.__process_csv(df)

          # Add to data
          self.data_filtered = pd.concat(
              [self.data_filtered, processed_df],
              ignore_index=True
         )
          print(f"{len(processed_df)} neue Bewertungen hinzugefügt.")

      except FileNotFoundError:
          print(f"Datei {file_path} nicht gefunden.")
      except Exception as e:
          print(f"Fehler: {str(e)}")

    def __process_csv(self, df):
      """Map UUIDs to numeric IDs"""
      # Rename columns
      df = df.rename(columns={
          "uuid": "user",
          "item_id": "item",
          "rating": "label"
      })
      

      df.loc[df["label"] < 0, "label"] = 0
      df.loc[df["label"] > 0, "label"] = 1


      # Determine current max ID from user_id_map
      current_max = max(self.user_id_map.values()) if self.user_id_map else 0

      # Generate new IDs for unknown UUIDs
      new_users = [uuid for uuid in df["user"].unique() if uuid not in self.user_id_map]
      num_new = len(new_users)

      if num_new > 0:
          new_ids = range(current_max + 1, current_max + num_new + 1)
          self.user_id_map.update(zip(new_users, new_ids))

      print("CSV erfolgreich geladen:")
      print(df.head())

      # Replace UUIDs with numeric IDs
      df["user"] = df["user"].map(self.user_id_map)

      return df

    def get_model(self):
      return self.model

    def get_data(self):
      return self.data_filtered
  
    def get_userid_map(self):
        return self.user_id_map