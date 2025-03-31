import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import BPR, UserCF, ItemCF, SVD, SVDpp, ALS
import kagglehub


class RecipeRecommenderCore:
    def __init__(self, data_path="shuyangli94/food-com-recipes-and-user-interactions"):
        # Hyperparameter mit BPR als Referenz
        self.embed_size = 64     # Für Embedding-basierte Modelle
        self.n_epochs = 8         # Für trainierbare Modelle
        self.lr = 5e-5            # Lernrate
        self.reg = 1e-5           # Regularisierung
        self.batch_size = 1024    # Batch-Größe
        self.num_neg = 20          # Negative Samples
        self.sampler = "unconsumed"   # Sampling-Methode
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


    def set_model(self, model_type):
        """Zentrale Methode zur Modellauswahl mit einheitlichen Parametern"""
        tf.compat.v1.reset_default_graph()
        self.__prepare_data(model_type)  # Daten für alle Modelle vorbereiten

        common_params = {
            "task": "ranking",
            "data_info": self.data_info
        }

        model_config = {
           "BPR": {
            "class": BPR,
            "params": {
                "loss_type": "bpr",
                "embed_size": self.embed_size,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "num_neg": self.num_neg,
                "reg": self.reg,
                "sampler": self.sampler
            }
        },
        "UserCF": {
            "class": UserCF,
            "params": {
                "k_sim": self.k_sim,
                "sim_type": self.sim_type
            }
        },
        "ItemCF": {
            "class": ItemCF,
            "params": {
                "k_sim": self.k_sim,
                "sim_type": self.sim_type
            }
        },
        "SVD": {
            "class": SVD,
            "params": {
                "embed_size": self.embed_size,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "reg": self.reg
            }
        },
        "SVDpp": {
            "class": SVDpp,
            "params": {
                "embed_size": self.embed_size,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "reg": self.reg,
            }
        },
        "ALS": {
            "class": ALS,
            "params": {
                "embed_size": self.embed_size,
                "n_epochs": self.n_epochs,
                "reg": self.reg,
                "alpha": 10,
                "use_cg": True,
                "n_threads": 1
            }
        }
    }
        config = model_config.get(model_type)
        if not config:
            raise ValueError(f"Unbekanntes Modell: {model_type}")

        self.model = config["class"](**common_params, **config["params"])


    def train(self):
      if not self.model:
          raise ValueError("Model not trained. Call set_model() first.")

     # Gemeinsame Parameter
      common_params = {
          "verbose": 2,
          "eval_data": self.eval_data,
          "metrics": ["loss", "roc_auc", "precision", "recall", "ndcg"]
     }

      # Modellspezifische Parameter
      if isinstance(self.model, (UserCF, ItemCF)):
          # Für Collaborative Filtering
          fit_params = {
              "neg_sampling": True,
              "verbose": 1
         }
      else:
          # Für Embedding-basierte Modelle: batch_size entfernen
          fit_params = {
              "neg_sampling": True,
              "shuffle": True,
             **common_params
         }

     # Training durchführen
      self.model.fit(
          self.train_data,
          **fit_params
      )
      self.item_popularity = self.data_filtered["item"].value_counts().to_dict()

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

        # Filter items
        item_counts = combined["item"].value_counts()
        items_to_keep = item_counts[item_counts >= min_interactions].index
        filtered = combined[combined["item"].isin(items_to_keep)]

        # Filter users
        user_counts = filtered["user"].value_counts()
        users_to_keep = user_counts[user_counts >= min_interactions].index
        self.data_filtered = filtered[filtered["user"].isin(users_to_keep)]

        print("Wie viele interactions gibt es?" + str(len(self.data_filtered)))

    def __prepare_data(self,model_type):
      # Convert ratings to 0/1 for UserCF and ItemCF
        if model_type in ["UserCF", "ItemCF"]:
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

      # Spalten filtern
      keep_cols = ["user", "item", "label"]
      df = df[keep_cols]

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
          processed_df = self.__process_ratings(df)

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

    def __process_ratings(self, df):
      """Map UUIDs to numeric IDs"""
      # Rename columns
      df = df.rename(columns={
          "uuid": "user",
          "item_id": "item",
          "rating": "label"
      })

      # Convert from range [-2,2] to [1,5]
      df["label"] = df["label"] + 3

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