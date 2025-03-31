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


class RecipeRecommenderCore:
    def __init__(self, data_path="shuyangli94/food-com-recipes-and-user-interactions"):
        # Hyperparameter mit BPR als Referenz
        self.embed_size = 64     # Für Embedding-basierte Modelle
        self.n_epochs = 8         # Für trainierbare Modelle
        self.lr = 5e-5            # Lernrate
        self.reg = 5e-6           # Regularisierung
        self.batch_size = 1024    # Batch-Größe
        self.num_neg = 10          # Negative Samples
        self.sampler = "random"   # Sampling-Methode
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
        self._load_recipe_names()


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

    def get_model(self):
      return self.model

    def get_data(self):
      return self.data_filtered

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


    def save_recommendations_as_csv(self,items_information,amount_of_recs, path):
      df = self.get_recommendations(items_information,amount_of_recs)
      df.to_csv(path, index=False)
      return df

    def get_recommendations(self, items_information, n_rec):
        """
        Holt Empfehlungen für alle User in user_id_map und speichert die Ergebnisse in einem DataFrame.
        """
        dfs = []
        for user_identifier in self.user_id_map:
            df = self.get_recommendation(user_identifier, n_rec, items_information)
            dfs.append(df)
        # Alle einzelnen DataFrames zusammenfügen
        final_df = pd.concat(dfs, ignore_index=True)
        return final_df

    def get_recommendation(self, user_identifier, n_rec, items_information):
        """Get personalized recommendations with popularity balancing"""
        try:
            # Modell-Check
            if not self.model:
                raise ValueError("Model not trained. Call train() first.")

            # User-ID Mapping
            user_id = user_identifier
            if isinstance(user_identifier, str):
                if user_identifier not in self.user_id_map:
                    raise ValueError(f"User UUID '{user_identifier}' not found")
                user_id = self.user_id_map[user_identifier]

            # Initiale Kandidaten mit 5x Überabtastung
            recommendations = self.model.recommend_user(
                user=user_id, n_rec=n_rec * 5, filter_consumed=True
            )
            # Convert each item to a Python integer to avoid numpy types
            candidate_items = [int(item) for item in recommendations.get(user_id, [])]

            # Check if candidate_items is empty (explicit length check)
            if len(candidate_items) == 0:
                print(f"No candidates for user {user_id}")
                return pd.DataFrame()

            # Berechnung der Long-Tail-Präferenz
            user_interactions = self.data_filtered[self.data_filtered['user'] == user_id]['item']
            user_popularity = [self.item_popularity.get(item, 0) for item in user_interactions]

            if user_popularity:
                median_pop = np.median(user_popularity)
                long_tail_ratio = sum(pop < median_pop for pop in user_popularity) / len(user_popularity)
                max_popularity = max(self.item_popularity.values())
            else:
                # Fallback für neue Nutzer
                long_tail_ratio = 0.5
                max_popularity = max(self.item_popularity.values()) if self.item_popularity else 1

            # Klassifizierung in Short-Head und Long-Tail
            sorted_popularity = sorted(self.item_popularity.values(), reverse=True)
            threshold_idx = int(0.2 * len(sorted_popularity))
            short_head_threshold = sorted_popularity[threshold_idx] if sorted_popularity else 0

            short_head = {item for item in candidate_items
                        if self.item_popularity.get(item, 0) >= short_head_threshold}
            long_tail = set(candidate_items) - short_head

            # Iteratives xQuAD-basiertes Re-Ranking
            lambda_param = 0.6  # Trade-off-Parameter
            selected = []
            remaining = candidate_items.copy()

            while len(selected) < n_rec and remaining:
                best_score = -1
                best_item = None

                for idx, item in enumerate(remaining):
                    # Relevanz basierend auf ursprünglichem Ranking
                    relevance = 1 - (idx / len(remaining))

                    # Diversitätsbonus für Long-Tail-Items
                    diversity_bonus = 1.0
                    if item in long_tail:
                        diversity_bonus += lambda_param * long_tail_ratio

                    # Popularitätsadjustierung
                    popularity = self.item_popularity.get(item, 0)
                    popularity_score = (1 - popularity / max_popularity)

                    total_score = (0.7 * relevance) + (0.3 * diversity_bonus * popularity_score)

                    if total_score > best_score:
                        best_score = total_score
                        best_item = item

                if best_item:
                    selected.append(best_item)
                    remaining.remove(best_item)

            # Metadaten-Anreicherung mit verbesserter Fehlerbehandlung
            records = []
            for item_id in selected[:n_rec]:
                try:
                    title, ingredients = self.__find_item_by_id(item_id, items_information)
                    records.append({
                        "uuid": user_identifier,
                        "item_id": item_id,
                        "item_title": title or "Unknown",
                        "item_ingredients": ingredients or []
                    })
                except (KeyError, IndexError) as e:
                    print(f"Metadata error for item {item_id}: {str(e)}")
                    continue

            # Berechnung der Evaluierungsmetriken
            if records:
                arp = sum(item['popularity'] for item in records) / len(records)
                aplt = sum(1 for item in records if item['popularity'] < short_head_threshold) / len(records)
                print(f"Recommender stats - ARP: {arp:.2f}, APLT: {aplt:.2f}")

            return pd.DataFrame(records).drop_duplicates().head(n_rec)

        except Exception as e:
            print(f"Recommendation error for user {user_id}: {str(e)}")
            return pd.DataFrame()

    def evaluate(self):
        """Evaluate model performance"""
        return evaluate(
            model=self.model,
            data=self.test_data,
            neg_sampling=True,
            metrics=["loss", "roc_auc", "precision", "recall", "ndcg"]
        )

    def info(self, UUID):
      """Gibt einen DataFrame mit allen Interaktionen des angegebenen Benutzers (UUID) zurück."""
      # Überprüfen, ob Daten geladen wurden
      if self.data_filtered is None or not isinstance(self.data_filtered, pd.DataFrame):
          return pd.DataFrame(columns=["user", "item", "label", "name"])

      # Prüfen, ob die UUID vorhanden ist
      if UUID not in self.user_id_map:
          return pd.DataFrame(columns=["user", "item", "label", "name"])

      # Numerische Benutzer-ID abrufen
      user_id = self.user_id_map[UUID]

      # Interaktionen filtern
      user_interactions = self.data_filtered[self.data_filtered['user'] == user_id].copy()

      if user_interactions.empty:
          return pd.DataFrame(columns=["user", "item", "label", "name"])

      # UUID statt numerischer ID setzen
      user_interactions['user'] = UUID

      # Rezeptnamen hinzufügen
      merged = user_interactions.merge(self.name_df, left_on='item', right_on='id', how='left')
      merged['name'] = merged['name'].fillna('Unknown Recipe')

      # Ergebnis formatieren
      result = merged[['user', 'item', 'label', 'name']]

      return result

    def save(self, storagepath):
      """Speichert Modell und Zustand"""
      if not self.model:
          raise ValueError("Modell nicht trainiert")

      os.makedirs(storagepath, exist_ok=True)

      # 1. Modell mit LibreCos eigener Methode speichern
      self.model.save(storagepath, model_name="BPR_model")

      # 2. User-Mapping als JSON
      with open(os.path.join(storagepath, "user_mapping.json"), "w") as f:
          json.dump(self.user_id_map, f)

      # 3. Rezeptnamen-Daten
      self.name_df.to_json(
          os.path.join(storagepath, "recipe_names.json"),
          orient="records"
      )

      # 4. Gefilterte Daten
      if self.data_filtered is not None:
          self.data_filtered.to_parquet(
             os.path.join(storagepath, "filtered_data.parquet")
          )

    @classmethod
    def get(cls, storagepath):
        """Lädt gespeicherte Instanz"""
        instance = cls.__new__(cls)
        instance.data_path = None  # Nicht mehr relevant

        # 1. Modell laden
        instance.model = BPR.load(
            path=storagepath,
            model_name="BPR_model",
            data_info=None  # Wird automatisch geladen
        )

        # 2. DataInfo aus dem Modell holen
        instance.data_info = instance.model.data_info

        # 3. User-Mapping laden
        with open(os.path.join(storagepath, "user_mapping.json"), "r") as f:
            instance.user_id_map = json.load(f)

        # 4. Rezeptnamen
        instance.name_df = pd.read_json(
            os.path.join(storagepath, "recipe_names.json"),
            orient="records"
        )

        # 5. Gefilterte Daten
        instance.data_filtered = pd.read_parquet(
            os.path.join(storagepath, "filtered_data.parquet")
        )

        return instance

#----------------------------------------------------------------------------


    def _load_recipe_names(self):
        """Load recipe ID to name mapping"""
        path = kagglehub.dataset_download(self.data_path)
        raw_recipes_path = os.path.join(path, "RAW_recipes.csv")
        self.name_df = pd.read_csv(raw_recipes_path)[["name", "id"]]

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

    def _get_user_interactions(self, user_id):
      """Get recipes rated by a user"""
      df = self.data_filtered[self.data_filtered['user'] == user_id]
      for _, row in df.iterrows():
         recipe = self._get_recipe_name(row['item'])
         rating = row['label']
         print(f"Recipe: {recipe}, Rating: {rating}")


    def _get_recipe_name(self, recipe_id):
        """Helper to get recipe name from ID"""
        name = self.name_df.loc[self.name_df['id'] == recipe_id, 'name']
        return name.values[0] if not name.empty else "Unknown Recipe"

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


    def __get_score(self,userid,itemid):
        return self.model.predict(userid,itemid)

    def __find_item_by_id(self,recipe_id, items_information):
        df = items_information.loc[items_information["id"] == recipe_id]
        return df['name'].values[0], df['ingredients'].values[0]
