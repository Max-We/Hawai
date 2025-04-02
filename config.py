CONCURRENT_WORKERS = 15
REQUEST_TIMEOUT = 3

# Step 1: Personas
MODEL_PERSONAS = "gpt-3.5-turbo-instruct"
TEMPERATURE_PERSONAS = 0.7
TOKENS_PERSONAS=250
NUM_PERSONAS = 1

# Step 2: Questionnaires
MODEL_QUESTIONNAIRES = "gpt-4o-mini"
TEMPERATURE_QUESTIONNAIRE = 0.2

# Step 3: Entropy sampling & rating
MODEL_RATING = "gpt-4o-mini"
TEMPERATURE_RATING = 0.2

# Data files
PERSONAS_FILE = "data/personas.csv"
QUESTIONNAIRES_FILE = "data/questionnaires.csv"
QUESTIONNAIRES_RECONSTRUCTED_FILE = "data/questionnaires_reconstructed.csv"
RATINGS_FILE = "data/ratings.csv"
EVAL_PLOTS = "evaluation/"

MIN_INTERACTIONS = 20