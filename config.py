# Step 1: Personas
MODEL_PERSONAS = "gpt-4o-mini"
TEMPERATURE_PERSONAS = 0.7
TOKENS_PERSONAS=500
NUM_PERSONAS = 5

# Step 2: Questionnaires (used in questionnaire generation AND reconstruction)
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

MIN_INTERACTIONS = 20