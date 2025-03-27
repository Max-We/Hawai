CONCURRENT_WORKERS = 15

# Step 1: Personas
MODEL_PERSONAS = "gpt-4o-mini"
TEMPERATURE_PERSONAS = 0.7
TOKENS_PERSONAS=777
NUM_PERSONAS = 10

# Step 2: Questionnaires (used in questionnaire generation AND reconstruction)
MODEL_QUESTIONNAIRES = "gpt-4o-mini"
TEMPERATURE_QUESTIONNAIRE = 0.2

# Step 3: Entropy sampling & rating
MODEL_RATING = "gpt-4o-mini"
TEMPERATURE_RATING = 0.2

# Data files
PERSONAS_FILE = "data/personas.csv"
QUESTIONNAIRES_FILE = "data/questionnaires.csv"
QUESTIONNAIRES_RECONSTRUCTED_FILE = "data/questionnaires_reconstructed_"
RATINGS_FILE = "data/ratings.csv"
RECOMMENDATIONS_FILE = "data/recommendations_"
EVAL_PLOTS = "evaluation/"

MIN_INTERACTIONS = 20
