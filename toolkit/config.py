# === Paths ===
RAW_DATA_DIR = "data/raw/"
PREDICTIONS_DIR = "outputs/predictions/"
MODELS_DIR = "outputs/models/"
SHAP_DIR = "outputs/shap_values/"
PERFORMANCE_DIR = "outputs/performance/"
PLOTS_DIR = "outputs/plots/"
COMPOUNDS_SPLIT_PATH = "data/processed/compounds.pkl"

# === Data scope ===
PDB_IDS = ["1B41", "4M0E", "6NTO", "6WUY", "8AEN"]
SCORE_TYPES = [
	"non_covalent_score",
	"pre_reaction_score", "post_reaction_score",
	"covdock_score"
]

# === Modelling ===
RANDOM_SEED = 1
TEST_PROP = 0.45
NUM_FOLDS = 5
N_TRIALS = 100

