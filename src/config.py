import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "march-machine-learning-mania-2026")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 33

ELO_INITIAL = 1500
ELO_K = 20
ELO_HOME_ADVANTAGE = 100

ROLLING_WINDOW = 14

TRAIN_SEASONS_END = 2023
VAL_SEASON = 2024
TEST_SEASON = 2025

#Gender prefix (M = Men, W = Women)
GENDER = "M"