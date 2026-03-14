#configurações globais do projeto
import os
#paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "march-machine-learning-mania-2026")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#reprodutibilidade
SEED = 33

#elo
ELO_INITIAL = 1500
ELO_K = 20
ELO_HOME_ADVANTAGE = 100

#features
ROLLING_WINDOW = 14

#validação
TRAIN_SEASONS_END = 2023 #treino até esta temporada (inclusive)
VAL_SEASON = 2024 #validação
TEST_SEASON = 2025 #(stage1) ou 2026 (stage2)

#Gender prefix (M = Men, W = Women)
GENDER = "M"