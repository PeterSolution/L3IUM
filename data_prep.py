import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# zad4
# parametry czytane z pliku .env, nic nie jest wpisane na sztywno w logice
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data/wine.csv")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "target")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

#zad1
def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_data(df, test_size=TEST_SIZE, seed=RANDOM_SEED):
    if df.empty:
        raise ValueError("DataFrame jest pusty")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Brak kolumny docelowej: {TARGET_COLUMN}")
    if not 0 < test_size < 1:
        raise ValueError("test_size musi być w przedziale (0, 1)")

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(x, y, test_size=test_size, random_state=seed)