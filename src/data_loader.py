import pandas as pd
import os

def load_data():
    # Get current file directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct full path to data file
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'imdb_sample.csv')

    # Load CSV
    return pd.read_csv(DATA_PATH)
