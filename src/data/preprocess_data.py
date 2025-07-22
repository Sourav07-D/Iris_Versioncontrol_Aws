import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        df = df.dropna()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info("Preprocessing complete")
    except Exception as e:
        logging.error("Preprocessing failed", exc_info=True)
        raise e

if __name__ == "__main__":
    preprocess_data("data/raw/iris.csv", "data/processed/iris_clean.csv")