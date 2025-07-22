import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

def ingest_modified_data(output_path):
    try:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame

        # Simulate change: remove class 2, add Gaussian noise to features
        df = df[df['target'] != 2]
        noise = np.random.normal(0, 0.2, df.iloc[:, :-1].shape)
        df.iloc[:, :-1] += noise

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info("Saved modified dataset.")
    except Exception as e:
        logging.error("Failed to modify and ingest data", exc_info=True)
        raise e

if __name__ == "__main__":
    ingest_modified_data("data/raw/iris_v2.csv")