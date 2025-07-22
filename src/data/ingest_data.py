import pandas as pd
import logging
import os

import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)
output_path = params["ingest_data"]["output_path"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ingest_data(output_path):
    try:
        logging.info("Reading Iris dataset from sklearn")
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved raw data to {output_path}")
    except Exception as e:
        logging.error("Data ingestion failed", exc_info=True)
        raise e

if __name__ == "__main__":
    ingest_data(output_path)