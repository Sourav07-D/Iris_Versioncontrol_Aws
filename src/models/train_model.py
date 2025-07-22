import pandas as pd
import pickle
import os
import logging
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)

def train_model(input_path, model_path):
    try:
        df = pd.read_csv(input_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        model = RandomForestClassifier()
        model.fit(X, y)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved at {model_path}")
    except Exception as e:
        logging.error("Model training failed", exc_info=True)
        raise e

if __name__ == "__main__":
    train_model("data/processed/iris_features.csv", "models/model.pkl")