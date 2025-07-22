import pandas as pd
import pickle
import logging
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)

def evaluate_model(input_path, model_path):
    try:
        df = pd.read_csv(input_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        preds = model.predict(X)
        report = classification_report(y, preds)
        logging.info("\n" + report)

        with open("reports/metrics.txt", "w") as f:
            f.write(report)
    except Exception as e:
        logging.error("Evaluation failed", exc_info=True)
        raise e

if __name__ == "__main__":
    evaluate_model("data/processed/iris_features.csv", "models/model.pkl")