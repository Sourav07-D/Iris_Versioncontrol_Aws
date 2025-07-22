import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)

def build_features(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        features = df.drop("target", axis=1)
        target = df["target"]
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
        df_scaled["target"] = target
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_scaled.to_csv(output_path, index=False)
        logging.info("Feature engineering complete")
    except Exception as e:
        logging.error("Feature engineering failed", exc_info=True)
        raise e

if __name__ == "__main__":
    build_features("data/processed/iris_clean.csv", "data/processed/iris_features.csv")