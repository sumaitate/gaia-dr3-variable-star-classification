import argparse

import joblib
import pandas as pd

from gaia_variable_stars.config import FEATURE_FILE, FEATURES, MODEL_FILE, PRED_FILE


def predict(feat):
    """Predicted class and class probabilities for a feature table."""
    model = joblib.load(MODEL_FILE)
    X = feat[FEATURES]

    out = pd.DataFrame(
        {"source_id": feat["source_id"].to_numpy(), "pred": model.predict(X)}
    )
    prob = model.predict_proba(X)

    for i, name in enumerate(model.classes_):
        out[name] = prob[:, i]

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(FEATURE_FILE))
    parser.add_argument("--output", default=str(PRED_FILE))
    args = parser.parse_args()

    out = predict(pd.read_csv(args.input))
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()