import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from gaia_variable_stars.config import (
    CLASSES,
    CV_FILE,
    FEATURE_FILE,
    FEATURES,
    FOLDS,
    MODEL_FILE,
    MODELS_ROOT,
    RANK_FILE,
    SEED,
    TEST_FILE,
    TEST_SIZE,
)


def load():
    feat = pd.read_csv(FEATURE_FILE)
    missing = set(CLASSES) - set(feat["class_name"])

    if missing:
        raise ValueError(f"No stars for {sorted(missing)}. Re-run notebooks 01 to 03.")

    return feat


def models():
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ),
        "svm": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(class_weight="balanced", random_state=SEED), ensemble=False
            ),
        ),
        "forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "boost": make_pipeline(
            HistGradientBoostingClassifier(random_state=SEED),
        ),
    }


def fit(feat, cols=FEATURES):

    train, test = train_test_split(
        feat,
        test_size=TEST_SIZE,
        stratify=feat["class_name"],
        random_state=SEED,
    )

    X, y = train[cols], train["class_name"].to_numpy(dtype=object)
    Xt, yt = test[cols], test["class_name"].to_numpy(dtype=object)

    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    pipes = models()

    rows = []

    for name, pipe in pipes.items():
        score = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
        rows.append({"model": name, "f1": score.mean(), "std": score.std()})

    cvs = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best = cvs.loc[0, "model"]
    model = pipes[best].fit(X, y)

    pred = model.predict(Xt)

    imp = permutation_importance(
        model, Xt, yt, scoring="f1_macro", n_repeats=20, random_state=SEED
    )
    rank = (
        pd.DataFrame(
            {
                "feature": cols,
                "drop": imp.importances_mean,
                "std": imp.importances_std,
            }
        )
        .sort_values("drop", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "cv": cvs,
        "best": best,
        "model": model,
        "acc": accuracy_score(yt, pred),
        "f1": f1_score(yt, pred, average="macro"),
        "matrix": pd.DataFrame(
            confusion_matrix(yt, pred, labels=CLASSES), index=CLASSES, columns=CLASSES
        ),
        "report": pd.DataFrame(
            classification_report(yt, pred, labels=CLASSES, output_dict=True)
        ).T,
        "test": pd.DataFrame(
            {"source_id": test["source_id"].to_numpy(), "true": yt, "pred": pred}
        ),
        "rank": rank,
    }


def save(out):
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    joblib.dump(out["model"], MODEL_FILE)
    out["cv"].to_csv(CV_FILE, index=False)
    out["rank"].to_csv(RANK_FILE, index=False)
    out["test"].to_csv(TEST_FILE, index=False)


def main():
    out = fit(load())
    save(out)

    print(out["cv"].to_string(index=False))
    print(f"\nbest: {out['best']}  acc={out['acc']:.3f}  f1={out['f1']:.3f}\n")
    print(out["rank"].to_string(index=False))


if __name__ == "__main__":
    main()