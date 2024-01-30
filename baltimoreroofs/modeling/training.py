import gc
import pickle
import uuid
from collections import namedtuple
from itertools import product
from pathlib import Path

import h5py
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from tqdm.auto import tqdm

from ..data import Database, fetch_blocklots_imaged
from .image_model import ImageModel
from .models import fetch_blocklots_imaged_and_labeled, write_completed_preds_to_db

SEED = 2


class Trainer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.param_values = {
            "n_estimators": [10, 100, 500],  # 1000
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 10, 20, 30, 50],
            "class_weight": [None, "balanced"],
            "max_features": [None],  # sqrt
        }
        self.param_space = calc_param_space(self.param_values)

    # TODO Don't train lots of models by default
    def train(self, X, y):
        """This actually trains a bunch of models"""
        model_scores = {}
        cv_splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for params in tqdm(self.param_space, desc="Training models", smoothing=0):
            model = RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)
            model.fit(X, y)
            model_path = self.save_model(model, params)
            # Uses stratified k-fold by default
            model_scores[model_path] = cross_validate(
                model,
                X,
                y,
                cv=cv_splitter,
                n_jobs=-1,
                scoring=("precision", "recall", "f1"),
            )
        return model_scores

    def save_model(self, model, params):
        model_id = str(uuid.uuid4())
        model_path = Path(self.model_dir) / f"{model_id}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            dump(
                {
                    "class": model.__class__,
                    "model": model.to_save() if hasattr(model, "to_save") else model,
                    "params": params,
                },
                f,
            )
        return model_path


# TODO organize

EvalCounts = namedtuple(
    "EvalCounts",
    ("n", "n_labeled", "n_labeled_pos", "n_labeled_neg", "precision", "recall"),
)


def pull_eval_counts(df, n_total_positive):
    label_counts = df.label.value_counts()
    precision = None
    recall = None
    if label_counts.sum() > 0:
        precision = label_counts.get(1.0, default=0) / label_counts.sum()
    if n_total_positive > 0:
        recall = label_counts.get(1.0, default=0) / n_total_positive
    return EvalCounts(
        n=int(df.shape[0]),
        n_labeled=int(label_counts.sum()),
        n_labeled_pos=int(label_counts.get(1.0, default=0)),
        n_labeled_neg=int(label_counts.get(0.0, default=0)),
        precision=precision,
        recall=recall,
    )


def build_evaluation(df, thresholds=None, ks=None):
    df = df[~df.score.isna()].copy()
    n_total_positive = df.label.sum()
    df["threshold"] = df.score.rank(ascending=False, pct=True)
    threshold_counts = {}
    for threshold in thresholds:
        to_eval = df[df.threshold < threshold]
        threshold_counts[threshold] = pull_eval_counts(to_eval, n_total_positive)

    top_k_counts = {}
    for k in ks:
        top_k = df.sort_values("score", ascending=False).head(k)
        top_k_counts[k] = pull_eval_counts(top_k, n_total_positive)

    return threshold_counts, top_k_counts


def build_score_df(preds):
    blocklots = list(preds.keys())
    labels = Labeler.load_labels(blocklots)
    return pd.DataFrame({"label": {b: l for b, l in labels.items()}, "score": preds})


def train_image_model(db: Database, models_path: Path, hdf5: Path):
    model = ImageModel(
        hdf5,
        batch_size=128,
        learning_rate=1e-4,
        angle_variations=[0, 20, 45, 60, 90],
        num_epochs=5,
        unfreeze=1,
        dropout=0.9,
    )
    blocklots = fetch_blocklots_imaged_and_labeled(db, hdf5)
    X, y = list(blocklots.keys()), list(blocklots.values())
    model.fit(X, y)
    with open(models_path / "image_model.pkl", "wb") as f:
        pickle.dump(model.to_save(), f, protocol=5)
    with h5py.File(hdf5) as f:
        blocklots = fetch_blocklots_imaged(f)
    gc.collect()
    preds = model.forward(blocklots)
    write_completed_preds_to_db(db, "image_model", preds)


def calc_param_space(param_values):
    param_names = list(param_values.keys())
    return [
        dict(zip(param_names, [None if v == "None" else v for v in values]))
        for values in product(*param_values.values())
    ]


def flatten_X_y(X: dict[str, dict[str, float]], y: dict[str, int]) -> tuple[list, list]:
    flat_X, flat_y = [], []
    feature_order = list(X.keys())
    blocklot_order = list(next(iter(X.values())).keys())
    for blocklot in blocklot_order:
        row = [X[feature][blocklot] for feature in feature_order]
        flat_X.append(row)
        flat_y.append(y.get(blocklot, None))
    return flat_X, flat_y
