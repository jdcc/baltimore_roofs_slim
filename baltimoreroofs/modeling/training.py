import gc
import pickle
import uuid
from collections import namedtuple
from itertools import product
from pathlib import Path

import h5py
from joblib import dump
import pandas as pd
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
            "n_estimators": [10, 100, 500, 1000],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [10, 20, 30, 50, None],
            "class_weight": ["balanced", None],
            "max_features": ["sqrt", None],
        }
        self.param_space = calc_param_space(self.param_values)

    # TODO Don't train lots of models by default
    def train(self, X, y):
        """This actually trains a bunch of models"""
        model_scores = {}
        model_params = {}
        cv_splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for params in tqdm(self.param_space, desc="Training models", smoothing=0):
            model = RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)
            model.fit(X, y)
            model_path = self.save_model(model, params)
            model_scores[model_path] = cross_validate(
                model,
                X,
                y,
                cv=cv_splitter,
                n_jobs=-1,
                scoring=("precision", "recall", "f1"),
            )
            model_params[model_path] = params
        return model_scores, model_params

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


def build_score_df(labels: dict[str, int], preds: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {b: {"label": labels.get(b, None), "score": preds[b]} for b in preds},
        orient="index",
    )


def train_image_model(db: Database, models_path: Path, hdf5: Path, seed: int):
    model = ImageModel(
        hdf5,
        batch_size=512,
        learning_rate=1e-5,
        angle_variations=[0, 20, 45, 80],
        num_epochs=20,
        unfreeze=1,
        dropout=0.9,
        seed=seed,
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


def flatten_X_y(
    X: dict[str, dict[str, float]], y: dict[str, int | None]
) -> tuple[list[list[float]], list[int | None], list[str]]:
    flat_X, flat_y = [], []
    feature_order = list(X.keys())
    blocklot_order = list(next(iter(X.values())).keys())
    for blocklot in blocklot_order:
        row = [X[feature][blocklot] for feature in feature_order]
        flat_X.append(row)
        flat_y.append(y.get(blocklot, None))
    return flat_X, flat_y, blocklot_order
