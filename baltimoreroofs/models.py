import h5py
import numpy as np
import torch
from psycopg2 import sql, errors
from torchvision.transforms.functional import rgb_to_grayscale

from .images import blocklots_in_hdf5

# Reorder given matplotlib and pytorch have different order of channel, height, width.
# pytorch:    [C, H, W]
# matplotlib: [H, W, C]
TENSOR_TO_NUMPY = [1, 2, 0]
NUMPY_TO_TENSOR = [2, 0, 1]


def tensor_to_numpy(t):
    return t.numpy().transpose(TENSOR_TO_NUMPY)


def numpy_to_tensor(n):
    return torch.from_numpy(n.transpose(NUMPY_TO_TENSOR))


def is_gpu_available():
    return torch.cuda.is_available()


def fetch_labels(db):
    query = sql.SQL("SELECT blocklot, label FROM {label_table}").format(
        label_table=sql.Identifier(db.CLEAN_SCHEMA, "ground_truth")
    )
    results = db.run_query(query)
    return {row[0]: row[1] for row in results}


def blocklots_with_image_and_label(db, hdf5_path) -> dict[str, int]:
    labels = fetch_labels(db)
    with h5py.File(hdf5_path) as f:
        bl_with_images = blocklots_in_hdf5(f)
        bl_with_image_and_label = set(
            bl for bl, label in labels.items() if label is not None
        ) & set(bl_with_images)
    return {bl: labels[bl] for bl in bl_with_image_and_label}


class DarkImageBaseline:
    def __init__(self, dark_threshold):
        self.threshold = dark_threshold

    def fit(self, X, y):
        return self

    def preprocess(self, X):
        numpy_to_tensor(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # How much of each image is darker than the darkness threshold?
        if X.ndim == 0:
            return None
        X = rgb_to_grayscale(numpy_to_tensor(X).unsqueeze(0))
        is_darker = torch.where(X.isnan(), X, X < self.threshold).to(torch.float32)
        return is_darker.nanmean(axis=(2, 3), dtype=torch.float32).squeeze().item()


def write_completed_preds_to_db(db, model_name, preds: dict[str, float]):
    create_predictions_table(db, model_name)
    rows = []
    for blocklot, pred in preds.items():
        if pred is not None:
            pred = round(pred, 6)
        rows.append([blocklot, pred])
    db.batch_insert(
        sql.SQL(
            """
            INSERT INTO {table} (blocklot, score) VALUES %s
            ON CONFLICT (blocklot) DO UPDATE SET score = EXCLUDED.score
        """
        ).format(table=sql.Identifier(db.OUTPUTS_SCHEMA, f"{model_name}_predictions")),
        rows,
    )


def create_predictions_table(db, model_name):
    db.run(
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {pred_schema}").format(
            pred_schema=sql.Identifier(db.OUTPUTS_SCHEMA)
        )
    )
    db.run(
        sql.SQL(
            """
        CREATE TABLE IF NOT EXISTS {preds_table} (
            blocklot varchar(10),
            score real
        )
    """
        ).format(
            preds_table=sql.Identifier(db.OUTPUTS_SCHEMA, f"{model_name}_predictions"),
        )
    )
    try:
        db.run(
            sql.SQL("CREATE UNIQUE INDEX {index} ON {dest} (blocklot)").format(
                index=sql.Identifier(f"{model_name}_predictions_blocklot_idx"),
                dest=sql.Identifier(db.OUTPUTS_SCHEMA, f"{model_name}_predictions"),
            )
        )
    except errors.DuplicateTable:
        pass
