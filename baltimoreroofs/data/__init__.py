from .blocklots import fetch_all_blocklots, split_blocklot
from .data_set_importer import GeodatabaseImporter, InspectionNotesImporter
from .database import Creds, Database
from .images import (
    ImageCropper,
    count_datasets_in_hdf5,
    fetch_blocklots_imaged,
    fetch_image_predictions,
    fetch_image_from_hdf5,
    numpy_to_tensor,
    tensor_to_numpy,
)
from .import_manager import ImportManager
from .matrix_creator import MatrixCreator


def build_dataset(
    db, hdf5, blocklots, labels: dict[str, int], max_date
) -> tuple[list[list[float]], list[int | None], list[str]]:
    """Build a nice modelable dataset"""
    X_creator = MatrixCreator(db, hdf5)
    features = X_creator.build_features(blocklots, max_date)
    return flatten_X_y(features, {b: labels.get(b, None) for b in blocklots})


def flatten_X_y(
    X: dict[str, dict[str, float]], y: dict[str, int | None]
) -> tuple[list[list[float]], list[int | None], list[str]]:
    """Take features and labels keyed by blocklot and flatten into modelable datasets"""
    flat_X, flat_y = [], []
    feature_order = list(X.keys())
    blocklot_order = list(next(iter(X.values())).keys())
    for blocklot in blocklot_order:
        row = [X[feature][blocklot] for feature in feature_order]
        flat_X.append(row)
        flat_y.append(y.get(blocklot, None))
    return flat_X, flat_y, blocklot_order
