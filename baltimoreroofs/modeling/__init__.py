import random
from collections import Counter
from pathlib import Path
from typing import Optional

from ..data import Database, fetch_all_blocklots
from .models import (
    fetch_blocklots_imaged_and_labeled,
    fetch_labels,
    fetch_blocklot_label,
    is_gpu_available,
)
from .training import train_image_model, Trainer, train_many_models


def build_label_str(labels: dict[str, Optional[int]]):
    """Format a nice string of labels and their counts"""
    return "\n  ".join(
        f"{label if label is not None else 'None':5}: {n:,}"
        for label, n in Counter(labels.values()).most_common()
    )


def get_modeling_status(db: Database, models_path: Path, hdf5: Path):
    """Report the status of the modeling process"""
    blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
    labels = fetch_labels(db)
    bl_with_image_and_label = fetch_blocklots_imaged_and_labeled(db, hdf5)
    blocklots_sample = random.sample(sorted(bl_with_image_and_label), k=3)
    image_model_exists = Path(models_path / "image_model.pkl").is_file()
    model_exists = Path(models_path / "model.pkl").is_file()

    gpu_status = " NOT" if not is_gpu_available() else ""
    image_model_status = " NOT" if not image_model_exists else ""
    overall_model_status = " NOT" if not model_exists else ""

    output = f"""
Modeling Status
{'=' * 50}
GPU acceleration is{gpu_status} available.
The database contains {len(blocklots):,} relevant blocklots.
The database contains the following ground-truth labels:
  {build_label_str(labels)}

There are {len(bl_with_image_and_label):,} blocklots with both images and labels.
They have these ground-truth labels:
  {build_label_str(bl_with_image_and_label)}
Here are a few: {blocklots_sample}

The image model does{image_model_status} exist in {models_path}.
The overall model does{overall_model_status} exist in {models_path}.
"""

    return output.strip()
