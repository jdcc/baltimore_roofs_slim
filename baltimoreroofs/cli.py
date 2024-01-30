import gc
import logging
import pickle
import random
from collections import Counter
from datetime import date
from pathlib import Path
from pprint import pprint
from typing import Optional

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from sklearn.model_selection import train_test_split

from .data import (
    Creds,
    Database,
    GeodatabaseImporter,
    ImageCropper,
    ImportManager,
    InspectionNotesImporter,
    MatrixCreator,
    count_datasets_in_hdf5,
    fetch_all_blocklots,
    fetch_image_from_hdf5,
)
from .modeling import get_modeling_status, train_image_model
from .modeling.image_model import ImageModel
from .modeling.models import (
    load_model,
)
from .modeling.training import Trainer, flatten_X_y

load_dotenv()

logging.basicConfig(level=logging.INFO)

this_dir = Path(__file__).resolve().parent
SEED = 0


# db group
@click.group()
@click.option("--user", envvar="PGUSER")
@click.option("--password", envvar="PGPASSWORD")
@click.option("--host", envvar="PGHOST")
@click.option("--port", envvar="PGPORT")
@click.option("--database", envvar="PGDATABASE")
@click.pass_context
def roofs(ctx, user, password, host, port, database):
    """Command line interface tools for setting up and detecting roof damage"""
    creds = Creds(user, password, host, port, database)
    ctx.obj = {"db": Database(creds)}


@roofs.group()
def db():
    """Tools for loading and maintaining the database"""
    pass


@roofs.group()
@click.option(
    "--models-path",
    "-m",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    help="path where models are stored",
    default=this_dir.parent / "models",
)
@click.pass_context
def modeling(ctx, models_path):
    """Tools for interacting with roof damage classification models"""
    ctx.obj["models_path"] = models_path


@modeling.command()
def predict():
    """Output roof damage predictions"""
    pass


@modeling.command(name="status")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.pass_obj
def modeling_status(obj, hdf5):
    """Status of the modeling pipeline

    HDF5 is the path to the hdf5 file containing blocklot images.
    """
    db, models_path = obj["db"], obj["models_path"]
    click.echo(get_modeling_status(db, models_path, hdf5))


@modeling.command(name="train-image-model")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.pass_obj
def cli_train_image_model(obj, hdf5):
    """Train an image classification model from aerial photos

    HDF5 is the path to the hdf5 file containing blocklot images.
    """
    train_image_model(obj["db"], obj["models_path"], hdf5)


@modeling.command()
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option(
    "--model-path",
    "-m",
    help="directory to save models into",
    default=Path(__file__).resolve().parent / "models",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--max-date",
    "-d",
    help="Most recent date to consider",
    default="2022-01-01",
    type=click.DateTime(),
)
@click.pass_obj
def train(obj, model_path, hdf5, max_date):
    """Train a new model to classify roof damage severity

    HDF5 is the path to the hdf5 file containing blocklot images.
    """
    blocklots = blocklots_with_image_and_label(obj["db"], hdf5)
    X_creator = MatrixCreator(obj["db"], hdf5)
    X = X_creator.build_features(list(blocklots.keys()), max_date)
    y = blocklots
    X, y = flatten_X_y(X, y)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    trainer = Trainer(model_path)
    scores = trainer.train(train_X, train_y)
    pprint(scores)
    graph_model_scores(scores)
    best_model = pull_out_best(scores)
    print(best_model)
    model, details = load_model(best_model)
    preds = model.predict_proba(test_X)
    print(preds)


# TODO Move this somewhere nice
def pull_out_best(scores):
    return sorted(scores, key=lambda k: np.mean(scores[k]["test_f1"]), reverse=True)[0]


def graph_model_scores(scores):
    # Initialize lists to store summary statistics
    f1_means, f1_stds = [], []
    precision_means, precision_stds = [], []
    recall_means, recall_stds = [], []
    fit_time_means, fit_time_stds = [], []
    score_time_means, score_time_stds = [], []

    # Iterate over models and calculate summary statistics
    for path, metrics in scores.items():
        f1_means.append(np.mean(metrics["test_f1"]))
        f1_stds.append(np.std(metrics["test_f1"]))

        precision_means.append(np.mean(metrics["test_precision"]))
        precision_stds.append(np.std(metrics["test_precision"]))

        recall_means.append(np.mean(metrics["test_recall"]))
        recall_stds.append(np.std(metrics["test_recall"]))

        fit_time_means.append(np.mean(metrics["fit_time"]))
        fit_time_stds.append(np.std(metrics["fit_time"]))

        score_time_means.append(np.mean(metrics["score_time"]))
        score_time_stds.append(np.std(metrics["score_time"]))

    # Plotting
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))  # 5 rows for 5 metrics

    # F1 Score Plot
    axs[0].errorbar(range(len(scores)), f1_means, yerr=f1_stds, fmt="o")
    axs[0].set_title("F1 Score of Models")
    axs[0].set_ylabel("F1 Score")

    # Precision Plot
    axs[1].errorbar(range(len(scores)), precision_means, yerr=precision_stds, fmt="o")
    axs[1].set_title("Precision of Models")
    axs[1].set_ylabel("Precision")

    # Recall Plot
    axs[2].errorbar(range(len(scores)), recall_means, yerr=recall_stds, fmt="o")
    axs[2].set_title("Recall of Models")
    axs[2].set_ylabel("Recall")

    # Fit Time Plot
    axs[3].errorbar(range(len(scores)), fit_time_means, yerr=fit_time_stds, fmt="o")
    axs[3].set_title("Fit Time of Models")
    axs[3].set_ylabel("Fit Time (s)")

    # Score Time Plot
    axs[4].errorbar(range(len(scores)), score_time_means, yerr=score_time_stds, fmt="o")
    axs[4].set_title("Score Time of Models")
    axs[4].set_ylabel("Score Time (s)")
    axs[4].set_xlabel("Model Index")

    plt.tight_layout()
    fig.savefig("model_performance_plot_4.png", dpi=300)


@roofs.group()
@click.pass_context
def images(ctx):
    """Tools for interacting with aerial images"""
    pass


@images.command()
@click.argument(
    "image-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
)
@click.option("--blocklots", "-b", type=str, multiple=True, default=[])
@click.option("--overwrite/--no-overwrite", default=False)
@click.pass_obj
def crop(obj, image_root, output, blocklots, overwrite):
    """Crop aerial image tiles into individual blocklot images

    IMAGE_ROOT is the path of the directory containing all the .tif images
    and their accompanying .tif.aux.xml files.

    OUTPUT is the path of the hdf5 file that will be created containing
    the blocklot images.
    """
    db = obj["db"]
    cropper = ImageCropper(db, image_root)
    if len(blocklots) == 0:
        blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
    cropper.write_h5py(output, blocklots, overwrite)


@images.command(name="status")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.pass_obj
def images_status(obj, hdf5):
    """Show the status of the blocklot image setup process

    hdf5 is the path to the hdf5 file containing all the blocklot image data.
    """
    db = obj["db"]
    blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
    click.echo("Images Status\n" + "=" * 50)
    click.echo(f"There are {len(blocklots):,} blocklots in the database.")
    click.echo(f"    Here are a few: {random.sample(blocklots, k=3)}")
    with h5py.File(hdf5) as f:
        n_datasets = count_datasets_in_hdf5(f)
        click.echo(f"\nThere are {n_datasets:,} blocklot images in the image database.")
        blocklots = random.sample(fetch_blocklots_in_hdf5(f), k=3)
        click.echo(f"    Here are a few: {blocklots}")


@images.command
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
)
@click.option("--blocklots", "-b", type=str, multiple=True, required=True)
def dump(hdf5, output, blocklots):
    """Dump JPEG images of blocklots to disk for further inspection

    HDF5 is the path to the hdf5 file containing blocklot images.
    OUTPUT is the directory into which images will be written.
    """

    with h5py.File(hdf5) as f:
        for blocklot in blocklots:
            array = fetch_image_from_hdf5(blocklot, f)
            pixels = np.nan_to_num(array[:], nan=255).astype("uint8")
            im = Image.fromarray(pixels)
            im.save(output / f"{blocklot}.jpeg")


@db.command()
@click.option(
    "-i",
    "--inspection-notes",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.pass_obj
def import_sheet(
    obj,
    inspection_notes: Optional[Path] = None,
):
    """Import spreadsheets of data

    Only handles inspection notes currently."""
    db = obj["db"]
    importers = []
    if inspection_notes:
        importers.append(InspectionNotesImporter.from_file(db, inspection_notes))
    manager = ImportManager(db, importers)
    manager.setup()
    click.echo(manager.status())


@db.command()
@click.argument(
    "geodatabase",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--building-outlines", help="layer name that shows building outlines")
@click.option("--building-permits", help="layer name that has construction permits")
@click.option("--code-violations", help="layer name that has building code violations")
@click.option("--data-311", help="layer name that contains 311 data")
@click.option("--demolitions", help="layer name that contains demolitions data")
@click.option("--ground-truth", help="layer name that includes roof quality labels")
@click.option("--real-estate", help="layer name that includes real estate transactions")
@click.option("--redlining", help="layer name that shows historical redlines")
@click.option(
    "--tax-parcel-address", help="layer name that contains tax parcel addresses"
)
@click.option(
    "--vacant-building-notices",
    help="layer name that contains the vacant building notices",
)
@click.pass_obj
def import_gdb(obj, geodatabase: Path, **layer_map):
    """Import a Geodatabase file"""
    db = obj["db"]
    if all(layer_map[layer] is None for layer in GeodatabaseImporter.REQUIRED_TABLES):
        raise click.UsageError("At least one layer name option must be specified.")
    layer_map = {
        table: layer for table, layer in layer_map.items() if layer is not None
    }
    importer = GeodatabaseImporter.from_file(db, geodatabase, layer_map)
    manager = ImportManager(db, [importer])
    manager.setup()
    click.echo(manager.status())


@db.command(name="status")
@click.pass_obj
def db_status(obj):
    """Show the status of the database"""
    db = obj["db"]
    click.echo(ImportManager(db).status())
    try:
        blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
        click.echo(f"There are {len(blocklots):,} row home blocklots in the database.")
    except Exception:
        pass


@db.command()
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
@click.pass_obj
def reset(obj):
    """Remove all data from the database"""
    db = obj["db"]
    ImportManager(db).reset()
    click.echo("The database has been reset.")


if __name__ == "__main__":
    roofs()
