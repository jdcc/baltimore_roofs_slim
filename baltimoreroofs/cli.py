import gc
import logging
import pickle
import random
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

import click
import h5py
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from .data_set_importer import GeodatabaseImporter, InspectionNodesImporter
from .database import Creds, Database
from .image_model import ImageModel
from .images import (
    ImageCropper,
    blocklots_in_hdf5,
    count_datasets_in_hdf5,
    fetch_image_from_hdf5,
)
from .import_manager import ImportManager

from .matrix_creator import MatrixCreator
from .models import (
    blocklots_with_image_and_label,
    fetch_labels,
    is_gpu_available,
    write_completed_preds_to_db,
)
from .utils import fetch_all_blocklots

load_dotenv()

logging.basicConfig(level=logging.INFO)

this_dir = Path(__file__).resolve().parent


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


def build_label_str(labels: dict[str, Optional[int]]):
    return "\n  ".join(
        f"{label if label is not None else 'None':5}: {n:,}"
        for label, n in Counter(labels.values()).most_common()
    )


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
    blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
    labels = fetch_labels(db)
    click.echo("Modeling Status\n" + "=" * 50 + "\n")
    click.echo(
        f"GPU acceleration is{ 'NOT' if not is_gpu_available() else ''} available.\n"
    )
    click.echo(f"The database contains {len(blocklots):,} relevant blocklots.\n")
    label_str = build_label_str(labels)
    click.echo(
        f"The database contains the following ground-truth labels:\n  {label_str}\n"
    )
    bl_with_image_and_label = blocklots_with_image_and_label(db, hdf5)
    click.echo(
        f"There are {len(bl_with_image_and_label):,} blocklots with both "
        "images and labels."
    )
    label_str = build_label_str(bl_with_image_and_label)
    click.echo(f"They have these ground-truth labels:\n  {label_str}\n")
    blocklots = random.sample(sorted(bl_with_image_and_label), k=3)
    click.echo(f"    Here are a few: {blocklots}")
    image_model_exists = Path(models_path / "image_model.pkl").is_file()
    click.echo(
        f"The image model does{' NOT' if not image_model_exists else ''} exist "
        f"in {models_path}."
    )
    model_exists = Path(models_path / "model.pkl").is_file()
    click.echo(
        f"The overall model does{' NOT' if not model_exists else ''} exist "
        f"in {models_path}."
    )


@modeling.command()
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.pass_obj
def train_image_model(obj, hdf5):
    """Train an image classification model from aerial photos

    HDF5 is the path to the hdf5 file containing blocklot images.
    """
    db, models_path = obj["db"], obj["models_path"]
    model = ImageModel(
        hdf5,
        batch_size=128,
        learning_rate=1e-4,
        angle_variations=[0, 20, 45, 60, 90],
        num_epochs=5,
        unfreeze=1,
        dropout=0.9,
    )
    blocklots = blocklots_with_image_and_label(db, hdf5)
    X, y = list(blocklots.keys()), list(blocklots.values())
    model.fit(X, y)
    with open(models_path / "image_model.pkl", "wb") as f:
        pickle.dump(model.to_save(), f, protocol=5)
    with h5py.File(hdf5) as f:
        blocklots = blocklots_in_hdf5(f)
    gc.collect()
    preds = model.forward(blocklots)
    write_completed_preds_to_db(db, "image_model", preds)


@modeling.command()
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option(
    "--max-date",
    "-d",
    help="Most recent date to consider",
    default="2022-01-01",
    type=click.DateTime(),
)
@click.pass_obj
def train(obj, hdf5, max_date):
    """Train a new model to classify roof damage severity

    HDF5 is the path to the hdf5 file containing blocklot images.
    """
    blocklots = blocklots_with_image_and_label(obj["db"], hdf5)
    X_creator = MatrixCreator(obj["db"], hdf5)
    X = X_creator.build_features(blocklots, max_date)
    print(X)


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
def crop_to_blocklots(obj, image_root, output, blocklots, overwrite):
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
        blocklots = random.sample(blocklots_in_hdf5(f), k=3)
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
        importers.append(InspectionNodesImporter.from_file(db, inspection_notes))
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
