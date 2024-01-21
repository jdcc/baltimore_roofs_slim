import logging

from pathlib import Path
import random
from typing import Optional

import click
from dotenv import load_dotenv
import h5py
import numpy as np
from PIL import Image

from .database import Database, Creds
from .data_set_importer import (
    GeodatabaseImporter,
    InspectionNodesImporter,
)
from .images import ImageCropper, count_datasets_in_hdf5, fetch_image_from_hdf5
from .import_manager import ImportManager
from .utils import fetch_all_blocklots

load_dotenv()

logging.basicConfig(level=logging.INFO)


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
    ctx.obj = Database(creds)


@roofs.group()
def db():
    """Tools for loading and maintaining the database"""
    pass


@roofs.group()
def models():
    """Tools for interacting with roof damage classification models"""
    pass


@models.command()
def predict():
    """Output roof damage predictions"""
    pass


@models.command()
def train():
    """Train a new model to classify roof damage severity"""
    pass


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
def crop_to_blocklots(db, image_root, output, blocklots, overwrite):
    """Crop aerial image tiles into individual blocklot images

    IMAGE_ROOT is the path of the directory containing all the .tif images
        and their accompanying .tif.aux.xml files.

    OUTPUT is the path of the hdf5 file that will be created containing
        the blocklot images.
    """
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
def images_status(db, hdf5):
    """Show the status of the blocklot image setup process

    hdf5 is the path to the hdf5 file containing all the blocklot image data.
    """
    blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
    click.echo("Images Status\n" + "=" * 50)
    click.echo(f"There are {len(blocklots):,} blocklots in the database.")
    click.echo(f"    Here are a few: {random.sample(blocklots, k=3)}")
    with h5py.File(hdf5) as f:
        n_datasets = count_datasets_in_hdf5(f)
        click.echo(f"\nThere are {n_datasets:,} blocklot images in the image database.")
        blocks = random.sample(list(f.keys()), k=3)
        blocklots = [f"{b:5}{list(f[b].keys())[0]}" for b in blocks]
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
    db,
    inspection_notes: Optional[Path] = None,
):
    """Import spreadsheets of data

    Only handles inspection notes currently."""
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
def import_gdb(db, geodatabase: Path, **layer_map):
    """Import a Geodatabase file"""
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
def db_status(db):
    """Show the status of the database"""
    click.echo(ImportManager(db).status())
    try:
        blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
        click.echo(f"There are {len(blocklots):,} row home blocklots in the database.")
    except Exception:
        pass


@db.command()
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
@click.pass_obj
def reset(db):
    """Remove all data from the database"""
    ImportManager(db).reset()
    click.echo("The database has been reset.")


if __name__ == "__main__":
    roofs()
