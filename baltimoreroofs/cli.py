import logging

from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from .database import Database, Creds
from .data_set_importer import (
    GeodatabaseImporter,
    InspectionNodesImporter,
)
from .import_manager import ImportManager

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
    creds = Creds(user, password, host, port, database)
    ctx.obj = Database(creds)


@roofs.group()
@click.pass_context
def db(ctx):
    pass


@click.option(
    "-i",
    "--inspection-notes",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-d",
    "--demolitions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.pass_obj
def import_sheets(
    db,
    vacant_building_notices: Optional[Path] = None,
    inspection_notes: Optional[Path] = None,
    demolitions: Optional[Path] = None,
):
    importers = []
    if vacant_building_notices:
        importers.append(VBNImporter.from_file(db, vacant_building_notices))
    if inspection_notes:
        importers.append(InspectionNodesImporter.from_file(db, inspection_notes))
    if demolitions:
        importers.append(DemolitionsImporter.from_file(db, demolitions))
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
    if all(layer_map[layer] is None for layer in GeodatabaseImporter.REQUIRED_TABLES):
        raise click.UsageError("At least one layer name option must be specified.")
    layer_map = {
        table: layer for table, layer in layer_map.items() if layer is not None
    }
    importer = GeodatabaseImporter.from_file(db, geodatabase, layer_map)
    manager = ImportManager(db, [importer])
    manager.setup()
    click.echo(manager.status())


@db.command()
@click.pass_obj
def status(db):
    click.echo(ImportManager(db).status())


@db.command()
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
@click.pass_obj
def reset(db):
    ImportManager(db).reset()
    click.echo("The database has been reset.")


if __name__ == "__main__":
    roofs()
