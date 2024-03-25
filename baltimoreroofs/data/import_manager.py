from psycopg2 import sql, errors

from .data_set_importer import (
    DataSetImporter,
    GeodatabaseImporter,
    InspectionNotesImporter,
    RAW_SCHEMA,
    CLEAN_SCHEMA,
)
from .database import Database


class ImportManager:
    def __init__(self, db: Database, importers: list[DataSetImporter] = []):
        self.db = db
        self.importers = importers
        self.all_importers = [
            GeodatabaseImporter(db),
            InspectionNotesImporter(db),
        ]

    def ensure_schemas(self):
        for schema in [RAW_SCHEMA, CLEAN_SCHEMA]:
            query = sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(
                schema=sql.Identifier(schema)
            )
            self.db.run(query)

    def setup(self):
        self.ensure_schemas()
        for importer in self.importers:
            if not importer.is_imported() and importer.is_ready_to_import():
                importer.import_raw()
                importer.clean()

    def reset(self):
        for importer in self.all_importers:
            importer.reset()

    def assert_setup(self):
        for importer in self.all_importers:
            importer.assert_imported()

    def status(self):
        WIDTH = 67
        output = "Import Status\n" + "=" * WIDTH + "\n"
        for importer in self.all_importers:
            output += f"{importer.data_desc:40}: {importer.is_imported()}\n"

        output += "\n"
        for importer in self.all_importers:
            try:
                importer.assert_imported()
            except AssertionError as e:
                output += f"{importer.data_desc}: {e}\n"

        output += "\nMost Recent Data\n" + "=" * WIDTH + "\n"
        for importer in self.all_importers:
            try:
                for table, date in importer.get_most_recent_data_date().items():
                    output += f"{table:40}: {date}\n"
            except errors.UndefinedTable:
                continue

        return output
