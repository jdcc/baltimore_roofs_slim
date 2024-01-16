import logging
from pathlib import Path
import subprocess
from typing import Optional

import ohio.ext.pandas  # noqa: F401
import pandas as pd
from psycopg2 import sql

from .database import Database

logger = logging.getLogger(__name__)

RAW_SCHEMA = "raw"
CLEAN_SCHEMA = "processed"
# All the data used is here:
# https://github.com/dssg/baltimore_roofs/blob/cd1b63f18ef1cb0a1df52e3ea41282ed82102703/src/pipeline/matrix_creator.py#L499


class DataSetImporter:
    def __init__(self, desc: str, db: Database, table: str = None):
        self.data_desc = desc
        self._db = db
        self.table_name = table

    def import_raw(self, src_filename: Path = None):
        """import a raw tabular data file into the database

        Args:
            filename (Path): path to the data source in a format Pandas understands.
                https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
        """
        src_filename = src_filename or self._src_filename
        assert src_filename is not None, "Need a source file to import"
        logger.info("Reading data from %s", src_filename)
        if src_filename.suffix in (".xls", ".xlsx"):
            df = pd.read_excel(src_filename)
        else:
            df = pd.read_csv(src_filename)

        logger.info("Importing %s rows from the input file", df.shape[0])
        df.pg_copy_to(self.table_name, self._db.engine, schema=RAW_SCHEMA)

    def is_ready_to_import(self):
        return self._src_filename is not None

    def reset(self):
        self._db.drop_table(RAW_SCHEMA, self.table_name)
        self._db.drop_table(CLEAN_SCHEMA, self.table_name)

    @classmethod
    def from_file(cls, db: Database, filename: Path):
        importer = cls(db)
        importer._src_filename = filename
        return importer

    def clean(self, table_name=None):
        """Default implemention just copies data from raw to cleaned schemas"""
        table_name = table_name or self.table_name
        self._db.run(
            sql.SQL("SELECT * INTO {dest} FROM {src}").format(
                dest=sql.Identifier(CLEAN_SCHEMA, table_name),
                src=sql.Identifier(RAW_SCHEMA, table_name),
            )
        )

    def _clean_with_select(self, select_sql: str, table_name: str = None):
        """Cleaning is often careful select statements into a new field"""
        table_name = table_name or self.table_name
        self._db.run(
            # Careful, SQL injection here. Only run with trusted data
            sql.SQL(f"SELECT {select_sql} INTO {{dest}} FROM {{src}}").format(
                dest=sql.Identifier(CLEAN_SCHEMA, table_name),
                src=sql.Identifier(RAW_SCHEMA, table_name),
            )
        )

    def _add_blocklot_index(self, table_name=None):
        table_name = table_name or self.table_name
        self._db.run(
            sql.SQL("CREATE INDEX {index} ON {dest} (blocklot)").format(
                index=sql.Identifier(f"{table_name}_blocklot_idx"),
                dest=sql.Identifier(CLEAN_SCHEMA, table_name),
            )
        )

    def is_imported_in_schema(self, schema):
        return self._db.table_exists(schema, self.table_name)

    def raw_is_imported(self):
        return self.is_imported_in_schema(RAW_SCHEMA)

    def is_cleaned(self):
        return self.is_imported_in_schema(CLEAN_SCHEMA)

    def is_imported(self):
        return self.raw_is_imported() and self.is_cleaned()

    def assert_imported(self):
        assert self.raw_is_imported(), "Raw data is missing"
        assert self.is_cleaned(), "Cleaned data is missing"


class InspectionNodesImporter(DataSetImporter):
    def __init__(self, db: Database):
        super().__init__("Inspection notes", db, "inspection_notes")

    def clean(self):
        self._clean_with_select(
            """
            "DateCreate"::timestamp AS created_date,
            rpad("Block", 5) || "Lot" AS blocklot,
            lower("Detail") AS lowered_detail"""
        )
        self._add_blocklot_index()


class GeodatabaseImporter(DataSetImporter):
    REQUIRED_TABLES = [
        "building_outlines",
        "building_permits",
        "code_violations",
        "data_311",
        "demolitions",
        "real_estate",
        "redlining",
        "tax_parcel_address",
        "vacant_building_notices",
    ]

    def __init__(self, db: Database):
        super().__init__("Geodatabase layers", db)

    @classmethod
    def from_file(cls, db: Database, filename: Path, layer_map: dict[str, str]):
        importer = cls(db)
        importer._src_filename = filename
        importer._layer_map = layer_map
        return importer

    def is_ready_to_import(self):
        return self._src_filename is not None and self._layer_map is not None

    def import_raw(
        self,
        src_filename: Optional[Path] = None,
        layer_map: Optional[dict[str, str]] = None,
    ):
        """import data from a geodatabase file

        Args:
            layer_map (dict[str, str]): a dictionary mapping the names of the
                required table names to the layers in the gdb file.
                The required tables are:
                    * building_outlines
                    * building_permits
                    * code_violations
                    * data_311
                    * real_estate
                    * redlining
                    * tax_parcel_address
                    * vacant_building_notices
        """
        src_filename = src_filename or self._src_filename
        layer_map = layer_map or self._layer_map

        for table_name, layer_name in layer_map.items():
            logger.info('Importing "%s" into table "%s"', layer_name, table_name)
            subprocess.run(
                'ogr2ogr -progress -f "PostgreSQL" '
                f"-lco SCHEMA={RAW_SCHEMA} "
                f'PG:"host={self._db.creds.host} '
                f"dbname={self._db.creds.database} "
                f"user={self._db.creds.user} "
                f'password={self._db.creds.password}" '
                f"-nln {table_name} "
                f"{src_filename} {layer_name}",
                shell=True,
            )

    def _clean_building_outlines(self):
        super().clean("building_outlines")
        self._db.run(
            sql.SQL(
                "CREATE INDEX building_outlines_shape_idx ON {dest} USING gist (shape)"
            ).format(dest=sql.Identifier(CLEAN_SCHEMA, "building_outlines"))
        )

    def _clean_building_permits(self):
        super().clean("building_permits")
        self._add_blocklot_index("building_permits")

    def _clean_code_violations(self):
        self._clean_with_select(
            """
            *,
            rpad("block", 5) || "lot" AS blocklot""",
            "code_violations",
        )
        self._add_blocklot_index("code_violations")

    def _clean_data_311(self):
        dest = sql.Identifier(CLEAN_SCHEMA, "data_311")
        query = sql.SQL(
            """
            SELECT
                objectid, ST_Transform(ST_SetSRID(ST_MakePoint(longitude, latitude), 4326), 2248) AS shape,
                sr_id, service_request_number, sr_type,
                created_date, sr_status,
                status_date, priority,
                due_date, week_number,
                last_activity,
                last_activity_date,
                outcome, method_received, source, street_address, zip_code,
                neighborhood, latitude, longitude, police_district, council_district,
                vri_focus_area, case_details, geo_census_tract, geo_bulk_pickup_route,
                geo_east_west, geo_fire_inspection_area, geo_hcd_inspection_district,
                geo_transportation_sector, geo_primary_snow_zone,
                geo_street_light_service_area, geo_mixed_refuse_schedule,
                geo_refuse_route_number, geo_tree_region, geo_water_area, geo_sw_quad,
                block_number_c, details, assigned_to, int_comments,
                close_date, chip_id, sf_source,
                contact_name, contact_email, contact_primary_phone, flex_summary,
                borough, additional_comments, community_stat_area, sr_parent_id,
                sr_duplicate_id, sr_parent_id_transfer, hashedrecord, agency
            INTO {dest}
            FROM {src}
            WHERE
                longitude IS NOT NULL
                AND latitude IS NOT NULL
            """
        ).format(src=sql.Identifier(RAW_SCHEMA, "data_311"), dest=dest)
        self._db.run(query)
        self._db.run(
            sql.SQL(
                "CREATE INDEX data_311_created_date_idx ON {dest} (created_date)"
            ).format(dest=dest)
        )
        self._db.run(
            sql.SQL(
                "CREATE INDEX data_311_shape_idx ON {dest} USING gist (shape)"
            ).format(dest=dest)
        )

    def _clean_demolitions(self):
        self._clean_with_select(
            "blocklot, id_demo_rfa, datedemofinish_group AS date_demo_finish",
            "demolitions",
        )
        self._add_blocklot_index("demolitions")

    def _clean_real_estate(self):
        self._clean_with_select(
            "blocklot, adjusted_price, date_of_deed AS deed_date",
            "real_estate",
        )
        self._add_blocklot_index("real_estate")

    def _clean_redlining(self):
        super().clean("redlining")

    def _clean_tax_parcel_address(self):
        dest = sql.Identifier(CLEAN_SCHEMA, "tax_parcel_address")
        query = sql.SQL(
            """
            WITH blocklot_row AS (
                SELECT
                    objectid,
                    row_number()
                OVER (
                    PARTITION BY blocklot
                    ORDER BY shape_area desc)
                FROM {src} tpa
            )
            SELECT
                tpa.objectid, pin, pinrelate, tpa.blocklot, block, lot, ward,
                section, assessor, taxbase, bfcvland, bfcvimpr, landexmp, imprexmp,
                citycred, statcred, ccredamt, scredamt, permhome, assesgrp, lot_size,
                no_imprv, currland, currimpr, exmpland, exmpimpr, fullcash, exmptype,
                exmpcode, usegroup, zonecode, sdatcode, artaxbas, distswch, dist_id,
                statetax, city_tax, ar_owner, deedbook, deedpage, saledate, owner_abbr,
                owner_1, owner_2, owner_3, fulladdr, stdirpre, st_name, st_type,
                bldg_no, fraction, unit_num, span_num, spanfrac, zip_code, extd_zip,
                dhcduse1, dhcduse2, dhcduse3, dhcduse4, dwelunit, eff_unit, roomunit,
                rpdeltag, salepric, propdesc, neighbor, srvccntr, year_build,
                structarea, ldate, ownmde, grndrent, subtype_geodb, sdatlink,
                blockplat, mailtoadd, vacind, projdemo, respagcy, releasedate,
                vbn_issued, name, shape, shape_length, shape_area
        INTO {dest}
        FROM {src} AS tpa
        JOIN blocklot_row AS br
            ON br.objectid = tpa.objectid
        WHERE
            br.row_number = 1
                """
        ).format(
            src=sql.Identifier(RAW_SCHEMA, "tax_parcel_address"),
            dest=dest,
        )
        self._db.run(query)
        self._db.run(
            sql.SQL(
                "CREATE INDEX tax_parcel_address_shape_idx ON {dest} USING gist (shape)"
            ).format(dest=dest)
        )
        # This is UNIQUE, so not using _add_blocklot_index
        self._db.run(
            sql.SQL(
                "CREATE UNIQUE INDEX tax_parcel_address_blocklot_idx "
                "ON {dest} (blocklot)"
            ).format(dest=dest)
        )

    def _clean_vacant_building_notices(self):
        self._clean_with_select(
            "noticenum, blocklot, datenotice AS created_date",
            "vacant_building_notices",
        )
        self._add_blocklot_index("vacant_building_notices")

    def clean(self):
        for table in self.REQUIRED_TABLES:
            if self._db.table_exists(RAW_SCHEMA, table):
                logger.info("Cleaning table %s", table)
                getattr(self, f"_clean_{table}")()

    def raw_is_imported(self):
        return all(
            self._db.table_exists(RAW_SCHEMA, table) for table in self.REQUIRED_TABLES
        )

    def is_cleaned(self):
        return all(
            self._db.table_exists(CLEAN_SCHEMA, table) for table in self.REQUIRED_TABLES
        )

    def assert_imported(self):
        for schema in [RAW_SCHEMA, CLEAN_SCHEMA]:
            for table in self.REQUIRED_TABLES:
                assert self._db.table_exists(
                    schema, table
                ), f'"{table}" is missing from "{schema}" schema'

    def reset(self):
        for schema in [RAW_SCHEMA, CLEAN_SCHEMA]:
            for table_name in self.REQUIRED_TABLES:
                self._db.drop_table(schema, table_name)
