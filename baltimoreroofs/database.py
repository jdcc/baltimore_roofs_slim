from collections import namedtuple
import logging
import os
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2 import sql
from sqlalchemy import create_engine

db = None

Creds = namedtuple("Creds", ("user", "password", "host", "port", "database"))

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, creds):
        self.creds = creds
        self.engine = create_engine(self.connection_string(self.creds))

    def table_exists(self, schema, table_name):
        return self.run_query(
            """
            SELECT EXISTS (
                SELECT FROM pg_tables
                WHERE  schemaname = %s
                AND    tablename  = %s
            );""",
            (schema, table_name),
        )[0][0]

    def drop_table(self, schema, table_name):
        self.run(
            sql.SQL("DROP TABLE IF EXISTS {table}").format(
                table=sql.Identifier(schema, table_name)
            )
        )

    def run(self, query: str, params: tuple[Any, ...] = None) -> None:
        """Run a query with no return.

        Args:
            query (str): The SQL query to run
            params (tuple[Any, ...], optional): The set of params for the query.
                Defaults to None.
        """
        conn = psycopg2.connect(
            self.connection_string(), cursor_factory=psycopg2.extras.DictCursor
        )
        cur = conn.cursor()

        logger.debug("Running query: %s", cur.mogrify(query, params))
        cur.execute(query, params)
        conn.commit()
        cur.close()
        conn.close()

    def run_query(
        self, query: str, params: tuple[Any, ...] = None
    ) -> list[psycopg2.extras.DictRow]:
        """Get results from the database.

        This creates a new database connection for every query, which should be fine for
        small things, but bad for large things.

        Args:
            query: The SQL query to run
            params: The set of parameters for the query

        Returns:
            A list of rows
        """
        conn = psycopg2.connect(
            self.connection_string(), cursor_factory=psycopg2.extras.DictCursor
        )
        cur = conn.cursor()

        logger.debug("Running query: %s", cur.mogrify(query, params))
        cur.execute(query, params)
        results = cur.fetchall()
        conn.commit()
        cur.close()
        conn.close()
        return results

    @staticmethod
    def connection_string(creds=None) -> str:
        """Get the PostgreSQL connection string from the environment.

        Returns:
            The PostgreSQL connection string
        """
        creds = creds or Creds(
            os.environ["PGUSER"],
            os.environ["PGPASSWORD"],
            os.environ["PGHOST"],
            os.environ["PGPORT"],
            os.environ["PGDATABASE"],
        )

        return (
            f"postgresql://{creds.user}:{creds.password}@"
            f"{creds.host}:{creds.port}/{creds.database}"
        )
