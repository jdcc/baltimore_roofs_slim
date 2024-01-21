from psycopg2 import sql


def split_blocklot(blocklot):
    block = blocklot[:5].strip()
    lot = blocklot[5:].strip()
    return block, lot


def fetch_all_blocklots(db, schema):
    query = sql.SQL(
        """
        SELECT DISTINCT(blocklot)
        FROM {tpa_table}
        ORDER BY blocklot"""
    ).format(tpa_table=sql.Identifier(schema, "tax_parcel_address"))
    results = db.run_query(query)
    return [row["blocklot"] for row in results]
