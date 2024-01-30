from collections import defaultdict
from datetime import timedelta

from psycopg2 import sql
from tqdm.auto import tqdm

from .images import fetch_image_from_hdf5
from ..modeling import DarkImageBaseline


def build_building_permits_features(db, blocklots, max_date, _):
    results = db.run_query(
        sql.SQL(
            """
        SELECT
            tp.blocklot,
            COUNT(DISTINCT csm_id) AS n_permits
        FROM {tpa} AS tp
        LEFT JOIN {table} bcp
            ON tp.blocklot = bcp.blocklot
            AND bcp.csm_issued_date <= %s
        WHERE
        tp.blocklot IN %s
        GROUP BY tp.blocklot
    """
        ).format(tpa=db.TPA, table=sql.Identifier(db.CLEAN_SCHEMA, "building_permits")),
        (max_date, tuple(blocklots)),
    )
    return {"n_building_permits": {r["blocklot"]: r["n_permits"] for r in results}}


def build_code_violations_features(db, blocklots, max_date, _):
    results = db.run_query(
        sql.SQL(
            """
        SELECT
            tpa.blocklot,
            count(distinct cva.noticenum) AS n_code_violations
        FROM {tpa} tpa
        LEFT JOIN {table} cva
            ON tpa.blocklot = cva.blocklot
            AND cva.datecreate <= %s
            AND cva.statusorig = 'NOTICE APPROVED'
        WHERE tpa.blocklot IN %s
        GROUP BY tpa.blocklot;
    """
        ).format(tpa=db.TPA, table=sql.Identifier(db.CLEAN_SCHEMA, "code_violations")),
        (max_date, tuple(blocklots)),
    )
    return {
        "n_code_violations": {
            row["blocklot"]: row["n_code_violations"] for row in results
        },
    }


def build_data_311_features(db, blocklots, max_date, args):
    assert "radii" in args
    features = {}
    for radius in args["radii"]:
        features.update(build_311_radius(db, blocklots, max_date, radius))
    features.update(build_311_type_features(db, blocklots, max_date, args["radii"]))
    return features


def build_311_radius(db, blocklots, max_date, radius):
    query = sql.SQL(
        """
        SELECT
            blocklot,
            COUNT(DISTINCT service_request_number) AS n_calls
        FROM {tpa} AS tp
        LEFT JOIN {table} AS d
            ON ST_DWithin(d.shape, tp.shape, %s)
            AND d.longitude IS NOT null
            AND created_date <= %s
        WHERE blocklot IN %s
        GROUP BY blocklot
    """
    ).format(tpa=db.TPA, table=sql.Identifier(db.CLEAN_SCHEMA, "data_311"))
    key = f"calls_to_311_{radius}ft"
    output = {
        key: {
            row["blocklot"]: row["n_calls"]
            for row in db.run_query(query, (radius, max_date, tuple(blocklots)))
        }
    }
    for blocklot in blocklots:
        if blocklot not in output[key]:
            output[key][blocklot] = 0
    return output


def build_311_type_features(db, blocklots, max_date, radii):
    types = [
        "HCD-Illegal Dumping",
        "HCD-Vacant Building",
        "HCD-Maintenance Structure",
        "HCD-Rodents",
        "HCD-Trees and Shrubs",
        "HCD-Abandoned Vehicle",
        "HCD-CCE Building Permit Complaint",
    ]
    features = {}
    for radius in radii:
        for type in types:
            results = db.run_query(
                sql.SQL(
                    """
                SELECT
                    blocklot,
                    COUNT(DISTINCT service_request_number) AS n_calls
                FROM {tpa} AS tp
                LEFT JOIN {table} AS d
                    ON ST_DWithin(d.shape, tp.shape, %s)
                    AND d.longitude IS NOT NULL
                    AND created_date <= %s
                    AND sr_type = %s
                WHERE blocklot IN %s
                GROUP BY blocklot
            """
                ).format(tpa=db.TPA, table=sql.Identifier(db.CLEAN_SCHEMA, "data_311")),
                (radius, max_date, type, tuple(blocklots)),
            )
            type_name = type.replace("HCD-", "").lower().replace(" ", "_")
            key = f"calls_to_311_for_{type_name}_{radius}ft"
            features[key] = {row["blocklot"]: row["n_calls"] for row in results}
    return features


def build_demolitions_features(db, blocklots, max_date, _):
    results = db.run_query(
        sql.SQL(
            """
            SELECT
                tp.blocklot,
                COUNT(DISTINCT id_demo_rfa) AS n_demos
            FROM {tpa} AS tp
            LEFT JOIN {table} demo
                ON tp.blocklot = demo.blocklot
                AND demo.date_demo_finish <= %s
            WHERE
            tp.blocklot IN %s
            GROUP BY tp.blocklot
        """
        ).format(tpa=db.TPA, table=sql.Identifier(db.CLEAN_SCHEMA, "demolitions")),
        (max_date, tuple(blocklots)),
    )
    return {"n_demolitions": {r["blocklot"]: r["n_demos"] for r in results}}


def build_inspection_notes_features(db, blocklots, max_date, args):
    assert "words" in args
    features = {}
    for word in args["words"]:
        results = db.run_query(
            sql.SQL(
                f"""
            SELECT tpa.blocklot, COUNT(DISTINCT lowered_detail) AS n_mentions
            FROM {{tpa}} tpa
            LEFT JOIN {{table}} insp
                ON tpa.blocklot = insp.blocklot
                AND insp.created_date <= %s
                AND insp.lowered_detail LIKE '%%{word}%%'
            WHERE tpa.blocklot IN %s
            GROUP BY tpa.blocklot
        """
            ).format(
                tpa=db.TPA, table=sql.Identifier(db.CLEAN_SCHEMA, "inspection_notes")
            ),
            (max_date, tuple(blocklots)),
        )
        features[f"n_insp_note_mentions_of_{word}"] = {
            r["blocklot"]: r["n_mentions"] for r in results
        }
    return features


def build_real_estate_features(db, blocklots, max_date, _):
    results = db.run_query(
        sql.SQL(
            """
        SELECT
            tp.blocklot,
            EXTRACT(EPOCH FROM(%s - COALESCE(LAST_VALUE(red.deed_date) OVER w,
                    '2011-09-01'::timestamp)))
                    AS secs_since_last_sale,
            COALESCE(LAST_VALUE(red.adjusted_price) OVER w, 0) AS last_sale_price,
            LAST_VALUE(red.adjusted_price) OVER w IS NULL AS last_sale_unknown
        FROM {tpa} AS tp
        LEFT JOIN {table} red
            ON tp.blocklot = red.blocklot
        AND red.deed_date <= %s
        WHERE tp.blocklot IN %s
        WINDOW w AS (PARTITION BY red.blocklot ORDER BY deed_date DESC)
        """
        ).format(tpa=db.TPA, table=sql.Identifier(db.CLEAN_SCHEMA, "real_estate")),
        (max_date, max_date, tuple(blocklots)),
    )
    return {
        "secs_since_last_sale": {
            r["blocklot"]: r["secs_since_last_sale"] for r in results
        },
        "last_sale_price": {r["blocklot"]: int(r["last_sale_price"]) for r in results},
        "last_sale_unknown": {r["blocklot"]: r["last_sale_unknown"] for r in results},
    }


def build_redlining_features(db, blocklots, *_):
    redline_classes = ["A", "B", "C", "D", "AUD", "BUD"]
    results = db.run_query(
        sql.SQL(
            """
        SELECT tpa.blocklot, red."class" AS redline_class
        FROM {tpa} tpa
        LEFT JOIN {table} AS red
        ON ST_Contains(red.shape, tpa.shape)
        WHERE tpa.blocklot IN %s
    """
        ).format(tpa=db.TPA, table=sql.Identifier(db.CLEAN_SCHEMA, "redlining")),
        (tuple(blocklots),),
    )
    features = {}
    for redline_class in redline_classes:
        features[f"in_redline_{redline_class}"] = {
            r["blocklot"]: (1 if r["redline_class"] == redline_class else 0)
            for r in results
        }
    return features


def build_vacant_building_notices_features(db, blocklots, max_date, args):
    if "interpolation_date" not in args:
        interpolation_date = first_vbn_notice_date(db) - timedelta(days=365)
    else:
        interpolation_date = args["interpolation_date"]
    results = db.run_query(
        sql.SQL(
            """
        SELECT
            tpa.blocklot,
            EXTRACT(
                EPOCH FROM (%s - COALESCE(min(created_date), %s)))
                AS secs_since_first_created_date,
            EXTRACT(
                EPOCH FROM (%s - COALESCE(max(created_date), %s)))
                AS secs_since_last_created_date,
            COUNT(distinct vbn.noticenum) AS n_vbns
        FROM {tpa} tpa
        LEFT JOIN {table} as vbn
            ON vbn.blocklot = tpa.blocklot
            AND vbn.created_date <= %s
        WHERE tpa.blocklot IN %s
        GROUP BY tpa.blocklot;
    """
            # TODO Don't need this anymore?
            # AND vbn."FileType" = 'New Violation Notice'
        ).format(
            tpa=db.TPA, table=sql.Identifier(db.CLEAN_SCHEMA, "vacant_building_notices")
        ),
        (
            max_date,
            interpolation_date,
            max_date,
            interpolation_date,
            max_date,
            tuple(blocklots),
        ),
    )
    return {
        "n_vbns": {row["blocklot"]: row["n_vbns"] for row in results},
        "secs_since_first_vbn": {
            row["blocklot"]: row["secs_since_first_created_date"] for row in results
        },
        "secs_since_last_vbn": {
            row["blocklot"]: row["secs_since_last_created_date"] for row in results
        },
    }


def first_vbn_notice_date(db):
    return db.run_query(
        sql.SQL(
            """
        SELECT min(created_date)
        FROM {table}
        """
        ).format(table=sql.Identifier(db.CLEAN_SCHEMA, "vacant_building_notices"))
    )[0][0]


def fetch_median_year_built(db):
    query = sql.SQL(
        """
        SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY year_build) AS year
        FROM {table}
    """
    ).format(table=db.TPA)
    return db.run_query(query)[0]["year"]


def build_year_built_features(db, blocklots, max_date, args):
    year_when_unknown = args.get("year_when_unknown", fetch_median_year_built(db))
    results = db.run_query(
        sql.SQL(
            """
        SELECT
            blocklot,
            CASE
                WHEN year_build = 0 THEN %s
                WHEN year_build > extract('year' FROM %s) THEN %s
                ELSE year_build
            END::int AS year_built,
            (year_build = 0)
                OR (year_build > extract('year' FROM %s)) AS year_built_unknown
        FROM {table} tpa
        WHERE blocklot IN %s
    """
        ).format(table=db.TPA),
        (
            year_when_unknown,
            max_date,
            year_when_unknown,
            max_date,
            tuple(blocklots),
        ),
    )
    return {
        "year_built": {row["blocklot"]: row["year_built"] for row in results},
        "year_built_unknown": {
            row["blocklot"]: row["year_built_unknown"] for row in results
        },
    }


def build_dark_pixels_features(_, blocklots, __, args):
    assert "thresholds" in args
    features = defaultdict(dict)
    threshold_models = {}
    for threshold in args["thresholds"]:
        threshold_models[threshold] = DarkImageBaseline(threshold)

    for blocklot in tqdm(blocklots, smoothing=0, desc="Loading images"):
        image = fetch_image_from_hdf5(blocklot, hdf5_filename=args["hdf5"])
        for threshold in args["thresholds"]:
            feature_name = f"pct_pixels_darker_than_{threshold}"
            model = threshold_models[threshold]
            score = model.predict_proba(image)
            if score is None:
                score = 0.0
            features[feature_name][blocklot] = score
    return features


def build_image_model_features(db, blocklots, max_date, args):
    resp = db.run_query(
        sql.SQL("SELECT blocklot, score FROM {table} WHERE blocklot IN %s").format(
            table=sql.Identifier(db.OUTPUT_SCHEMA, "image_model_predictions")
        ),
        (tuple(blocklots),),
    )
    blocklot_scores = {row["blocklot"]: row["score"] for row in resp}
    return {
        "image_model_score": blocklot_scores,
        "image_model_score_unknown": {
            b: int(b not in blocklot_scores) for b in blocklots
        },
    }
