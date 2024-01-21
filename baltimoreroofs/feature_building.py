from collections import defaultdict
from datetime import timedelta

from tqdm.auto import tqdm

from database import run_query
from models import DarkImageBaseline


def build_building_permits_features(blocklots, max_date, _):
    results = run_query(
        """
        SELECT
            tp.blocklot,
            COUNT(DISTINCT id_permit) AS n_permits
        FROM processed.tax_parcel_address AS tp
        LEFT JOIN raw.building_construction_permits bcp
            ON tp.blocklot = bcp.blocklot
            AND bcp.csm_issued_date <= %s
        WHERE
        tp.blocklot IN %s
        GROUP BY tp.blocklot
    """,
        (max_date, tuple(blocklots)),
    )
    return {"n_construction_permits": {r["blocklot"]: r["n_permits"] for r in results}}


def build_code_violations_features(blocklots, max_date, _):
    results = run_query(
        """
        SELECT
            tpa.blocklot,
            count(distinct cva.noticenum) AS n_code_violations
        FROM processed.tax_parcel_address tpa
        LEFT JOIN processed.code_violations_after_2017 cva
            ON tpa.blocklot = cva.blocklot
            AND cva.datecreate <= %s
            AND cva.statusorig = 'NOTICE APPROVED'
        WHERE tpa.blocklot IN %s
        GROUP BY tpa.blocklot;
    """,
        (max_date, tuple(blocklots)),
    )
    return {
        "n_code_violations": {
            row["blocklot"]: row["n_code_violations"] for row in results
        },
    }


def build_data_311_features(blocklots, max_date, args):
    assert "radii" in args
    features = {}
    for radius in args["radii"]:
        features.update(build_311_radius(blocklots, max_date, radius))
    features.update(build_311_type_features(blocklots, max_date, args["radii"]))
    return features


def build_311_radius(blocklots, max_date, radius):
    query = """
        SELECT
            blocklot,
            COUNT(DISTINCT service_request_number) AS n_calls
        FROM processed.tax_parcel_address AS tp
        LEFT JOIN processed.data_311 AS d
            ON ST_DWithin(d.shape, tp.wkb_geometry, %s)
            AND d.longitude IS NOT null
            AND created_date <= %s
        WHERE blocklot IN %s
        GROUP BY blocklot
    """
    key = f"calls_to_311_{radius}ft"
    output = {
        key: {
            row["blocklot"]: row["n_calls"]
            for row in run_query(query, (radius, max_date, tuple(blocklots)))
        }
    }
    for blocklot in blocklots:
        if blocklot not in output[key]:
            output[key][blocklot] = 0
    return output


def build_311_type_features(blocklots, max_date, radii):
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
            results = run_query(
                """
                SELECT
                    blocklot,
                    COUNT(DISTINCT service_request_number) AS n_calls
                FROM processed.tax_parcel_address AS tp
                LEFT JOIN processed.data_311 AS d
                    ON ST_DWithin(d.shape, tp.wkb_geometry, %s)
                    AND d.longitude IS NOT NULL
                    AND created_date <= %s
                    AND sr_type = %s
                WHERE blocklot IN %s
                GROUP BY blocklot
            """,
                (radius, max_date, type, tuple(blocklots)),
            )
            type_name = type.replace("HCD-", "").lower().replace(" ", "_")
            key = f"calls_to_311_for_{type_name}_{radius}ft"
            features[key] = {row["blocklot"]: row["n_calls"] for row in results}
    return features


def build_demolition_features(blocklots, max_date, _):
    results = run_query(
        """
            SELECT
                tp.blocklot,
                COUNT(distinct "ID_Demo_RFA") AS n_demos
            FROM processed.tax_parcel_address AS tp
            LEFT JOIN raw.demolitions_as_of_20220706 demo
                ON tp.blocklot = demo."BlockLot"
                AND demo."DateDemoFinish" <= %s
            WHERE
            tp.blocklot IN %s
            GROUP BY tp.blocklot
        """,
        (max_date, tuple(blocklots)),
    )
    return {"n_demolitions": {r["blocklot"]: r["n_demos"] for r in results}}


def build_inspection_notes_features(blocklots, max_date, args):
    assert "words" in args
    features = {}
    for word in args["words"]:
        results = run_query(
            f"""
            SELECT tpa.blocklot, COUNT(DISTINCT lowered_detail) AS n_mentions
            FROM processed.tax_parcel_address tpa
            LEFT JOIN processed.inspection_notes insp
                ON tpa.blocklot = insp.blocklot
                AND insp.created_date <= %s
                AND insp.lowered_detail LIKE '%%{word}%%'
            WHERE tpa.blocklot IN %s
            GROUP BY tpa.blocklot
        """,
            (max_date, tuple(blocklots)),
        )
        features[f"n_insp_note_mentions_of_{word}"] = {
            r["blocklot"]: r["n_mentions"] for r in results
        }
    return features


def build_real_estate_features(blocklots, max_date, _):
    results = run_query(
        """
        SELECT
            tp.blocklot,
            EXTRACT(EPOCH FROM(%s - COALESCE(LAST_VALUE(red.deed_date) OVER w,
                    '2011-09-01'::timestamp)))
                    AS secs_since_last_sale,
            COALESCE(LAST_VALUE(red.adjusted_price) OVER w, 0) AS last_sale_price,
            LAST_VALUE(red.adjusted_price) OVER w IS NULL AS last_sale_unknown
        FROM processed.tax_parcel_address AS tp
        LEFT JOIN processed.real_estate_data red
            ON tp.blocklot = red.blocklot
        AND red.deed_date <= %s
        WHERE tp.blocklot IN %s
        WINDOW w AS (PARTITION BY red.blocklot ORDER BY deed_date DESC)
        """,
        (max_date, max_date, tuple(blocklots)),
    )
    return {
        "secs_since_last_sale": {
            r["blocklot"]: r["secs_since_last_sale"] for r in results
        },
        "last_sale_price": {r["blocklot"]: int(r["last_sale_price"]) for r in results},
        "last_sale_unknown": {r["blocklot"]: r["last_sale_unknown"] for r in results},
    }


def build_redlining_features(blocklots, *_):
    redline_classes = ["A", "B", "C", "D", "AUD", "BUD"]
    results = run_query(
        """
        SELECT tpa.blocklot, red."class" AS redline_class
        FROM processed.tax_parcel_address tpa
        LEFT JOIN raw.redlining as red
        ON ST_Contains(red.shape, tpa.wkb_geometry)
        WHERE tpa.blocklot IN %s
    """,
        (tuple(blocklots),),
    )
    features = {}
    for redline_class in redline_classes:
        features[f"in_redline_{redline_class}"] = {
            r["blocklot"]: (1 if r["redline_class"] == redline_class else 0)
            for r in results
        }
    return features


def build_vacant_building_notices_features(blocklots, max_date, args):
    if "interpolation_date" not in args:
        interpolation_date = first_vbn_notice_date() - timedelta(days=365)
    else:
        interpolation_date = args["interpolation_date"]
    results = run_query(
        """
        SELECT
            tpa.blocklot,
            EXTRACT(
                EPOCH FROM (%s - COALESCE(min(created_date), %s)))
                AS secs_since_first_created_date,
            EXTRACT(
                EPOCH FROM (%s - COALESCE(max(created_date), %s)))
                AS secs_since_last_created_date,
            COUNT(distinct vbn."NoticeNum") AS n_vbns
            FROM processed.tax_parcel_address tpa
        LEFT JOIN processed.all_vacantbuilding_notices as vbn
            ON vbn.blocklot = tpa.blocklot
            AND vbn.created_date <= %s
            AND vbn."FileType" = 'New Violation Notice'
        WHERE tpa.blocklot IN %s
        GROUP BY tpa.blocklot;
    """,
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


def first_vbn_notice_date():
    return run_query(
        """
        SELECT min(created_date)
        FROM processed.all_vacantbuilding_notices
        """
    )[0][0]


def fetch_median_year_built():
    query = """
        SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY year_build) AS year
        FROM processed.tax_parcel_address
    """
    return run_query(query)[0]["year"]


def build_year_built_features(blocklots, max_date, args):
    assert "median_year_built" in args
    results = run_query(
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
        FROM processed.tax_parcel_address tpa
        WHERE blocklot IN %s
    """,
        (
            args["median_year_built"],
            max_date,
            args["median_year_built"],
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


def build_dark_pixels_features(blocklots, _, args):
    assert "thresholds" in args
    features = defaultdict(dict)
    threshold_models = {}
    for threshold in args["thresholds"]:
        threshold_models[threshold] = DarkImageBaseline(threshold)

    for blocklot in tqdm(blocklots, smoothing=0, desc="Loading images"):
        image = fetch_image(blocklot)
        for threshold in args["thresholds"]:
            feature_name = f"pct_pixels_darker_than_{threshold}"
            model = threshold_models[threshold]
            score = model.predict_proba(image)
            if score is None:
                score = 0.0
            features[feature_name][blocklot] = score
    return features


def build_image_model_features(blocklots, max_date, _):
    if self.transfer_learned_score.model_group_id != "None":
        scores = Evaluator(config.evaluator).model_group_scores(
            self.transfer_learned_score.model_group_id,
            self.transfer_learned_score.schema_prefix,
        )
        blocklot_scores = scores.score.to_dict()
    elif self.transfer_learned_score.model_id:
        blocklot_scores = Predictor.load_all_preds(
            self.transfer_learned_score.model_id,
            self.transfer_learned_score.schema_prefix,
        )
    return {
        "transfer_learned_score": {b: blocklot_scores.get(b, 0.0) for b in blocklots},
        "transfer_learned_score_unknown": {
            b: int(b not in blocklot_scores) for b in blocklots
        },
    }
