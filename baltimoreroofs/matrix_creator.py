from pathlib import Path

import pandas as pd

from . import feature_building

# To make predictions, I need data and a model. No labels. No splits. No tables in the database.

# Takes in a config, a table name, and a max date. Optionally blocklots

# The config includes:
#    * radii of 311
#    * inspection note words
#    * dark pixel thresholds
#    * transfer_learned_score, which is the model whose preds to use
#    * disabled features
#    * use cache
#    * year built imputed
#    * schema prefix
#    * evaluator, which makes the precision and recall curve graphs
#    * directories for pickle and cache

# Outputs a pickle file


class MatrixCreator:
    REQUIRED_TABLES = [
        "building_permits",
        "code_violations",
        "data_311",
        "demolitions",
        "inspection_notes",
        "real_estate",
        "redlining",
        "vacant_building_notices",
    ]

    def __init__(self, db, hdf5):
        self.db = db
        self.hdf5 = hdf5
        self.config = {
            "data_311": {"radii": [10, 50, 100, 200]},
            "inspection_notes": {
                "words": [
                    "roof",
                    "collaps",
                    "fall",
                    "brick",
                    "damage",
                    "unsafe",
                    "porch",
                    "caved",
                    "caving",
                ]
            },
            "dark_pixels": {
                "thresholds": [10, 20, 30, 40, 60, 80, 130],
                "hdf5": self.hdf5,
            },
        }

    def build_features(self, blocklots, max_date):
        features = {}
        for table in self.REQUIRED_TABLES + [
            "dark_pixels",
            "year_built",
            "image_model",
        ]:
            features.update(
                getattr(feature_building, f"build_{table}_features")(
                    self.db, blocklots, max_date, self.config.get(table, {})
                )
            )

        return features

    @staticmethod
    def save_matrix_to_disk(matrix, id):
        filename = f"{id}.pkl"
        directory = Path(config.matrix_creator.matrix_dir) / config.schema_prefix
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        matrix.to_pickle(path)
        return path

    @classmethod
    def calc_feature_matrix(self, max_date, blocklots):
        if blocklots is None:
            blocklots = self._blocklots_for_split(split_id)

        if imputed_year is None:
            imputed_year = self.year_built_imputed
        if imputed_year is None:
            imputed_year = feature_building.fetch_median_year_built(self.db)
        assert imputed_year is not None

        if max_date is None:
            max_date = self.max_date

        matrix = {}

        output = pd.DataFrame(matrix)
        return output

    def write_feature_matrices(self, max_date, blocklots=None):
        matrix = self.calc_feature_matrix(blocklots, max_date)
        self.save_matrix(matrix)


if __name__ == "__main__":
    sample_blocklots = [
        "0001 028",
        "0001 033C",
        "0001 043",
        "0002 043",
        "0002 059",
        "0003 001",
        "0003 002",
        "0003 062",
        "0003 067A",
        "0004 046",
        "0006 025",
        "0006 029B",
        "0006 032",
        "0007 001",
        "0008 019",
        "0008 032",
        "0008 062",
        "0008 064",
        "0009 007",
        "0009 024",
    ]
    creator = MatrixCreator(config.matrix_creator, "model_prep.split_kinds")
    creator.write_feature_matrices(creator.max_date, blocklots=sample_blocklots)
