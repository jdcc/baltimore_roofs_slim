from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psycopg2 import sql
from sklearn.metrics import precision_recall_curve

from ..data import fetch_image_from_hdf5
from ..modeling import fetch_blocklot_label


class Reporter:
    def __init__(self, db, hdf5: Path):
        self.db = db
        self.hdf5 = hdf5

    def blocklot_lat_lon(self, blocklots: list[str]) -> dict[str, list[float]]:
        """Get the latitude and longitude of the center of a blocklot"""
        results = self.db.run_query(
            sql.SQL(
                """
            SELECT blocklot,
                ST_X(ST_TRANSFORM(ST_CENTROID(tpa.shape), 4326)) AS lon,
                ST_Y(ST_TRANSFORM(ST_CENTROID(tpa.shape), 4326)) AS lat
            FROM {table} tpa
            WHERE blocklot IN %s
        """
            ).format(table=self.db.TPA),
            (tuple(blocklots),),
        )
        return {row["blocklot"]: [row["lat"], row["lon"]] for row in results}

    def pictometry_url_for_blocklot(self, blocklot: str) -> str:
        """Get the aerial photo link of a blocklot"""
        lat_lon = self.blocklot_lat_lon([blocklot])[blocklot]
        return self.pictometry_url(lat_lon[0], lat_lon[1])

    @classmethod
    def pictometry_url(cls, lat: float, lon: float) -> str:
        """Get the aerial photo link of a coordinate"""
        return (
            "https://explorer.pictometry.com/index.php?lat="
            f"{lat:.6f}&lon={lon:.6f}&angle=Or&zoom=21"
        )

    def codemap_url_for_blocklot(cls, blocklot: str) -> str:
        """Get the Codemap URL for a blocklot

        This is a link to the internal codemap"""
        lat_lon = cls.blocklot_lat_lon([blocklot])[blocklot]
        return cls.codemap_url(lat_lon[1], lat_lon[0])

    @classmethod
    def codemap_url(cls, lat: float, lon: float) -> str:
        """Get the Codemap URL for a coordinate"""
        return (
            "https://cels.baltimorehousing.org/codemapv2/?center="
            f"{lat}%2C{lon}%2C4326&level=20"
        )

    @classmethod
    def codemap_ext_url(cls, lat, lon):
        """Get the Codemap URL for a coordinate

        This gets a link to the external instance of Codemap"""
        return (
            "https://cels.baltimorehousing.org/codemapv2ext/?center="
            f"{lat}%2C{lon}%2C4326&level=20"
        )

    def codemap_ext_urls(self, blocklots: list[str]) -> dict[str, str]:
        """Get the Codemap (external instance) links for a bunch of blocklots"""
        lat_lon = self.blocklot_lat_lon(blocklots)
        return {
            b: self.codemap_ext_url(lat_lon[b][1], lat_lon[b][0]) for b in blocklots
        }

    def codemap_urls(self, blocklots):
        lat_lon = self.blocklot_lat_lon(blocklots)
        return {b: self.codemap_url(lat_lon[b][1], lat_lon[b][0]) for b in blocklots}

    def pictometry_urls(self, blocklots):
        lat_lon = self.blocklot_lat_lon(blocklots)
        return {b: self.pictometry_url(lat_lon[b][0], lat_lon[b][1]) for b in blocklots}

    def codemap_ext_url_for_blocklot(self, blocklot):
        lat_lon = self.blocklot_lat_lon([blocklot])[blocklot]
        return self.codemap_ext_url(lat_lon[1], lat_lon[0])

    def plot_blocklot(self, blocklot, title=None):
        image = fetch_image_from_hdf5(blocklot, hdf5_filename=self.hdf5)
        pixels = np.nan_to_num(image[:], nan=255).astype("uint8")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(pixels)
        if title is None:
            title = f"{blocklot} - {fetch_blocklot_label(self.db, blocklot)}"
        ax.set_title(title)
        print(self.pictometry_url_for_blocklot(blocklot))
        ax.axis("off")
        plt.show()

    def plot_score_distribution(self, scores, name=None):
        fig, ax = plt.subplots(4, 1, figsize=(6, 8), sharex=True, sharey=False)
        _, bins, _ = ax[0].hist(
            scores[scores.label == 0].score, bins=60, color="darkred"
        )
        ax[0].legend(['Labeled "No damage"'])
        ax[1].hist(scores[scores.label == 1].score, bins=bins, color="darkgreen")
        ax[1].legend(['Labeled "Damage"'])
        ax[2].hist(scores[scores.label.isna()].score, bins=bins, color="gray")
        ax[2].legend(["No label"])
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[2].set_yscale("log")
        ax[1].set_ylabel("Number of blocklots (log scale)")
        if name is None:
            ax[0].set_title("Distribution of scores")
        else:
            ax[0].set_title(name)

        _, bins, neg = ax[3].hist(
            scores[scores.label == 0].score, bins=60, color="darkred", alpha=0.3
        )
        twin_ax = ax[3].twinx()
        # ax[0].legend(['Labeled "No damage"'])
        _, _, pos = twin_ax.hist(
            scores[scores.label == 1].score, bins=bins, color="darkgreen", alpha=0.3
        )
        ax[3].set_yscale("log")
        twin_ax.set_yscale("log")
        # ax[3].set_ylabel("# of blocklots (No damage)")
        # twin_ax.set_ylabel("# of blocklots (Damage)")
        ax[3].set_xlabel("Score")

        fig.tight_layout()
        plt.show()

    def plot_overlapping_score_distribution(self, scores, name=None):
        fig, ax = plt.subplots(1, figsize=(8, 3), sharex=True, sharey=False)
        if name is None:
            ax[0].set_title("Distribution of scores")
        else:
            ax[0].set_title(name)
        fig.tight_layout()
        plt.show()

    def get_scores_and_labels(self, model_id):
        preds = Predictor.load_preds(model_id, split_id, schema_prefix)
        return self.evaluator.build_score_df(preds)

    def get_top_k(self, model_id, k):
        return (
            self.get_scores_and_labels(model_id, split_id)
            .sort_values("score", ascending=False)
            .head(k)
        )

    def plot_prk_curve_for_model_group(self, group_id, schema_prefix, *args, **kwargs):
        e = self.evaluator.load_group_eval(group_id, schema_prefix)
        self.plot_prk_curve(e, *args, **kwargs)

    def plot_prk_curve_for_split(self, model_id, split_id, *args, **kwargs):
        e = self.evaluator.load_eval(model_id, split_id)
        self.plot_prk_curve(e, *args, **kwargs)

    def plot_prk_curve(
        self, eval_metrics, title=None, xmin=-0.03, xmax=1.03, legend=True
    ):
        e = eval_metrics
        p = e[
            e.index.str.startswith("threshold") & (e.metric == "precision")
        ].sort_values("ref")
        r = e[e.index.str.startswith("threshold") & (e.metric == "recall")].sort_values(
            "ref"
        )

        x = p.ref
        base_rate = p.loc[p.ref == 1, "value"].values[0]
        n = e.loc[(e.ref == 1) & (e.metric == "n"), "value"].values[0]
        k = config.evaluator.top_k

        plt.rc("font", size=13)
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(x, p.value, "b")
        ax1.set_xlabel("Proportion of population (ordered by score)")
        ax1.set_ylabel("Precision", color="b")
        ax1.hlines(base_rate, -0.03, 1.03, linestyle="--", color="b", alpha=0.6)
        cutoff = ax1.vlines(k / n, 0, 1.05, linestyle="--", color="black", alpha=0.5)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(xmin, xmax)

        ax2 = ax1.twinx()
        ax2.plot(x, r.value, "r")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="r", alpha=0.6)
        ax2.set_ylabel("Recall", color="r")
        ax2.set_ylim(0, 1.05)
        if title:
            ax1.set_title(title)
        if legend:
            ax1.legend([cutoff], [f"Top {k} cutoff"])
        plt.show()

    @staticmethod
    def plot_precision_recall_n(y_true, y_prob, model_name):
        """
        y_true: ls
            ls of ground truth labels
        y_prob: ls
            ls of predic proba from model
        model_name: str
            str of model name (e.g, LR_123)
        """
        y_score = y_prob
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_true, y_score
        )
        precision_curve = precision_curve[:-1]
        recall_curve = recall_curve[:-1]
        pct_above_per_thresh = []
        number_scored = len(y_score)
        for value in pr_thresholds:
            num_above_thresh = len(y_score[y_score >= value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)
        plt.clf()

        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_per_thresh, precision_curve, "b")
        ax1.set_xlabel("percent of population")
        ax1.set_ylabel("precision", color="b")
        ax1.hlines(y_true.mean(), 0, 1, linestyle="--", color="b", alpha=0.6)

        ax1.set_ylim(0, 1.05)
        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, recall_curve, "r")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="r", alpha=0.6)
        ax2.set_ylabel("recall", color="r")
        ax2.set_ylim(0, 1.05)

        name = model_name
        plt.title(name)
        plt.show()
        plt.clf()
