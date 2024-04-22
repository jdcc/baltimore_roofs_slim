from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

from ..data import fetch_blocklots_imaged
from ..modeling import fetch_blocklots_imaged_and_labeled
from .reporting import Reporter

SEED = 0
EvalCounts = namedtuple(
    "EvalCounts",
    ("n", "n_labeled", "n_labeled_pos", "n_labeled_neg", "precision", "recall"),
)


def pull_eval_counts(df: pd.DataFrame, n_total_positive: int) -> EvalCounts:
    label_counts = df.label.value_counts()
    precision = None
    recall = None
    if label_counts.sum() > 0:
        precision = label_counts.get(1.0, default=0) / label_counts.sum()
    if n_total_positive > 0:
        recall = label_counts.get(1.0, default=0) / n_total_positive
    return EvalCounts(
        n=int(df.shape[0]),
        n_labeled=int(label_counts.sum()),
        n_labeled_pos=int(label_counts.get(1.0, default=0)),
        n_labeled_neg=int(label_counts.get(0.0, default=0)),
        precision=precision,
        recall=recall,
    )


def build_evaluation(
    df: pd.DataFrame, thresholds=None, ks=None
) -> tuple[dict[float, EvalCounts], dict[int, EvalCounts]]:
    """Return a data structure that gives performance metrics at various cuts

    A "cut" removes any predictions below a certain score. So for example, a cut
    at top-1000 sorts all the predictions by score so that highest scoring
    are at the top, cuts off the bottom of the list after item 1,000, and then
    evaluates the performance of remaining predictions.

    This returns two types of cuts: percentage cuts which are useful for plotting
    performance by list length, and top-k cuts which are useful for assessing
    performance of real-life list outputs."""
    df = df[~df.score.isna()].copy()
    n_total_positive = df.label.sum()
    df["threshold"] = df.score.rank(ascending=False, pct=True)
    threshold_counts = {}
    for threshold in thresholds:
        to_eval = df[df.threshold < threshold]
        threshold_counts[threshold] = pull_eval_counts(to_eval, n_total_positive)

    top_k_counts = {}
    for k in ks:
        top_k = df.sort_values("score", ascending=False).head(k)
        top_k_counts[k] = pull_eval_counts(top_k, n_total_positive)

    return threshold_counts, top_k_counts


def evaluate(db, model_path, hdf5, max_date, top_k, eval_test_only):
    """Evaluate the performance of a given model"""
    reporter = Reporter(db)
    blocklot_labels = fetch_blocklots_imaged_and_labeled(db, hdf5)
    if eval_test_only:
        blocklots, labels = list(blocklot_labels.keys()), list(blocklot_labels.values())
        _, blocklots = train_test_split(
            blocklots, test_size=0.3, stratify=labels, random_state=SEED
        )
    else:
        with h5py.File(hdf5) as f:
            blocklots = fetch_blocklots_imaged(f)
    preds = reporter.predictions(model_path, hdf5, blocklots, max_date)
    df = pd.DataFrame(
        {"label": {b: l for b, l in blocklot_labels.items()}, "score": preds}
    )
    return build_evaluation(df, thresholds=np.linspace(0.0, 1.0, 200), ks=top_k)


def graph_model_scores(scores):
    # Initialize lists to store summary statistics
    f1_means, f1_stds = [], []
    precision_means, precision_stds = [], []
    recall_means, recall_stds = [], []
    fit_time_means, fit_time_stds = [], []
    score_time_means, score_time_stds = [], []

    # Iterate over models and calculate summary statistics
    for path, metrics in scores.items():
        f1_means.append(np.mean(metrics["test_f1"]))
        f1_stds.append(np.std(metrics["test_f1"]))

        precision_means.append(np.mean(metrics["test_precision"]))
        precision_stds.append(np.std(metrics["test_precision"]))

        recall_means.append(np.mean(metrics["test_recall"]))
        recall_stds.append(np.std(metrics["test_recall"]))

        fit_time_means.append(np.mean(metrics["fit_time"]))
        fit_time_stds.append(np.std(metrics["fit_time"]))

        score_time_means.append(np.mean(metrics["score_time"]))
        score_time_stds.append(np.std(metrics["score_time"]))

    # Plotting
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))  # 5 rows for 5 metrics

    # F1 Score Plot
    axs[0].errorbar(range(len(scores)), f1_means, yerr=f1_stds, fmt="o")
    axs[0].set_title("F1 Score of Models")
    axs[0].set_ylabel("F1 Score")

    # Precision Plot
    axs[1].errorbar(range(len(scores)), precision_means, yerr=precision_stds, fmt="o")
    axs[1].set_title("Precision of Models")
    axs[1].set_ylabel("Precision")

    # Recall Plot
    axs[2].errorbar(range(len(scores)), recall_means, yerr=recall_stds, fmt="o")
    axs[2].set_title("Recall of Models")
    axs[2].set_ylabel("Recall")

    # Fit Time Plot
    axs[3].errorbar(range(len(scores)), fit_time_means, yerr=fit_time_stds, fmt="o")
    axs[3].set_title("Fit Time of Models")
    axs[3].set_ylabel("Fit Time (s)")

    # Score Time Plot
    axs[4].errorbar(range(len(scores)), score_time_means, yerr=score_time_stds, fmt="o")
    axs[4].set_title("Score Time of Models")
    axs[4].set_ylabel("Score Time (s)")
    axs[4].set_xlabel("Model Index")

    plt.tight_layout()
    return fig
