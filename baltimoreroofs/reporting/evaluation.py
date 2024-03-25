from collections import namedtuple

import pandas as pd

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
