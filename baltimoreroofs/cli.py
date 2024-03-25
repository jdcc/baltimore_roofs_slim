import base64
import logging
import io
import random
from pathlib import Path
from pprint import pprint
import string
from typing import Optional, Iterable

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .data import (
    Creds,
    Database,
    GeodatabaseImporter,
    ImageCropper,
    ImportManager,
    InspectionNotesImporter,
    MatrixCreator,
    count_datasets_in_hdf5,
    fetch_all_blocklots,
    fetch_blocklots_imaged,
    fetch_image_from_hdf5,
)
from .modeling import get_modeling_status, train_image_model
from .modeling.models import (
    fetch_blocklots_imaged_and_labeled,
    load_model,
    fetch_labels,
)
from .modeling.training import Trainer, build_score_df, flatten_X_y
from .reporting import build_evaluation, Reporter

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

this_dir = Path(__file__).resolve().parent
SEED = 0


# db group
@click.group()
@click.option("--user", envvar="PGUSER")
@click.option("--password", envvar="PGPASSWORD")
@click.option("--host", envvar="PGHOST")
@click.option("--port", envvar="PGPORT")
@click.option("--database", envvar="PGDATABASE")
@click.pass_context
def roofs(ctx, user, password, host, port, database):
    """Command line interface tools for setting up and detecting roof damage"""
    creds = Creds(user, password, host, port, database)
    ctx.obj = {"db": Database(creds)}


@roofs.group()
def db():
    """Tools for loading and maintaining the database"""
    pass


@roofs.group()
@click.option(
    "--models-path",
    "-m",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    help="path where models are stored",
    default=this_dir.parent / "models",
)
@click.pass_context
def train(ctx, models_path):
    """Tools for training roof damage classification models"""
    ctx.obj["models_path"] = models_path


@train.command(name="status")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.pass_obj
def modeling_status(obj, hdf5):
    """Status of the training pipeline

    HDF5 is the path to the hdf5 file containing blocklot images.
    """
    db, models_path = obj["db"], obj["models_path"]
    click.echo(get_modeling_status(db, models_path, hdf5))


@train.command(name="image-model")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option("--seed", default=SEED, type=int)
@click.pass_obj
def cli_train_image_model(obj, hdf5, seed):
    """Train an image classification model from aerial photos

    HDF5 is the path to the hdf5 file containing blocklot images.
    """
    train_image_model(obj["db"], obj["models_path"], hdf5, seed=seed)


@train.command()
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--model-path",
    "-m",
    help="directory to save models into",
    default=Path(__file__).resolve().parent.parent / "models",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--max-date",
    "-d",
    help="Most recent date to consider",
    default="2023-01-03",
    type=click.DateTime(),
)
@click.pass_obj
def model(obj, hdf5, output, model_path, max_date):
    """Train a new model to classify roof damage severity

    HDF5 is the path to the hdf5 file containing blocklot images.
    OUTPUT is the filename of the CSV into which to write scores

    In addition to the CSV, a PNG of the same name will be written
    that graphs the scores.
    """
    blocklot_labels = fetch_blocklots_imaged_and_labeled(obj["db"], hdf5)
    print(f"Fetched {len(blocklot_labels):,} labeled blocklots")
    blocklots, labels = list(blocklot_labels.keys()), list(blocklot_labels.values())
    train, test = train_test_split(
        blocklots, test_size=0.3, stratify=labels, random_state=SEED
    )
    train_X, train_y, _ = build_dataset(
        obj["db"], hdf5, train, blocklot_labels, max_date
    )
    print(f"Training with {len(train):,} examples, {sum(train_y):,} with damage")
    trainer = Trainer(model_path)
    scores, params = trainer.train(train_X, train_y)
    mean_scores = {
        model: ({name: sum(numbers) / len(numbers) for name, numbers in metric.items()})
        for model, metric in scores.items()
    }
    score_df = pd.DataFrame.from_dict(mean_scores, orient="index")
    df = score_df.join(pd.DataFrame.from_dict(params, orient="index"))
    graph_filename = output.with_suffix(".png")
    graph_model_scores(scores, graph_filename)
    df.to_csv(output)
    print(f"Wrote scores to {output} and graph to {graph_filename}")


def build_dataset(db, hdf5, blocklots, labels: dict[str, int], max_date):
    X_creator = MatrixCreator(db, hdf5)
    features = X_creator.build_features(blocklots, max_date)
    return flatten_X_y(features, {b: labels.get(b, None) for b in blocklots})


@roofs.group()
@click.option(
    "--models-path",
    "-m",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    help="path where models are stored",
    default=this_dir.parent / "models",
)
@click.pass_context
def report(ctx, models_path):
    """Write out reports using trained models"""
    ctx.obj["models_path"] = models_path


@report.command()
@click.argument(
    "model_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, exists=False, path_type=Path),
)
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option("--blocklots", "-b", type=str, multiple=True, default=[])
@click.option(
    "--max-date",
    "-d",
    help="Most recent date to consider",
    default="2023-01-03",
    type=click.DateTime(),
)
@click.pass_obj
def predictions(obj, model_path, output, hdf5, blocklots, max_date):
    """Generate roof damage scores from a given model

    MODEL_PATH is the path to a trained model
    OUTPUT is the path at which the output CSV of predictions should be written
    HDF5 is the path to the hdf5 file containing blocklot images
    """
    X_creator = MatrixCreator(obj["db"], hdf5)
    if len(blocklots) == 0:
        with h5py.File(hdf5) as f:
            blocklots = fetch_blocklots_imaged(f)
    print(f"Predicting on {len(blocklots):,} blocklots")
    X = X_creator.build_features(blocklots, max_date)
    X, _, _ = flatten_X_y(X, {})
    model, _ = load_model(model_path)
    preds = model.predict_proba(X)
    df = pd.DataFrame(preds, index=blocklots, columns=["not_damaged", "damaged"])
    df.index.name = "blocklot"
    reporter = Reporter(obj["db"], hdf5)
    df["codemap"] = pd.Series(reporter.codemap_urls(df.index))
    df["codemap_ext"] = pd.Series(reporter.codemap_ext_urls(df.index))
    df["pictometry"] = pd.Series(reporter.pictometry_urls(df.index))
    df.sort_values("damaged", ascending=False, inplace=True)
    df.to_csv(output)
    print(f"Wrote predictions to {output}")


@report.command()
@click.argument(
    "model_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, exists=False, path_type=Path),
)
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.option(
    "--max-date",
    "-d",
    help="Most recent date to consider",
    default="2023-01-03",
    type=click.DateTime(),
)
@click.option(
    "--top-k",
    "-k",
    help="Run evaluations at this top-k value",
    multiple=True,
    default=[1000, 2500, 3000, 5000],
    type=int,
)
@click.option(
    "--eval-test-only/--eval-on-all",
    default=False,
    help="Evaluate the model performance only on the test set",
)
@click.pass_obj
def evals(obj, model_path, output, hdf5, max_date, top_k, eval_test_only):
    """Evaluate the performance of a given model

    MODEL_PATH is the path to a trained model
    HDF5 is the path to the hdf5 file containing blocklot images

    The --eval-test-only option is useful for gauging the performance of the model
    on unseen data. It will run an evaluation only on labeled data that the model
    did not see during training (assuming the same seed). Without the option,
    performance will be evaluated on all data, which will include data that was in
    the training set. This means that if the model overfit to the training data,
    the performance will look better than you'll be able to expect on new data.
    """
    blocklot_labels = fetch_blocklots_imaged_and_labeled(obj["db"], hdf5)
    logger.info("Building dataset")
    if eval_test_only:
        blocklots, labels = list(blocklot_labels.keys()), list(blocklot_labels.values())
        _, test = train_test_split(
            blocklots, test_size=0.3, stratify=labels, random_state=SEED
        )
        X, _, blocklots = build_dataset(
            obj["db"], hdf5, test, blocklot_labels, max_date
        )
    else:
        blocklots = fetch_blocklots_imaged(hdf5)
        X, _, blocklots = build_dataset(
            obj["db"], hdf5, blocklots, blocklot_labels, max_date
        )
    logger.info(f"Evaluating model on {len(blocklots):,} blocklots")
    model, _ = load_model(model_path)
    preds = dict(zip(blocklots, model.predict_proba(X)[:, 1]))
    df = build_score_df(blocklot_labels, preds)
    thresholds = np.linspace(0.0, 1.0, 50)
    threshold_eval, top_k_eval = build_evaluation(df, thresholds, top_k)
    eval = pd.concat(
        [
            pd.DataFrame.from_dict(top_k_eval, orient="index"),
            pd.DataFrame.from_dict(threshold_eval, orient="index"),
        ]
    )
    eval.index.name = "cut"
    eval.to_csv(output, index_label="cut")
    print(f"Wrote evaluation details to {output}")


def numpy_to_base64(arr):
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64_string


@report.command()
@click.argument(
    "preds",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
)
@click.argument(
    "image-root",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.option(
    "--top-n",
    type=int,
    help="How many blocklots to add to report from the prediction CSV",
    default=100,
)
@click.pass_obj
def html(obj, preds, image_root, top_n):
    """Generate an HTML report of predictions

    PREDS is the predictions CSV that results from `roofs report predictions`
    IMAGE_ROOT is the root directory of the aerial images"""
    blocklot_labels = fetch_labels(obj["db"])
    preds = pd.read_csv(preds, index_col="blocklot")
    blocklots = preds.head(top_n).index.values
    cropper = ImageCropper(obj["db"], image_root)
    blocklot_sections = []
    for blocklot in tqdm(blocklots, desc="Writing blocklots", smoothing=0):
        pixels = cropper.pixels_for_blocklot(blocklot, buffer=20)
        np.nan_to_num(pixels, nan=255.0, copy=False)
        base64_data = numpy_to_base64(pixels.astype(np.uint8))
        blocklot_sections.append(
            f"""<tr>
            <td>{blocklot}</td>
            <td>{preds.at[blocklot,'damaged']:.5}</td>
            <td>{blocklot_labels.get(blocklot, '')}</td>
            <td><a href="{preds.at[blocklot, 'codemap_ext']}" target="_blank">CoDeMap</a></td>
            <td><img src="data:image/png;base64,{base64_data}"></td>
            <td><input type="checkbox" name="{blocklot}"/></td>
            </tr>"""
        )
    output = "report.html"
    with open(output, "w") as f:
        f.write(
            f"""
                <html>
                <head>
                    <style>
                    html {{
                        font-family: sans-serif;
                        margin: 20px;
                    }}
                    img {{
                        max-width: 800px;
                    }}
                    </style>
                </head>
                <body>
                    <h1>Blocklots</h1>
                    <button id="download">Download CSV</button>
                    <table>
                    <thead>
                    <tr>
                    <th>Blocklot</th>
                    <th>Score</th>
                    <th>Label</th>
                    <th>Link</th>
                    <th>Image</th>
                    <th>Damaged?</th>
                    </tr>
                    </thead>
                    <tbody>
                    {''.join(blocklot_sections)}
                    </tbody>
                    </table>
                    <script>
                        const downloadBtn = document.getElementById('download');

                        downloadBtn.addEventListener('click', () => {{
                        const checkboxes = document.querySelectorAll('input[type="checkbox"]');
                        const csvData = [];

                        checkboxes.forEach((checkbox) => {{
                            const {{ name, value, checked }} = checkbox;
                            csvData.push(`${{name}},${{value}},${{checked}}`);
                        }});

                        const csvContent = 'Blocklot,Value,Checked\\n' + csvData.join('\\n');
                        const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                        const url = URL.createObjectURL(blob);
                        const link = document.createElement('a');
                        link.setAttribute('href', url);
                        link.setAttribute('download', 'checkbox_states.csv');
                        link.style.visibility = 'hidden';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        }});
                    </script>
                </body>
                </html>"""
        )
    print(f"Wrote output to {output}")


# TODO Move this somewhere nice
def pull_out_best(scores):
    return sorted(scores, key=lambda k: np.mean(scores[k]["test_f1"]), reverse=True)[0]


def graph_model_scores(scores, filename):
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
    fig.savefig(filename, dpi=300)


@roofs.group()
@click.pass_context
def images(ctx):
    """Tools for interacting with aerial images"""
    pass


@images.command()
@click.argument(
    "image-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
)
@click.option("--blocklots", "-b", type=str, multiple=True, default=[])
@click.option("--overwrite/--no-overwrite", default=False)
@click.pass_obj
def crop(obj, image_root, output, blocklots, overwrite):
    """Crop aerial image tiles into individual blocklot images

    IMAGE_ROOT is the path of the directory containing all the .tif images
    and their accompanying .tif.aux.xml files.

    OUTPUT is the path of the hdf5 file that will be created containing
    the blocklot images.
    """
    db = obj["db"]
    cropper = ImageCropper(db, image_root)
    if len(blocklots) == 0:
        blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
    cropper.write_h5py(output, blocklots, overwrite)


@images.command(name="status")
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.pass_obj
def images_status(obj, hdf5):
    """Show the status of the blocklot image setup process

    hdf5 is the path to the hdf5 file containing all the blocklot image data.
    """
    db = obj["db"]
    blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
    click.echo("Images Status\n" + "=" * 50)
    click.echo(f"There are {len(blocklots):,} blocklots in the database.")
    click.echo(f"    Here are a few: {random.sample(blocklots, k=3)}")
    with h5py.File(hdf5) as f:
        n_datasets = count_datasets_in_hdf5(f)
        click.echo(f"\nThere are {n_datasets:,} blocklot images in the image database.")
        blocklots = random.sample(fetch_blocklots_imaged(f), k=3)
        click.echo(f"    Here are a few: {blocklots}")


@images.command
@click.argument(
    "hdf5",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
)
@click.option("--blocklots", "-b", type=str, multiple=True, required=True)
def dump(hdf5, output, blocklots):
    """Dump JPEG images of blocklots to disk for further inspection

    HDF5 is the path to the hdf5 file containing blocklot images.
    OUTPUT is the directory into which images will be written.
    """

    with h5py.File(hdf5) as f:
        for blocklot in blocklots:
            array = fetch_image_from_hdf5(blocklot, f)
            pixels = np.nan_to_num(array[:], nan=255).astype("uint8")
            im = Image.fromarray(pixels)
            im.save(output / f"{blocklot}.jpeg")


@db.command()
@click.option(
    "-i",
    "--inspection-notes",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.pass_obj
def import_sheet(
    obj,
    inspection_notes: Optional[Path] = None,
):
    """Import spreadsheets of data

    Only handles inspection notes currently."""
    db = obj["db"]
    importers = []
    if inspection_notes:
        importers.append(InspectionNotesImporter.from_file(db, inspection_notes))
    manager = ImportManager(db, importers)
    manager.setup()
    click.echo(manager.status())


@db.command()
@click.argument(
    "geodatabase",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--building-outlines", help="layer name that shows building outlines")
@click.option("--building-permits", help="layer name that has construction permits")
@click.option("--code-violations", help="layer name that has building code violations")
@click.option("--data-311", help="layer name that contains 311 data")
@click.option("--demolitions", help="layer name that contains demolitions data")
@click.option("--ground-truth", help="layer name that includes roof quality labels")
@click.option("--real-estate", help="layer name that includes real estate transactions")
@click.option("--redlining", help="layer name that shows historical redlines")
@click.option(
    "--tax-parcel-address", help="layer name that contains tax parcel addresses"
)
@click.option(
    "--vacant-building-notices",
    help="layer name that contains the vacant building notices",
)
@click.pass_obj
def import_gdb(obj, geodatabase: Path, **layer_map):
    """Import a Geodatabase file"""
    db = obj["db"]
    if all(layer_map[layer] is None for layer in GeodatabaseImporter.REQUIRED_TABLES):
        raise click.UsageError("At least one layer name option must be specified.")
    layer_map = {
        table: layer for table, layer in layer_map.items() if layer is not None
    }
    importer = GeodatabaseImporter.from_file(db, geodatabase, layer_map)
    manager = ImportManager(db, [importer])
    manager.setup()
    click.echo(manager.status())


@db.command(name="status")
@click.pass_obj
def db_status(obj):
    """Show the status of the database"""
    db = obj["db"]
    click.echo(ImportManager(db).status())
    try:
        blocklots = fetch_all_blocklots(db, db.CLEAN_SCHEMA)
        click.echo(f"There are {len(blocklots):,} row home blocklots in the database.")
    except Exception:
        pass


@db.command()
@click.confirmation_option(prompt="Are you sure you want to reset the database?")
@click.pass_obj
def reset(obj):
    """Remove all data from the database"""
    db = obj["db"]
    ImportManager(db).reset()
    click.echo("The database has been reset.")


@roofs.group()
def misc():
    """Miscellaneous helpful widgets"""
    pass


@misc.command()
@click.argument(
    "output",
    type=click.Path(file_okay=True, dir_okay=False, exists=False, path_type=Path),
)
@click.argument(
    "sheets",
    nargs=-1,
    type=click.Path(file_okay=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--suffixes", multiple=True, help="Suffixes for the fields from each of the sheets"
)
def merge_sheets(output, sheets, suffixes):
    """Merge a number of CSVs or Excel files on blocklot

    All sheets must have a column titled "blocklot"."""
    dfs = []
    if len(suffixes) == 0:
        suffixes = [f"{letter}_" for letter in string.ascii_lowercase]

    for sheet in sheets:
        if ".xls" in str(sheet):
            df = pd.read_excel(sheet)
        else:
            df = pd.read_csv(sheet)
        df["blocklot"] = df["blocklot"].astype(str)
        dfs.append(df)

    merged = dfs[0]
    for i, df in enumerate(dfs[1:]):
        merged = pd.merge(
            merged,
            df,
            on="blocklot",
            how="outer",
            suffixes=(suffixes[0], suffixes[i + 1]),
        )
    if "damaged" in merged.columns:
        merged = merged.sort_values("damaged", ascending=False)
    merged.to_csv(output, index=False)
    print(f"Merged {len(sheets)} sheets and wrote to {output}")


if __name__ == "__main__":
    roofs()
