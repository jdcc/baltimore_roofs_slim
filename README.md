# Baltimore Roof Damage

This is a project to locate row houses in Baltimore that have extensive roof damage. It's a slimmed down and refactored version of a larger project done as part of [Data Science for Social Good, 2022](https://www.dssgfellowship.org/project/improving-community-safety-and-economic-well-being-by-remediating-buildings-with-roof-damage-in-baltimore/).

## Overview

This project handles loading and cleaning data, training models that estimate the likelihood of roof damage, and outputting predictions from those models.

There are three data sources:
  * A Geodatabase (gdb) file
  * Tabular files (CSVs or Excel files)
  * Aerial images

And two models:
  * An image model that predicts the likelihood of roof damage given an aerial photograph
  * An overall model that takes in data from many sources, including the outputs of the image model, and predicts the likelihood of roof damage

## Installation

### System Requirements

* At least 200GB of hard drive space. This is mostly occupied by the aerial images.
* At least 32GB of RAM (this could be relaxed in the future).
* A CUDA-enabled GPU is optional, but recommended.

### Docker

The easiest way to get this project running is with Docker. Make sure you have [Docker](https://docs.docker.com/manuals/) ([Docker Engine](https://docs.docker.com/engine/), either by itself or through [Docker Desktop](https://docs.docker.com/desktop/)) and [Docker Compose](https://docs.docker.com/compose/) installed. From a command shell in the project directory, just run `docker-compose up`. This will install and run everything required to use this project.

Once the containers are running, you can start interacting with the app by running bash in the main container with `docker-compose run roofs bash`.

For GPU acceleration of model training and inference, you need a GPU. If you've using a CUDA-enabled NVIDIA GPU, you can install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the existing code and Docker setup should handle running everything on the GPU.

### Bare metal

TODO. Look at `docker-compose.yaml` and the `Dockerfile` for now.

## Running
In the root of the project, create a `.env` file with the following keys:
```
PGUSER=user
PGPASSWORD=password
PGHOST=db
PGPORT=5432
PGDATABASE=roofs
```

The main mode of interacting with this project is through a series of command-line interface commands. The full list of commands is:
```
roofs --help                      Shows help documentation. Works for all subcommands

roofs db filter                   Filter the ground truth to just the row homes we're interested in
roofs db import-gdb               Import a Geodatabase file
roofs db import-sheet             Import spreadsheets of data
roofs db reset                    Remove all data from the database
roofs db status                   Show the status of the database

roofs images crop                 Crop aerial image tiles into individual blocklot images
roofs images dump                 Dump JPEG images of blocklots to disk for further...
roofs images status               Show the status of the blocklot image setup process
roofs images predict              Make predictions using just the image model

roofs train status                Status of the training pipeline
roofs train image-model           Train an image classification model from aerial photos
roofs train model                 Train a new model to classify roof damage severity

roofs report predictions          Generate roof damage scores from a given model
roofs report evals                Evaluate the performance of a given model
roofs report html                 Generate an HTML report of predictions

roofs misc merge-sheets           Merge a number of CSVs or Excel files on blocklot
```

A full run of the project from data loading to model training to prediction looks similar to this series of commands. Use the `--help` argument to each of these commands for a full understanding of what's going on and the `status` command of each subcommand to check that everything is proceeding normally.
```bash
$ roofs db import-sheet --inspection-notes data/InspectorNotes_Roof.xlsx
$ roofs db import-gdb data/roofdata_2024.gdb \
  --building-outlines=building_outline_2010 \
  --building-permits=building_construction_permits \
  --code-violations=code_violation_data_after_2017 \
  --data-311=Housing_311_SR_Data \
  --demolitions=completed_city_demolition \
  --ground-truth=roof_data_2018 \
  --real-estate=real_estate_data \
  --tax-parcel-address=tax_parcel_address \
  --redlining=redlining \
  --vacant-building-notices=open_notice_vacant
$ roofs db filter
$ roofs db status
$ roofs images crop data/aerial_images data/images.hdf5
$ roofs images status data/images.hdf5
$ roofs images dump data/images.hdf5 . -b "1152 011"
$ roofs train status data/images.hdf5
$ roofs train image-model data/images.hdf5
$ roofs train model data/images.hdf5 models.csv 
$ roofs report evals models/6c87d283-bfee-4075-bd72-a0d4355d356a.pkl 6c87_eval.csv data/images.hdf5
$ roofs report predictions models/6c87d283-bfee-4075-bd72-a0d4355d356a.pkl 6c87_preds.csv data/images.hdf5
$ roofs report html 6c87_preds.csv data/aerial_images 6c87_report.html
$ roofs misc merge-sheets merged.csv 6c87_preds.csv 2022_data.xlsx
```

If you already have the cropped images (at `data/images.hdf5`), an image model (at `models/image_model.pth`), and an overall model (at `f940...0.pkl`), the series of commands looks like this:
```bash
$ roofs db import-sheet --inspection-notes data/InspectorNotes_Roof.xlsx
$ roofs db import-gdb data/roofdata_2024.gdb \
  --building-outlines=building_outline_2010 \
  --building-permits=building_construction_permits \
  --code-violations=code_violation_data_after_2017 \
  --data-311=Housing_311_SR_Data \
  --demolitions=completed_city_demolition \
  --ground-truth=roof_data_2018 \
  --real-estate=real_estate_data \
  --tax-parcel-address=tax_parcel_address \
  --redlining=redlining \
  --vacant-building-notices=open_notice_vacant
$ roofs db filter
$ roofs db status
$ roofs images status data/images.hdf5
$ roofs images dump data/images.hdf5 . -b "1152 011"
$ roofs train status data/images.hdf5
$ roofs images predict data/images.hdf5
$ roofs report evals models/f940d7a5-e5c0-4267-8764-c02fa3542730.pkl f940_eval.csv data/images.hdf5
$ roofs report predictions models/f940d7a5-e5c0-4267-8764-c02fa3542730.pkl f940_preds.csv data/images.hdf5
$ roofs report html f940_preds.csv data/aerial_images f940_report.html
$ roofs misc merge-sheets merged.csv f940_preds.csv 2022_data.xlsx
```

## Open Questions
* License?