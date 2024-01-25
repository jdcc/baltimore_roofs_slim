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

### Docker

The easiest way to get this project running is with Docker. Make sure you have [Docker](https://docs.docker.com/manuals/) ([Docker Engine](https://docs.docker.com/engine/), either by itself or through [Docker Desktop](https://docs.docker.com/desktop/)) and [Docker Compose](https://docs.docker.com/compose/) installed. From a command shell in the project directory, just run `docker-compose up`. This will install and run everything required to use this project.

Once the containers are running, you can start interacting with the app by running bash in the main container with `docker-compose run roofs bash`.

For GPU acceleration of model training and inference, you need a GPU. If you've using a CUDA-enabled NVIDIA GPU, you can install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the existing code and Docker setup should handle running everything on the GPU.

### Bare metal

TODO. Look at `docker-compose.yaml` and the `Dockerfile` for now.

## Running
The main mode of interacting with this project is through a series of command-line interface commands. The full list of commands is:
```
roofs --help                      Shows help documentation. Works for all subcommands

roofs db import-gdb               Import a Geodatabase file
roofs db import-sheet             Import spreadsheets of data
roofs db reset                    Remove all data from the database
roofs db status                   Show the status of the database

roofs images crop-to-blocklots    Crop aerial image tiles into individual blocklot images
roofs images dump                 Dump JPEG images of blocklots to disk for further...
roofs images status               Show the status of the blocklot image setup process

roofs modeling predict            Output roof damage predictions
roofs modeling status             Status of the modeling pipeline
roofs modeling train              Train a new model to classify roof damage severity
roofs modeling train-image-model  Train an image classification model from aerial photos
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
$ roofs db status
$ roofs images crop-to-blocklots data/images data/images.hdf5
$ roofs images status data/images.hdf5
$ roofs modeling train-image-model data/images.hdf5
```

## Open Questions
* USEGROUP R vs. USEGROUP E? Dropping blocklots like 4544D030 and 4577B002
* License?