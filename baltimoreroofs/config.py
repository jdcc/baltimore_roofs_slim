from pathlib import Path

import yaml

currdir = Path(__file__).resolve().parent
fname = currdir.parent / "config.yaml"
with open(fname) as f:
    config = yaml.safe_load(f)
