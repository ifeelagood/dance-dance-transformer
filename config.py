import json
import pathlib
from types import SimpleNamespace

# json decoding hook to convert json objects to SimpleNamespace objects
_object_hook = lambda d: SimpleNamespace(**d)

# load config.json
with open("config.json") as f:
    config = json.load(f, object_hook=_object_hook)

# set config.paths.* to pathlib.Path objects
for k, v in config.paths.__dict__.items():
    config.paths.__dict__[k] = pathlib.Path(v)

