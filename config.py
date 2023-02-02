import json
from types import SimpleNamespace

global config
with open("config.json", "r") as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))