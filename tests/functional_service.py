"""Read existing shared acceptance projects; no fixtures or inference at startup."""

import os

from pipeline import Pipeline
from tkd_poomsae.api import create_app

pipe = Pipeline()
app = create_app(
    pipe,
    allowed_roots={"acceptance": pipe.store.root.path},
    trusted_origins={os.environ.get("TKD_BROWSER_ORIGIN", "http://127.0.0.1:5350")},
)
