"""Disposable real local service for browser acceptance; no external inputs."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn

from contracts.models import Semantics
from storage import ArtifactKey, hash_file
from tests.test_inspection_api import parser_inspection
from tkd_poomsae.api import create_app


def main() -> None:
    with TemporaryDirectory(prefix="tkd-timeline-") as directory:
        root = Path(directory)
        index, automatic = parser_inspection(root)
        # Both projects share one store and source clock, but distinct automatic
        # parser artifacts (ordinary asymmetric arms and a two-track special).
        special, special_automatic = parser_inspection(
            root, centered=True, project="special"
        )
        originals = [automatic, special_automatic]
        hashes = {
            str(p): hash_file(p)
            for a in originals
            for p in a.path.rglob("*")
            if p.is_file()
        }
        headers, _ = index.headers("demo")
        special.pipe.register("raw", {side: root / side for side in ("left", "right")})
        special.register(
            "raw",
            {
                name: ArtifactKey(**header["key"])
                for name, header in headers.items()
                if name != "semantics"
            },
        )
        app = create_app(
            special.pipe,
            allowed_roots={"local": root},
            trusted_origins={
                f"http://127.0.0.1:{os.environ.get('GROUND_PORT', '5186')}"
            },
        )

        @app.get("/fixture/originals")
        def original_hashes() -> dict[str, object]:
            assert isinstance(automatic.metadata, Semantics)
            assert isinstance(special_automatic.metadata, Semantics)
            return {
                "unchanged": all(
                    hash_file(Path(p)) == digest for p, digest in hashes.items()
                ),
                "demo": automatic.metadata.model_dump(mode="json"),
                "special": special_automatic.metadata.model_dump(mode="json"),
            }

        uvicorn.run(app, host="127.0.0.1", port=5197)


if __name__ == "__main__":
    main()
