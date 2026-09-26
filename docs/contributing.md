# Development checks

Use Python 3.11, [uv](https://docs.astral.sh/uv/), and Node.js 22 with npm. A fresh checkout needs no GPU, model weights, or dataset.

```sh
uv sync --frozen --group dev
uv run --frozen tkd-poomsae --help
uv run --frozen tkd-poomsae serve
```

The local API listens at `http://127.0.0.1:8000`; `GET /health` reports readiness. In a separate terminal:

```sh
cd web
npm ci
npm run dev
```

Before a PR, run the same offline checks as application CI:

```sh
uv run --frozen pytest
uv run --frozen ruff check src tests contracts storage media sync calibration pose reconstruction
uv run --frozen mypy
cd web
npm run typecheck
npx tsc --project ../contracts/tsconfig.json
npm run test
npm run build
```

Core pytest cases reject network connections. To verify locked dependencies are already cached before running tests, use `UV_OFFLINE=1 uv run --frozen pytest`.

The package and viewer inspect persisted motion products. Use the canonical
`tkd-poomsae run` path in the
[local runbook](local-runbook.md#canonical-offline-project-command) to wire
provisioned producers and persist inspection. Product
behavior and component ownership remain in `spec/`.

The optional MMPose environment and model setup are documented in
[`vision/README.md`](../vision/README.md). Core checks do not download models or
install the vision stack.
