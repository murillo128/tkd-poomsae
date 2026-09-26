Ground browser acceptance uses a small offline fixture exported from the shared
synthetic pivot scenario and the authoritative ground-view producer:

```
uv run --frozen python -m tests.fixtures.export_ground_view
cd web
npm ci
npx playwright install chromium
npm run test:browser
```

No dataset or model is downloaded. Tests mock only the HTTP transport and inspect
the real React/SVG app in Chromium. They verify XY projection and foot orientation,
fixed bounds, dynamic/summary agreement, shared selection, layer visibility,
unresolved scale and unavailable ground. The fixture retains six native snapshots
and original complete placement/pivot geometry, rather than estimating geometry
in a test or browser. Python transport tests separately cover all native rows,
gaps, pagination, revision conflicts and stale clocks.

Screenshots (`dynamic.png`, `summary.png`, `arbitrary.png`) are bounded to the
ground panel and stay outside Git, under `/tmp/tkd-ground-browser-evidence` by
default. Set `GROUND_EVIDENCE_DIR` to retain them elsewhere; `CHROMIUM_PATH` may
select a locally provisioned browser. CI installs Chromium and runs this suite.
