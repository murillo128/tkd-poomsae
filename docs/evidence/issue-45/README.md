# 3D panel acceptance evidence

`rendered-panel.png` is a bounded capture of the actual Chromium WebGL panel,
not a diagram or mocked canvas. It comes from `web/browser/three.spec.ts` using
`geometryFixture.ts`: known XY-ground/Z-up skeleton, both finger chains, detailed
feet, world-parent head orientation, camera extrinsics, native root/landmark paths,
one missing finger tip, interpolated thumb points and low/unknown evidence.
The capture shows the 0.500 s native sample with shared right-leg highlighting.

Reproduce offline after installing npm dependencies and Playwright Chromium:

```sh
cd web
npm test
npm run build
npm run test:browser
```

The four 3D browser tests exercise axis identity/handedness, calibrated camera
placement, all hand edges (40 minus the edge to the missing tip), all 10 foot
edges, layer toggles against actual rendered pixels, selected paths, ray picking,
landmark selection, native cursor/timeline/inspector synchronization, missing-region text,
orbit/pan/zoom, project context release and late-response rejection. Inspector
regressions compare coordinates, confidence, native time and contributing camera
PTS after a seek; missing/off-grid instants clear old values/evidence, and delayed
entity responses cannot restore them after a seek or project change. Their API
fixtures replace only the local service transport; the production panel, scene
builder, controls and renderer execute in Chromium. No ML/data asset is loaded.

Focused scene tests also verify artifact immutability, missing-joint edge omission,
unknown styling, absence of invented poses between native instants, and resource
disposal. API integration tests verify calibration registration, bounded reads,
revision conflicts, unavailable/stale state and incompatible reconstruction frames.

Intentional limits: bounded native trajectory page (256 rows / two seconds),
16 displayed cameras, no off-grid pose interpolation, pinhole frusta without lens
distortion, no fitted mesh. Limits and missing inputs remain visible in the panel.

Local validation after integration reconciliation: 18 web unit tests, 8 browser
tests (4 3D and 4 ground), 21 focused API tests, web/portable-contract type checks,
build, Ruff and mypy passed. The Vite build
reports the existing default chunk-size warning for the renderer bundle.

Each browser project owns its Vite server and refuses to reuse unrelated running
servers. This revision was reproduced locally with `TKD_BROWSER_PORT=5185` and
`GROUND_PORT=5286`; ground evidence used `/tmp/issue45-ground-browser-evidence`.
CI uses the project-owned default ports. Integration reconciliation preserves the
ground suite and its Playwright version and includes it in normal browser validation.
