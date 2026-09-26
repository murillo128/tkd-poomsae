# Local web inspection

Run `npm ci`, then `npm run dev` with Node 22.12+ and the local service at
`127.0.0.1:8000`. The 3D panel reads registered persisted inspection products;
register the final reconstruction and optional calibrated world as described in
[the service contract](../docs/local-service.md).

The panel uses Three.js directly with XY ground and Z-up (identity presentation
conversion). Coordinates and measurements are never transformed or changed.
Native samples within a two-second local window are requested with a revision
pin, a 256-row page and cancellation on window/project changes. Windows are
anchored to whole seconds so playback reuses a bounded batch. A page truncation
is explicit; the UI does not claim a complete trajectory. At an off-grid cursor
it displays paths but no invented body pose. Select a native instant or use the
shared reconstruction stepping controls to inspect recorded geometry.

Body, both full finger chains, foot landmarks, world-parent head orientation,
resolved ground, root/selected paths and calibrated cameras are independent
layers. Missing coordinates and masks remove joints/edges. Unknown, interpolated,
inferred and low/unspecified-score evidence remain visually distinct. Missing
landmarks and unavailable head/ground/calibration information have text reasons.
Camera frusta use registered oriented image bounds with pinhole intrinsics;
lens distortion is not rendered. Camera metadata reads/display are capped at 16
with an explicit truncation notice. Arbitrary-scale geometry is labeled accordingly.

Orbit with left drag, pan with right drag, zoom with the wheel. Click a joint or
use the landmark chooser to select its persisted entity ID in the shared inspector
and highlight its physical track. Action buttons select persisted action IDs and
seek their interval start; selected paths remain within the returned window and
selected action interval. Inspector evidence is fetched from the revision-pinned
entity endpoint. Controls/observers/listeners/materials/geometries are disposed,
and the WebGL context is released when changing projects or unmounting.

Validation: `npm test`, `npm run build`, then `npx playwright install chromium`
and `npm run test:browser`. Browser tests use local synthetic API responses and
actual Chromium WebGL rendering without model/data assets or external services.
The browser suite captures `test-results/**/rendered-panel.png`; CI retains it
as the `web-browser-evidence` artifact. Compact recorded evidence is in
[docs/evidence/issue-45](../docs/evidence/issue-45/README.md).
