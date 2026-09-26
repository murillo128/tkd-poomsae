# Provenance and development inspection

The shared inspector reads registered entity evidence independently of the 3D
renderer. Camera landmarks expose native pixel values and raw scores; reconstructed
landmarks, placements and semantic events expose their structured values, duration
when applicable, quality, artifact origin, edit revision and generating provenance.
The entity endpoint includes coordinate units and the producer's algorithm, model,
schema, configuration, calibration and synchronization revisions. Unresolved scale
is shown as arbitrary world units. Detector scores and internal reprojection
consistency are not accuracy measurements.

Expand **Contributing camera evidence** and choose a camera/PTS to open that exact
persisted frame. This pauses playback and selects its camera while preserving the
global cursor and entity selection. The inspector shows the signed difference
between each contributing frame's effective global time and the cursor. Native
frame identity, raw scores and provenance remain available separately. Missing
contributors, unavailable frames and bounded/truncated evidence are explicit.
Contributing physical-sample links open the persisted upstream values on the same
cursor, including when native camera observations are unavailable.

**Development controls** sets the local trajectory radius shared by 3D and dynamic
ground views (0.5–10 seconds). Summary ground mode retains the whole execution.
Camera observations, 3D layers and ground analytical layers have panel controls.

Each camera's synchronization controls show its automatic estimate, effective
offset, confidence and saved edit provenance. Editing pins the expected sync
revision at the first input. Saving requires a finite offset, author and reason.
Failures preserve the draft; a conflict refreshes current artifacts without
rebasing the user's draft. Discard the draft and reload synchronization to
intentionally adopt a new edit base. Unsupported synchronization stays unavailable.

Successful sync edits clear incompatible geometry and visibly stale products.
Timeline edits use the timeline's existing optimistic editor and automatic base;
the shared panels refresh after success or conflicts without resetting global
time or losing rejected timing drafts. Reopening reads persisted edit provenance.
These controls only read persisted products or save explicit edits; they do not
run inference or fetch data/model assets.

Browser acceptance combines a real temporary local service (generated camera
clips, native observations and revision-bound reconstruction), the persisted
semantic-edit service, and deterministic ground rendering fixtures. Run
`cd web && npm run test:browser`. Large browser reports and generated media remain
outside Git. Other validation uses the normal Python checks, web unit tests,
TypeScript checks and production build.
