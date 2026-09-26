# Camera inspection

The web workspace renders every registered camera and uses persisted synchronization
mappings from the local inspection service. Each panel separates requested global
and source time from delivered ordinal, PTS, source/global time, and sample mismatch.
Excluded/unmapped sources and times outside native coverage have explicit unavailable
states. Missing observations remain absent.

Paused seeking and native stepping use the exact PNG frame service. Delivered time
and overlays appear after that image loads and its response headers match the indexed
source hash, ordinal, and PTS. Native stepping uses a bounded 256-frame page around
the current sample; each paused seek refreshes that page. Exact image requests are
queued through two client decoder slots. No recordings are copied or estimated in
the browser.

During playback the shared clock controls video rate and corrects drift exceeding
80 ms. Presented-frame callbacks supply source time; only matching native PTS and
source identity can enable observations. A new presentation or seek invalidates
pending observations. Slow requests may leave overlays unavailable during playback.
Display refresh and skipped video frames do not add temporal evidence. Unsupported
codecs, nonzero source timeline origins, missing presented-frame callbacks, and
browser/native orientation differences require paused exact inspection. Browser
playback support does not establish frame identity on its own.

Observation points, foot/head axes, and crop provenance use display-oriented
original-image pixels, as defined by the observation producer. Images and SVGs share
one contain/letterbox transform; stored rotation is already applied by the exact
frame service and is not applied again to observations. Points use pixel-center
coordinates. Raw model scores/visibility and derived quality remain separate.
Landmarks can be selected by pointer or keyboard; the inspector identifies their
source sample and quality. Crop boxes describe original-image provenance and are
not used to rescale refined points a second time.

For browser acceptance, first run `uv sync --frozen --group dev`, then in `web/`:

```sh
npm ci
npx playwright install chromium
npm run test:browser
```

The suite starts its own fixture service on port 18044 and Vite on port 5173. It
creates two-, three-, and four-camera H.264 recordings, different PTS grids/offsets,
rotation, known colored pixels and persisted observation points in temporary storage.
It exercises native stepping, seeking, selection, resize/letterboxing, exclusion,
coverage, stale requests, identity rejection, missing observations, and playback
limitations. Generated media, storage, and browser reports are not committed.
`VITE_API_ROOT` optionally selects another loopback service for isolated testing;
production defaults to `http://127.0.0.1:8000`.
