# Real native observation acceptance

The compact reports here record detection-driven local MMPose execution of the
registered Mendeley Taegeuk 1 frontal/lateral selections. They establish runtime
integration, native timing, typed persistence, offline reload and cache reuse.
They do not measure accuracy or compare predictions with CSV annotations.

Dense observation payloads, source videos, checkpoints and the interpreter remain
in persistent shared storage under `~/.local/share/tkd-poomsae` on the exercised
host. Reported absolute paths identify that host's reusable artifacts; another
host selects its own `TKD_DATA_ROOT` and registers the dataset there.

## Observed results

| Selection | Native frames | Windows | Usable body: frontal / lateral |
| --- | ---: | ---: | ---: |
| smoke-short | 122 | 4 | 54 / 25 |
| demo-full | 1,536 | 48 | 255 / 240 |

Both final recipes reloaded offline and repeated from another cwd with zero
inference and network calls. The full CPU run took 1,111.395 seconds and invoked
the detector and wholebody model 1,536 times each, plus 2,624 uncached hand calls.
It selected a practitioner in 1,530 frames; the six remaining frames retain
missing/ambiguous selection. Detailed hand states include 1,398 low-evidence,
1,637 refined and 25 skipped observations. Region usability can still be false
for a refined hand, for example when side identity is ambiguous.

The prerequisite acceptance had left its assets and interpreter under
`/tmp/tkd-issue10-assets`. The host still had all 14 registry assets and the
locked package cache. Recovery reused those bytes through the supported asset
bootstrap, verified their hashes, and provisioned the shared interpreter from
the vision/media locks. No model or dataset was placed in this worktree, and
the real acceptance harness attempted no network connection.

## Reproduce

Provision the shared interpreter and models with the explicit commands in
[vision/README.md](../../../vision/README.md), and register the dataset using
[the dataset bootstrap instructions](../../dataset-bootstrap.md). From a checkout:

```sh
export TKD_DATA_ROOT="${TKD_DATA_ROOT:-$HOME/.local/share/tkd-poomsae}"
export PYTHONPATH="$PWD/src:$PWD"
"$TKD_DATA_ROOT/runtime/vision/bin/python" -m pose.acceptance smoke-short \
  --output "$TKD_DATA_ROOT/runs/observation-acceptance/smoke-short.json"
"$TKD_DATA_ROOT/runtime/vision/bin/python" -m pose.acceptance demo-full \
  --output "$TKD_DATA_ROOT/runs/observation-acceptance/demo-full.json"
```

The harness denies socket connections, counts actual detector/wholebody/hand
calls, verifies every published window offline, then repeats the selection from
a fresh temporary cwd and requires zero additional inference or network calls.
A previously completed selection is a cache hit even on the first harness pass;
the checked-in reports capture the initial inference for their recipe identities.
`smoke-short-initial.json` records the first score-domain-corrected smoke, including
257 real hand calls. `smoke-short.json` records the final tracking-origin recipe;
it reused those verified hand-crop artifacts while executing detector and
wholebody inference again. The full demo also reuses existing hand crops where
their content and model/configuration identities match.
Window size is 32 native frames; models remain loaded for each camera session.
Tracking origin is part of each window's identity because preceding frames affect
subject and side continuity. Sync offsets are absent from native inference keys.

## Inspection

[smoke-regions.svg](smoke-regions.svg) shows the retained derived pixel landmarks
at 6.0 seconds for each camera on a blank canvas, with per-region evidence status.
Each camera uses its own coordinate frame, so the panels do not imply shared 3D
geometry. Missing landmarks are omitted; source predictions and masks remain
available in the external window payloads. Small or occluded hand regions remain
low evidence, even when coarse wholebody points exist.

CPU was explicitly selected. The host's NVIDIA driver was unavailable; CUDA was
not exercised. Low-evidence frames are retained rather than discarded. Usable
body evidence is required in each view for successful integration acceptance;
that condition is not an accuracy threshold or a technique judgement.
