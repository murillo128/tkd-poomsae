# Real native observation acceptance

The compact reports here record detection-driven local MMPose execution of the
registered Mendeley Taegeuk 1 frontal/lateral selections. They establish runtime
integration, native timing, typed persistence, offline reload and cache reuse.
They do not measure accuracy or compare predictions with CSV annotations.

Dense observation payloads, source videos, checkpoints and the interpreter remain
in persistent shared storage under `~/.local/share/tkd-poomsae` on the exercised
host. Reported absolute paths identify that host's reusable artifacts; another
host selects its own `TKD_DATA_ROOT` and registers the dataset there.

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
