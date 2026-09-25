# Shared test media selections

`datasets/mendeley-bjy7vr4xkt-v1/selections.json` pins three named selections
to the registered version-1 source SHA-256 hashes, camera IDs, and measured
native PTS ranges:

- `smoke-short`: both Taegeuk 1 views from 5 to 7 seconds;
- `demo-full`: both Taegeuk 1 views from 0 through the common final PTS;
- `all-forms`: all eight paired forms, each through its common final PTS.

These are virtual windows. `tkd-poomsae datasets selections NAME` resolves
source paths from the shared registered projects and validates hashes and time
bounds. It does not create clips or fetch the dataset. Set one `TKD_DATA_ROOT`
for every worktree and run `tkd-poomsae datasets bootstrap
mendeley-bjy7vr4xkt-v1` followed by `tkd-poomsae datasets register` once on
that host. Thereafter selection resolution works offline.

For a specific codec, rate, size, offset, pre/post-roll, dropped-frame, or
corrupt-tail input, write a version-1 JSON recipe using the fields of
`VariantRecipe` and run:

```sh
tkd-poomsae datasets variant smoke-short mendeley-bjy7vr4xkt-v1-taegeuk-1 mendeley-frontal --recipe recipe.json
```

The recipe and original source hash identify one directory under shared
`derived/media-variants/`. A file lock serializes producers; a completion
receipt binds the clip and manifest hashes, and later callers verify both
before reuse. The manifest records each output
frame's actual PTS and corresponding original ordinal, PTS, and time base.
Recipes never alter originals. The output removes audio, and any generated
view remains the **same camera evidence** as its original. Pre/post-roll may
stop at the source boundary. Requested rates above native evidence are
rejected; temporal interpolation is not used. Intentional offsets test
transformation handling. The original pair has no exact alignment oracle;
absolute synchronization tests use synthetic fixtures. Three- and four-view
geometry tests must use distinct synthetic camera projections.

Default `pytest` runs synthetic tests only and prints the number of test calls
run in each suite. Explicitly select local or full data with `pytest -m
local_data` or `pytest -m full_data`; missing local registration fails with a
bootstrap hint. `model_required` is reserved for tests that actually use the
provisioned model runtime. The report prints zero when a suite did not run.
