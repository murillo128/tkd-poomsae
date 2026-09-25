# Optional vision runtime

This Linux x86_64, Python 3.11 CPU reference environment is separate from the
application's `uv.lock` and CI checks. It pins PyTorch 2.1.0+cpu, torchvision
0.16.0+cpu, MMCV 2.1.0, MMEngine 0.10.3, MMDetection 3.3.0, and MMPose 1.3.2.
`requirements.lock` pins the transitive dependencies and hashes. The pinned
MMCV wheel is built for PyTorch 2.1.0. CUDA wheels must be resolved and tested
against the actual host driver and GPU before claiming CUDA support.

From the repository root:

```sh
uv venv --python 3.11 .venv-vision
uv pip install --python .venv-vision/bin/python pip setuptools wheel
uv pip sync --python .venv-vision/bin/python vision/requirements.lock --no-build-isolation
export PYTHONPATH=src:.
export TKD_DATA_ROOT=/absolute/shared/data/root
.venv-vision/bin/python -m tkd_poomsae.cli doctor
.venv-vision/bin/python -m tkd_poomsae.cli models bootstrap
.venv-vision/bin/python -m tkd_poomsae.cli models smoke --input /absolute/local/image.png --device cpu
```

The legacy `chumpy` source package imports `pip` in its build script without
declaring it, so the bootstrap installs `pip`, `setuptools`, and `wheel` before
the locked sync. The `--no-build-isolation` setting lets that source build see
them. The vision interpreter runs repository source via `PYTHONPATH`; it does
not install the core project, whose NumPy 2 requirement would conflict with
this PyTorch 2.1 environment.

`models bootstrap` is the only model download command. The registry in
`src/tkd_poomsae/vision/registry.json` records exact OpenMMLab source commits,
config/checkpoint URLs, full SHA-256 values, and topologies. All files are
published under `${TKD_DATA_ROOT}/models/` after hash verification, a shared
process lock, and atomic rename. A complete cache hit reads no network.
Inference verifies all assets and passes only local config and checkpoint paths
to the OpenMMLab APIs. The original official training configs contain remote
backbone initialization URLs; inference explicitly clears `init_cfg` before
model construction, then loads the pinned local checkpoints.

The detector selects at most one COCO person (class 0) per image. Whole-body
inference uses the 133-point COCO-WholeBody layout: 17 body, 6 foot, 68 face,
and 21 points for each hand. Visible hand landmarks define expanded bounding
boxes that the hand model crops from the original source image at its native
resolution. The hand model yields 21 points. `smoke` forces a whole-frame box
when detection finds nobody, so it tests loading and execution rather than
prediction quality. Normal `models infer` emits no pose or hand samples when
detection finds nobody. No accuracy claim follows from the smoke result.

Use `models infer` with the same arguments for the normal detection-driven path.
CPU threads are capped at two by default and one person/one image is processed
per call. A CUDA job asks the CUDA driver for its actual visible-ordinal UUID,
verifies it against the physical `GPU-...` UUIDs from `nvidia-smi`, and obtains
a file lease under the shared models root. Only one job per GPU proceeds; waits
are bounded, cancellation is checked between stages, an exiting process releases
the lease, and CUDA OOM gives an actionable error. MIG visibility is rejected
until a reliable parent-GPU mapping is available.

The exercised host reported `nvidia-smi` unable to communicate with the NVIDIA
driver. PyTorch reported `cuda_available=false`; the model smoke and offline
repeat therefore exercised `cpu`. GPU operation was not tested here.

The OpenMMLab source repositories use Apache-2.0. The model checkpoints are
downloaded from official OpenMMLab links and kept outside Git; consult the
upstream model and training-dataset terms before redistribution. Model identities
come from the [MMPose model lists](https://github.com/open-mmlab/mmpose/tree/v1.3.2/configs)
and [MMDetection RTMDet table](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/rtmdet/README.md).
