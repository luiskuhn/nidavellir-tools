# Nidavellir Tools

Nidavellir Tools packages trained PyTorch models with reproducibility metadata,
portable test tensors, and BioImage.IO-compatible artifacts. It also stages and
loads packaged models for transfer-learning runs.

The library is intentionally independent of any model architecture, training
framework, microscopy modality, or dataset. Consuming projects remain
responsible for their model, DataLoader, augmentation, model RDF, and model card.

> **Status:** the initial API may change before version 1.0.

## Installation

Install the core package from PyPI:

```bash
python -m pip install nidavellir-tools
```

Optional integrations are installed explicitly, for example:

```bash
python -m pip install "nidavellir-tools[bioimageio,huggingface,mlflow]"
```

For development from a checkout, use `python -m pip install -e ".[dev]"`.

## Command line

```bash
nidavellir --help
nidavellir build --help
nidavellir samples --help
nidavellir inspect --help
nidavellir validate --help
```

The existing command names remain available during migration:

```bash
nidavellir-build --help
nidavellir-registry --help
nidavellir-samples --help
```

## Dataset reading (unreleased)

`nidavellir_tools.data_loading` contains the readers extracted from
`nuxnet-training` at commit `3e74a613fbcdf9f7ef6a902867175060d6b7fd65`.
This addition is Python-only and requires no new dependencies.

```python
from pathlib import Path
from nidavellir_tools.data_loading import (
    VolumePair, OMEVolume, read_bia_pairs, extract_dataset_archive,
    _read_ome, _read_voxel_size_um,
)

for pair in read_bia_pairs(Path("dataset")):
    image = _read_ome(pair.image, mask=False)  # CZYX, original dtype
    mask = _read_ome(pair.annotation, mask=True)  # ZYX, original dtype
    spacing = image.voxel_size_um  # Z,Y,X in micrometers

# Keep the temporary extraction alive while consuming the files.
with extract_dataset_archive("dataset.zip") as root:
    pairs = read_bia_pairs(root)
    image = _read_ome(pairs[0].image, mask=False)
```

Exactly one directory below the supplied root must contain both `images.tsv`
and `annotations.tsv`. Paths resolve relative to that directory or its parent,
as in NuxNet's `bia/` and `data/` submission layout. Existing column aliases,
identifier/path matching, split/group fields, and OME-TIFF suffix checks remain.
These readers support this particular layout, not arbitrary BIA submissions or
BioImage.IO dataset RDF validation.

OME reading uses the first series, requires positive PhysicalSizeZ/Y/X metadata,
and retains the existing restrictions on time/sample axes and mask channels.
It does not resample, normalize, or impose binary labels. The underscore-prefixed
reader names are intentionally retained for NuxNet import compatibility.

When adopting this module, replace NuxNet's matching definitions and constants
with imports above. Keep its download, splitting, preprocessing, dataset and
Lightning classes in NuxNet. The published 0.1.0 release does not include this
module yet; install a wheel built from this checkout to test the extraction.

## Monte Carlo dropout uncertainty (unreleased)

`nidavellir_tools.uncertainty` provides application-independent PyTorch inference
utilities. It does not load checkpoints, move tensors between devices, preprocess
images, or export files. No additional CLI or dependencies are required.

```python
from nidavellir_tools.uncertainty import mc_dropout_predict, select_class_uncertainty

# model and images are already loaded and on the same device.
result = mc_dropout_predict(
    model,
    images,
    num_samples=30,
    output_transform=lambda logits: logits.softmax(dim=1),
)
segmentation = result.mean.argmax(dim=1)
uncertainty = select_class_uncertainty(result.std, segmentation, class_dim=1)
```

Supply sigmoid for single-channel binary or multilabel probabilities; use no
transform for regression outputs or already-normalized probabilities. For tuple
outputs, an adapter such as `lambda outputs: outputs[0].softmax(dim=1)` selects
the desired tensor. No activation, channel axis, or output layout is inferred.
Mean, variance, and standard deviation retain the full output shape and device.
Single-channel binary standard deviation is already an uncertainty map and needs
no class-selection step. To reproduce deterministic-class selection, compute
labels separately in ordinary evaluation mode and pass those labels instead of
the MC mean labels to `select_class_uncertainty`.

Statistics use streaming accumulation (float32, or float64 for float64 outputs).
`correction=1` is the default sample variance convention; `correction=0` selects
population variance. At least two passes are required. `return_samples=True`
additionally returns an `[MC, ...output_shape]` tensor and increases memory use
linearly with the pass count. The result also records `num_samples` and
`correction`. Seed PyTorch once before calling if repeatability is needed; never
reset the seed on every pass. Results need not be identical across hardware.

### Model requirements and integration boundaries

- Train with appropriately placed `nn.Dropout`, `Dropout1d`, `Dropout2d`, or
  `Dropout3d` modules and nonzero, nonunit probabilities. The library preserves
  these rates and does not insert dropout or modify weights. Adding dropout only
  after training is not equivalent to training with it.
- Inference enables only these dropout modules, with gradients disabled and other
  modules in evaluation mode. BatchNorm running buffers remain unchanged.
  BatchNorm configured without tracked running statistics still uses batch
  statistics during evaluation, so predictions can depend on batch composition.
- Functional dropout, AlphaDropout, internal attention/RNN dropout, and custom
  stochastic layers are not automatically supported. A supported stochastic
  dropout module is required, but its presence does not prove that a particular
  forward path uses it or guarantee nonzero output variance.
- Original module training flags are restored even on failure. Do not run
  concurrent inference/training on the same model instance, or change model modes
  inside callbacks. `mc_dropout_mode(model)` is also available as a standalone
  context manager; callers then manage their own gradient context.
- `predict_fn(model, inputs)` can adapt multiple arguments or perform one complete
  sliding-window reconstruction per MC pass. Tiling and stitching remain the
  application's responsibility. Standard dropout resamples between window
  forwards; a single shared mask over an entire volume is not enforced.
- Variability is an approximate model-uncertainty measure, not calibrated error
  probability or a complete measure of uncertainty. Validate its usefulness and
  sampling convergence on your task; 30 passes is a starting point, not a guarantee.

The sampling approach follows the earlier
[RTS uncertainty implementation](https://github.com/qbic-pipelines/rts-prediction-package/blob/main/rts_package/utils/uncertainty.py),
but deliberately preserves trained rates and enables only dropout, not full
training mode. See also [Gal and Ghahramani (2016)](https://proceedings.mlr.press/v48/gal16.html).
This module is not included in the published 0.1.0 release; use a wheel built from
this checkout until the next release.

## Model sample image axes

Sample TIFF creation supports model tensors described by explicit BioImage.IO
axes. Standard 2D and 3D layouts include `BCYX` and `BCZYX`. The exact NPY
tensors retain the complete model input/output boundary, including the batch
dimension; TIFFs are presentation samples derived from one selected batch item.

## Containerized development

```bash
docker build -t nidavellir-tools:dev .
docker run --rm nidavellir-tools:dev --help
```

Run tests without modifying the host Python environment:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  python:3.12-slim \
  bash -lc 'python -m pip install -e ".[dev]" && pytest -q'
```

## Origin

The project was extracted from
[`luiskuhn/nuxnet-training`](https://github.com/luiskuhn/nuxnet-training), where
the workflow was first exercised for 3D nuclei segmentation and parent-to-child
fine-tuning. The standalone package is derived from that implementation while
removing NuxNet-specific assumptions.
