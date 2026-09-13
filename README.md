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
