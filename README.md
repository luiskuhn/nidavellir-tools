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

## Image axes

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
