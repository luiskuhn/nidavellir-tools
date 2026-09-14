# Nidavellir Tools

Nidavellir Tools is a Python library and command-line toolkit for packaging,
sharing, and reusing trained PyTorch models. It combines exact inference fixtures,
model metadata, checksums, and provenance with reusable OME-TIFF dataset readers
and Monte Carlo dropout inference.

It is independent of training frameworks and model architectures. Your project
owns training, preprocessing, augmentation, scientific validation, and deployment.
The dataset reader supports a specific table-based layout rather than arbitrary
datasets; the uncertainty API operates on PyTorch modules and tensors.

> **Version:** this README describes `0.3.0`, which adds structured official
> BioImage.IO validation and optional build-time validation. Dataset reading and
> uncertainty were introduced in `0.2.0`.
> Installing `nidavellir-tools==0.1.0` will not provide these two new modules.
> The API may change before version 1.0.

## Contents

- [Installation](#installation)
- [CLI command reference](#cli-command-reference)
- [Build a reproducible model package](#build-a-reproducible-model-package)
- [Stage, inspect, and load packages](#stage-inspect-and-load-packages)
- [Export a fine-tuned child package](#export-a-fine-tuned-child-package)
- [Validate and publish](#validate-and-publish)
- [Dataset reading](#dataset-reading)
- [Monte Carlo dropout uncertainty](#monte-carlo-dropout-uncertainty)
- [Sample TIFFs and tensor axes](#sample-tiffs-and-tensor-axes)
- [Security and operational limitations](#security-and-operational-limitations)
- [Development and Docker](#development-and-docker)

## Installation

Requires Python **3.10 or newer**. The core dependency ranges are NumPy
`>=1.26,<3`, PyYAML `>=6,<7`, tifffile `>=2024.8`, and PyTorch `>=2.4,<3`.
Use an isolated environment and install the PyTorch build appropriate for your
hardware before installing this package when GPU support is needed.

Install the released package from PyPI:

```bash
python -m pip install nidavellir-tools
```

The distribution name is `nidavellir-tools`; Python imports use `nidavellir_tools`.
Optional integrations are explicit:

| Extra | Used for |
| --- | --- |
| `bioimageio` | Official BioImage.IO validation via `bioimageio.core` |
| `huggingface` | Hugging Face downloads and model-repository uploads |
| `mlflow` | Downloading run artifacts through MLflow |
| `dev` | Tests, linting, wheel/sdist builds, and metadata checks |

```bash
python -m pip install "nidavellir-tools[bioimageio,huggingface,mlflow]"
```

For development, install from a checkout:

```bash
python -m pip install -e ".[dev]"
```

For a non-editable installation, build a wheel with `python -m build` and install
that wheel in the consuming environment. Replacing an installation with a
development wheel carrying the same version may require `--force-reinstall`.
Do not mistake a development wheel for the published release. Pin the eventual
released version in production dependency files.

## CLI command reference

| Command | Purpose | Python equivalent |
| --- | --- | --- |
| `nidavellir build` | Reconstruct, test, and package model weights | `build_model_package.build_model_package` |
| `nidavellir samples` | Regenerate presentation TIFFs from staged NPY files | `create_sample_tensors.create_samples` |
| `nidavellir stage` | Copy/download and unpack a package into a stable directory | `model_package_registry.stage` |
| `nidavellir inspect` | Print RDF metadata and check declared hashes | `model_package_registry.verify` |
| `nidavellir load` | Reconstruct a model; optionally export weights and metadata | `model_package_registry.load_package` |
| `nidavellir export-child` | Package fine-tuned weights with parent lineage | `model_package_registry.export_child_package` |
| `nidavellir validate` | Check hashes and run official BioImage.IO tests | `validation.validate_bioimageio` |
| `nidavellir publish-hf` | Upload a package to a Hugging Face model repository | `model_package_registry.publish_huggingface` |

Module paths in the table are relative to `nidavellir_tools`. Dataset reading and
uncertainty are **Python-only**; there are no corresponding CLI commands.

```bash
nidavellir --help
nidavellir build --help
nidavellir samples --help
nidavellir inspect --help
nidavellir validate --help
```

The standalone entry points are equivalent alternatives:

```bash
nidavellir-build --help
nidavellir-registry --help
nidavellir-samples --help
```

Use `nidavellir-registry stage ...` for registry subcommands; use
`nidavellir-build ...` and `nidavellir-samples ...` without an extra subcommand.

## Build a reproducible model package

### 1. Describe the model contract

Your project supplies a BioImage.IO model RDF YAML specification, reconstructible
architecture, environment declaration, and model card. A starting point is the
[example specification](src/nidavellir_tools/examples/model-package.example.yaml)
and [model-card template](src/nidavellir_tools/model-card-template.md).

**Important:** remove the example's `weights.torchscript` block. The current
builder accepts only `weights.pytorch_state_dict`. The legacy `--trace-input`
option is accepted but currently unused; it does not generate TorchScript.
The example is a template, not a ready-to-publish model specification.

Replace placeholder authors, maintainers, citations, license, model description,
version, channel names, axes, shapes, and preprocessing/postprocessing with your
actual model contract. Set `pytorch_version` and environment dependencies to the
versions used for reconstruction. Architecture can be declared as either:

```yaml
architecture:
  source: src/model_architecture.py
  callable: Model
  kwargs: {input_channels: 1, output_channels: 1}
```

or an installed callable such as `your_package.models:Model` without `source`.
For the latter, the consuming environment must have that package installed.
Local architecture, dependency, cover, and sample paths are relative to the
specification. A copied architecture file must be self-contained or have its
imports satisfied by the target environment; its import dependencies are not
automatically bundled.

### 2. Capture artifacts from a trained model in Python

Call this at the end of training with the **bare inference model**, not a trainer
wrapper. `trained_model` and `example_input` below are supplied by your project:

```python
from pathlib import Path
from nidavellir_tools.run_artifacts import prepare_model_package_artifacts

run_dir = prepare_model_package_artifacts(
    model=trained_model,
    sample_input=example_input,
    cli_parameters={"seed": 42, "epochs": 100},
    specification_path=Path("model-package.yaml"),
    output_dir=Path("artifacts/run-001"),
    model_card_path=Path("MODEL_CARD.md"),
    sample_layout="auto",
    sample_batch_index=0,
    sample_channel_policy="squeeze-singleton",
)
```

This writes `weights.pt`, `test-input.npy`, `test-output.npy`, presentation TIFFs,
`model-package.yaml`, `README.md`, `cli-parameters.json`, and
`run-provenance.json`, plus copied architecture/environment/cover artifacts.
The provenance includes tensor shapes/dtypes, the weight digest, a Git commit
when available, and sample-generation settings. Optional `provenance` accepts a
JSON path or mapping and is recorded as parent-model information.

The helper calls `model.eval()` and **leaves it in evaluation mode**. It evaluates
one tensor input and expects one tensor output. The NPY pair captures the direct
model boundary, without external preprocessing or postprocessing. For a logits
model, save logits—not thresholded labels—as the expected output. The input must
have the correct dtype and shape; it is moved to the model parameter device for
evaluation. Image sample generation requires batch, channel, X and Y axes, with
optional Z. Use the conventional test filenames shown above in this staged RDF.

### 3. Build from those artifacts

```bash
nidavellir build \
  --run-artifacts-dir artifacts/run-001 \
  --output-dir packages/model-v1
nidavellir inspect packages/model-v1
```

The builder reconstructs the declared model on CPU, strictly loads its weights,
and compares its output with the supplied NPY fixture (`rtol=1e-4`, `atol=1e-5`,
matching shapes required). It then creates the package directory and adjacent
`packages/model-v1.zip`. Use CPU-compatible model code and test inputs.

Typical package contents are:

```text
model-v1/
├── rdf.yaml                   # Final metadata and artifact hashes
├── weights.pt                 # Tensor-only state dictionary
├── test-input.npy             # Exact model input, including batch axis
├── test-output.npy            # Exact direct model output
├── sample-input.tif           # Presentation image from one batch item
├── sample-output.tif
├── README.md                  # Model card, not this library's README
├── provenance.json
├── SHA256SUMS
├── cli-parameters.json
└── ... architecture, environment and cover files declared by the RDF
```

Artifact names follow your RDF; these are the conventional defaults.
`--run-artifacts-dir` cannot be combined with explicit specification, checkpoint,
test-input/output, model-card, or provenance arguments.

### Alternative: build from existing files

```bash
nidavellir build \
  --specification model-package.yaml \
  --checkpoint checkpoints/best.ckpt \
  --state-dict-key state_dict \
  --strip-prefix model. \
  --test-input test-input.npy \
  --test-output test-output.npy \
  --model-card MODEL_CARD.md \
  --output-dir packages/model-v1
```

Omit `--state-dict-key` and `--strip-prefix` for an already bare state dictionary.
Prefix handling **filters** to matching keys and removes that prefix; choose the
prefix for your inference network, not the entire training wrapper. Although
test-input/output flags are repeatable, current inference verification supports
exactly **one input and one output**. Supply pre-generated sample TIFFs beside
the specification at the declared paths. `--extra-file` can be repeated to copy
additional files to the package root; avoid filename collisions.

Equivalent library call using the staged artifacts:

```python
from pathlib import Path
from nidavellir_tools.build_model_package import build_model_package

run_dir = Path("artifacts/run-001")
package_dir, archive_path = build_model_package(
    specification=run_dir / "model-package.yaml",
    checkpoint=run_dir / "weights.pt",
    test_input=run_dir / "test-input.npy",
    test_output=run_dir / "test-output.npy",
    model_card=run_dir / "README.md",
    output=Path("packages/model-v1"),
    provenance=run_dir / "run-provenance.json",
    extra_files=[run_dir / "cli-parameters.json"],
)
```

Choose either the CLI or Python build for a given destination; existing nonempty
destinations are refused unless overwrite is explicitly requested.

## Stage, inspect, and load packages

Stage a directory or ZIP before loading it. The source must contain exactly one
discoverable `rdf.yaml` (or have one at its root).

```bash
nidavellir stage packages/model-v1.zip staged/model-v1
nidavellir inspect staged/model-v1
nidavellir load staged/model-v1 \
  --representation pytorch_state_dict \
  --weights-output initialization/weights.pt \
  --metadata-output initialization/metadata.json
```

Remote source forms (replace the placeholders with real identifiers):

```bash
nidavellir stage https://example.org/model.zip staged/from-http
nidavellir stage hf://organisation/model-repository staged/from-hf --revision COMMIT_SHA
nidavellir stage mlflow://RUN_ID/model-artifacts staged/from-mlflow
```

HTTP sources must resolve to a ZIP, not an HTML landing page. Hugging Face and
MLflow use their respective optional extras and existing authentication/configuration.
`--revision` applies to Hugging Face downloads. Pin a commit for reproducible
retrieval. Staging copies files and checks RDF-declared hashes; it does not install
the model's environment.

```python
from pathlib import Path
import torch
from nidavellir_tools.model_package_registry import stage, verify, load_package

root = stage("packages/model-v1.zip", Path("staged/python-model"))
rdf = verify(root)
loaded = load_package(root, representation="pytorch_state_dict")
model = loaded.model  # CPU, evaluation mode

# Supply your own prepared tensor with the declared shape and dtype.
with torch.inference_mode():
    prediction = model(example_input.cpu())

metadata = loaded.rdf
lineage = loaded.provenance
# For fine-tuning, call model.train() and create your optimizer in your project.
```

`LoadedModelPackage` contains `model`, `rdf`, `provenance`, `root`, and
`representation`. `load(root, representation=...)` returns just the model.
`auto` prefers TorchScript if present, otherwise state-dict reconstruction.
Choose `pytorch_state_dict` explicitly for fine-tuning, initialization weight
export, or MC dropout on ordinary Python modules. Loading does not execute RDF
preprocessing/postprocessing or automatically apply tiling or activation functions.

## Export a fine-tuned child package

Use this when new weights retain the parent's architecture and tensor contract:

```bash
nidavellir export-child staged/model-v1 checkpoints/finetuned.pt packages/model-v2 \
  --test-output child-test-output.npy \
  --model-card CHILD_MODEL_CARD.md \
  --version 0.2.0 \
  --parent-identifier https://example.org/model-v1
```

```python
from pathlib import Path
from nidavellir_tools.model_package_registry import export_child_package

child = export_child_package(
    parent=Path("staged/model-v1"),
    checkpoint=Path("checkpoints/finetuned.pt"),
    test_output=Path("child-test-output.npy"),
    model_card=Path("CHILD_MODEL_CARD.md"),
    destination=Path("packages/model-v2"),
    version="0.2.0",
    parent_identifier="https://example.org/model-v1",
)
```

The version here is the **model's version**, not the `nidavellir-tools` version.
This strictly checks state-dict architecture compatibility, replaces the first
output test tensor and model card, retains the parent input/architecture metadata,
records lineage, removes other weight representations, and creates a ZIP.
Checkpoint-key/prefix options are also supported.

Generate `child-test-output.npy` by evaluating the child on the retained parent
test input. Unlike the main builder, child export does **not** compare that output
numerically with a forward pass. It also retains presentation samples from the
parent; refresh affected samples and their RDF hashes or rebuild from newly staged
artifacts when necessary. Validate the child before publishing. If architecture,
channels, preprocessing, or the model contract changes, use a new specification
and the main builder instead of this shortcut.

## Validate and publish

`inspect`/`verify` check that the RDF describes a model and verify local `source`
entries that declare SHA-256 hashes. They do not validate the complete schema,
check every file against `SHA256SUMS`, or establish scientific correctness.

### Structured official validation

There are three distinct levels of checking:

| Check | What it establishes | What it does not establish |
| --- | --- | --- |
| `inspect` / `verify` | RDF identifies a model; files with declared hashes match | Full specification compliance or successful inference |
| `build` | Strict model reconstruction reproduces the supplied raw output fixture | Compatibility with the complete BioImage.IO processing contract |
| `validate` / `build --validate-bioimageio` | Official metadata and inference tests pass in the selected environment | Scientific accuracy, environment recreatability, or Zoo acceptance |

Internally, validation computes the input digest, prepares a temporary snapshot,
checks local integrity, calls the official validator, and returns a
`ValidationReport`. The optional JSON file is written before a validation-failure
exception is raised. The original package is never intentionally edited.

The structured reports and build flags below require version **0.3.0 or newer**.
Version 0.2.0 provides the earlier `nidavellir validate` CLI wrapper, without
structured reports or build-time validation. Install the extra in the same
environment as the model's runtime dependencies.

For official BioImage.IO testing, install the extra and validate a directory or ZIP:

```bash
python -m pip install "nidavellir-tools[bioimageio]==0.3.0"
nidavellir validate packages/model-v1 --report reports/model-v1.json
nidavellir validate packages/model-v1.zip --report reports/model-v1-zip.json
```

The command checks declared artifact hashes and calls the official
`bioimageio.core.test_description` API for metadata and dynamic inference tests.
It defaults to `pytorch_state_dict` weights on CPU in the **currently active
environment**. Choose a representation with `--weight-format torchscript` or a
device with `--device cuda:0` (repeatable). The selected representation must exist.
The validator uses the model's declared processing contract; this is distinct
from the builder's direct raw-input/raw-output comparison.

A passing result requires the official summary status to be `passed`: format-only
validity is not sufficient. Failed checks and execution errors both cause a nonzero
CLI exit status. Reports distinguish `failed` (integrity or official checks) from
`error` (input, missing dependencies, or execution problems). Inspect both the
`error` field and official `summary.details` when diagnosing a failure.

Each JSON report contains schema version 1, status, original package path,
SHA-256 identity, timestamp, Python/platform and library versions, requested
settings, and the official JSON summary including its warnings/errors. ZIP
identity hashes the archive bytes. Directory identity hashes a sorted list of
`[relative POSIX path, file SHA-256]` pairs encoded as compact ASCII-escaped JSON
in UTF-8; it excludes directory metadata and is not expected to equal a ZIP hash.

Reports must be **outside** the input package and cannot overwrite existing files.
Each validation operates on a temporary copy, leaving input artifacts unchanged.
Directories with internal symlinks are rejected. ZIP extraction rejects traversal;
no download, extraction-size quota, or security sandbox is provided. Copying and
hashing large packages requires additional disk space and I/O.

### Python API and build-time gate

```python
from nidavellir_tools.validation import validate_bioimageio, BioimageioValidationError

try:
    report = validate_bioimageio(
        "packages/model-v1.zip",
        report_path="reports/model-v1.json",
        devices=["cpu"],
    )
    print(report.status, report.package_sha256)
except BioimageioValidationError as exc:
    # The requested report is saved before this exception is raised.
    print(exc.report.to_dict())
    raise
```

Use `raise_on_failure=False` to inspect a returned report without an exception;
the caller must then check `report.status == "passed"`. Invalid report locations,
existing reports, and report-write errors still raise directly. Existing imports
from `model_package_registry.validate_bioimageio` remain supported.

For an opt-in package-build gate:

```bash
nidavellir build \
  --run-artifacts-dir artifacts/run-001 \
  --output-dir packages/model-v1 \
  --validate-bioimageio \
  --validation-report reports/model-v1-build.json
```

The Python builder accepts `validate_bioimageio=True` and optional
`validation_report=Path(...)`. The default report is adjacent to the output,
named `<output-directory-name>.validation.json`. A report path without validation
enabled is rejected. Validation runs after assembly and local checks, **before ZIP
creation**. On failure, the directory and diagnostic report remain for inspection
but no new ZIP is created. A ZIP left over from a previous run is not removed;
use fresh output/report paths and never treat a failed build's old archive as new.
Successful validation does not change the builder's `(directory, archive)` return.

Keep this opt-in for routine training exports; require it before production
publication. Neither `export-child` nor `publish-hf` automatically invokes it.
Validate the exact final package after any edits, and associate the external report
with that package digest. This toolkit does not automatically submit models to
the BioImage Model Zoo or deposit datasets in BioImage Archive.

### Environment and trust boundaries

Validation executes model code and may perform network I/O for referenced
resources. Run only trusted packages, preferably in a disposable container with
appropriate network/device limits. The current API runs in-process and official
testing may affect framework/RNG state; do not run it concurrently with training
in the same process. A separate validation CLI/container is preferable.

The optional `bioimageio` extra remains outside core dependencies. Model runtime
dependencies must already be installed. This first implementation does not create
Conda environments or prove that the declared environment can be recreated; that
is a separate portability test. CPU success does not imply GPU support. Reports
identify the installed validator versions because results can vary by version.
No validation result establishes scientific quality, calibrated uncertainty,
absence of data leakage, or automatic acceptance by the Model Zoo.

### Using validation from a training project

Training projects do not need to change their model or data-loader code to use
these checks. Update the project's dependency pin to 0.3.0 or newer and install
its `bioimageio` extra in the packaging/validation
environment, and rebuild any container image. Core-only training environments can
remain unchanged if validation runs in a separate environment.

For an image that already includes the new API, the extra, and your model's
dependencies, mount the package read-only and persist the report separately:

```bash
mkdir -p reports
docker run --rm \
  --entrypoint nidavellir \
  -v "$PWD/packages/model-v1:/package:ro" \
  -v "$PWD/reports:/reports" \
  your-training-image \
  validate /package --report /reports/model-v1.json
```

Use a fresh report filename for another attempt. A report left only in a disposable
container is lost when that container exits. Retain reports with the corresponding
package archives, outside their contents.

Existing `bioimageio test ...` instructions remain valid if a project prefers the
upstream CLI directly. Replacing them with `nidavellir validate ... --report ...`
adds the structured report and consistent local checks. Merely upgrading the
dependency does not enable validation automatically: add the validation command
or opt in with `build --validate-bioimageio`. Validate final parent and child
packages before publication, and repeat validation after editing package files.

### Publish to Hugging Face

To upload to Hugging Face, configure credentials with repository write access
through the Hugging Face client, then explicitly run:

```bash
python -m pip install "nidavellir-tools[huggingface]"
nidavellir publish-hf packages/model-v1 organisation/model-repository --private
```

```python
from pathlib import Path
from nidavellir_tools.model_package_registry import (
    verify, validate_bioimageio, publish_huggingface,
)

package = Path("packages/model-v1")
verify(package)
validate_bioimageio(package)
# This creates/reuses a remote repository and uploads files.
result_url = publish_huggingface(package, "organisation/model-repository", private=True)
```

Publishing verifies hashes but does **not** automatically run BioImage.IO tests.
`--revision` selects the upload revision (default `main`); `--commit-message`
customizes the upload message. Without `--private`, newly created repositories
are public. `--private` is not a visibility-change operation for existing repositories.
Never include credentials, private training data, or unwanted artifacts in the
package directory: the upload sends its folder contents.

## Dataset reading

`nidavellir_tools.data_loading` contains the readers extracted from
`nuxnet-training` at commit `3e74a613fbcdf9f7ef6a902867175060d6b7fd65`.
This addition is Python-only and requires no new dependencies.

Supported submission layout, with paths in the tables relative to `dataset/`:

```text
dataset/
├── bia/
│   ├── images.tsv
│   └── annotations.tsv
└── data/
    ├── image-001.ome.tif
    └── mask-001.ome.tif
```

`images.tsv` (columns are separated by literal tabs):

```text
image_id	filename	split	resolution_group
image-001	data/image-001.ome.tif	train	group-a
```

`annotations.tsv`:

```text
image_id	filename
image-001	data/mask-001.ome.tif
```

The same tables can live beside the image files with filenames relative to that
directory. `read_bia_pairs(root)` returns a list of immutable `VolumePair`
records: `image_id`, image/annotation `Path`s, optional `split`, and optional
`group`. It resolves metadata and file paths without loading image pixels.

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

Specifically, `_read_ome(path, mask=...)` returns an `OMEVolume` with `array`
and `voxel_size_um`. Non-singleton T/S axes are rejected, masks must have only
one channel, and missing Z is inserted as a singleton dimension. Images without
C gain a singleton channel axis. Calibration is still required for all three
spatial axes, including singleton Z. Accepted units include µm/um, nm, mm, and
their micrometer/nanometer/millimeter spelling variants; missing units default
to µm. No image/mask shape or spacing consistency check is performed across a pair.

Duplicate/ambiguous references, unknown annotation sources, multiple annotations
per image, missing files, unsupported suffixes, and empty pair results raise
errors. Images without annotations are not returned. Optional `split` and
`resolution_group` (or `subset`) are metadata only: the reader does not partition
the dataset. ZIP extraction rejects path traversal but does not impose archive
size limits. Use trusted datasets and keep extraction alive until all readers or
DataLoader workers finish accessing their files.

When adopting this module, replace NuxNet's matching definitions and constants
with imports above. Keep its download, splitting, preprocessing, dataset and
Lightning classes in NuxNet. This module requires version 0.2.0 or newer.

## Monte Carlo dropout uncertainty

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
This module requires version 0.2.0 or newer.

### Lower-level context and custom prediction

Use the context manager for your own inference loop:

```python
import torch
from nidavellir_tools.uncertainty import mc_dropout_mode

with mc_dropout_mode(model), torch.inference_mode():
    first = model(images)
    second = model(images)
# All original module training flags are restored here.
```

Multiple inputs and tiled reconstruction can be adapted without changing the core:

```python
result = mc_dropout_predict(
    model,
    (images, auxiliary_input),
    predict_fn=lambda net, inputs: net(*inputs),
    output_transform=lambda outputs: outputs[0].softmax(dim=1),
    num_samples=30,
    correction=0,
    return_samples=True,
)
```

Here the application supplies a model accepting two inputs and returning a tuple.
Each transformed prediction must be a finite floating-point tensor with constant
shape, device, and dtype. The default call is simply `model(inputs)`; tuples are
not unpacked automatically. `MCPrediction` exposes `mean`, `variance`, `std`,
`num_samples`, `correction`, and `samples` (`None` unless requested).
For `select_class_uncertainty`, labels must be int64, on the same device, and have
the output shape with only the class dimension removed. Class indices must be in
range. Negative `class_dim` values are supported.

## Sample TIFFs and tensor axes

Sample TIFF creation supports model tensors described by explicit BioImage.IO
axes. Standard 2D and 3D layouts include `BCYX` and `BCZYX`. The exact NPY
tensors retain the complete model input/output boundary, including the batch
dimension; TIFFs are presentation samples derived from one selected batch item.

Regenerate them from an existing staged run directory without loading the model:

```bash
nidavellir samples \
  --run-artifacts-dir artifacts/run-001 \
  --sample-layout auto \
  --sample-batch-index 0 \
  --sample-channel-policy preserve
```

This reads `model-package.yaml` and the declared NPY test tensors, overwrites the
declared sample TIFFs, and updates `sample_tensors` in `run-provenance.json`.
Run it **before** packaging so the builder captures the new sample hashes.

For a single array in Python:

```python
from pathlib import Path
import numpy as np
from nidavellir_tools.sample_tensors import create_sample_tensor

record = create_sample_tensor(
    np.load("artifacts/run-001/test-input.npy", allow_pickle=False),
    axes=["b", "c", "y", "x"],  # Must describe this array's actual axis order.
    destination=Path("preview/sample.tif"),
    layout="auto",
    batch_index=0,
    channel_policy="squeeze-singleton",
)
```

`auto` follows RDF axes and reorders them for TIFF storage. Explicit `bcyx` and
`bczyx` require exactly those declared axis orders. Output is CYX/CZYX, or YX/ZYX
when `squeeze-singleton` removes a channel of size one. `preserve` retains that
channel. Pixel values and spatial dimensions are unchanged; no rescaling, color
mapping, or segmentation postprocessing is applied. These presentation TIFFs do
not automatically gain the physical calibration required by the dataset reader.

## Security and operational limitations

- **Only load trusted model packages.** Reconstructing packaged architectures
  executes Python code; TorchScript loading also crosses a trust boundary.
  Checksums detect changes relative to metadata, not malicious content or author
  authenticity. Do not treat the package tools as a sandbox.
- Use trusted specifications and fresh, dedicated output directories. The tools
  do not validate every possible artifact path or impose download/extraction size
  quotas. Dataset table paths may be absolute and are not confined to the dataset root.
- `--overwrite` on build/stage/export-child deletes and recreates a nonempty
  destination. The Python functions expose `overwrite=True`; artifact capture
  instead reads `overwrite_model_package`/`overwrite` from `cli_parameters`.
  Never use an important source directory as the output. Adjacent ZIP files may
  be replaced when packages are built; partial output may remain after a failure.
- Package metadata and environment declarations are not an environment installer.
  Install architecture dependencies yourself, then validate in the environment
  you plan to distribute. Local verification does not run RDF transforms.
- Provenance and hashes support FAIR-oriented workflows, but do not by themselves
  guarantee FAIR compliance, reproducibility on every device, scientific validity,
  a complete RO-Crate, or acceptance by a model repository.

Common problems:

| Symptom | Check |
| --- | --- |
| Missing `data_loading` or `uncertainty` import | Install version 0.2.0 or newer; remove any vendored package shadowing it |
| State-dict missing/unexpected keys | Architecture kwargs, checkpoint key, and prefix filtering must match the bare network |
| Test output mismatch | Use the exact raw model input/output, evaluation mode, matching dtype, and compatible dependencies |
| Missing sample TIFF | Generate it at the RDF-declared path beside the specification before building |
| Checksum mismatch | Files changed after packaging; rebuild and validate rather than ignoring the mismatch |
| Invalid OME calibration | Provide positive PhysicalSizeX/Y/Z metadata and supported units |
| MC dropout rejected or zero variance | Use trained supported dropout modules; check that the forward path actually uses them |

## Development and Docker

```bash
docker build -t nidavellir-tools:dev .
docker run --rm nidavellir-tools:dev --help
```

The image includes the BioImage.IO and Hugging Face extras. Mount a working
directory explicitly for package operations:

```bash
docker run --rm \
  -v "$PWD:/work" -w /work \
  nidavellir-tools:dev inspect packages/model-v1
```

The image entry point is already `nidavellir`; do not repeat it after the image
name. GPU configuration and model-specific dependencies remain your responsibility.

Run tests without modifying the host Python environment:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  python:3.12-slim \
  bash -lc 'python -m pip install -e ".[dev]" && ruff check src tests && pytest -q && python -m build && twine check dist/*.whl dist/*.tar.gz'
```

This mounts the checkout read/write and generates build artifacts in `dist/`.
Run the real-validator integration test explicitly (it is skipped in core-only tests):

```bash
python -m pip install -e ".[dev,bioimageio]"
NIDAVELLIR_TEST_BIOIMAGEIO=1 python -m pytest -q tests/test_validation_official.py
```

The integration test builds a synthetic model, checks directory and ZIP validation,
and verifies that incorrect expected outputs fail the official inference test even
when local checksums are valid.

Optional-service operations require their extras and credentials and are not
implied by the core test suite. Never place credentials in source files or images.

## Origin

The project was extracted from
[`luiskuhn/nuxnet-training`](https://github.com/luiskuhn/nuxnet-training), where
the workflow was first exercised for 3D nuclei segmentation and parent-to-child
fine-tuning. The standalone package is derived from that implementation while
removing NuxNet-specific assumptions.
