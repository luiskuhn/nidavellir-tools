# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

- Add Python-only Monte Carlo dropout inference with temporary dropout-only
  evaluation, streaming mean/variance/std, optional samples, and class selection.

- Add Python-only `nidavellir_tools.data_loading` readers extracted from
  NuxNet: BIA table pairing, temporary ZIP extraction, and OME-TIFF reading.
  Existing folder conventions, pixel values, calibration, and errors are preserved.

## [0.1.0] - 2026-09-13

- Extract the reusable Nidavellir tools from `nuxnet-training`.
- Adopt a `src/` Python package layout and `pyproject.toml` build.
- Add installable CLI entry points and a standalone container definition.
- Build BioImage.IO-compatible model packages with provenance and test tensors.
- Stage, inspect, validate, and publish model packages.
- Load parent packages and export fine-tuned child packages.

[Unreleased]: https://github.com/luiskuhn/nidavellir-tools/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/luiskuhn/nidavellir-tools/releases/tag/v0.1.0
