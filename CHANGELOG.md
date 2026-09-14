# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

## [0.3.0] - 2026-09-14

- Add structured official BioImage.IO validation for directories and ZIPs, external
  JSON reports, explicit validation/error status, and optional build-time gating.
  BioImage.IO remains an optional dependency; validation uses the active environment.
- Add `ValidationReport` and `BioimageioValidationError`, CLI report/device/weight
  selection, and `build --validate-bioimageio` with an optional report path.
- Preserve the existing registry validation import while returning a structured
  report. Validation now defaults to CPU and `pytorch_state_dict`; select another
  weight representation explicitly when needed.
- Add real-validator integration coverage and documentation for report persistence,
  failure handling, trusted-code execution, and downstream training workflows.

## [0.2.0] - 2026-09-14

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

[Unreleased]: https://github.com/luiskuhn/nidavellir-tools/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/luiskuhn/nidavellir-tools/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/luiskuhn/nidavellir-tools/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/luiskuhn/nidavellir-tools/releases/tag/v0.1.0
