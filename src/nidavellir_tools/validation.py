"""Optional official BioImage.IO validation in the currently active environment.

Only trusted packages should be tested: inference executes packaged model code.
Input artifacts are copied to temporary storage; reports stay outside the input.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationReport:
    """Nidavellir envelope around the unmodified official JSON test summary."""

    status: str
    package: str
    package_sha256: str | None
    digest_kind: str
    created_utc: str
    environment: dict[str, str | None]
    settings: dict[str, Any]
    summary: dict[str, Any] | None
    error: dict[str, str] | None

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}


class BioimageioValidationError(RuntimeError):
    """Validation did not pass; inspect ``report`` for failure/error details."""

    def __init__(self, report: ValidationReport):
        self.report = report
        detail = report.error["message"] if report.error else "official tests did not pass"
        super().__init__(f"BioImage.IO validation {report.status}: {detail}")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_digest(path: Path) -> str:
    if path.is_file():
        return _file_digest(path)
    records = []
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink():
            raise ValueError("Validation does not accept symlinks inside package directories")
        if entry.is_file():
            records.append([entry.relative_to(path).as_posix(), _file_digest(entry)])
    # Directory identity is SHA256 of this UTF-8 JSON file manifest, not ZIP bytes.
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _versions() -> dict[str, str | None]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("nidavellir-tools", "bioimageio.core", "bioimageio.spec", "torch", "numpy"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def check_report_path(package: Path, report_path: Path | None) -> None:
    """Reject input mutation and accidental replacement of existing report files."""
    if report_path is None:
        return
    source, destination = package.resolve(), report_path.resolve()
    if destination == source or (not source.is_file() and destination.is_relative_to(source)):
        raise ValueError("Validation report must be outside the package")
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"Validation report already exists: {report_path}")


def validate_bioimageio(
    package: str | Path,
    *,
    report_path: str | Path | None = None,
    weight_format: str = "pytorch_state_dict",
    devices: list[str] | None = None,
    raise_on_failure: bool = True,
) -> ValidationReport:
    """Check local integrity and run official metadata and inference tests.

    Accept a local directory or ZIP; remote download and environment creation are
    deliberately not performed. Defaults to CPU and PyTorch state dictionaries.
    Requires the optional ``nidavellir-tools[bioimageio]`` extra. Runtime/model
    dependencies must already be installed. Official tests may access the network
    for referenced metadata and execute arbitrary trusted architecture code.

    Status is ``passed``, ``failed`` (integrity/official checks), or ``error``
    (dependency/execution errors). Requested JSON reports are written on failure
    too, before raising BioimageioValidationError. Invalid or existing report paths
    fail before testing. Reports are never embedded in or allowed to overwrite the
    package. A temporary snapshot is not a security sandbox.
    """
    from nidavellir_tools.model_package_registry import _rdf_root, _safe_extract, verify

    source = Path(package).resolve()
    report_path = Path(report_path) if report_path is not None else None
    check_report_path(source, report_path)
    selected_devices = list(devices) if devices is not None else ["cpu"]
    if not selected_devices:
        raise ValueError("devices must contain at least one device")
    settings = {
        "runtime_env": "currently-active",
        "weight_format": weight_format,
        "devices": selected_devices,
        "determinism": "seed_only",
        "expected_type": "model",
    }
    digest = None
    summary = None
    error = None
    status = "error"
    stage = "input"
    try:
        if not source.exists():
            raise FileNotFoundError(source)
        digest = _package_digest(source)
        with tempfile.TemporaryDirectory(prefix="nidavellir-validation-") as work:
            snapshot = Path(work) / "package"
            if source.is_dir():
                shutil.copytree(source, snapshot)
                if _package_digest(snapshot) != digest:
                    raise ValueError("Package changed while preparing validation snapshot")
            elif zipfile.is_zipfile(source):
                archive = Path(work) / "input.zip"
                shutil.copyfile(source, archive)
                if _file_digest(archive) != digest:
                    raise ValueError("Package changed while preparing validation snapshot")
                snapshot.mkdir()
                _safe_extract(archive, snapshot)
            else:
                raise ValueError("Expected a model package directory or ZIP")
            stage = "integrity"
            root = _rdf_root(snapshot)
            rdf = verify(root)
            if weight_format not in rdf.get("weights", {}):
                raise ValueError(f"Package has no {weight_format!r} weights")
            stage = "dependency"
            try:
                core = importlib.import_module("bioimageio.core")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Official validation requires nidavellir-tools[bioimageio] "
                    "and its dependencies in the active environment"
                ) from exc
            stage = "official"
            (Path(work) / "tests").mkdir()
            result = core.test_description(
                root / "rdf.yaml", **settings, working_dir=Path(work) / "tests"
            )
            summary = result.model_dump(mode="json")
            status = "passed" if summary.get("status") == "passed" else "failed"
    except Exception as exc:
        status = "failed" if stage == "integrity" else "error"
        error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    report = ValidationReport(
        status=status,
        package=str(source),
        package_sha256=digest,
        digest_kind="directory-file-manifest-sha256" if source.is_dir() else "file-sha256",
        created_utc=datetime.now(timezone.utc).isoformat(),
        environment=_versions(),
        settings=settings,
        summary=summary,
        error=error,
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
            handle.write("\n")
    if raise_on_failure and status != "passed":
        raise BioimageioValidationError(report)
    return report
