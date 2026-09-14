import json
import shutil
from types import SimpleNamespace

import pytest
import yaml

from nidavellir_tools import model_package_registry as registry
from nidavellir_tools import validation


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "weights.pt").write_bytes(b"fixture")
    (root / "rdf.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "model",
                "weights": {
                    "pytorch_state_dict": {
                        "source": "weights.pt",
                        "sha256": validation._file_digest(root / "weights.pt"),
                    }
                },
            }
        )
    )
    return root


def mock_core(monkeypatch, status="passed", callback=None):
    def test_description(source, **kwargs):
        if callback:
            callback(source, kwargs)
        return SimpleNamespace(model_dump=lambda **_: {"status": status, "details": []})

    monkeypatch.setattr(
        validation.importlib,
        "import_module",
        lambda _: SimpleNamespace(test_description=test_description),
    )


@pytest.mark.parametrize("archive", [False, True])
def test_report_and_unchanged_input(package, tmp_path, monkeypatch, archive):
    before = validation._package_digest(package)
    source = package
    if archive:
        source = type(package)(shutil.make_archive(str(tmp_path / "input"), "zip", package))
    source_digest = validation._package_digest(source)

    def check(path, settings):
        assert path != package / "rdf.yaml"
        assert settings["runtime_env"] == "currently-active"
        assert settings["devices"] == ["cpu"]
        path.write_text("modified by validator")

    mock_core(monkeypatch, callback=check)
    report_path = tmp_path / "reports" / "validation.json"
    result = registry.validate_bioimageio(source, report_path=report_path)
    assert result.status == "passed"
    assert result.package_sha256 == source_digest
    assert validation._package_digest(package) == before
    assert validation._package_digest(source) == source_digest
    assert json.loads(report_path.read_text())["summary"]["status"] == "passed"


@pytest.mark.parametrize("status", ["failed", "valid-format", "unknown"])
def test_non_pass_is_failure_and_report_saved(package, tmp_path, monkeypatch, status):
    mock_core(monkeypatch, status)
    destination = tmp_path / "failure.json"
    with pytest.raises(validation.BioimageioValidationError) as error:
        validation.validate_bioimageio(package, report_path=destination)
    assert error.value.report.status == "failed"
    assert json.loads(destination.read_text())["status"] == "failed"


def test_corruption_fails_before_official_tests(package, monkeypatch):
    def unexpected(*args):
        pytest.fail("official module must not load for corrupt artifacts")

    monkeypatch.setattr(validation.importlib, "import_module", unexpected)
    (package / "weights.pt").write_bytes(b"changed")
    result = validation.validate_bioimageio(package, raise_on_failure=False)
    assert result.status == "failed" and result.error["stage"] == "integrity"


@pytest.mark.parametrize("failure", [ModuleNotFoundError("missing"), RuntimeError("crash")])
def test_dependency_errors_are_not_validation_passes(package, monkeypatch, failure):
    def fail(*args):
        raise failure

    monkeypatch.setattr(validation.importlib, "import_module", fail)
    result = validation.validate_bioimageio(package, raise_on_failure=False)
    assert result.status == "error" and result.error["stage"] == "dependency"


def test_official_execution_error(package, monkeypatch):
    def fail(*args):
        raise RuntimeError("inference failed to execute")

    mock_core(monkeypatch, callback=fail)
    result = validation.validate_bioimageio(package, raise_on_failure=False)
    assert result.status == "error" and result.error["stage"] == "official"


def test_report_path_safety(package, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        validation.validate_bioimageio(package, report_path=package / "report.json")
    destination = tmp_path / "existing.json"
    destination.write_text("keep")
    with pytest.raises(FileExistsError):
        validation.validate_bioimageio(package, report_path=destination)
    assert destination.read_text() == "keep"


def test_cli_failure_exit_and_json_report(package, monkeypatch, capsys):
    mock_core(monkeypatch, "failed")
    with pytest.raises(SystemExit) as error:
        registry.main(["validate", str(package)])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_cli_pass_with_device_and_report(package, tmp_path, monkeypatch):
    def check(path, kwargs):
        assert kwargs["devices"] == ["cuda:0"]

    mock_core(monkeypatch, callback=check)
    destination = tmp_path / "cli.json"
    registry.main(["validate", str(package), "--device", "cuda:0", "--report", str(destination)])
    assert json.loads(destination.read_text())["status"] == "passed"


def test_missing_package_and_unsafe_zip(tmp_path):
    import zipfile

    missing = validation.validate_bioimageio(tmp_path / "missing", raise_on_failure=False)
    assert missing.status == "error"
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape", "bad")
    result = validation.validate_bioimageio(archive, raise_on_failure=False)
    assert result.status == "error"
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("location", ["inside", "archive", "existing", "disabled"])
def test_build_report_preflight_preserves_existing_output(tmp_path, location):
    from nidavellir_tools.build_model_package import build_model_package

    output = tmp_path / "package"
    output.mkdir()
    original = output / "keep.txt"
    original.write_text("keep")
    report = {
        "inside": output / "report.json",
        "archive": output.with_suffix(".zip"),
        "existing": tmp_path / "existing.json",
        "disabled": tmp_path / "unused.json",
    }[location]
    if location == "existing":
        report.write_text("keep report")
    with pytest.raises((ValueError, FileExistsError)):
        build_model_package(
            tmp_path / "missing.yaml",
            tmp_path / "missing.pt",
            tmp_path / "input.npy",
            tmp_path / "output.npy",
            tmp_path / "card.md",
            output,
            overwrite=True,
            validate_bioimageio=location != "disabled",
            validation_report=report,
        )
    assert original.read_text() == "keep"
