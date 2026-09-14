"""Opt-in tests against the real optional validator (may perform network I/O)."""

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
import yaml

from nidavellir_tools.build_model_package import build_model_package
from nidavellir_tools.validation import _file_digest, validate_bioimageio

pytestmark = pytest.mark.skipif(
    os.environ.get("NIDAVELLIR_TEST_BIOIMAGEIO") != "1",
    reason="Set NIDAVELLIR_TEST_BIOIMAGEIO=1 with the bioimageio extra installed",
)


def test_official_inference_and_wrong_expected_output(tmp_path):
    template = (
        Path(__file__).parents[1] / "src/nidavellir_tools/examples/model-package.example.yaml"
    )
    rdf = yaml.safe_load(template.read_text())
    rdf.pop("covers", None)
    rdf.pop("cite", None)
    rdf["name"] = "Nidavellir validation fixture"
    rdf["description"] = "A synthetic model for testing validation, not a scientific model."
    rdf["maintainers"] = [{"name": "Test Maintainer", "github_user": "bioimage-io"}]
    rdf["weights"].pop("torchscript")
    state = rdf["weights"]["pytorch_state_dict"]
    state["pytorch_version"] = torch.__version__.split("+")[0]
    state["architecture"] = {"source": "network.py", "callable": "Model", "kwargs": {}}
    (tmp_path / "network.py").write_text(
        "import torch\n"
        "class Model(torch.nn.Conv2d):\n"
        "    def __init__(self):\n"
        "        super().__init__(1, 1, 1, bias=False)\n"
    )
    (tmp_path / "environment.yml").write_text("dependencies:\n  - python=3.12\n  - pytorch\n")
    (tmp_path / "card.md").write_text(
        "# Validation fixture\nSynthetic identity-like convolution.\n"
    )
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(rdf))
    model = torch.nn.Conv2d(1, 1, 1, bias=False).eval()
    with torch.no_grad():
        model.weight.fill_(2)
    x = np.arange(256, dtype=np.float32).reshape(1, 1, 16, 16) / 256
    y = model(torch.from_numpy(x)).detach().numpy()
    np.save(tmp_path / "input.npy", x)
    np.save(tmp_path / "output.npy", y)
    tifffile.imwrite(tmp_path / "sample-input.tif", x[0, 0], photometric="minisblack")
    tifffile.imwrite(tmp_path / "sample-output.tif", y[0, 0], photometric="minisblack")
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    package, archive = build_model_package(
        tmp_path / "model.yaml",
        tmp_path / "weights.pt",
        tmp_path / "input.npy",
        tmp_path / "output.npy",
        tmp_path / "card.md",
        tmp_path / "package",
        validate_bioimageio=True,
    )
    assert (tmp_path / "package.validation.json").is_file()
    report = validate_bioimageio(archive, raise_on_failure=False)
    assert report.status == "passed", report.to_dict()

    # Keep integrity checks valid but deliberately break inference equivalence.
    output = package / "test-output.npy"
    np.save(output, y + 100)
    packaged_rdf = yaml.safe_load((package / "rdf.yaml").read_text())
    packaged_rdf["outputs"][0]["test_tensor"]["sha256"] = _file_digest(output)
    (package / "rdf.yaml").write_text(yaml.safe_dump(packaged_rdf))
    failed = validate_bioimageio(package, raise_on_failure=False)
    assert failed.status == "failed", failed.to_dict()
    assert failed.summary is not None
