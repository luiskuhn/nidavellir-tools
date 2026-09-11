from importlib.resources import files

from nidavellir_tools import cli


def test_cli_help(capsys):
    cli.main(["--help"])

    assert "usage: nidavellir COMMAND [OPTIONS]" in capsys.readouterr().out


def test_packaged_templates_are_available():
    package_files = files("nidavellir_tools")

    assert package_files.joinpath("model-card-template.md").is_file()
    assert package_files.joinpath("examples/model-package.example.yaml").is_file()

