"""Unified command-line entry point for Nidavellir Tools."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from nidavellir_tools import build_model_package, create_sample_tensors
from nidavellir_tools import model_package_registry as registry

_REGISTRY_COMMANDS = {
    "stage",
    "inspect",
    "load",
    "validate",
    "publish-hf",
    "export-child",
}


def _usage() -> str:
    return """usage: nidavellir COMMAND [OPTIONS]

Commands:
  build          Build a model package from staged run artifacts
  samples        Create presentation TIFFs from exact NPY test tensors
  stage          Stage a local or remote model package
  inspect        Inspect and verify a package
  load           Load a package or export initialization weights
  validate       Run BioImage.IO validation
  publish-hf     Publish a package to Hugging Face
  export-child   Export a fine-tuned child package

Run 'nidavellir COMMAND --help' for command-specific options.
"""


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_usage())
        return

    command, remaining = arguments[0], arguments[1:]
    handler: Callable[[Sequence[str] | None], None]
    if command == "build":
        handler = build_model_package.main
        handler(remaining)
    elif command == "samples":
        handler = create_sample_tensors.main
        handler(remaining)
    elif command in _REGISTRY_COMMANDS:
        registry.main(arguments)
    else:
        raise SystemExit(f"unknown command {command!r}\n\n{_usage()}")


if __name__ == "__main__":
    main()
