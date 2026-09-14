"""Dataset readers extracted from nuxnet-training (3e74a613).

Supports its images.tsv/annotations.tsv layout; this is not a general BIA
submission validator. Names and behavior are retained for import compatibility.
"""

from __future__ import annotations

import csv
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

_FILE_COLUMNS = (
    "filename",
    "file_name",
    "file_path",
    "filepath",
    "path",
    "uri",
    "file",
    "files",
)
_IMAGE_FILE_COLUMNS = (*_FILE_COLUMNS, "image")
_ANNOTATION_FILE_COLUMNS = (*_FILE_COLUMNS, "annotation", "mask", "label")
_IMAGE_ID_COLUMNS = ("image_id", "image_uuid", "id", "name")
_SOURCE_COLUMNS = (
    "image_id",
    "source_image_id",
    "source_image_uuid",
    "source_image",
    "image",
    "source",
)


@dataclass(frozen=True)
class VolumePair:
    image_id: str
    image: Path
    annotation: Path
    split: str | None = None
    group: str | None = None


@dataclass(frozen=True)
class OMEVolume:
    """An array and its OME physical voxel size, expressed as Z,Y,X in µm."""

    array: np.ndarray
    voxel_size_um: tuple[float, float, float]


def _normalise_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        return [
            {_normalise_key(str(key)): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _column(row: dict[str, str], choices: Iterable[str], table: str) -> str:
    for choice in choices:
        if row.get(choice):
            return row[choice]
    raise ValueError(f"{table} must contain one of these populated columns: {', '.join(choices)}")


def _metadata_root(root: Path) -> Path:
    candidates = sorted(root.rglob("images.tsv"))
    matches = [path.parent for path in candidates if (path.parent / "annotations.tsv").is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one directory containing images.tsv and annotations.tsv "
            f"below {root}; found {len(matches)}"
        )
    return matches[0]


def _resolve_file(metadata_root: Path, filename: str) -> Path:
    """Resolve paths relative to the TSV folder or its submission-package root."""
    path = Path(filename)
    if path.is_absolute():
        return path
    local = metadata_root / path
    package_relative = metadata_root.parent / path
    return local if local.exists() or not package_relative.exists() else package_relative


def extract_dataset_archive(archive: str | Path) -> tempfile.TemporaryDirectory:
    """Safely extract a downloaded BIA/BioImage.IO ZIP for this process."""
    temporary = tempfile.TemporaryDirectory(prefix="nuxnet_dataset_")
    destination = Path(temporary.name).resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                temporary.cleanup()
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        zipped.extractall(destination)
    return temporary


def read_bia_pairs(root: str | Path) -> list[VolumePair]:
    """Join BIA ``images.tsv`` and ``annotations.tsv`` records by source image ID."""
    metadata_root = _metadata_root(Path(root))
    images = _read_tsv(metadata_root / "images.tsv")
    annotations = _read_tsv(metadata_root / "annotations.tsv")
    image_index: dict[str, tuple[str, Path, str | None, str | None]] = {}
    for row in images:
        filename = _column(row, _IMAGE_FILE_COLUMNS, "images.tsv")
        group = row.get("resolution_group") or row.get("subset") or None
        pair_id = row.get("pair_id")
        image_id = next((row[column] for column in _IMAGE_ID_COLUMNS if row.get(column)), None)
        image_id = image_id or (f"{group}:{pair_id}" if group and pair_id else filename)
        if image_id in image_index:
            raise ValueError(f"Duplicate image identifier in images.tsv: {image_id}")
        image = _resolve_file(metadata_root, filename)
        # BIA annotation tables in the wild reference either the stable image
        # identifier, the relative file path, or only the source basename.
        for reference in (image_id, filename, Path(filename).name):
            previous = image_index.get(reference)
            if previous and previous[0] != image_id:
                raise ValueError(f"Ambiguous image reference in images.tsv: {reference}")
            image_index[reference] = (image_id, image, row.get("split") or None, group)

    pairs: list[VolumePair] = []
    seen: set[str] = set()
    for row in annotations:
        source = _column(row, _SOURCE_COLUMNS, "annotations.tsv")
        filename = _column(row, _ANNOTATION_FILE_COLUMNS, "annotations.tsv")
        source_record = image_index.get(source) or image_index.get(Path(source).name)
        if source_record is None:
            raise ValueError(f"Annotation references unknown image identifier or path: {source}")
        image_id, image, split, group = source_record
        if image_id in seen:
            raise ValueError(f"More than one segmentation annotation for image: {image_id}")
        annotation = _resolve_file(metadata_root, filename)
        for kind, candidate in (("image", image), ("annotation", annotation)):
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"{kind.title()} file for {image_id} not found: {candidate}"
                )
            if not candidate.name.lower().endswith((".ome.tif", ".ome.tiff")):
                raise ValueError(f"{kind.title()} file must be OME-TIFF: {candidate}")
        pairs.append(VolumePair(image_id, image, annotation, split, group))
        seen.add(image_id)
    if not pairs:
        raise ValueError("No image/annotation pairs found in the BIA metadata tables")
    return pairs


_UNIT_TO_UM = {
    "µm": 1.0,
    "um": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometre": 1e-3,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimetre": 1e3,
}


def _read_voxel_size_um(ome_xml: str | None, path: Path) -> tuple[float, float, float]:
    """Return OME ``PhysicalSizeZ/Y/X`` values as µm per voxel."""
    try:
        pixels = next(
            element
            for element in ET.fromstring(ome_xml or "").iter()
            if element.tag.endswith("Pixels")
        )
        xyz = []
        for axis in "ZYX":
            value = float(pixels.attrib[f"PhysicalSize{axis}"])
            unit = pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm").lower()
            factor = _UNIT_TO_UM[unit]
            if not np.isfinite(value) or value <= 0:
                raise ValueError
            xyz.append(value * factor)
        return tuple(xyz)
    except (ET.ParseError, StopIteration, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Valid PhysicalSizeX, PhysicalSizeY and PhysicalSizeZ OME metadata "
            f"is required for {path}"
        ) from error


def _read_ome(path: Path, *, mask: bool) -> OMEVolume:
    """Read the first OME-TIFF series in canonical CZYX or ZYX order."""
    with tifffile.TiffFile(path) as tif:
        if not tif.is_ome:
            raise ValueError(f"Expected OME-TIFF metadata in {path}")
        array = tif.series[0].asarray()
        axes = tif.series[0].axes.upper()
        voxel_size_um = _read_voxel_size_um(tif.ome_metadata, path)
    for axis in "ST":
        if axis in axes:
            index = axes.index(axis)
            if array.shape[index] != 1:
                raise ValueError(
                    f"{path} has non-singleton {axis}; "
                    "export each scene/timepoint as a separate record"
                )
            array = np.take(array, 0, axis=index)
            axes = axes[:index] + axes[index + 1 :]
    if mask and "C" in axes:
        index = axes.index("C")
        if array.shape[index] != 1:
            raise ValueError(f"Segmentation mask must have one channel: {path}")
        array = np.take(array, 0, axis=index)
        axes = axes[:index] + axes[index + 1 :]
    wanted = "ZYX" if mask else "CZYX"
    if not mask and "C" not in axes:
        array, axes = np.expand_dims(array, 0), "C" + axes
    if "Z" not in axes:
        index = axes.index("Y")
        array, axes = np.expand_dims(array, index), axes[:index] + "Z" + axes[index:]
    if set(axes) != set(wanted) or len(axes) != len(wanted):
        raise ValueError(f"Unsupported axes {axes!r} in {path}; expected {wanted}")
    array = np.transpose(array, tuple(axes.index(axis) for axis in wanted))
    return OMEVolume(array=array, voxel_size_um=voxel_size_um)
