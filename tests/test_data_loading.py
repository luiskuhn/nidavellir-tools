"""Reading tests migrated from NuxNet, plus OME boundary checks."""

import csv
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from nidavellir_tools.data_loading import (
    _read_ome,
    _read_voxel_size_um,
    extract_dataset_archive,
    read_bia_pairs,
)


def write_tsv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    image = np.arange(60, dtype=np.uint16).reshape(2, 5, 6)
    mask = (image > 10).astype(np.uint8)
    tifffile.imwrite(
        root / "image.ome.tiff",
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    tifffile.imwrite(
        root / "mask.ome.tiff",
        mask,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    write_tsv(
        root / "images.tsv",
        [{"Image ID": "volume-1", "File Path": "image.ome.tiff", "split": "train"}],
    )
    write_tsv(
        root / "annotations.tsv",
        [
            {
                "annotation_id": "mask-1",
                "Source Image": "volume-1",
                "File": "mask.ome.tiff",
            }
        ],
    )


def test_zip_archive_is_safely_extracted_and_discovered(tmp_path: Path):
    source = tmp_path / "source" / "nested"
    make_dataset(source)
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in source.iterdir():
            zipped.write(path, f"submission/nested/{path.name}")
    extracted = extract_dataset_archive(archive)
    try:
        assert len(read_bia_pairs(extracted.name)) == 1
    finally:
        extracted.cleanup()


def test_bia_filename_references_and_bioimage_columns_are_supported(tmp_path: Path):
    image = np.zeros((4, 8, 8), dtype=np.uint8)
    tifffile.imwrite(
        tmp_path / "raw.ome.tiff",
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    tifffile.imwrite(
        tmp_path / "labels.ome.tiff",
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    write_tsv(tmp_path / "images.tsv", [{"image": "raw.ome.tiff"}])
    write_tsv(
        tmp_path / "annotations.tsv",
        [{"source image": "raw.ome.tiff", "annotation": "labels.ome.tiff"}],
    )

    [pair] = read_bia_pairs(tmp_path)

    assert pair.image_id == "raw.ome.tiff"
    assert pair.annotation.name == "labels.ome.tiff"


def test_numorph_submission_layout_is_resolved_from_bia_tables(tmp_path: Path):
    package = tmp_path / "NUMORPH_SEM_SEG_DATASET"
    image_dir = package / "data" / "C075" / "images"
    mask_dir = package / "data" / "C075" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    volume = np.zeros((4, 8, 8), dtype=np.uint8)
    tifffile.imwrite(
        image_dir / "raw.ome.tif",
        volume,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    tifffile.imwrite(
        mask_dir / "mask.ome.tif",
        volume,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    bia = package / "bia"
    bia.mkdir()
    write_tsv(
        bia / "images.tsv",
        [
            {
                "Files": "data/C075/images/raw.ome.tif",
                "Resolution group": "C075",
                "Pair ID": "0001",
            }
        ],
    )
    write_tsv(
        bia / "annotations.tsv",
        [
            {
                "Files": "data/C075/masks/mask.ome.tif",
                "source_image": "data/C075/images/raw.ome.tif",
            }
        ],
    )

    [pair] = read_bia_pairs(tmp_path)

    assert pair.image_id == "C075:0001"
    assert pair.group == "C075"
    assert pair.image == image_dir / "raw.ome.tif"
    assert pair.annotation == mask_dir / "mask.ome.tif"


def test_unknown_image_reference_is_rejected(tmp_path: Path):
    write_tsv(
        tmp_path / "images.tsv",
        [{"image_id": "volume-1", "filename": "image.ome.tiff"}],
    )
    write_tsv(
        tmp_path / "annotations.tsv",
        [{"image_id": "missing", "filename": "mask.ome.tiff"}],
    )
    with pytest.raises(ValueError, match="unknown image"):
        read_bia_pairs(tmp_path)


def test_read_pair_preserves_pixels_calibration_and_split(tmp_path):
    make_dataset(tmp_path)
    [pair] = read_bia_pairs(tmp_path)
    image = _read_ome(pair.image, mask=False)
    mask = _read_ome(pair.annotation, mask=True)
    expected = np.arange(60, dtype=np.uint16).reshape(2, 5, 6)
    np.testing.assert_array_equal(image.array, expected[None])
    np.testing.assert_array_equal(mask.array, (expected > 10).astype(np.uint8))
    assert image.array.dtype == np.uint16
    assert image.voxel_size_um == mask.voxel_size_um == (3.2, 1.0, 1.0)
    assert pair.split == "train"


def test_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe path"):
        extract_dataset_archive(archive)
    assert not (tmp_path / "escape").exists()


def test_physical_units_convert_to_micrometers():
    xml = (
        '<OME><Pixels PhysicalSizeZ="3000" PhysicalSizeZUnit="nm" '
        'PhysicalSizeY="0.001" PhysicalSizeYUnit="mm" PhysicalSizeX="1"/></OME>'
    )
    assert _read_voxel_size_um(xml, Path("image.ome.tif")) == (3.0, 1.0, 1.0)


@pytest.mark.parametrize("xml", [None, "<OME/>", '<OME><Pixels PhysicalSizeZ="0"/></OME>'])
def test_missing_or_invalid_calibration_is_rejected(xml):
    with pytest.raises(ValueError, match="Valid PhysicalSize"):
        _read_voxel_size_um(xml, Path("image.ome.tif"))


def test_non_ome_tiff_is_rejected(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((5, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected OME-TIFF"):
        _read_ome(path, mask=False)


@pytest.mark.parametrize(
    "axes,shape,mask,message",
    [
        ("TZYX", (2, 2, 5, 6), False, "non-singleton T"),
        ("CZYX", (2, 2, 5, 6), True, "one channel"),
    ],
)
def test_unsupported_axes_are_rejected(tmp_path, axes, shape, mask, message):
    path = tmp_path / "image.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros(shape, dtype=np.uint8),
        ome=True,
        metadata={"axes": axes, "PhysicalSizeZ": 1, "PhysicalSizeY": 1, "PhysicalSizeX": 1},
    )
    with pytest.raises(ValueError, match=message):
        _read_ome(path, mask=mask)
