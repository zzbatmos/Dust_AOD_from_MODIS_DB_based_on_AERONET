from PIL import Image
import pytest

from paper.audit_review_bundle import font_check, raster_check


def test_true_type_embedded_font_passes():
    report = "header\n----\nNAME CID TrueType Identity-H yes yes yes 35 0\n"
    assert font_check(report) == 1


@pytest.mark.parametrize("line", ["NAME CID TrueType Identity-H no yes yes 35 0", "NAME Type 3 Custom yes yes no 1 0", ""])
def test_invalid_pdf_fonts_fail(line):
    with pytest.raises(ValueError):
        font_check("header\n----\n" + line)


def test_tiff_resolution_size_mode_and_compression(tmp_path):
    path = tmp_path / "test.tif"
    image = Image.new("RGB", (600, 1200))
    image.save(path, compression="tiff_lzw", dpi=(600, 600))
    result = raster_check(path, 25.4, 50.8, tiff=True)
    assert result["mode"] == "RGB"
    image.save(path, dpi=(600, 600))
    with pytest.raises(ValueError, match="LZW"):
        raster_check(path, 25.4, 50.8, tiff=True)
    image.save(path, compression="tiff_lzw", dpi=(300, 300))
    with pytest.raises(ValueError, match="DPI"):
        raster_check(path, 25.4, 50.8, tiff=True)


def test_preview_physical_size(tmp_path):
    path = tmp_path / "test.png"
    Image.new("RGB", (300, 600)).save(path, dpi=(300, 300))
    raster_check(path, 25.4, 50.8)
    with pytest.raises(ValueError, match="dimensions"):
        raster_check(path, 95, 100)
