from PIL import Image
import json
import pytest

from paper.audit_review_bundle import font_check, raster_check
from paper.review_bundle import missing_supplementary


def test_supplement_requires_both_seasonal_pages():
    complete = {f"FigS{i:02}" for i in range(1, 19)} - {"FigS13"}
    assert missing_supplementary(complete) == ["FigS13"]
    assert missing_supplementary(complete | {"FigS13a"}) == ["FigS13"]
    assert missing_supplementary(complete | {"FigS13a", "FigS13b"}) == []
    assert missing_supplementary(complete | {"FigS13"}) == ["FigS13"]


def test_assemble_only_never_rerenders_or_compiles(tmp_path, monkeypatch):
    import subprocess
    import sys
    from paper import review_bundle
    from paper.numbers import digest
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    root = tmp_path / "bundle"
    folder = root / "si"
    folder.mkdir(parents=True)
    (folder / "FigS13a_maps.pdf").write_bytes(b"fake PDF for command routing test")
    (folder / "FigS13a_provenance.json").write_text(json.dumps({"registry_sha256": digest(registry)}))
    (folder / "FigS13a_caption.md").write_text("A caption.")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[0] in {"pdfunite", "pdffonts"}
        return subprocess.CompletedProcess(command, 0, "fonts", "")
    monkeypatch.setattr(review_bundle.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["bundle", "--registry", str(registry), "--output-dir", str(root),
                                     "--assemble-only", "--supplement-tables"])
    review_bundle.main()
    manifest = json.loads((root / "review_manifest.json").read_text())
    assert "FigS13" in manifest["missing_supplementary_figures"]
    assert len(manifest["figures"]) == 1
    assert [c[0] for c in calls] == ["pdfunite", "pdffonts"]


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
    json.dumps(result, allow_nan=False)
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
