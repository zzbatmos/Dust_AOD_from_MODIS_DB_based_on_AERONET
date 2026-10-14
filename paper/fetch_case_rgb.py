"""Fetch dated NASA GIBS context imagery; never use it as retrieval input.

The WMS 1.1.1 EPSG:4326 bounding-box order is longitude,latitude. GIBS imagery
is a daily platform mosaic, not a guaranteed single-granule overpass image.
Official interface: https://nasa-gibs.github.io/gibs-api-docs/access-basics/
"""
import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import requests

from paper.numbers import digest


CASES = {
    "Kansas_20250314": ("Terra", "2025-03-14", [-110, 28, -93, 45]),
    "Amazonia_20240922": ("Terra", "2024-09-22", [-70, -27, -53, -9]),
    "NorthAmerica_20170829": ("Aqua", "2017-08-29", [-125, 39, -100, 50]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Save review imagery outside public code worktree")
    output.mkdir(parents=True, exist_ok=True)
    for name, (platform, date, bounds) in CASES.items():
        image_path = output / f"{name}_GIBS_truecolor.png"
        provenance = output / f"{name}_GIBS_truecolor.json"
        width = 1200
        height = round(width * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0]))
        params = dict(SERVICE="WMS", VERSION="1.1.1", REQUEST="GetMap", SRS="EPSG:4326",
                      BBOX=",".join(map(str, bounds)), WIDTH=width, HEIGHT=height,
                      FORMAT="image/png", LAYERS=f"MODIS_{platform}_CorrectedReflectance_TrueColor",
                      STYLES="", TIME=date, TRANSPARENT="TRUE")
        if image_path.exists() or provenance.exists():
            if not (image_path.exists() and provenance.exists()):
                raise ValueError("Partial cache; inspect before replacing")
            old = json.loads(provenance.read_text())
            if old["parameters"] != params or old["sha256"] != digest(image_path):
                raise ValueError("Cached imagery changed or uses different request")
            print(f"Verified existing {image_path}")
            continue
        response = requests.get("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
                                params=params, timeout=(20, 120))
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.load()
        if image.format != "PNG" or image.size != (width, height):
            raise ValueError("Wrong GIBS image format or dimensions")
        if image.mode == "RGBA" and image.getchannel("A").getextrema()[1] == 0:
            raise ValueError("GIBS returned entirely missing imagery")
        image_path.write_bytes(response.content)
        provenance.write_text(json.dumps({
            "source": "NASA GIBS daily corrected-reflectance true-color visualization",
            "url": response.url, "parameters": params, "platform": platform, "date": date,
            "extent_lonlat": [bounds[0], bounds[2], bounds[1], bounds[3]], "origin": "upper",
            "sha256": digest(image_path), "script_sha256": digest(Path(__file__)),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "caveat": "Same platform/date daily mosaic; not guaranteed exact single L2 overpass. "
            "Context only, never an input to inference or a validation reference."
        }, indent=2) + "\n")
        print(image_path)


if __name__ == "__main__":
    main()
