"""
test_trocr_cropped_line.py

Crops a single line/region out of a scanned invoice (using relative
coordinates, so it works regardless of render DPI) and runs TrOCR on
just that crop — proving the model works correctly on the kind of
input it was actually trained for (one line of text), rather than a
full multi-line page.

Usage:
    python test_trocr_cropped_line.py <path_to_invoice.pdf>

Saves a debug image (cropped_region.png) so you can SEE exactly what
was fed to the model before trusting the OCR result.
"""

import sys
import time
from pdf2image import convert_from_path
from tools.trocr_engine import trocr_extract

# Relative crop box (fractions of page width/height: left, top, right, bottom).
# Defaults target the "Invoice No: SDS/2026-27-102" line near the top of
# a typical invoice like scanned_invoice_02. Adjust these if your crop
# misses — check cropped_region.png after running to see what got cropped.
CROP_BOX = (0.10, 0.19, 0.55, 0.23)


def main(pdf_path: str):
    print(f"Rasterizing {pdf_path} ...")
    images = convert_from_path(pdf_path, dpi=200)
    page = images[0]
    w, h = page.size

    left = int(CROP_BOX[0] * w)
    top = int(CROP_BOX[1] * h)
    right = int(CROP_BOX[2] * w)
    bottom = int(CROP_BOX[3] * h)

    crop = page.crop((left, top, right, bottom))
    crop.save("cropped_region.png")
    print(f"Saved crop as cropped_region.png ({crop.size[0]}x{crop.size[1]}px) "
          f"— open it to confirm it shows the intended line.")

    print("Running TrOCR on the cropped line ...")
    start = time.time()
    text, confidence = trocr_extract(crop)
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("TrOCR result on cropped line")
    print("=" * 60)
    print(f"Local inference time : {elapsed*1000:.1f} ms")
    print(f"Confidence           : {confidence:.1f}/100")
    print(f"Extracted text: {text!r}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_trocr_cropped_line.py <path_to_invoice.pdf>")
        sys.exit(1)
    main(sys.argv[1])
