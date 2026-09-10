"""
test_trocr_standalone.py

Tests TrOCR in complete isolation — no Gemini, no Groq, no API calls at
all. This proves the on-device OCR step works on its own, independent
of the LLM structuring step (which is a separate concern and separate
rate limit).

Usage:
    python test_trocr_standalone.py sample_invoices/scanned_invoice_02_SDS-2026-27-102.pdf
"""

import sys
import time
from agents.extraction_agent import pdf_page_to_image
from tools.trocr_engine import trocr_extract


def main(pdf_path: str):
    print(f"Rasterizing {pdf_path} ...")
    image = pdf_page_to_image(pdf_path)

    print("Running TrOCR (on-device, no network calls) ...")
    start = time.time()
    text, confidence = trocr_extract(image)
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"TrOCR result for: {pdf_path}")
    print("=" * 60)
    print(f"Local inference time : {elapsed*1000:.1f} ms")
    print(f"Confidence           : {confidence:.1f}/100")
    print(f"Extracted text ({len(text)} chars):")
    print("-" * 60)
    print(text)
    print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_trocr_standalone.py <path_to_scanned_invoice.pdf>")
        sys.exit(1)
    main(sys.argv[1])
