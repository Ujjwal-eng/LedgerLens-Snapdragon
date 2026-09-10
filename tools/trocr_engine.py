"""
TrOCR (Qualcomm AI Hub) — local, on-device OCR replacing the previous
cloud/Tesseract tier. Same (text, confidence) signature as the old
ocr_extract() so extraction_agent.py needs no other changes.
"""

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

_processor = None
_model = None

# "printed" checkpoint suits invoices; swap to trocr-base-handwritten
# if you expect handwritten fields.
MODEL_NAME = "microsoft/trocr-small-printed"


# def _load():
#     global _processor, _model
#     if _model is None:
#         _processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=False)
#         _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
#         _model.eval()
#     return _processor, _model

# def _load():
#     global _processor, _model
#     if _model is None:
#         from transformers import ViTImageProcessor, RobertaTokenizer

#         # Load components directly instead of via TrOCRProcessor/AutoTokenizer —
#         # AutoTokenizer's auto-detection was failing to build the tokenizer
#         # for this checkpoint. Loading RobertaTokenizer explicitly bypasses
#         # that broken detection entirely and needs no sentencepiece at all.
#         image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
#         tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
#         _processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

#         _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
#         _model.eval()
#     return _processor, _model

def _load():
    global _processor, _model
    if _model is None:
        from transformers import ViTImageProcessor, XLMRobertaTokenizer

        image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
        _processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

        _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _processor, _model


def trocr_extract(image: Image.Image) -> tuple[str, float]:
    processor, model = _load()
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values

    with torch.no_grad():
        out = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=512,
        )
    # print(f"[DEBUG] raw token ids: {out.sequences[0].tolist()}")
    # print(f"[DEBUG] pixel_values stats: min={pixel_values.min().item():.3f}, "
    #       f"max={pixel_values.max().item():.3f}, mean={pixel_values.mean().item():.3f}")
    # print(f"[DEBUG] tokens: {processor.tokenizer.convert_ids_to_tokens(out.sequences[0].tolist())}")
    # print(f"[DEBUG] decode no skip: {processor.batch_decode(out.sequences, skip_special_tokens=False)[0]!r}")

    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
    confidence = _score_to_confidence(out.scores)
    return text, confidence


def _score_to_confidence(scores) -> float:
    """Average max-token-probability across generation steps, mapped to
    the same 0-100 scale OCR_MIN_CONFIDENCE already expects."""
    if not scores:
        return 0.0
    probs = [torch.softmax(step[0], dim=-1).max().item() for step in scores]
    return round((sum(probs) / len(probs)) * 100, 1)