"""Token-classification PII detector — the contextual layer of the pipeline.

Default model is ``iiiorg/piiranha-v1-detect-personal-information`` (mDeBERTa-
v3-base fine-tuned on AI4Privacy).  Any HuggingFace token-classification
checkpoint with a BIO/IO label scheme will work as long as the labels appear
in ``_LABEL_TO_CATEGORY`` below.
"""
from __future__ import annotations

# Stub the _lzma C module if missing — pyenv-built Pythons sometimes lack it
# because liblzma headers were unavailable at compile time, and torchvision
# (a transitive transformers dep) imports lzma at module load.  We never
# decompress xz data here, so a stub is safe.
import sys as _sys

if "_lzma" not in _sys.modules:
    try:  # pragma: no cover - import-time platform check
        import _lzma  # noqa: F401
    except ModuleNotFoundError:
        import types as _types

        _stub = _types.ModuleType("_lzma")

        class _LZMAError(Exception):
            pass

        class _Stub:
            def __init__(self, *_a, **_k) -> None:
                raise _LZMAError("lzma support is not available in this Python build")

        for _name in (
            "FORMAT_AUTO", "FORMAT_XZ", "FORMAT_ALONE", "FORMAT_RAW",
            "CHECK_NONE", "CHECK_CRC32", "CHECK_CRC64", "CHECK_SHA256",
            "CHECK_ID_MAX", "CHECK_UNKNOWN",
            "FILTER_LZMA1", "FILTER_LZMA2", "FILTER_DELTA", "FILTER_X86",
            "FILTER_POWERPC", "FILTER_IA64", "FILTER_ARM", "FILTER_ARMTHUMB",
            "FILTER_SPARC",
            "MF_HC3", "MF_HC4", "MF_BT2", "MF_BT3", "MF_BT4",
            "MODE_FAST", "MODE_NORMAL",
            "PRESET_DEFAULT", "PRESET_EXTREME",
        ):
            setattr(_stub, _name, 0)
        _stub.LZMAError = _LZMAError
        _stub.LZMACompressor = _Stub
        _stub.LZMADecompressor = _Stub
        _stub._encode_filter_properties = lambda *_a, **_k: (_ for _ in ()).throw(
            _LZMAError("stub")
        )
        _stub._decode_filter_properties = lambda *_a, **_k: (_ for _ in ()).throw(
            _LZMAError("stub")
        )
        _sys.modules["_lzma"] = _stub

import logging
from typing import Callable, Protocol

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable

logger = logging.getLogger(__name__)


# Map common PII NER labels (Piranha + generic NER conventions) to the
# autodact internal categories.  Strip B-/I- prefix before lookup.
_LABEL_TO_CATEGORY: dict[str, str] = {
    # Piranha (mDeBERTa-v3) schema
    "GIVENNAME": "NAME",
    "SURNAME": "NAME",
    "MIDDLENAME": "NAME",
    "USERNAME": "NAME",
    "TITLE": "JOBTITLE",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "DATEOFBIRTH": "DOB",
    "DATE": "DOB",
    "TIME": "DOB",
    "CITY": "LOCATION",
    "STREET": "LOCATION",
    "BUILDINGNUM": "LOCATION",
    "COUNTRY": "LOCATION",
    "STATE": "LOCATION",
    "ZIPCODE": "POSTCODE",
    "SOCIALNUM": "ID",
    "IDCARDNUM": "ID",
    "ACCOUNTNUM": "ID",
    "CREDITCARDNUMBER": "ID",
    "DRIVERLICENSENUM": "ID",
    "TAXNUM": "ID",
    "PASSWORD": "ID",
    # Generic NER conventions (spaCy / OntoNotes / others)
    "PER": "NAME",
    "PERSON": "NAME",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "GPE": "LOCATION",
    "ADDRESS": "LOCATION",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "URL": "URL",
    "IP": "IP",
    "IPADDRESS": "IP",
    "DOB": "DOB",
}

# Skip very short spans to avoid over-replacement.
_MIN_ENTITY_LEN = 3

# DeBERTa max context.  Long lines are sliced with stride for context.
_MAX_CHUNK_TOKENS = 512
_CHUNK_STRIDE = 64

# Per-token softmax threshold below which a label is treated as O.
_DEFAULT_CONFIDENCE = 0.5

# Default category set covers everything Piranha can predict.  Restrict via
# the ``categories`` constructor arg to a narrower set if needed.
_DEFAULT_CATEGORIES = frozenset({
    "NAME", "ORG", "LOCATION", "JOBTITLE", "EMAIL", "PHONE",
    "DOB", "POSTCODE", "IP", "URL", "ID",
})


class DebertaDetector:
    def __init__(
        self,
        model_name: str = "iiiorg/piiranha-v1-detect-personal-information",
        device: str | None = None,
        confidence: float = _DEFAULT_CONFIDENCE,
        categories: list[str] | None = None,
    ) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        import torch

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._model.eval()

        self._device = device or self._auto_device(torch)
        self._model.to(self._device)

        self._id2label = self._model.config.id2label
        self._confidence = confidence
        self._categories = (
            set(categories) if categories else set(_DEFAULT_CATEGORIES)
        )

    @staticmethod
    def _auto_device(torch) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def set_categories(self, categories: list[str]) -> None:
        """Update the active category filter without reloading the model."""
        self._categories = set(categories) if categories else set(_DEFAULT_CATEGORIES)

    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        on_progress: Callable[[int, int], None] | None = None,
        mapped_columns: dict[int, str] | None = None,
    ) -> list[int]:
        flagged: list[int] = []
        total = len(doc.lines)
        cache: dict[str, list[tuple[str, str]]] = {}

        for i, line in enumerate(doc.lines):
            text = line.text
            stripped = text.strip()
            if not stripped:
                if on_progress:
                    on_progress(i + 1, total)
                continue

            if stripped in cache:
                detections = cache[stripped]
            else:
                try:
                    detections = self._detect(text)
                except Exception:
                    logger.warning(
                        "DeBERTa inference failed on line %d, flagging",
                        line.line_number,
                    )
                    flagged.append(line.line_number)
                    if on_progress:
                        on_progress(i + 1, total)
                    continue
                cache[stripped] = detections

            for original, category in detections:
                if category not in self._categories:
                    continue
                if len(original) < _MIN_ENTITY_LEN:
                    continue
                if original not in line.text:
                    continue
                lookup.register(
                    original, category,
                    doc.source_path.name, line.line_number,
                )

            if on_progress:
                on_progress(i + 1, total)

        return flagged

    def _detect(self, text: str) -> list[tuple[str, str]]:
        torch = self._torch
        enc = self._tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=_MAX_CHUNK_TOKENS,
            stride=_CHUNK_STRIDE,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)

        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = self._model(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits

        probs = torch.softmax(logits, dim=-1)
        scores, preds = probs.max(dim=-1)

        spans: list[tuple[int, int, str]] = []
        for c in range(preds.size(0)):
            self._merge_chunk_spans(
                preds[c].tolist(),
                scores[c].tolist(),
                offsets[c].tolist(),
                spans,
            )

        return _spans_to_pairs(text, spans)

    def _merge_chunk_spans(
        self,
        preds: list[int],
        scores: list[float],
        offsets: list[list[int]],
        out: list[tuple[int, int, str]],
    ) -> None:
        """Group consecutive same-category tokens into character-level spans."""
        cur_cat: str | None = None
        cur_start: int | None = None
        cur_end: int | None = None

        def flush() -> None:
            nonlocal cur_cat, cur_start, cur_end
            if cur_cat is not None and cur_start is not None and cur_end is not None:
                out.append((cur_start, cur_end, cur_cat))
            cur_cat = None
            cur_start = None
            cur_end = None

        for pid, score, span in zip(preds, scores, offsets):
            start, end = span[0], span[1]
            # Special tokens (CLS/SEP/PAD) get (0, 0) offsets.
            if start == 0 and end == 0:
                flush()
                continue

            label_raw = self._id2label[pid] if pid in self._id2label else "O"
            if label_raw == "O" or score < self._confidence:
                flush()
                continue

            base = label_raw[2:] if label_raw[:2] in ("B-", "I-") else label_raw
            cat = _LABEL_TO_CATEGORY.get(base.upper())
            if cat is None:
                flush()
                continue

            is_b = label_raw.startswith("B-")
            if cur_cat == cat and not is_b:
                cur_end = end
            else:
                flush()
                cur_cat = cat
                cur_start = start
                cur_end = end

        flush()


def _spans_to_pairs(
    text: str, spans: list[tuple[int, int, str]],
) -> list[tuple[str, str]]:
    """Deduplicate spans (chunk overlap can repeat) and trim whitespace."""
    seen: set[tuple[int, int, str]] = set()
    out: list[tuple[str, str]] = []
    for start, end, cat in spans:
        key = (start, end, cat)
        if key in seen:
            continue
        seen.add(key)
        snippet = text[start:end].strip()
        if snippet:
            out.append((snippet, cat))
    return out


class _SupportsProcess(Protocol):
    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        on_progress: Callable[[int, int], None] | None = ...,
        mapped_columns: dict[int, str] | None = ...,
    ) -> list[int]: ...
