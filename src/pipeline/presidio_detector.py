from __future__ import annotations

import logging
import re
from typing import Callable

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable

logger = logging.getLogger(__name__)

# Regex for company names: one or more title-cased words followed by a
# well-known corporate suffix.  Used as a heuristic to catch organisations
# that spaCy NER misses.
_COMPANY_SUFFIXES = (
    r"(?:PLC|Ltd|Limited|Inc|Incorporated|Corp|Corporation|LLC|LLP|"
    r"GmbH|AG|SA|NV|BV|SE|SRL|Pty|Co\.?|"
    r"Group|Holdings|Partners|Associates|Enterprises|& Co\.?|"
    r"University|Institute|Hospital|Council|Foundation|Academy|"
    r"Centre|Center|Clinic|College|School|Trust|Authority|Bureau|"
    r"Commission|Agency|Ministry|Department)"
)
_COMPANY_RE = re.compile(
    rf"\b(?:[A-Z][a-zA-Z]*[-']?[a-zA-Z]*[\s-]?){{1,5}}{_COMPANY_SUFFIXES}\b"
)

# Simple email regex for pre-pass detection.  Presidio's EmailRecognizer
# rejects emails on public-suffix domains (gov.ie, nhs.uk, …) because
# tldextract returns an empty FQDN for them.  This pre-pass catches those
# before Presidio ever sees the text.
_EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w{2,}\b")

# Regex for title + name: captures the name portion (not the title itself).
# Handles both "Dr Smith" (UK) and "Dr. Smith" (US) via optional period.
_TITLE_NAME_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Rev|Reverend|"
    r"Sgt|Sergeant|Lt|Lieutenant|Cpl|Corporal|Capt|Captain|"
    r"Col|Colonel|Gen|General|Cmdr|Commander|Adm|Admiral|"
    r"Supt|Superintendent|Insp|Inspector|Det|Detective|"
    r"Hon|Honourable|Honorable|Sen|Senator|Rep|Representative|"
    r"Gov|Governor|Amb|Ambassador|Cllr|Councillor|"
    r"Fr|Father|Sr|Sister|Br|Brother|Dame|Lord|Lady|Sir)"
    r"\.?\s+"
    r"((?:(?:[A-Z][a-z]*(?:['\u2019-][A-Z][a-z]+)+|[A-Z][a-z]+)\s*){1,3})",
)

# Regex patterns for structured PII that Presidio either has no recognizer
# for or handles unreliably.  Applied as a pre-pass so the text is safely
# tagged before Presidio's NER can misclassify it.
_STRUCTURED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # URLs (match before phones so https://... isn't partially grabbed)
    ("URL", re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)),
    ("URL", re.compile(r"ftp://[^\s<>\"')\]]+", re.IGNORECASE)),
    ("URL", re.compile(r"\bwww\.[\w.-]+\.\w{2,}(?:/[^\s<>\"')\]]*)?", re.IGNORECASE)),
    # IP addresses (IPv4)
    ("IP", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    )),
    # Phone numbers: specific structural patterns only.
    ("PHONE", re.compile(
        r"(?:"
        # International: must start with + (strong indicator)
        r"\+\d{1,3}[\s.-]?\(?\d{1,5}\)?(?:[\s.-]?\d{1,5}){1,4}"
        r"|"
        # US/CA with parentheses: (555) 123-4567
        r"\(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}"
        r"|"
        # US/CA with separators: 555-123-4567 or 555.123.4567
        r"\b\d{3}[-]\d{3}[-]\d{4}\b"
        r"|"
        # UK landline: 020 1234 5678 / 01onal 123456
        r"\b0\d{2,4}[\s.-]\d{3,4}[\s.-]?\d{3,4}\b"
        r")"
    )),
    # UK National Insurance number (Presidio has NO recognizer for this).
    # First char: not D,F,I,Q,U,V.  Second char: not F,I,O,Q,U,V (D is valid).
    ("ID", re.compile(
        r"\b[A-CEGHJ-PR-TW-Z][A-EGHJ-NPR-TW-Z]"
        r"\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
        re.IGNORECASE,
    )),
    # US Social Security Number (Presidio's recognizer rejects test data)
    ("ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Credit card numbers: 4 groups of 4 digits (most common format)
    ("ID", re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b")),
    # Credit card numbers: 13-19 contiguous digits (validated by Luhn later)
    ("ID", re.compile(r"\b\d{13,19}\b")),
    # UK postcodes
    ("POSTCODE", re.compile(
        r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
        re.IGNORECASE,
    )),
    # Irish Eircode (routing key + unique ID, e.g. D02 VK60)
    ("POSTCODE", re.compile(
        r"\b[A-Z]\d[\dW]\s?[A-Z\d]{4}\b",
        re.IGNORECASE,
    )),
    # US ZIP+4 (e.g. 90210-1234) — unambiguous format
    ("POSTCODE", re.compile(r"\b\d{5}-\d{4}\b")),
    # US 5-digit ZIP after two-letter state abbreviation (e.g. "CA 90210")
    ("POSTCODE", re.compile(
        r"(?<=\b[A-Z]{2}\s)\d{5}\b"
    )),
    # Passport numbers: single letter + 7 digits, only when preceded by a
    # contextual keyword (e.g. "Passport number: L1234567").  Without context
    # the pattern is too broad and would match product codes / reference IDs.
    # Uses named group "pii" so the pre-pass extracts just the number.
    ("ID", re.compile(
        r"(?i)(?:passport|travel\s+document)\s*(?:no\.?|number|num|#)?[\s:]*"
        r"(?P<pii>[A-Z]\d{7})\b",
    )),
    # Dates (dd/mm/yyyy, mm-dd-yyyy, etc.) — pre-tagged so Presidio's
    # DateRecognizer cannot misclassify non-dates as DATE_TIME.
    ("DOB", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
]

# Map Presidio entity types → our internal category names.
_ENTITY_MAP: dict[str, str] = {
    "PERSON": "NAME",
    "LOCATION": "LOCATION",
    "ORG": "ORG",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "IP_ADDRESS": "IP",
    "URL": "URL",
    "DATE_TIME": "DOB",
    "US_SSN": "ID",
    "UK_NINO": "ID",
    "CREDIT_CARD": "ID",
    "IBAN_CODE": "ID",
}

# Entity types that need a lower score threshold to avoid misses.
_LENIENT_ENTITIES: frozenset[str] = frozenset({
    "PHONE_NUMBER", "US_SSN", "UK_NINO",
})
_LENIENT_THRESHOLD = 0.3

# Minimum entity length to avoid noisy short matches.
_MIN_ENTITY_LEN = 3

# Street / address words used to reclassify PERSON → LOCATION when spaCy
# misidentifies a street name as a person (e.g. "Oak Lane", "Anna Salai").
# Expanded based on benchmark data: Faker-generated addresses use suffixes
# like "Haven", "Hollow", "Inlet", "Colonnade" that spaCy tags as PERSON.
_ADDRESS_WORDS = frozenset({
    # English suffixes
    "street", "st", "road", "rd", "lane", "ln", "drive", "dr",
    "boulevard", "blvd", "court", "ct", "place", "pl", "way",
    "circle", "crescent", "terrace", "close", "green", "gardens",
    "park", "square", "row", "mews", "walk", "hill",
    # Additional suffixes seen in Faker-generated addresses
    "hollow", "haven", "glen", "amble", "colonnade", "cliffs",
    "inlet", "gates", "heights", "shores", "crest", "ridge",
    "crossing", "junction", "pass", "path", "bend", "apt", "suite",
    "circles", "hills", "formation", "locks", "knoll", "knolls",
    "flats", "pines", "manor", "meadow", "meadows", "vista",
    "run", "fork", "forks", "trace", "trail", "turnpike",
    "bypass", "overpass", "underpass", "burg", "burgh",
    # Prefixes (English / French / Spanish)
    "avenue", "ave", "rue", "via", "calle",
    # Directional prefixes
    "north", "south", "east", "west",
    # Place-name prefixes
    "lake", "port", "mount", "saint", "fort", "cape",
    # South Asian
    "salai", "marg", "nagar", "chowk",
})

# Regex for synthetic place names ending in common geographic suffixes
# (e.g. "Gonzalezland", "Christopherborough", "Smithville").
_PLACE_SUFFIX_RE = re.compile(
    r"(?:burg|burgh|borough|ville|stad|town|field|ford|bridge|"
    r"chester|shire|mouth|port|land|view|side|dale|wood|ston)$",
    re.IGNORECASE,
)

# Priority for resolving same-span entity conflicts (lower = higher priority).
_ENTITY_TYPE_PRIORITY: dict[str, int] = {
    "PERSON": 0,
    "EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1,
    "US_SSN": 1, "UK_NINO": 1,
    "IP_ADDRESS": 2, "URL": 2,
    "CREDIT_CARD": 2, "IBAN_CODE": 2,
    "DATE_TIME": 3,
    "LOCATION": 4,
    "ORG": 5,
}


def _looks_like_address(text: str) -> bool:
    """Return True if *text* contains address indicators.

    Checks for:
    1. First or last word is a known address word (e.g. "Oak Lane").
    2. Text ends with a geographic suffix (e.g. "Gonzalezland", "Smithville").
    """
    words = text.lower().split()
    if not words:
        return False
    if words[0] in _ADDRESS_WORDS or words[-1] in _ADDRESS_WORDS:
        return True
    # Check if any word has a geographic suffix (catches synthetic Faker places)
    for w in words:
        if _PLACE_SUFFIX_RE.search(w):
            return True
    return False


def _is_plausible_numeric_date(s: str) -> bool:
    """Check if a pure-digit string could plausibly be a date.

    Accepts 6-digit (DDMMYY / MMDDYY / YYMMDD) and 8-digit
    (DDMMYYYY / MMDDYYYY / YYYYMMDD) forms.  Returns False for
    numbers like ``600002`` or ``560001`` whose digit groups cannot
    form a valid month (1–12) and day (1–31).
    """
    if len(s) == 6:
        candidates = [(s[2:4], s[4:6]), (s[:2], s[2:4]), (s[2:4], s[:2])]
    elif len(s) == 8:
        candidates = [
            (s[4:6], s[6:8]),   # YYYYMMDD
            (s[2:4], s[:2]),    # DDMMYYYY
            (s[:2], s[2:4]),    # MMDDYYYY
        ]
    else:
        return False
    for m_s, d_s in candidates:
        if 1 <= int(m_s) <= 12 and 1 <= int(d_s) <= 31:
            return True
    return False


def _is_plausible_date(s: str) -> bool:
    """Check if a separator-based date string could plausibly be a date.

    Accepts dd/mm/yyyy, mm-dd-yyyy, etc.  Tries both DD/MM and MM/DD
    interpretations and returns True if at least one is valid.
    """
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", s)
    if not m:
        return False
    p1, p2, year_str = int(m.group(1)), int(m.group(2)), m.group(3)
    # 3-digit year is not a real date format.
    if len(year_str) == 3:
        return False
    # 4-digit year must be in a plausible range.
    if len(year_str) == 4 and not (1900 <= int(year_str) <= 2099):
        return False
    # Try DD/MM and MM/DD interpretations.
    for month, day in [(p1, p2), (p2, p1)]:
        if 1 <= month <= 12 and 1 <= day <= 31:
            return True
    return False


def _passes_luhn(s: str) -> bool:
    """Validate a number string using the Luhn algorithm (credit cards)."""
    digits = [int(d) for d in re.sub(r"\D", "", s)]
    if len(digits) < 13:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _is_plausible_phone(s: str) -> bool:
    """Check if a phone regex match is plausible (not a version number, etc.)."""
    digits = re.sub(r"\D", "", s)
    # ITU-T E.164: 7–15 digits for any real phone number.
    if len(digits) < 7 or len(digits) > 15:
        return False
    # Reject pure dot-separated short-group patterns (version numbers).
    if re.match(r"^\d{1,3}(\.\d{1,3}){2,}$", s):
        return False
    # Require at least one structural phone indicator:
    # + prefix, parentheses, or extension marker.
    if not re.search(r"[+()]", s) and "ext" not in s.lower() and "x" not in s.lower():
        # For numbers without structural indicators, require dash separators
        # in a phone-like grouping (3-3-4 or similar).
        if not re.match(r"^\d{3}[-]\d{3}[-]\d{4}$", s.strip()):
            return False
    return True


def _build_analyzer(model_name: str = "en_core_web_md") -> AnalyzerEngine:
    """Create a Presidio AnalyzerEngine with spaCy + custom ORG recognizer."""
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model_name}],
    })
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

    # Presidio's default SpacyRecognizer doesn't expose ORG.  Add one.
    org_recognizer = SpacyRecognizer(
        supported_entities=["ORG"],
        supported_language="en",
        check_label_groups=[({"ORG"}, {"ORG"})],
    )
    analyzer.registry.add_recognizer(org_recognizer)

    return analyzer


class PresidioDetector:
    """PII detector backed by Microsoft Presidio (spaCy NER + regex)."""

    def __init__(
        self,
        analyzer: AnalyzerEngine | None = None,
        score_threshold: float = 0.5,
        enabled_categories: set[str] | None = None,
    ) -> None:
        self._analyzer = analyzer or _build_analyzer()
        self._score_threshold = score_threshold
        self._enabled_categories = enabled_categories

    @property
    def nlp(self):
        """Expose the underlying spaCy Language object for reuse."""
        return self._analyzer.nlp_engine.nlp["en"]

    def pre_detect_emails(
        self,
        doc: Document,
        lookup: LookupTable,
    ) -> None:
        """Regex pre-pass that catches emails Presidio would miss.

        Must run before column-detector name replacements so that
        email addresses are safely tagged before name-derived
        variations (e.g. ``james.oconnor``) are replaced in the text.
        """
        for line in doc.lines:
            for match in _EMAIL_RE.finditer(line.text):
                original = match.group()
                if original.startswith("[") and original.endswith("]"):
                    continue
                existing = lookup.lookup(original)
                if existing:
                    doc.replace_all(original, existing)
                else:
                    tag = lookup.register(
                        original, "EMAIL",
                        doc.source_path.name, line.line_number,
                    )
                    doc.replace_all(original, tag)

    def pre_detect_orgs(
        self,
        doc: Document,
        lookup: LookupTable,
    ) -> None:
        """Heuristic pass that catches company names by corporate suffix."""
        for line in doc.lines:
            for match in _COMPANY_RE.finditer(line.text):
                original = match.group()
                if original.startswith("[") and original.endswith("]"):
                    continue
                existing = lookup.lookup(original)
                if existing:
                    doc.replace_all(original, existing)
                else:
                    tag = lookup.register(
                        original, "ORG",
                        doc.source_path.name, line.line_number,
                    )
                    doc.replace_all(original, tag)

    def pre_detect_titled_names(
        self,
        doc: Document,
        lookup: LookupTable,
    ) -> None:
        """Heuristic pass: title (Mr/Dr/Prof/…) + name → register as NAME.

        Extracts the name portion only (not the title itself).
        For multi-word names, individual words are registered as aliases.
        """
        for line in doc.lines:
            for match in _TITLE_NAME_RE.finditer(line.text):
                name = match.group(1).strip()
                if not name:
                    continue
                if name.startswith("[") and name.endswith("]"):
                    continue

                existing = lookup.lookup(name)
                if existing:
                    doc.replace_all(name, existing)
                else:
                    tag = lookup.register(
                        name, "NAME",
                        doc.source_path.name, line.line_number,
                    )
                    doc.replace_all(name, tag)

                    # Register individual words as aliases for multi-word names.
                    words = name.split()
                    if len(words) > 1:
                        for word in words:
                            if len(word) >= 3 and not lookup.lookup(word):
                                lookup.register_alias(
                                    word, tag, "NAME",
                                    doc.source_path.name, line.line_number,
                                )

    def pre_detect_patterns(
        self,
        doc: Document,
        lookup: LookupTable,
    ) -> None:
        """Regex pre-pass for structured PII that Presidio misses.

        Catches phones, NI numbers, SSNs, postcodes, IPs, and URLs
        before the main Presidio pass so they cannot be misclassified
        (e.g. SSN → DATE_TIME) or silently dropped.
        """
        for line in doc.lines:
            for category, pattern in _STRUCTURED_PATTERNS:
                for match in pattern.finditer(line.text):
                    # Prefer named group "pii" if present (e.g. passport
                    # regex captures just the number, not the keyword).
                    try:
                        original = match.group("pii")
                    except IndexError:
                        original = match.group()
                    if original.startswith("[") and original.endswith("]"):
                        continue
                    if category == "DOB" and not _is_plausible_date(original):
                        continue
                    if category == "PHONE" and not _is_plausible_phone(original):
                        continue
                    # Credit card / long-digit patterns: require Luhn validity.
                    if category == "ID" and re.fullmatch(r"[\d\s-]{13,}", original):
                        if not _passes_luhn(original):
                            continue
                    existing = lookup.lookup(original)
                    if existing:
                        doc.replace_all(original, existing)
                    else:
                        tag = lookup.register(
                            original, category,
                            doc.source_path.name, line.line_number,
                        )
                        doc.replace_all(original, tag)

    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        total = len(doc.lines)
        for i, line in enumerate(doc.lines):
            text = line.text
            if not text.strip():
                if on_progress:
                    on_progress(i + 1, total)
                continue

            results: list[RecognizerResult] = self._analyzer.analyze(
                text=text, language="en",
            )

            results = _remove_overlapping(results)

            for result in results:
                threshold = (
                    _LENIENT_THRESHOLD
                    if result.entity_type in _LENIENT_ENTITIES
                    else self._score_threshold
                )
                if result.score < threshold:
                    continue

                our_category = _ENTITY_MAP.get(result.entity_type)
                if our_category is None:
                    continue
                if (
                    self._enabled_categories
                    and our_category not in self._enabled_categories
                ):
                    continue

                original = text[result.start:result.end]
                if len(original) < _MIN_ENTITY_LEN:
                    continue
                # Skip text that is already an existing tag.
                if original.startswith("[") and original.endswith("]"):
                    continue

                # Reclassify street names that spaCy tagged as PERSON.
                if our_category == "NAME" and _looks_like_address(original):
                    our_category = "LOCATION"

                # Reject PERSON entities containing digits — real names
                # don't have digits, but IDs like "MRN-926439" do.
                if our_category == "NAME" and re.search(r"\d", original):
                    continue

                # Reject PERSON entities containing underscores (field names
                # like "date_of_birth").
                if our_category == "NAME" and "_" in original:
                    continue

                # Reject bare 4-digit years (e.g. "2022") — Presidio's
                # DATE_TIME catches these but they are not meaningful DOBs.
                if (
                    our_category == "DOB"
                    and re.fullmatch(r"\d{4}", original)
                ):
                    continue

                # Reject implausible numeric dates (e.g. postcodes 600002).
                if (
                    our_category == "DOB"
                    and original.isdigit()
                    and not _is_plausible_numeric_date(original)
                ):
                    continue

                existing = lookup.lookup(original)
                if existing:
                    doc.replace_all(original, existing)
                else:
                    tag = lookup.register(
                        original,
                        our_category,
                        doc.source_path.name,
                        line.line_number,
                    )
                    doc.replace_all(original, tag)

            if on_progress:
                on_progress(i + 1, total)


def _remove_overlapping(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Drop entities fully contained within a longer entity.

    Prevents e.g. a URL ``acme.com`` from being registered when it is
    already part of the longer EMAIL ``jane@acme.com``.  Keeps the
    longer (more specific) entity.  When two entities share the same
    span, the one with higher priority wins (PERSON > LOCATION > ORG).
    """
    results.sort(
        key=lambda r: (
            -(r.end - r.start),
            _ENTITY_TYPE_PRIORITY.get(r.entity_type, 99),
        ),
    )
    kept: list[RecognizerResult] = []
    for result in results:
        if any(result.start >= k.start and result.end <= k.end for k in kept):
            continue
        kept.append(result)
    return kept
