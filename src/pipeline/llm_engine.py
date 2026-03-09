from __future__ import annotations

import logging
import re

import json_repair

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distil-PII system prompt (fixed — from model training).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a problem solving model working on task_description XML block:
<task_description>
Produce a redacted version of texts, removing sensitive personal data while \
preserving operational signals. The model must return a single json blob with:

* **redacted_text** is the input with minimal, in-place replacements of \
redacted entities.
* **entities** as an array of objects with exactly three fields \
{value: original_value, replacement_token: replacement, reason: reasoning}.

## What to redact (-> replacement token)

* **PERSON** -- customer/patient/person names (first/last/full; identifying \
initials) -> `[PERSON]`
* **EMAIL** -- any email, including obfuscated `name(at)domain(dot)com` \
-> `[EMAIL]`
* **PHONE** -- any international/national format (separators/emoji bullets \
allowed) -> `[PHONE]`
* **ADDRESS** -- street + number; full postal lines; apartment/unit numbers \
-> `[ADDRESS]`
* **SSN** -- US Social Security numbers -> `[SSN]`
* **ID** -- national IDs (PESEL, NIN, Aadhaar, DNI, etc.) when personal \
-> `[ID]`
* **UUID** -- person-scoped system identifiers (e.g., MRN/NHS/patient \
IDs/customer UUIDs) -> `[UUID]`
* **CREDIT_CARD** -- 13-19 digits (spaces/hyphens allowed) \
-> `[CARD_LAST4:####]` (keep last-4 only)
* **IBAN** -- IBAN/bank account numbers -> `[IBAN_LAST4:####]` \
(keep last-4 only)
* **GENDER** -- self-identification (male/female/non-binary/etc.) \
-> `[GENDER]`
* **AGE** -- stated ages ("I'm 29", "age: 47", "29 y/o") \
-> `[AGE_YEARS:##]`
* **RACE** -- race/ethnicity self-identification -> `[RACE]`
* **MARITAL_STATUS** -- married/single/divorced/widowed/partnered \
-> `[MARITAL_STATUS]`

## Keep (do not redact)

* Card **last-4** when only last-4 is present (e.g., "ending 9021", \
"~~~~ 9021").
* Operational IDs: order/ticket/invoice numbers, shipment tracking, device \
serials, case IDs.
* Non-personal org info: company names, product names, team names.
* Cities/countries alone (redact full street+number, not plain city/country \
mentions).
</task_description>"""

USER_PROMPT_TEMPLATE = """\
Now for the real task, solve the task in question block based on the context \
in context block.
Generate only the solution, do not generate anything else
<context>
{context}
</context>
<question>Redact provided text according to the task description and return \
redacted elements.</question>"""

# ---------------------------------------------------------------------------
# Map Distil-PII replacement tokens → our internal categories.
# ---------------------------------------------------------------------------
_TOKEN_TO_CATEGORY: dict[str, str] = {
    "PERSON": "NAME",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "ADDRESS": "LOCATION",
    "SSN": "ID",
    "ID": "ID",
    "UUID": "ID",
    "CREDIT_CARD": "ID",
    "CARD_LAST4": "ID",
    "CARD": "ID",
    "IBAN": "ID",
    "IBAN_LAST4": "ID",
    "GENDER": "NAME",      # treat as sensitive personal info
    "AGE": "NAME",
    "AGE_YEARS": "NAME",
    "RACE": "NAME",
    "MARITAL_STATUS": "NAME",
}

# Matches text that looks like an existing replacement tag (e.g. "EMAIL 1")
_EXISTING_TAG = re.compile(r"^[A-Z]+ \d+$")

# Minimum entity length to avoid over-replacement of short strings
_MIN_ENTITY_LEN = 4

# Bare category/tag words that should never be treated as PII entities
_CATEGORY_WORDS = frozenset({
    "NAME", "ORG", "LOCATION", "JOBTITLE",
    "EMAIL", "PHONE", "DOB", "POSTCODE", "IP", "URL", "ID",
    "PERSON", "ADDRESS", "SSN", "UUID", "CREDIT_CARD", "IBAN",
    "GENDER", "AGE", "RACE", "MARITAL_STATUS",
})


def build_user_prompt(text: str) -> str:
    """Build the user-role message for the Distil-PII model."""
    return USER_PROMPT_TEMPLATE.format(context=text)


class LLMEngine:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 8,
    ) -> None:
        import os
        import sys

        from llama_cpp import Llama

        # Suppress C-level ggml/metal log spam to stderr
        stderr_fd = sys.stderr.fileno()
        old_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        try:
            self._llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)
            os.close(devnull)

    def detect_pii(
        self, user_prompt: str, categories: list[str]
    ) -> list[dict[str, str]]:
        """Send text to the Distil-PII model and return detected entities.

        Returns a list of ``{"original": ..., "category": ...}`` dicts
        using our internal category names.
        """
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"]["content"]
        logger.debug("LLM raw response: %s", content[:500])
        return self._parse_response(content, set(categories) if categories else None)

    @staticmethod
    def _parse_response(
        content: str, valid_categories: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Parse a Distil-PII JSON response into normalised entity dicts."""
        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        parsed = json_repair.loads(content)
        if parsed is None or (isinstance(parsed, str) and not parsed.strip()):
            logger.warning(
                "Failed to parse LLM response as JSON: %s", content[:200]
            )
            return []

        # Extract the entities list from the response dict.
        entities: list = []
        if isinstance(parsed, dict):
            entities = parsed.get("entities", [])
            if not isinstance(entities, list):
                entities = []
        elif isinstance(parsed, list):
            entities = parsed

        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in entities:
            if not isinstance(item, dict):
                continue

            # Distil-PII format: {"value": ..., "replacement_token": ..., "reason": ...}
            original = item.get("value", "") or item.get("original", "")
            token = item.get("replacement_token", "") or item.get("category", "")
            if not original or not token:
                continue

            # Map the replacement token to our internal category.
            category = _map_token_to_category(token)
            if not category:
                continue

            # Skip short entities that cause over-replacement
            if len(original) < _MIN_ENTITY_LEN:
                continue
            # Skip entities that look like existing replacement tags
            if _EXISTING_TAG.match(original):
                continue
            # Skip entities that contain existing [TAG] brackets — the LLM
            # sometimes re-tags already-anonymised text like "[POSTCODE 1]".
            if "[" in original:
                continue
            # Skip bare category words
            if original.upper() in _CATEGORY_WORDS:
                continue
            # Skip NAME entities containing digits — real names don't have
            # digits, but IDs like "F8183076" or "K-502256-S" do.
            if category == "NAME" and re.search(r"\d", original):
                continue
            # Skip categories not in the requested set
            if valid_categories and category not in valid_categories:
                continue

            if original not in seen:
                seen.add(original)
                results.append({"original": original, "category": category})

        return results


def _map_token_to_category(token: str) -> str | None:
    """Map a Distil-PII replacement token to our internal category.

    Handles tokens like ``[PERSON]``, ``[AGE_YEARS:29]``, ``PERSON``, etc.
    """
    # Strip brackets: "[CARD_LAST4:1234]" → "CARD_LAST4:1234"
    inner = token.strip("[]")
    # Strip value suffix: "CARD_LAST4:1234" → "CARD_LAST4"
    base = inner.split(":")[0].upper()
    # Try exact match first (handles CREDIT_CARD, MARITAL_STATUS, etc.)
    if base in _TOKEN_TO_CATEGORY:
        return _TOKEN_TO_CATEGORY[base]
    # Try first word before underscore (e.g. AGE_UNKNOWN → AGE)
    prefix = base.split("_")[0]
    return _TOKEN_TO_CATEGORY.get(prefix)
