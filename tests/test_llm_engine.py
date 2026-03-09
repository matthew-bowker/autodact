"""Tests for LLMEngine._parse_response and Distil-PII helpers."""

import json

from src.pipeline.llm_engine import LLMEngine, build_user_prompt, _map_token_to_category


def test_parse_distil_pii_format():
    """Standard Distil-PII response with entities list."""
    data = json.dumps({
        "redacted_text": "[PERSON] works at Acme Corp.",
        "entities": [
            {"value": "Jane Smith", "replacement_token": "[PERSON]", "reason": "name"},
        ],
    })
    result = LLMEngine._parse_response(data)
    assert result == [{"original": "Jane Smith", "category": "NAME"}]


def test_parse_multiple_entities():
    data = json.dumps({
        "redacted_text": "[PERSON] lives at [ADDRESS].",
        "entities": [
            {"value": "Jane Smith", "replacement_token": "[PERSON]", "reason": "name"},
            {"value": "123 Main Street", "replacement_token": "[ADDRESS]", "reason": "address"},
        ],
    })
    result = LLMEngine._parse_response(data)
    assert len(result) == 2
    assert result[0] == {"original": "Jane Smith", "category": "NAME"}
    assert result[1] == {"original": "123 Main Street", "category": "LOCATION"}


def test_parse_bare_array():
    """Bare JSON array (no dict wrapper) is still handled."""
    data = json.dumps([
        {"value": "Jane Smith", "replacement_token": "[PERSON]", "reason": "name"},
    ])
    result = LLMEngine._parse_response(data)
    assert result == [{"original": "Jane Smith", "category": "NAME"}]


def test_parse_empty_array():
    result = LLMEngine._parse_response("[]")
    assert result == []


def test_parse_empty_entities_dict():
    data = json.dumps({"redacted_text": "nothing here", "entities": []})
    result = LLMEngine._parse_response(data)
    assert result == []


def test_parse_dict_wrapper():
    data = json.dumps({
        "entities": [{"value": "Jane", "replacement_token": "[PERSON]", "reason": "name"}],
    })
    result = LLMEngine._parse_response(data)
    assert result == [{"original": "Jane", "category": "NAME"}]


def test_parse_markdown_fenced():
    inner = json.dumps({
        "entities": [{"value": "Jane", "replacement_token": "[PERSON]", "reason": "name"}],
    })
    data = f"```json\n{inner}\n```"
    result = LLMEngine._parse_response(data)
    assert result == [{"original": "Jane", "category": "NAME"}]


def test_parse_with_extra_text():
    inner = json.dumps({
        "entities": [{"value": "Jane", "replacement_token": "[PERSON]", "reason": "name"}],
    })
    data = f"Here are the results: {inner} done."
    result = LLMEngine._parse_response(data)
    assert result == [{"original": "Jane", "category": "NAME"}]


def test_parse_skips_existing_tags():
    """Entities that look like existing replacement tags should be ignored."""
    data = json.dumps({
        "entities": [
            {"value": "EMAIL 1", "replacement_token": "[PERSON]", "reason": "tag"},
            {"value": "PHONE 23", "replacement_token": "[PERSON]", "reason": "tag"},
        ],
    })
    result = LLMEngine._parse_response(data)
    assert result == []


def test_parse_skips_short_entities():
    """Entities shorter than 4 chars are skipped to avoid over-replacement."""
    data = json.dumps({
        "entities": [
            {"value": "42", "replacement_token": "[ADDRESS]", "reason": "number"},
            {"value": "Bob", "replacement_token": "[PERSON]", "reason": "name"},
        ],
    })
    result = LLMEngine._parse_response(data)
    assert result == []


def test_parse_keeps_4char_entities():
    data = json.dumps({
        "entities": [{"value": "Jane", "replacement_token": "[PERSON]", "reason": "name"}],
    })
    result = LLMEngine._parse_response(data)
    assert result == [{"original": "Jane", "category": "NAME"}]


def test_parse_skips_category_words():
    """Bare category words like 'NAME', 'Location', 'PERSON' should be rejected."""
    data = json.dumps({
        "entities": [
            {"value": "NAME", "replacement_token": "[PERSON]", "reason": "word"},
            {"value": "PERSON", "replacement_token": "[PERSON]", "reason": "word"},
            {"value": "email", "replacement_token": "[EMAIL]", "reason": "word"},
        ],
    })
    result = LLMEngine._parse_response(data)
    assert result == []


def test_parse_filters_by_valid_categories():
    data = json.dumps({
        "entities": [
            {"value": "Jane", "replacement_token": "[PERSON]", "reason": "name"},
            {"value": "jane@example.com", "replacement_token": "[EMAIL]", "reason": "email"},
        ],
    })
    result = LLMEngine._parse_response(data, valid_categories={"NAME"})
    assert len(result) == 1
    assert result[0]["category"] == "NAME"


def test_parse_deduplicates():
    data = json.dumps({
        "entities": [
            {"value": "Jane", "replacement_token": "[PERSON]", "reason": "name"},
            {"value": "Jane", "replacement_token": "[PERSON]", "reason": "name again"},
        ],
    })
    result = LLMEngine._parse_response(data)
    assert len(result) == 1


def test_parse_invalid_json_returns_empty():
    """Unparseable text returns empty list (json_repair handles gracefully)."""
    result = LLMEngine._parse_response("not json at all")
    assert result == []


def test_parse_truncated_json_recovers():
    """Truncated JSON (e.g. from max_tokens cutoff) recovers partial entities."""
    truncated = '{"redacted_text": "[PERSON]", "entities": [{"value": "Jane", "replacement_token": "[PERSON]", "reason": "name"}, {"value": "Acme'
    result = LLMEngine._parse_response(truncated)
    # Should recover at least the complete first entity
    assert len(result) >= 1
    assert result[0] == {"original": "Jane", "category": "NAME"}


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------

def test_build_user_prompt_includes_text():
    prompt = build_user_prompt("Jane Smith works at Acme Corp.")
    assert "Jane Smith works at Acme Corp." in prompt
    assert "<context>" in prompt
    assert "<question>" in prompt


def test_build_user_prompt_template_structure():
    prompt = build_user_prompt("test text")
    assert prompt.startswith("Now for the real task")
    assert "<context>\ntest text\n</context>" in prompt


# ---------------------------------------------------------------------------
# _map_token_to_category
# ---------------------------------------------------------------------------

def test_map_token_person():
    assert _map_token_to_category("[PERSON]") == "NAME"
    assert _map_token_to_category("PERSON") == "NAME"


def test_map_token_address():
    assert _map_token_to_category("[ADDRESS]") == "LOCATION"


def test_map_token_email():
    assert _map_token_to_category("[EMAIL]") == "EMAIL"


def test_map_token_phone():
    assert _map_token_to_category("[PHONE]") == "PHONE"


def test_map_token_id_variants():
    assert _map_token_to_category("[SSN]") == "ID"
    assert _map_token_to_category("[ID]") == "ID"
    assert _map_token_to_category("[UUID]") == "ID"
    assert _map_token_to_category("[CREDIT_CARD]") == "ID"
    assert _map_token_to_category("[IBAN]") == "ID"


def test_map_token_with_suffix():
    """Tokens like [AGE_YEARS:29] or [CARD_LAST4:1234] are handled."""
    assert _map_token_to_category("[AGE_YEARS:29]") == "NAME"
    assert _map_token_to_category("[CARD_LAST4:1234]") == "ID"
    assert _map_token_to_category("[IBAN_LAST4:5678]") == "ID"


def test_map_token_unknown():
    assert _map_token_to_category("[UNKNOWN]") is None
    assert _map_token_to_category("NONSENSE") is None
