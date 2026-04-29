import json
from pathlib import Path
from unittest.mock import patch

from src.config import (
    AVAILABLE_MODELS,
    AppConfig,
    get_app_data_dir,
    get_model_by_id,
    models_downloaded,
)


def test_default_config():
    config = AppConfig()
    assert config.output_format == "preserve"
    assert config.lookup_mode == "per_file"
    assert config.review_enabled is True
    assert config.model_path == ""
    assert "EMAIL" in config.enabled_categories
    assert config.device == "auto"
    assert config.max_line_chars == 500
    assert config.selected_model == "piranha"


def test_save_and_load(tmp_path: Path):
    config_path = tmp_path / "config.json"
    with patch("src.config.get_config_path", return_value=config_path):
        config = AppConfig(output_format="txt", lookup_mode="persist", device="cpu")
        config.save()
        assert config_path.exists()
        loaded = AppConfig.load()
        assert loaded.output_format == "txt"
        assert loaded.lookup_mode == "persist"
        assert loaded.device == "cpu"


def test_load_missing_file(tmp_path: Path):
    config_path = tmp_path / "nonexistent" / "config.json"
    with patch("src.config.get_config_path", return_value=config_path):
        config = AppConfig.load()
        assert config.output_format == "preserve"


def test_load_corrupt_json(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("not valid json{{{")
    with patch("src.config.get_config_path", return_value=config_path):
        config = AppConfig.load()
        assert config.output_format == "preserve"


def test_load_ignores_unknown_fields(tmp_path: Path):
    config_path = tmp_path / "config.json"
    data = {"output_format": "txt", "unknown_field": "value"}
    config_path.write_text(json.dumps(data))
    with patch("src.config.get_config_path", return_value=config_path):
        config = AppConfig.load()
        assert config.output_format == "txt"


def test_effective_model_source_default():
    config = AppConfig()
    assert config.effective_model_source() == AVAILABLE_MODELS[0].repo


def test_effective_model_source_custom_overrides_selected():
    """model_path (custom) always takes priority over selected_model."""
    config = AppConfig(
        selected_model="piranha", model_path="/local/path/to/snapshot",
    )
    assert config.effective_model_source() == "/local/path/to/snapshot"


def test_effective_model_source_unknown_selected_falls_back():
    config = AppConfig(selected_model="nonexistent")
    assert config.effective_model_source() == AVAILABLE_MODELS[0].repo


# ── Model registry ────────────────────────────────────────────────


def test_get_model_by_id_piranha():
    model = get_model_by_id("piranha")
    assert model is not None
    assert model.id == "piranha"
    assert "DeBERTa" in model.name or "Piranha" in model.name


def test_get_model_by_id_missing():
    assert get_model_by_id("nonexistent") is None


def test_models_downloaded_when_cached(tmp_path: Path):
    fake_cached_path = tmp_path / "config.json"
    fake_cached_path.write_text("{}")
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        return_value=str(fake_cached_path),
    ):
        downloaded = models_downloaded()
    assert AVAILABLE_MODELS[0].id in downloaded


def test_models_downloaded_when_missing():
    with patch(
        "huggingface_hub.try_to_load_from_cache",
        return_value=None,
    ):
        downloaded = models_downloaded()
    assert downloaded == []
