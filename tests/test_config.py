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
    assert config.window_size == 2
    assert 2 <= config.n_threads <= 8  # default_thread_count(): half cores, clamped
    assert config.max_line_chars == 500


def test_save_and_load(tmp_path: Path):
    config_path = tmp_path / "config.json"
    with patch("src.config.get_config_path", return_value=config_path):
        config = AppConfig(output_format="txt", lookup_mode="persist", window_size=3)
        config.save()
        assert config_path.exists()
        loaded = AppConfig.load()
        assert loaded.output_format == "txt"
        assert loaded.lookup_mode == "persist"
        assert loaded.window_size == 3


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


def test_effective_model_path_default():
    config = AppConfig()
    path = config.effective_model_path()
    assert path.name == "Distil-PII-Llama-3.2-1B-Instruct.Q4_K_M.gguf"


def test_effective_model_path_custom():
    config = AppConfig(model_path="/custom/model.gguf")
    assert config.effective_model_path() == Path("/custom/model.gguf")


# ── Model registry ────────────────────────────────────────────────


def test_get_model_by_id_standard():
    model = get_model_by_id("standard")
    assert model is not None
    assert model.id == "standard"
    assert "Llama" in model.name


def test_get_model_by_id_missing():
    assert get_model_by_id("nonexistent") is None


def test_effective_model_path_custom_overrides_selected():
    """model_path (custom) always takes priority over selected_model."""
    config = AppConfig(selected_model="standard", model_path="/my/model.gguf")
    assert config.effective_model_path() == Path("/my/model.gguf")


def test_effective_model_path_unknown_selected_falls_back():
    config = AppConfig(selected_model="nonexistent")
    path = config.effective_model_path()
    # Falls back to first available model (standard)
    assert path.name == AVAILABLE_MODELS[0].local_name


def test_models_downloaded(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # Create only the first model file
    (models_dir / AVAILABLE_MODELS[0].local_name).touch()
    with patch("src.config.get_models_dir", return_value=models_dir):
        downloaded = models_downloaded()
    assert AVAILABLE_MODELS[0].id in downloaded


def test_models_downloaded_all_available(tmp_path: Path):
    """Test that all available models are detected when downloaded."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for m in AVAILABLE_MODELS:
        (models_dir / m.local_name).touch()
    with patch("src.config.get_models_dir", return_value=models_dir):
        downloaded = models_downloaded()
    assert "standard" in downloaded
    # Should contain all available model IDs
    assert len(downloaded) == len(AVAILABLE_MODELS)


def test_fast_model_selection(tmp_path: Path):
    """Test that 'fast' is a valid model selection."""
    config_path = tmp_path / "config.json"
    data = {"selected_model": "fast", "output_format": "txt"}
    config_path.write_text(json.dumps(data))
    with patch("src.config.get_config_path", return_value=config_path):
        loaded = AppConfig.load()
        assert loaded.selected_model == "fast"
        assert loaded.output_format == "txt"
