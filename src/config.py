from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_NAME = "Autodact"

ALL_CATEGORIES = [
    "NAME", "ORG", "LOCATION", "JOBTITLE",
    "EMAIL", "PHONE", "ID", "DOB", "POSTCODE", "IP", "URL",
]

# spaCy model used for NER.  "md" saves ~185 MB vs "lg" with negligible
# quality loss — DeBERTa, the name dictionary, and regex layers compensate.
SPACY_MODEL = "en_core_web_md"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    repo: str           # HuggingFace repo id (e.g. "iiiorg/piiranha-v1-...")
    size_hint: str
    description: str


AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="piranha",
        name="Piranha v1 (mDeBERTa-v3-base)",
        repo="iiiorg/piiranha-v1-detect-personal-information",
        size_hint="~750 MB",
        description="Multilingual DeBERTa-v3 fine-tuned for PII detection.",
    ),
]

DEFAULT_MODEL_REPO = AVAILABLE_MODELS[0].repo


def get_model_by_id(model_id: str) -> ModelInfo | None:
    """Look up a model by its short ID (e.g. ``"piranha"``)."""
    for m in AVAILABLE_MODELS:
        if m.id == model_id:
            return m
    return None


def model_is_cached(repo: str) -> bool:
    """True when the HuggingFace repo is fully cached locally."""
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE  # noqa: F401

    # config.json is small and always present — its existence in the cache
    # is a reliable proxy for "the snapshot is downloaded".
    cached = try_to_load_from_cache(
        repo_id=repo,
        filename="config.json",
        cache_dir=str(get_models_dir()),
    )
    return isinstance(cached, str) and Path(cached).exists()


def models_downloaded() -> list[str]:
    """Return IDs of models whose weights are cached locally."""
    return [m.id for m in AVAILABLE_MODELS if model_is_cached(m.repo)]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_app_data_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif system == "Windows":
        base = Path(
            __import__("os").environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
        )
    else:
        base = Path.home() / ".config"
    return base / APP_NAME


def get_config_path() -> Path:
    return get_app_data_dir() / "config.json"


def get_models_dir() -> Path:
    """HuggingFace cache directory for downloaded encoder models."""
    return get_app_data_dir() / "models"


def get_sessions_dir() -> Path:
    return get_app_data_dir() / "sessions"


def get_custom_lists_path() -> Path:
    return get_app_data_dir() / "custom_lists.json"


# ---------------------------------------------------------------------------
# Application config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    output_format: str = "preserve"
    lookup_mode: str = "per_file"
    review_enabled: bool = True
    selected_model: str = "piranha"
    model_path: str = ""          # non-empty ⇒ local snapshot dir or HF repo override
    device: str = "auto"          # auto | cpu | mps | cuda
    max_line_chars: int = 500
    enabled_categories: list[str] = field(default_factory=lambda: list(ALL_CATEGORIES))
    custom_lists_enabled: bool = True
    fuzzy_matching_enabled: bool = False

    def save(self) -> None:
        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> AppConfig:
        path = get_config_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                known_fields = {f.name for f in cls.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in known_fields}

                return cls(**filtered)
            except (json.JSONDecodeError, TypeError):
                return cls()
        return cls()

    def effective_model_source(self) -> str:
        """Return the HF repo id or local path the detector should load.

        Precedence: explicit ``model_path`` (custom override) → selected model
        from ``AVAILABLE_MODELS`` → fall back to the first available entry.
        """
        if self.model_path:
            return self.model_path
        model = get_model_by_id(self.selected_model)
        if model:
            return model.repo
        return AVAILABLE_MODELS[0].repo
