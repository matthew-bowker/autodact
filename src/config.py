from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.power import default_thread_count

APP_NAME = "Autodact"

ALL_CATEGORIES = ["NAME", "ORG", "LOCATION", "JOBTITLE"]

# spaCy model used for NER.  "md" saves ~185 MB vs "lg" with negligible
# quality loss — the LLM, name dictionary, and regex layers compensate.
SPACY_MODEL = "en_core_web_md"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    repo: str
    filename: str       # filename inside the HuggingFace repo
    local_name: str     # filename we store locally (avoids generic "model.gguf")
    size_hint: str
    description: str


AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="fast",
        name="Fast (SmolLM2 135M)",
        repo="distil-labs/Distil-PII-SmolLM2-135M-Instruct-gguf",
        filename="model.gguf",
        local_name="Distil-PII-SmolLM2-135M-Instruct.gguf",
        size_hint="~540 MB",
        description="Lightweight and fast. Best for low-end hardware.",
    ),
    ModelInfo(
        id="standard",
        name="Standard (Llama 1B)",
        repo="mradermacher/Distil-PII-Llama-3.2-1B-Instruct-GGUF",
        filename="Distil-PII-Llama-3.2-1B-Instruct.Q4_K_M.gguf",
        local_name="Distil-PII-Llama-3.2-1B-Instruct.Q4_K_M.gguf",
        size_hint="~808 MB",
        description="More accurate. Best for thorough anonymisation.",
    ),
]

# Backward-compat aliases (used by a few imports elsewhere).
DEFAULT_MODEL_REPO = AVAILABLE_MODELS[0].repo
DEFAULT_MODEL_FILE = AVAILABLE_MODELS[0].local_name


def get_model_by_id(model_id: str) -> ModelInfo | None:
    """Look up a model by its short ID (e.g. ``"standard"``)."""
    for m in AVAILABLE_MODELS:
        if m.id == model_id:
            return m
    return None


def models_downloaded() -> list[str]:
    """Return IDs of models whose local file exists in the models dir."""
    models_dir = get_models_dir()
    return [m.id for m in AVAILABLE_MODELS if (models_dir / m.local_name).exists()]


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
    selected_model: str = "standard"
    model_path: str = ""          # non-empty ⇒ custom file overrides selected_model
    window_size: int = 2
    n_threads: int = field(default_factory=default_thread_count)
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

    def effective_model_path(self) -> Path:
        if self.model_path:
            return Path(self.model_path)
        model = get_model_by_id(self.selected_model)
        if model:
            return get_models_dir() / model.local_name
        return get_models_dir() / AVAILABLE_MODELS[0].local_name
