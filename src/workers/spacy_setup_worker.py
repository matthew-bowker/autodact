from __future__ import annotations

import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.config import get_app_data_dir
from src.pipeline.presidio_detector import PresidioDetector

logger = logging.getLogger(__name__)


def _make_ssl_context():
    """Create an SSL context that works in frozen PyInstaller builds."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download_spacy_model(model_name: str, target_dir: Path) -> None:
    """Download a spaCy model wheel from GitHub and extract it."""
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from spacy.cli.download import get_compatibility, get_version
        compat = get_compatibility()
        version = get_version(model_name, compat)
    except Exception:
        import spacy
        minor = ".".join(spacy.__version__.split(".")[:2])
        version = f"{minor}.0"

    whl_name = f"{model_name}-{version}-py3-none-any.whl"
    url = (
        f"https://github.com/explosion/spacy-models/releases/download/"
        f"{model_name}-{version}/{whl_name}"
    )

    logger.info("Downloading %s from %s", model_name, url)
    whl_path = target_dir / whl_name

    ctx = _make_ssl_context()
    with urllib.request.urlopen(url, context=ctx) as resp, open(whl_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    with zipfile.ZipFile(whl_path) as zf:
        zf.extractall(target_dir)

    whl_path.unlink()

    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))

    logger.info("spaCy model %s installed to %s", model_name, target_dir)


def _ensure_spacy_model(model_name: str = "en_core_web_lg") -> None:
    """Download the spaCy model if it is not already installed."""
    import spacy.util

    spacy_dir = get_app_data_dir() / "spacy_models"
    if spacy_dir.exists() and str(spacy_dir) not in sys.path:
        sys.path.insert(0, str(spacy_dir))

    if spacy.util.is_package(model_name):
        return

    if getattr(sys, "frozen", False):
        logger.info("Downloading spaCy model %s (frozen mode) ...", model_name)
        _download_spacy_model(model_name, spacy_dir)
    else:
        logger.info("Downloading spaCy model %s ...", model_name)
        from spacy.cli import download
        download(model_name)


class SpacySetupWorker(QObject):
    """Download spaCy model (if needed) and create PresidioDetector off the main thread."""

    finished = pyqtSignal(object)  # PresidioDetector instance
    error = pyqtSignal(str)

    @pyqtSlot()
    def run(self) -> None:
        try:
            _ensure_spacy_model()
            presidio = PresidioDetector()
            self.finished.emit(presidio)
        except Exception as e:
            self.error.emit(str(e))
