from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.config import get_models_dir


class DownloadWorker(QObject):
    """Download a single GGUF model from HuggingFace."""

    progress = pyqtSignal(str)   # status message
    finished = pyqtSignal(str)   # downloaded file path
    error = pyqtSignal(str)

    def __init__(
        self,
        repo_id: str,
        filename: str,
        local_name: str,
    ) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._filename = filename
        self._local_name = local_name

    @pyqtSlot()
    def run(self) -> None:
        try:
            from huggingface_hub import hf_hub_download

            models_dir = get_models_dir()
            models_dir.mkdir(parents=True, exist_ok=True)

            self.progress.emit(f"Downloading {self._local_name}...")

            path = hf_hub_download(
                repo_id=self._repo_id,
                filename=self._filename,
                local_dir=str(models_dir),
            )

            # Rename if the repo filename differs from our desired local name
            # (e.g. "model.gguf" → "Distil-PII-gemma-3-270m-it.gguf").
            downloaded = Path(path)
            target = models_dir / self._local_name
            if downloaded != target:
                downloaded.rename(target)
                path = str(target)

            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))
