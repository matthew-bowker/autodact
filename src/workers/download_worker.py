from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.config import get_models_dir


class DownloadWorker(QObject):
    """Download a HuggingFace model snapshot into the app's models cache."""

    progress = pyqtSignal(int, str)  # pct (0-100, or -1 for indeterminate), message
    finished = pyqtSignal(str)       # local snapshot directory
    error = pyqtSignal(str)

    def __init__(self, repo_id: str) -> None:
        super().__init__()
        self._repo_id = repo_id

    @pyqtSlot()
    def run(self) -> None:
        try:
            from huggingface_hub import snapshot_download

            models_dir = get_models_dir()
            models_dir.mkdir(parents=True, exist_ok=True)

            self.progress.emit(-1, f"Downloading {self._repo_id}…")

            # snapshot_download writes its own tqdm progress to stderr; we
            # surface a single indeterminate message to the UI rather than
            # piping the per-file tqdm output through the Qt signal layer.
            local_dir = snapshot_download(
                repo_id=self._repo_id,
                cache_dir=str(models_dir),
                # Skip alternate-framework weights — we only use PyTorch
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.ckpt"],
            )

            self.finished.emit(str(local_dir))
        except Exception as e:
            self.error.emit(str(e))
