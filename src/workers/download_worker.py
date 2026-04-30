from __future__ import annotations

import contextlib
import io

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.config import get_models_dir, model_is_cached


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
            # In a windowed PyInstaller bundle stderr is detached, so tqdm's
            # progress writes raise BrokenPipeError partway through the
            # download. We render our own progress, so silence the library's
            # bars and swallow any remaining stderr writes.
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()

            from huggingface_hub import snapshot_download

            models_dir = get_models_dir()
            models_dir.mkdir(parents=True, exist_ok=True)

            self.progress.emit(-1, f"Downloading {self._repo_id}…")

            with contextlib.redirect_stderr(io.StringIO()):
                local_dir = snapshot_download(
                    repo_id=self._repo_id,
                    cache_dir=str(models_dir),
                    # Skip alternate-framework weights — we only use PyTorch
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.ckpt"],
                )

            self.finished.emit(str(local_dir))
        except Exception as e:
            # Some stderr writes happen at fd-level and bypass redirect_stderr.
            # If the cache is fully populated despite the error, the download
            # actually succeeded — don't show a misleading red error.
            if model_is_cached(self._repo_id):
                self.finished.emit(str(get_models_dir()))
                return
            self.error.emit(str(e))
