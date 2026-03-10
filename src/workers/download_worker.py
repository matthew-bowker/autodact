from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.config import get_models_dir


class DownloadWorker(QObject):
    """Download a single GGUF model from HuggingFace with real progress."""

    progress = pyqtSignal(int, str)  # pct (0-100, or -1 for indeterminate), message
    finished = pyqtSignal(str)       # downloaded file path
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
            import requests
            from huggingface_hub import hf_hub_url
            from huggingface_hub.utils import build_hf_headers

            models_dir = get_models_dir()
            models_dir.mkdir(parents=True, exist_ok=True)
            target = models_dir / self._local_name

            self.progress.emit(-1, "Connecting to HuggingFace…")

            url = hf_hub_url(self._repo_id, self._filename)
            headers = build_hf_headers()
            resp = requests.get(url, headers=headers, stream=True, timeout=30)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            start_time = time.monotonic()
            last_emit = 0.0

            with open(target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.monotonic()
                    if now - last_emit >= 0.25:
                        last_emit = now
                        elapsed = now - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        mb_done = downloaded / 1_048_576
                        if total:
                            pct = int(downloaded * 100 / total)
                            mb_total = total / 1_048_576
                            speed_str = f" · {speed / 1_048_576:.1f} MB/s" if speed > 0 else ""
                            msg = f"{mb_done:.0f} MB / {mb_total:.0f} MB{speed_str}"
                            self.progress.emit(pct, msg)
                        else:
                            self.progress.emit(-1, f"{mb_done:.0f} MB downloaded…")

            self.finished.emit(str(target))
        except Exception as e:
            self.error.emit(str(e))
