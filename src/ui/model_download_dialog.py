from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from src.config import AVAILABLE_MODELS, ModelInfo, get_models_dir
from src.ui.styles import PRIMARY_BLUE, PRIMARY_BLUE_HOVER, button_style
from src.workers.download_worker import DownloadWorker


class _ModelRow(QHBoxLayout):
    """A single row showing model info + download/status button."""

    def __init__(self, model: ModelInfo) -> None:
        super().__init__()
        self.model = model

        label_text = (
            f"<b>{model.name}</b> ({model.size_hint})<br/>"
            f"<span style='color:#555'>{model.description}</span>"
        )
        self._label = QLabel(label_text)
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setWordWrap(True)
        self.addWidget(self._label, stretch=1)

        self._btn = QPushButton("Download")
        self._btn.setFixedWidth(120)
        self.addWidget(self._btn)

    @property
    def button(self) -> QPushButton:
        return self._btn

    def mark_downloaded(self) -> None:
        self._btn.setText("Downloaded")
        self._btn.setEnabled(False)

    def mark_downloading(self) -> None:
        self._btn.setText("Downloading…")
        self._btn.setEnabled(False)


class ModelDownloadDialog(QDialog):
    """Dialog offering download of one or both PII-detection models."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Setup")
        self.setMinimumWidth(500)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self._info_label = QLabel(
            "Choose a model for PII detection. The <b>Fast</b> model (135M params) "
            "runs well on low-end hardware. The <b>Standard</b> model (1B params) "
            "is more accurate but slower. You can switch in Settings at any time."
        )
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        # Per-model rows
        self._rows: dict[str, _ModelRow] = {}
        for model in AVAILABLE_MODELS:
            row = _ModelRow(model)
            row.button.clicked.connect(lambda _checked, m=model: self._queue_single(m))
            layout.addLayout(row)
            self._rows[model.id] = row

        # Download Both button (only show if multiple models available)
        if len(AVAILABLE_MODELS) > 1:
            self._download_both_btn = QPushButton("Download Both")
            self._download_both_btn.setDefault(True)
            self._download_both_btn.setStyleSheet(button_style(PRIMARY_BLUE, PRIMARY_BLUE_HOVER))
            self._download_both_btn.clicked.connect(self._queue_all)
            layout.addWidget(self._download_both_btn)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #555;")
        self._status_label.hide()
        layout.addWidget(self._status_label)

        # Browse / Skip
        bottom = QHBoxLayout()
        self._browse_btn = QPushButton("Browse for Local File…")
        self._browse_btn.clicked.connect(self._browse_local)
        bottom.addWidget(self._browse_btn)
        bottom.addStretch()
        self._skip_btn = QPushButton("Skip (configure later)")
        self._skip_btn.setStyleSheet(
            "QPushButton { color: #666; background: transparent; border: 1px solid #ccc; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background: #f0f0f0; }"
        )
        self._skip_btn.clicked.connect(self.reject)
        bottom.addWidget(self._skip_btn)
        layout.addLayout(bottom)

        # Internal state
        self._queue: list[ModelInfo] = []
        self._thread: QThread | None = None
        self._worker: DownloadWorker | None = None

        # Refresh already-downloaded state
        self._refresh_rows()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh_rows(self) -> None:
        models_dir = get_models_dir()
        all_present = True
        for model_id, row in self._rows.items():
            if (models_dir / row.model.local_name).exists():
                row.mark_downloaded()
            else:
                all_present = False
        if all_present:
            self._download_both_btn.setText("All Downloaded")
            self._download_both_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Queuing
    # ------------------------------------------------------------------

    def _queue_single(self, model: ModelInfo) -> None:
        models_dir = get_models_dir()
        if (models_dir / model.local_name).exists():
            return
        self._queue = [model]
        self._start_next()

    def _queue_all(self) -> None:
        models_dir = get_models_dir()
        self._queue = [
            m for m in AVAILABLE_MODELS
            if not (models_dir / m.local_name).exists()
        ]
        if not self._queue:
            self.accept()
            return
        self._start_next()

    # ------------------------------------------------------------------
    # Download lifecycle
    # ------------------------------------------------------------------

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._download_both_btn.setEnabled(enabled)
        self._browse_btn.setEnabled(enabled)
        self._skip_btn.setEnabled(enabled)
        for row in self._rows.values():
            if row.button.text() != "Downloaded":
                row.button.setEnabled(enabled)

    def _start_next(self) -> None:
        if not self._queue:
            # All queued downloads finished
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(100)
            self._status_label.setText("All downloads complete.")
            self._refresh_rows()
            self.accept()
            return

        model = self._queue.pop(0)
        self._rows[model.id].mark_downloading()
        self._set_buttons_enabled(False)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.show()
        self._status_label.show()
        self._status_label.setStyleSheet("color: #555;")
        self._status_label.setText(f"Downloading {model.name}…")

        self._thread = QThread()
        self._worker = DownloadWorker(
            repo_id=model.repo,
            filename=model.filename,
            local_name=model.local_name,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_download_finished)
        self._worker.error.connect(self._on_download_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _on_progress(self, message: str) -> None:
        self._status_label.setText(message)

    def _on_download_finished(self, path: str) -> None:
        self._refresh_rows()
        self._start_next()

    def _on_download_error(self, message: str) -> None:
        self._progress_bar.hide()
        self._status_label.setText(f"Download failed: {message}")
        self._status_label.setStyleSheet("color: red;")
        self._set_buttons_enabled(True)
        self._queue.clear()
        self._refresh_rows()

    # ------------------------------------------------------------------
    # Browse
    # ------------------------------------------------------------------

    def _browse_local(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF model file", "", "GGUF files (*.gguf);;All files (*)"
        )
        if path:
            # Store path — controller will pick it up via config.model_path
            self._selected_custom_path = path
            self._status_label.show()
            self._status_label.setText(f"Using model: {path}")
            self.accept()

    @property
    def custom_model_path(self) -> str | None:
        """Path chosen via Browse, if any. Check after dialog.exec()."""
        return getattr(self, "_selected_custom_path", None)
