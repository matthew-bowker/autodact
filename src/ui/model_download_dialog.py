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
from src.ui.styles import (
    PRIMARY_BLUE,
    PRIMARY_BLUE_HOVER,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    button_style,
)
from src.workers.download_worker import DownloadWorker


class _ModelRow(QHBoxLayout):
    """A single row showing model info + download/status button."""

    def __init__(self, model: ModelInfo) -> None:
        super().__init__()
        self.model = model

        label_text = (
            f"<b>{model.name}</b> &nbsp;<span style='color:{TEXT_TERTIARY}'>{model.size_hint}</span><br/>"
            f"<span style='color:{TEXT_SECONDARY}'>{model.description}</span>"
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
        self._btn.setText("✓ Downloaded")
        self._btn.setEnabled(False)
        self._btn.setStyleSheet("color: #5cb85c; font-weight: bold;")

    def mark_downloading(self) -> None:
        self._btn.setText("Downloading…")
        self._btn.setEnabled(False)


class ModelDownloadDialog(QDialog):
    """Dialog offering download of one or both PII-detection models."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("First-time Setup — Download Model")
        self.setMinimumWidth(520)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QLabel("AI model required")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #222;")
        layout.addWidget(header)

        intro = QLabel(
            "Autodact uses a local AI model to detect PII — your data never leaves "
            "your machine. Choose a model to download once; it's stored locally and "
            "reused on every run."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(intro)

        # Separator
        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #e8e8e8;")
        layout.addWidget(sep)

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

        # Progress area
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ border: none; border-radius: 4px; background: #e8e8e8; }}"
            f"QProgressBar::chunk {{ border-radius: 4px; background-color: {PRIMARY_BLUE}; }}"
        )
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.hide()
        layout.addWidget(self._status_label)

        # Browse / Skip
        bottom = QHBoxLayout()
        self._browse_btn = QPushButton("Use Local File…")
        self._browse_btn.clicked.connect(self._browse_local)
        bottom.addWidget(self._browse_btn)
        bottom.addStretch()
        self._skip_btn = QPushButton("Skip for now")
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

        self._refresh_rows()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh_rows(self) -> None:
        models_dir = get_models_dir()
        all_present = True
        for row in self._rows.values():
            if (models_dir / row.model.local_name).exists():
                row.mark_downloaded()
            else:
                all_present = False
        if all_present and len(AVAILABLE_MODELS) > 1:
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
        if len(AVAILABLE_MODELS) > 1:
            self._download_both_btn.setEnabled(enabled)
        self._browse_btn.setEnabled(enabled)
        self._skip_btn.setEnabled(enabled)
        for row in self._rows.values():
            if row.button.text() not in ("✓ Downloaded", "Downloading…"):
                row.button.setEnabled(enabled)

    def _start_next(self) -> None:
        if not self._queue:
            self._progress_bar.setValue(100)
            self._status_label.setText("Download complete — you're all set.")
            self._status_label.setStyleSheet("color: #5cb85c; font-size: 11px;")
            self._refresh_rows()
            self.accept()
            return

        model = self._queue.pop(0)
        self._rows[model.id].mark_downloading()
        self._set_buttons_enabled(False)

        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._status_label.show()
        self._status_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        self._status_label.setText(f"Starting download of {model.name}…")

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

    def _on_progress(self, pct: int, message: str) -> None:
        if pct == -1:
            self._progress_bar.setRange(0, 0)  # indeterminate
        else:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(pct)
        self._status_label.setText(message)

    def _on_download_finished(self, path: str) -> None:
        self._refresh_rows()
        self._start_next()

    def _on_download_error(self, message: str) -> None:
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        self._status_label.setText(f"Download failed: {message}")
        self._status_label.setStyleSheet("color: #d9534f; font-size: 11px;")
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
            self._selected_custom_path = path
            self._status_label.show()
            self._status_label.setText(f"Using: {Path(path).name}")
            self.accept()

    @property
    def custom_model_path(self) -> str | None:
        """Path chosen via Browse, if any. Check after dialog.exec()."""
        return getattr(self, "_selected_custom_path", None)
