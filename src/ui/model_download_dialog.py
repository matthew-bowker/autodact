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

from src.config import AVAILABLE_MODELS, ModelInfo, model_is_cached
from src.ui.styles import (
    BORDER_LIGHT,
    ERROR_RED,
    PRIMARY_BLUE,
    PRIMARY_BLUE_HOVER,
    SUCCESS_GREEN,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    button_style,
)
from src.workers.download_worker import DownloadWorker


class _ModelRow(QHBoxLayout):
    """A single row showing model info + download/status indicator."""

    def __init__(self, model: ModelInfo) -> None:
        super().__init__()
        self.model = model

        label_text = (
            f"<b>{model.name}</b> &nbsp;"
            f"<span style='color:{TEXT_TERTIARY}'>{model.size_hint}</span><br/>"
            f"<span style='color:{TEXT_SECONDARY}'>{model.description}</span>"
        )
        self._label = QLabel(label_text)
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setWordWrap(True)
        self.addWidget(self._label, stretch=1)

        self._btn = QPushButton("Download")
        self._btn.setFixedWidth(120)
        self._btn.setStyleSheet(button_style(PRIMARY_BLUE, PRIMARY_BLUE_HOVER))
        self.addWidget(self._btn)

    @property
    def button(self) -> QPushButton:
        return self._btn

    def mark_downloaded(self) -> None:
        self._btn.setText("✓ Downloaded")
        self._btn.setEnabled(False)
        self._btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; "
            f"color: {SUCCESS_GREEN}; font-weight: bold; "
            f"border: 1px solid {SUCCESS_GREEN}; border-radius: 4px; padding: 6px 12px; }}"
        )

    def mark_downloading(self) -> None:
        self._btn.setText("Downloading…")
        self._btn.setEnabled(False)


class ModelDownloadDialog(QDialog):
    """Dialog offering download of the PII-detection model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download detection model")
        self.setMinimumWidth(540)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 16)

        # Header
        header = QLabel("One-time setup")
        header.setStyleSheet("font-size: 17px; font-weight: bold;")
        layout.addWidget(header)

        intro = QLabel(
            "Autodact runs a local PII-detection model on your machine — your "
            "data never leaves your computer. Download it once and it's reused "
            "every run."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(intro)

        # Separator
        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {BORDER_LIGHT};")
        layout.addWidget(sep)

        # Per-model rows
        self._rows: dict[str, _ModelRow] = {}
        for model in AVAILABLE_MODELS:
            row = _ModelRow(model)
            row.button.clicked.connect(lambda _checked, m=model: self._queue_single(m))
            layout.addLayout(row)
            self._rows[model.id] = row

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ border: none; border-radius: 4px; "
            f"background: {BORDER_LIGHT}; }}"
            f"QProgressBar::chunk {{ border-radius: 4px; "
            f"background-color: {PRIMARY_BLUE}; }}"
        )
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.hide()
        layout.addWidget(self._status_label)

        # Bottom row: local file + skip
        bottom = QHBoxLayout()
        self._browse_btn = QPushButton("Use local snapshot…")
        self._browse_btn.setStyleSheet(
            f"QPushButton {{ color: {TEXT_SECONDARY}; "
            f"background: transparent; border: 1px solid {BORDER_LIGHT}; "
            f"padding: 7px 14px; border-radius: 4px; }}"
            f"QPushButton:hover {{ background: #f4f4f4; }}"
        )
        self._browse_btn.clicked.connect(self._browse_local)
        bottom.addWidget(self._browse_btn)
        bottom.addStretch()
        self._skip_btn = QPushButton("Later")
        self._skip_btn.setStyleSheet(
            f"QPushButton {{ color: {TEXT_SECONDARY}; "
            f"background: transparent; border: 1px solid {BORDER_LIGHT}; "
            f"padding: 7px 14px; border-radius: 4px; }}"
            f"QPushButton:hover {{ background: #f4f4f4; }}"
        )
        self._skip_btn.setToolTip(
            "Close this dialog. You won't be able to start processing until "
            "the model is downloaded."
        )
        self._skip_btn.clicked.connect(self.reject)
        bottom.addWidget(self._skip_btn)
        layout.addLayout(bottom)

        self._queue: list[ModelInfo] = []
        self._thread: QThread | None = None
        self._worker: DownloadWorker | None = None

        self._refresh_rows()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _refresh_rows(self) -> None:
        for row in self._rows.values():
            if model_is_cached(row.model.repo):
                row.mark_downloaded()

    def _queue_single(self, model: ModelInfo) -> None:
        if model_is_cached(model.repo):
            return
        self._queue = [model]
        self._start_next()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._browse_btn.setEnabled(enabled)
        self._skip_btn.setEnabled(enabled)
        for row in self._rows.values():
            if row.button.text() not in ("✓ Downloaded", "Downloading…"):
                row.button.setEnabled(enabled)

    def _start_next(self) -> None:
        if not self._queue:
            self._progress_bar.setValue(100)
            self._status_label.setText("All set.")
            self._status_label.setStyleSheet(
                f"color: {SUCCESS_GREEN}; font-size: 11px;"
            )
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
        self._status_label.setText(f"Downloading {model.name}…")

        self._thread = QThread()
        self._worker = DownloadWorker(repo_id=model.repo)
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
            self._progress_bar.setRange(0, 0)
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
        self._status_label.setStyleSheet(f"color: {ERROR_RED}; font-size: 11px;")
        self._set_buttons_enabled(True)
        self._queue.clear()
        self._refresh_rows()

    # ------------------------------------------------------------------
    # Browse
    # ------------------------------------------------------------------

    def _browse_local(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select model snapshot directory", "",
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
