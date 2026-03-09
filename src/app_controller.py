from __future__ import annotations

import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.config import AppConfig, get_app_data_dir
from src.pipeline.llm_detector import LLMDetector
from src.pipeline.llm_engine import LLMEngine
from src.pipeline.lookup_table import LookupTable
from src.pipeline.name_detector import NameDictionaryDetector
from src.pipeline.orchestrator import Orchestrator, PipelineResult
from src.pipeline.presidio_detector import PresidioDetector
from src.session.session_manager import SessionManager
from src.ui.main_window import MainWindow
from src.workers.model_load_worker import ModelLoadWorker
from src.workers.processing_worker import ProcessingWorker

logger = logging.getLogger(__name__)

_STRUCTURED_EXTS = {".csv", ".xlsx"}


def _download_spacy_model(model_name: str, target_dir: Path) -> None:
    """Download a spaCy model wheel from GitHub and extract it.

    Used in frozen PyInstaller builds where pip is not available.
    After extraction the model package lives in *target_dir* and is
    importable once *target_dir* is on ``sys.path``.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the compatible model version via spaCy's own machinery.
    try:
        from spacy.cli.download import get_compatibility, get_version
        compat = get_compatibility()
        version = get_version(model_name, compat)
    except Exception:
        # Fallback: guess from spaCy's minor version (usually correct).
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
    urllib.request.urlretrieve(url, whl_path)

    with zipfile.ZipFile(whl_path) as zf:
        zf.extractall(target_dir)

    whl_path.unlink()

    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))

    logger.info("spaCy model %s installed to %s", model_name, target_dir)


class AppController:
    def __init__(self, window: MainWindow, config: AppConfig) -> None:
        self._window = window
        self._config = config
        self._file_paths: list[Path] = []
        self._thread: QThread | None = None
        self._worker: ProcessingWorker | None = None
        self._model_load_worker: ModelLoadWorker | None = None
        self._llm_engine: LLMEngine | None = None
        self._presidio: PresidioDetector | None = None
        self._results: list[PipelineResult] = []
        self._column_mappings: dict[str, dict[int, str]] = {}
        self._session_manager = SessionManager()
        self._current_session_path: Path | None = None

        # Connect signals
        self._window.files_dropped.connect(self._on_files_dropped)
        self._window.start_clicked.connect(self._on_start)
        self._window.pause_clicked.connect(self._on_pause)
        self._window.resume_clicked.connect(self._on_resume)
        self._window.discard_clicked.connect(self._on_discard)
        self._window.settings_panel.settings_changed.connect(self._on_settings_changed)

    def initialize(self) -> None:
        # Check for existing sessions
        sessions = self._session_manager.list_sessions()
        if sessions:
            # Show resume option for most recent session
            self._window.show_resume_option(sessions[0])

        model_path = self._config.effective_model_path()
        if not model_path.exists():
            self._show_download_dialog()

    def _on_files_dropped(self, paths: list[Path]) -> None:
        self._file_paths = paths
        self._window.set_file_list(paths)

    def _on_settings_changed(self) -> None:
        settings = self._window.get_settings()
        self._config.output_format = settings["output_format"]
        self._config.lookup_mode = settings["lookup_mode"]
        self._config.review_enabled = settings["review_enabled"]
        # Invalidate cached engine when model selection changes.
        model_changed = (
            self._config.model_path != settings["model_path"]
            or self._config.selected_model != settings["selected_model"]
        )
        if model_changed:
            self._llm_engine = None
        self._config.selected_model = settings["selected_model"]
        self._config.model_path = settings["model_path"]
        self._config.window_size = settings["window_size"]
        self._config.enabled_categories = settings["enabled_categories"]
        self._config.save()

    def _on_start(self) -> None:
        if not self._file_paths:
            return

        model_path = self._config.effective_model_path()
        if not model_path.exists():
            self._show_download_dialog()
            if not model_path.exists():
                return

        # Show column mapping dialog for CSV/XLSX files.
        self._column_mappings = {}
        for fp in self._file_paths:
            if fp.suffix.lower() in _STRUCTURED_EXTS:
                mapping = self._show_column_mapping(fp)
                if mapping is None:
                    return  # User cancelled
                self._column_mappings[str(fp)] = mapping

        # Ensure Presidio is initialised (loads spaCy model once).
        # Presidio always detects all entity types (emails, phones, IPs, etc.)
        # regardless of the LLM category checkboxes.
        if self._presidio is None:
            self._ensure_spacy_model()
            self._presidio = PresidioDetector()

        if self._llm_engine is None:
            self._load_model_then_process(str(model_path))
        else:
            self._start_processing()

    # ------------------------------------------------------------------
    # Column mapping
    # ------------------------------------------------------------------

    def _show_column_mapping(self, file_path: Path) -> dict[int, str] | None:
        from src.ui.column_mapping_dialog import (
            ColumnMappingDialog,
            read_csv_headers_and_samples,
            read_xlsx_headers_and_samples,
        )

        ext = file_path.suffix.lower()
        if ext == ".csv":
            headers, samples = read_csv_headers_and_samples(file_path)
        elif ext == ".xlsx":
            headers, samples = read_xlsx_headers_and_samples(file_path)
        else:
            return {}

        if not headers:
            return {}

        dialog = ColumnMappingDialog(headers, samples, parent=self._window)
        if dialog.exec():
            return dialog.get_mapping()
        return None  # Cancelled

    # ------------------------------------------------------------------
    # spaCy / Presidio setup
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_spacy_model(model_name: str = "en_core_web_lg") -> None:
        """Download the spaCy model if it is not already installed."""
        import spacy.util

        # Make previously-downloaded models discoverable via sys.path.
        spacy_dir = get_app_data_dir() / "spacy_models"
        if spacy_dir.exists() and str(spacy_dir) not in sys.path:
            sys.path.insert(0, str(spacy_dir))

        if spacy.util.is_package(model_name):
            return

        if getattr(sys, "frozen", False):
            # Frozen PyInstaller app — pip is not available, download
            # the model wheel directly from GitHub and extract it.
            logger.info("Downloading spaCy model %s (frozen mode) ...", model_name)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                _download_spacy_model(model_name, spacy_dir)
            finally:
                QApplication.restoreOverrideCursor()
        else:
            logger.info("Downloading spaCy model %s ...", model_name)
            from spacy.cli import download

            download(model_name)

    # ------------------------------------------------------------------
    # LLM model loading
    # ------------------------------------------------------------------

    def _load_model_then_process(self, model_path: str) -> None:
        self._window.set_processing_state(True)
        self._window.set_progress("Loading model", 0, 1)

        self._model_load_thread = QThread()
        self._model_load_worker = ModelLoadWorker(
            model_path, n_threads=self._config.n_threads
        )
        self._model_load_worker.moveToThread(self._model_load_thread)

        self._model_load_thread.started.connect(self._model_load_worker.run)
        self._model_load_worker.finished.connect(self._on_model_loaded)
        self._model_load_worker.error.connect(self._on_model_load_error)
        self._model_load_worker.finished.connect(self._model_load_thread.quit)
        self._model_load_worker.error.connect(self._model_load_thread.quit)
        self._model_load_thread.finished.connect(self._on_model_thread_finished)

        self._model_load_thread.start()

    def _on_model_loaded(self, engine: LLMEngine) -> None:
        self._llm_engine = engine

    def _on_model_thread_finished(self) -> None:
        if self._llm_engine is not None:
            self._start_processing()

    def _on_model_load_error(self, message: str) -> None:
        self._window.set_processing_state(False)
        QMessageBox.critical(
            self._window, "Model Error",
            f"Failed to load LLM model:\n{message}"
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _start_processing(self) -> None:
        orchestrator = Orchestrator(
            presidio_detector=self._presidio,
            llm_detector=LLMDetector(
                self._llm_engine,
                categories=self._config.enabled_categories,
            ),
            name_detector=NameDictionaryDetector(),
            max_line_chars=self._config.max_line_chars,
        )
        lookup = LookupTable()

        output_dir = self._file_paths[0].parent

        # Create config snapshot for session
        from dataclasses import asdict
        config_snapshot = asdict(self._config)

        self._results = []
        self._thread = QThread()
        self._worker = ProcessingWorker(
            orchestrator=orchestrator,
            file_paths=self._file_paths,
            lookup=lookup,
            output_dir=output_dir,
            preserve_format=self._config.output_format == "preserve",
            reset_per_file=self._config.lookup_mode == "per_file",
            column_mappings=self._column_mappings,
            session_manager=self._session_manager,
            config_snapshot=config_snapshot,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.file_finished.connect(self._on_file_finished)
        self._worker.all_finished.connect(self._on_all_finished)
        self._worker.error.connect(self._on_error)
        self._worker.paused.connect(self._on_paused)
        self._worker.all_finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.paused.connect(self._thread.quit)

        self._window.set_processing_state(True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_progress(self, phase: str, current: int, total: int) -> None:
        self._window.set_progress(phase, current, total)

    def _on_stats(self, stats: dict[str, int]) -> None:
        self._window.set_stats(stats)

    def _on_file_finished(self, result: PipelineResult) -> None:
        self._results.append(result)
        self._window.add_output_file(result.output_path)
        self._window.add_output_file(result.lookup_path)

    def _on_all_finished(self) -> None:
        self._window.set_complete(
            f"Done. Processed {len(self._results)} file(s)."
        )

        if self._config.review_enabled and self._results:
            all_entries = []
            for r in self._results:
                all_entries.extend(r.lookup_table.all_entries())
            if all_entries:
                self._show_review_dialog()

        # Cleanup session on successful completion
        self._cleanup_session()

        self._window.set_processing_state(False)

    def _on_error(self, message: str) -> None:
        self._window.set_processing_state(False)
        QMessageBox.critical(
            self._window, "Processing Error",
            f"An error occurred during processing:\n{message}"
        )

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _show_download_dialog(self) -> None:
        from src.ui.model_download_dialog import ModelDownloadDialog

        dialog = ModelDownloadDialog(self._window)
        dialog.exec()
        # If the user browsed a local file, store it as custom model path.
        if dialog.custom_model_path:
            self._config.model_path = dialog.custom_model_path
            self._config.save()

    def _show_review_dialog(self) -> None:
        from src.ui.review_dialog import ReviewDialog

        all_entries = []
        for result in self._results:
            all_entries.extend(result.lookup_table.all_entries())

        if not all_entries:
            return

        dialog = ReviewDialog(all_entries, self._window)
        if dialog.exec():
            decisions = dialog.get_decisions()
            self._apply_review_decisions(decisions)

    def _apply_review_decisions(
        self, decisions: list[tuple[str, str, str]]
    ) -> None:
        for result in self._results:
            for original_term, action, new_category in decisions:
                if action == "reject":
                    tag = result.lookup_table.lookup(original_term)
                    if tag:
                        result.document.replace_all(tag, original_term)
                        result.lookup_table.remove(original_term)

            from src.pipeline.writers import write_file

            write_file(
                result.document,
                result.output_path,
                preserve_format=self._config.output_format == "preserve",
            )
            result.lookup_table.export_csv(result.lookup_path)

    # ------------------------------------------------------------------
    # Pause/Resume handlers
    # ------------------------------------------------------------------

    def _on_pause(self) -> None:
        """Handle pause button click."""
        if self._worker:
            self._worker.request_pause()

    def _on_paused(self) -> None:
        """Handle paused signal from worker."""
        self._window.set_paused_state()
        QMessageBox.information(
            self._window,
            "Processing Paused",
            "Processing has been paused. You can resume later or close the app.\n\n"
            "Your progress has been saved.",
        )

    def _on_resume(self) -> None:
        """Handle resume button click."""
        # Load most recent session
        sessions = self._session_manager.list_sessions()
        if not sessions:
            QMessageBox.warning(
                self._window, "No Session", "No saved session found."
            )
            return

        session_id = sessions[0]
        session_state = self._session_manager.load_session(session_id)
        if not session_state:
            QMessageBox.warning(
                self._window, "Load Error", "Failed to load session."
            )
            return

        # Validate session
        from dataclasses import asdict
        current_config = asdict(self._config)
        validation = self._session_manager.validate_session(
            session_state, current_config
        )

        if not validation.is_valid:
            QMessageBox.critical(
                self._window,
                "Cannot Resume",
                "Cannot resume session:\n\n" + "\n".join(validation.errors),
            )
            return

        if validation.warnings:
            reply = QMessageBox.warning(
                self._window,
                "Session Warnings",
                "Warnings about the saved session:\n\n"
                + "\n".join(validation.warnings)
                + "\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Restore file paths
        self._file_paths = [
            Path(p) for p in session_state.metadata.file_paths
            if Path(p).exists()
        ]

        # Restore lookup table
        lookup = self._session_manager.restore_lookup_table(session_state)

        # Determine resume point
        resume_file_index = session_state.metadata.current_file_index

        # Ensure model is loaded
        model_path = self._config.effective_model_path()
        if not model_path.exists():
            QMessageBox.critical(
                self._window, "Model Missing", "LLM model not found."
            )
            return

        if self._presidio is None:
            self._ensure_spacy_model()
            self._presidio = PresidioDetector()

        if self._llm_engine is None:
            self._load_model_then_resume(
                str(model_path), session_state, lookup, resume_file_index
            )
        else:
            self._resume_processing(session_state, lookup, resume_file_index)

    def _load_model_then_resume(
        self,
        model_path: str,
        session_state,
        lookup: LookupTable,
        resume_file_index: int,
    ) -> None:
        """Load model then resume processing."""
        self._window.set_processing_state(True)
        self._window.set_progress("Loading model", 0, 1)

        # Store resume parameters
        self._resume_params = (session_state, lookup, resume_file_index)

        self._model_load_thread = QThread()
        self._model_load_worker = ModelLoadWorker(
            model_path, n_threads=self._config.n_threads
        )
        self._model_load_worker.moveToThread(self._model_load_thread)

        self._model_load_thread.started.connect(self._model_load_worker.run)
        self._model_load_worker.finished.connect(self._on_model_loaded_for_resume)
        self._model_load_worker.error.connect(self._on_model_load_error)
        self._model_load_worker.finished.connect(self._model_load_thread.quit)
        self._model_load_worker.error.connect(self._model_load_thread.quit)

        self._model_load_thread.start()

    def _on_model_loaded_for_resume(self, engine: LLMEngine) -> None:
        """Handle model loaded for resume."""
        self._llm_engine = engine
        session_state, lookup, resume_file_index = self._resume_params
        self._resume_processing(session_state, lookup, resume_file_index)

    def _resume_processing(
        self, session_state, lookup: LookupTable, resume_file_index: int
    ) -> None:
        """Resume processing from saved state."""
        orchestrator = Orchestrator(
            presidio_detector=self._presidio,
            llm_detector=LLMDetector(
                self._llm_engine,
                categories=self._config.enabled_categories,
            ),
            name_detector=NameDictionaryDetector(),
            max_line_chars=self._config.max_line_chars,
        )

        output_dir = self._file_paths[0].parent

        from dataclasses import asdict
        config_snapshot = asdict(self._config)

        self._results = []
        self._thread = QThread()
        self._worker = ProcessingWorker(
            orchestrator=orchestrator,
            file_paths=self._file_paths,
            lookup=lookup,
            output_dir=output_dir,
            preserve_format=self._config.output_format == "preserve",
            reset_per_file=self._config.lookup_mode == "per_file",
            column_mappings=self._column_mappings,
            session_manager=self._session_manager,
            config_snapshot=config_snapshot,
        )

        # Set resume parameters
        self._worker.resume_from_state(
            session_state, resume_file_index, resume_document=None
        )

        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.file_finished.connect(self._on_file_finished)
        self._worker.all_finished.connect(self._on_all_finished)
        self._worker.error.connect(self._on_error)
        self._worker.paused.connect(self._on_paused)
        self._worker.all_finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.paused.connect(self._thread.quit)

        self._window.set_processing_state(True)
        self._thread.start()

    def _on_discard(self) -> None:
        """Handle discard button click."""
        reply = QMessageBox.question(
            self._window,
            "Discard Session",
            "Are you sure you want to discard the saved session?\n\n"
            "This will delete all progress and cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            sessions = self._session_manager.list_sessions()
            if sessions:
                self._session_manager.delete_session(sessions[0])
            self._window.hide_resume_option()

    def _cleanup_session(self) -> None:
        """Delete session on successful completion."""
        sessions = self._session_manager.list_sessions()
        if sessions:
            for session_id in sessions:
                self._session_manager.delete_session(session_id)
