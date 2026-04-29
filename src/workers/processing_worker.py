from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable
from src.pipeline.orchestrator import Orchestrator, PipelineResult
from src.power import keep_awake, low_priority
from src.session.session_manager import SessionManager
from src.session.session_state import FileProgress, SessionState


class ProcessingWorker(QObject):
    progress = pyqtSignal(str, int, int)  # (phase, current, total)
    stats_updated = pyqtSignal(dict)  # category counts
    file_finished = pyqtSignal(object)  # PipelineResult
    all_finished = pyqtSignal()
    error = pyqtSignal(str)
    paused = pyqtSignal()  # Emitted when paused
    checkpoint = pyqtSignal(dict)  # In-memory progress updates

    def __init__(
        self,
        orchestrator: Orchestrator,
        file_paths: list[Path],
        lookup: LookupTable,
        output_dir: Path,
        preserve_format: bool,
        reset_per_file: bool,
        column_mappings: dict[str, dict[int, str]] | None = None,
        session_manager: SessionManager | None = None,
        config_snapshot: dict | None = None,
    ) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._file_paths = file_paths
        self._lookup = lookup
        self._output_dir = output_dir
        self._preserve_format = preserve_format
        self._reset_per_file = reset_per_file
        self._column_mappings = column_mappings or {}
        self._session_manager = session_manager or SessionManager()
        self._config_snapshot = config_snapshot or {}

        # Pause/resume state
        self._pause_requested = False
        self._resume_from_file_index: int | None = None
        self._resume_document: Document | None = None
        self._session_state: SessionState | None = None

    @pyqtSlot()
    def run(self) -> None:
        with keep_awake(), low_priority():
            self._run_inner()

    def _run_inner(self) -> None:
        try:
            # Create session state if not resuming
            if not self._session_state:
                self._session_state = SessionState.create_new(
                    self._file_paths,
                    self._config_snapshot,
                )

            lookup_filename = "session_lookup.csv" if not self._reset_per_file else None

            for i, file_path in enumerate(self._file_paths):
                # Skip completed files if resuming
                if self._resume_from_file_index is not None and i < self._resume_from_file_index:
                    continue

                # Check pause between files (safe checkpoint)
                if self._pause_requested:
                    self._save_checkpoint(current_file_index=i)
                    self.paused.emit()
                    return

                # Update file progress
                self._session_state.metadata.current_file_index = i
                self._session_state.file_progress[i].status = "processing"

                if self._reset_per_file:
                    self._lookup.reset()

                col_mapping = self._column_mappings.get(str(file_path))

                # Resume mid-file if applicable
                if i == self._resume_from_file_index and self._resume_document:
                    # TODO: Implement resume_file in orchestrator
                    # For now, just process normally
                    pass

                result = self._orchestrator.process_file(
                    file_path,
                    self._lookup,
                    self._output_dir,
                    preserve_format=self._preserve_format,
                    lookup_filename=lookup_filename,
                    column_mapping=col_mapping,
                    on_column_progress=lambda c, t: self.progress.emit(
                        "Column mapping", c, t
                    ),
                    on_presidio_progress=lambda c, t: (
                        self.progress.emit("Presidio pass", c, t),
                        self.stats_updated.emit(self._lookup.category_counts()),
                    ),
                    on_syntactic_progress=lambda c, t: (
                        self.progress.emit("Syntactic pass", c, t),
                        self.stats_updated.emit(self._lookup.category_counts()),
                    ),
                    on_custom_list_progress=lambda c, t: (
                        self.progress.emit("Custom lists", c, t),
                        self.stats_updated.emit(self._lookup.category_counts()),
                    ),
                    on_deberta_progress=lambda c, t: (
                        self.progress.emit("DeBERTa pass", c, t),
                        self.stats_updated.emit(self._lookup.category_counts()),
                    ),
                )

                # Mark file as completed
                self._session_state.file_progress[i].status = "completed"
                self.file_finished.emit(result)

            self.all_finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def request_pause(self) -> None:
        """Thread-safe pause request."""
        self._pause_requested = True

    def _save_checkpoint(self, current_file_index: int) -> None:
        """Save complete session state to disk."""
        if not self._session_state:
            return

        try:
            self._session_manager.save_session(
                self._session_state,
                self._lookup,
                current_document=None,  # TODO: Add current document if mid-file
            )
        except Exception as e:
            self.error.emit(f"Failed to save session: {e}")

    def resume_from_state(
        self,
        session_state: SessionState,
        resume_file_index: int,
        resume_document: Document | None = None,
    ) -> None:
        """Set resume parameters."""
        self._session_state = session_state
        self._resume_from_file_index = resume_file_index
        self._resume_document = resume_document
