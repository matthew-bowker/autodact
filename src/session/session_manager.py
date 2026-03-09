from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from src.config import get_sessions_dir
from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable
from src.session.session_state import SessionState, ValidationResult


class SessionManager:
    """Manages persistence and restoration of processing sessions."""

    def __init__(self) -> None:
        self._sessions_dir = get_sessions_dir()
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(
        self,
        session_state: SessionState,
        lookup_table: LookupTable,
        current_document: Document | None = None,
    ) -> Path:
        """Save complete session state atomically to disk.

        Returns the session directory path.
        """
        session_id = session_state.metadata.session_id
        session_dir = self._sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create .incomplete marker for atomic save
        incomplete_marker = session_dir / ".incomplete"
        incomplete_marker.touch()

        try:
            # Update paused timestamp
            session_state.metadata.paused_at = datetime.now().isoformat()

            # Serialize lookup table
            session_state.lookup_table_data = lookup_table.to_dict()

            # Serialize current document if provided
            if current_document:
                session_state.current_document_data = current_document.to_dict()

            # Write session metadata
            metadata_path = session_dir / "session_metadata.json"
            metadata_path.write_text(
                json.dumps(session_state.to_dict(), indent=2),
                encoding="utf-8"
            )

            # Remove .incomplete marker (indicates success)
            incomplete_marker.unlink()

            return session_dir

        except Exception:
            # Cleanup on failure
            if session_dir.exists():
                shutil.rmtree(session_dir)
            raise

    def load_session(self, session_id: str) -> SessionState | None:
        """Load and deserialize session state.

        Returns None if session doesn't exist or is incomplete.
        """
        session_dir = self._sessions_dir / session_id

        # Check if session exists
        if not session_dir.exists():
            return None

        # Reject incomplete sessions
        if (session_dir / ".incomplete").exists():
            return None

        # Load session metadata
        metadata_path = session_dir / "session_metadata.json"
        if not metadata_path.exists():
            return None

        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            return SessionState.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def restore_lookup_table(self, session_state: SessionState) -> LookupTable:
        """Restore LookupTable from session state."""
        if not session_state.lookup_table_data:
            return LookupTable()
        return LookupTable.from_dict(session_state.lookup_table_data)

    def restore_document(self, session_state: SessionState) -> Document | None:
        """Restore Document from session state."""
        if not session_state.current_document_data:
            return None
        return Document.from_dict(session_state.current_document_data)

    def list_sessions(self) -> list[str]:
        """List available session IDs, sorted by creation time (newest first)."""
        sessions = []
        for session_dir in self._sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            # Skip incomplete sessions
            if (session_dir / ".incomplete").exists():
                continue
            # Check for metadata file
            if (session_dir / "session_metadata.json").exists():
                sessions.append(session_dir.name)

        # Sort by directory name (which contains timestamp)
        return sorted(sessions, reverse=True)

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its data."""
        session_dir = self._sessions_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def validate_session(
        self,
        session_state: SessionState,
        current_config: dict,
    ) -> ValidationResult:
        """Validate session against current environment.

        Checks:
        - Missing files
        - Model changes
        - Config differences
        """
        warnings: list[str] = []
        errors: list[str] = []

        # Check for missing files
        missing_files = []
        for file_path in session_state.metadata.file_paths:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            if len(missing_files) == len(session_state.metadata.file_paths):
                errors.append("All files from the session are missing.")
            else:
                warnings.append(
                    f"{len(missing_files)} file(s) are missing and will be skipped: "
                    + ", ".join(Path(f).name for f in missing_files[:3])
                    + ("..." if len(missing_files) > 3 else "")
                )

        # Check for model changes
        saved_model = session_state.metadata.config_snapshot.get("selected_model")
        saved_model_path = session_state.metadata.config_snapshot.get("model_path")
        current_model = current_config.get("selected_model")
        current_model_path = current_config.get("model_path")

        if saved_model_path != current_model_path or saved_model != current_model:
            warnings.append(
                f"Model has changed (was: {saved_model or saved_model_path}, "
                f"now: {current_model or current_model_path})"
            )

        # Check for config differences
        config_diffs = []
        for key in ["enabled_categories", "window_size"]:
            saved_value = session_state.metadata.config_snapshot.get(key)
            current_value = current_config.get(key)
            if saved_value != current_value:
                config_diffs.append(f"{key}: {saved_value} → {current_value}")

        if config_diffs:
            warnings.append(
                "Settings have changed: " + ", ".join(config_diffs)
            )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, warnings=warnings, errors=errors)

    def cleanup_old_sessions(self, days: int = 7) -> None:
        """Delete sessions older than specified days."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)

        for session_dir in self._sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            # Check modification time
            if session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir)
