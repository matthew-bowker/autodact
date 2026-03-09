from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class FileProgress:
    """Progress information for a single file."""

    file_path: str
    status: str  # "pending", "processing", "completed"
    stage: str | None = None  # "parsing", "llm_detection", "writing", etc.
    current_line: int = 0
    total_lines: int = 0


@dataclass
class SessionMetadata:
    """Metadata about the processing session."""

    session_id: str
    created_at: str  # ISO format timestamp
    paused_at: str | None = None  # ISO format timestamp
    file_paths: list[str] = field(default_factory=list)
    current_file_index: int = 0
    config_snapshot: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of session validation."""

    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.warnings or self.errors)


@dataclass
class SessionState:
    """Complete state of a processing session."""

    metadata: SessionMetadata
    file_progress: list[FileProgress] = field(default_factory=list)
    lookup_table_data: dict | None = None
    current_document_data: dict | None = None
    current_stage: str | None = None
    current_line: int = 0

    @classmethod
    def create_new(
        cls,
        file_paths: list[Path],
        config_snapshot: dict,
    ) -> SessionState:
        """Create a new session state."""
        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        created_at = datetime.now().isoformat()

        metadata = SessionMetadata(
            session_id=session_id,
            created_at=created_at,
            file_paths=[str(p) for p in file_paths],
            current_file_index=0,
            config_snapshot=config_snapshot,
        )

        file_progress = [
            FileProgress(
                file_path=str(p),
                status="pending",
            )
            for p in file_paths
        ]

        return cls(
            metadata=metadata,
            file_progress=file_progress,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "metadata": {
                "session_id": self.metadata.session_id,
                "created_at": self.metadata.created_at,
                "paused_at": self.metadata.paused_at,
                "file_paths": self.metadata.file_paths,
                "current_file_index": self.metadata.current_file_index,
                "config_snapshot": self.metadata.config_snapshot,
            },
            "file_progress": [
                {
                    "file_path": fp.file_path,
                    "status": fp.status,
                    "stage": fp.stage,
                    "current_line": fp.current_line,
                    "total_lines": fp.total_lines,
                }
                for fp in self.file_progress
            ],
            "lookup_table_data": self.lookup_table_data,
            "current_document_data": self.current_document_data,
            "current_stage": self.current_stage,
            "current_line": self.current_line,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionState:
        """Deserialize from dictionary."""
        metadata = SessionMetadata(
            session_id=data["metadata"]["session_id"],
            created_at=data["metadata"]["created_at"],
            paused_at=data["metadata"].get("paused_at"),
            file_paths=data["metadata"]["file_paths"],
            current_file_index=data["metadata"]["current_file_index"],
            config_snapshot=data["metadata"]["config_snapshot"],
        )

        file_progress = [
            FileProgress(
                file_path=fp["file_path"],
                status=fp["status"],
                stage=fp.get("stage"),
                current_line=fp.get("current_line", 0),
                total_lines=fp.get("total_lines", 0),
            )
            for fp in data["file_progress"]
        ]

        return cls(
            metadata=metadata,
            file_progress=file_progress,
            lookup_table_data=data.get("lookup_table_data"),
            current_document_data=data.get("current_document_data"),
            current_stage=data.get("current_stage"),
            current_line=data.get("current_line", 0),
        )
