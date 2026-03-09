from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.styles import (
    BACKGROUND_HIGHLIGHT,
    BACKGROUND_PANEL,
    BORDER_LIGHT,
    ERROR_RED,
    PRIMARY_BLUE,
    RADIUS_MD,
    RADIUS_SM,
    TEXT_MUTED,
)

FORMAT_ICONS = {
    ".txt": "TXT",
    ".csv": "CSV",
    ".xlsx": "XLSX",
    ".docx": "DOCX",
}


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.0f} {unit}" if unit == "B" else f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


class FileListPanel(QGroupBox):
    files_changed = pyqtSignal(list)  # emits updated list[Path]

    def __init__(self, parent=None):
        super().__init__("Selected files", parent)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(4)
        self._files: list[Path] = []
        self._rows: list[QWidget] = []

        self._empty_label = QLabel("No files selected")
        self._empty_label.setStyleSheet("color: #999; font-style: italic; padding: 4px;")
        self._layout.addWidget(self._empty_label)

        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.setFixedWidth(70)
        self._clear_btn.setStyleSheet("font-size: 11px;")
        self._clear_btn.clicked.connect(self.clear)
        self._clear_btn.hide()

        self.hide()

    def set_files(self, files: list[Path]) -> None:
        self._clear_rows()
        self._files = list(files)

        if not self._files:
            self._empty_label.show()
            self._clear_btn.hide()
            self.hide()
            return

        self._empty_label.hide()
        self.show()

        # Header row with clear button
        header = QHBoxLayout()
        count_label = QLabel(f"{len(self._files)} file(s) selected")
        count_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #333;")
        header.addWidget(count_label)
        header.addStretch()
        header.addWidget(self._clear_btn)
        self._clear_btn.show()
        header_widget = QWidget()
        header_widget.setLayout(header)
        self._layout.addWidget(header_widget)
        self._rows.append(header_widget)

        for path in self._files:
            row = self._make_file_row(path)
            self._layout.addWidget(row)
            self._rows.append(row)

    def _make_file_row(self, path: Path) -> QFrame:
        row = QFrame()
        row.setFrameStyle(QFrame.Shape.StyledPanel)
        row.setStyleSheet(
            f"QFrame {{ background-color: {BACKGROUND_PANEL}; "
            f"border: 1px solid {BORDER_LIGHT}; border-radius: {RADIUS_MD}px; "
            f"padding: 6px; }} "
            f"QFrame:hover {{ border-color: {PRIMARY_BLUE}; background-color: {BACKGROUND_HIGHLIGHT}; }}"
        )
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 4, 8, 4)
        row_layout.setSpacing(8)

        # Format badge
        ext = path.suffix.lower()
        badge_text = FORMAT_ICONS.get(ext, ext.upper().lstrip("."))
        badge = QLabel(badge_text)
        badge.setFixedWidth(42)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"background-color: {PRIMARY_BLUE}; color: white; font-weight: bold; "
            f"font-size: 11px; border-radius: {RADIUS_SM}px; padding: 2px 4px; border: none;"
        )
        row_layout.addWidget(badge)

        # File name
        name_label = QLabel(path.name)
        name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #222; border: none;")
        row_layout.addWidget(name_label)

        # File size
        try:
            size = _human_size(path.stat().st_size)
        except OSError:
            size = "?"
        size_label = QLabel(size)
        size_label.setStyleSheet("color: #888; font-size: 11px; border: none;")
        row_layout.addWidget(size_label)

        # Path (truncated)
        dir_label = QLabel(str(path.parent))
        dir_label.setStyleSheet("color: #aaa; font-size: 10px; border: none;")
        dir_label.setToolTip(str(path))
        row_layout.addWidget(dir_label, stretch=1)

        # Remove button
        remove_btn = QPushButton("x")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setStyleSheet(
            f"QPushButton {{ color: {TEXT_MUTED}; border: none; "
            f"font-weight: bold; font-size: 14px; background-color: transparent; "
            f"border-radius: {RADIUS_SM}px; }} "
            f"QPushButton:hover {{ color: {ERROR_RED}; background-color: #fee; }}"
        )
        remove_btn.setToolTip("Remove file")
        remove_btn.clicked.connect(lambda: self._remove_file(path))
        row_layout.addWidget(remove_btn)

        return row

    def _remove_file(self, path: Path) -> None:
        if path in self._files:
            self._files.remove(path)
            self.set_files(self._files)
            self.files_changed.emit(self._files)

    def clear(self) -> None:
        self._files = []
        self._clear_rows()
        self._empty_label.show()
        self._clear_btn.hide()
        self.hide()
        self.files_changed.emit(self._files)

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    def get_files(self) -> list[Path]:
        return list(self._files)
