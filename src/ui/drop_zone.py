from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QFrame, QLabel, QVBoxLayout

from src.ui.styles import (
    BACKGROUND_ACTIVE,
    BACKGROUND_HIGHLIGHT,
    BACKGROUND_LIGHT,
    BORDER_DARK,
    PRIMARY_BLUE,
    RADIUS_LG,
    TEXT_TERTIARY,
)

SUPPORTED_EXTENSIONS = {".txt", ".csv", ".xlsx", ".docx"}
FILE_FILTER = "Supported files (*.txt *.csv *.xlsx *.docx);;All files (*)"


class DropZone(QFrame):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setMinimumHeight(120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel(
            "Drag & drop files here\n(.txt .csv .xlsx .docx)\n\nor click to browse"
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(f"color: {TEXT_TERTIARY}; font-size: 14px; line-height: 1.6;")
        layout.addWidget(self._label)

        self.setStyleSheet(
            f"DropZone {{ border: 2px dashed {BORDER_DARK}; border-radius: {RADIUS_LG}px; "
            f"background-color: {BACKGROUND_LIGHT}; }} "
            f"DropZone:hover {{ border-color: {PRIMARY_BLUE}; background-color: {BACKGROUND_HIGHLIGHT}; }}"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                f"DropZone {{ border: 2px solid {PRIMARY_BLUE}; border-radius: {RADIUS_LG}px; "
                f"background-color: {BACKGROUND_ACTIVE}; }}"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            f"DropZone {{ border: 2px dashed {BORDER_DARK}; border-radius: {RADIUS_LG}px; "
            f"background-color: {BACKGROUND_LIGHT}; }} "
            f"DropZone:hover {{ border-color: {PRIMARY_BLUE}; background-color: {BACKGROUND_HIGHLIGHT}; }}"
        )

    def dropEvent(self, event):
        self.dragLeaveEvent(event)
        paths = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(path)
        if paths:
            self.files_dropped.emit(paths)

    def mousePressEvent(self, event):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select files to anonymize", "", FILE_FILTER
        )
        if file_paths:
            paths = [Path(p) for p in file_paths if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS]
            if paths:
                self.files_dropped.emit(paths)
