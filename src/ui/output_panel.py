from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.styles import PRIMARY_BLUE, PRIMARY_BLUE_HOVER, RADIUS_SM, panel_style


class OutputPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Output files", parent)
        self.setStyleSheet(panel_style())
        self._layout = QVBoxLayout(self)
        self._entries: list[QWidget] = []
        self.hide()

    def add_file(self, path: Path) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        # Add file type icon
        icon = self._get_file_icon(path)
        icon_label = QLabel(icon)
        row_layout.addWidget(icon_label)

        label = QLabel(path.name)
        label.setToolTip(str(path))
        row_layout.addWidget(label, stretch=1)

        open_btn = QPushButton("Open")
        open_btn.setFixedWidth(70)
        open_btn.setStyleSheet(
            f"QPushButton {{ background-color: {PRIMARY_BLUE}; color: white; "
            f"font-size: 12px; border-radius: {RADIUS_SM}px; padding: 4px 8px; }} "
            f"QPushButton:hover {{ background-color: {PRIMARY_BLUE_HOVER}; }}"
        )
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))))
        row_layout.addWidget(open_btn)

        folder_btn = QPushButton("Show")
        folder_btn.setFixedWidth(70)
        folder_btn.setStyleSheet(
            f"QPushButton {{ background-color: {PRIMARY_BLUE}; color: white; "
            f"font-size: 12px; border-radius: {RADIUS_SM}px; padding: 4px 8px; }} "
            f"QPushButton:hover {{ background-color: {PRIMARY_BLUE_HOVER}; }}"
        )
        folder_btn.clicked.connect(lambda: self._reveal_in_finder(path))
        row_layout.addWidget(folder_btn)

        self._layout.addWidget(row)
        self._entries.append(row)
        self.show()

    def clear(self) -> None:
        for entry in self._entries:
            self._layout.removeWidget(entry)
            entry.deleteLater()
        self._entries.clear()
        self.hide()

    @staticmethod
    def _get_file_icon(path: Path) -> str:
        """Get emoji icon for file type."""
        ext = path.suffix.lower()
        if ext == ".csv":
            return "📊"
        elif ext == ".xlsx":
            return "📊"
        elif ext == ".docx":
            return "📝"
        else:
            return "📄"

    @staticmethod
    def _reveal_in_finder(path: Path) -> None:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
