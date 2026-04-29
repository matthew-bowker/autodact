from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
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
    BACKGROUND_PANEL,
    BORDER_LIGHT,
    PRIMARY_BLUE,
    PRIMARY_BLUE_HOVER,
    RADIUS_MD,
    RADIUS_SM,
    TEXT_PRIMARY,
    panel_style,
)


def _reveal_label() -> str:
    """Platform-appropriate label for revealing a file in its folder."""
    if sys.platform == "darwin":
        return "Show in Finder"
    if sys.platform == "win32":
        return "Show in Explorer"
    return "Show in folder"


def _open_dir(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", str(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _reveal_in_finder(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


_FILE_ICONS = {
    ".csv": "📊",
    ".xlsx": "📊",
    ".docx": "📝",
    ".txt": "📄",
}


class OutputPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Output files", parent)
        self.setStyleSheet(panel_style())
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(6)
        self._entries: list[QWidget] = []
        self._output_dir: Path | None = None

        # "Open output folder" button — added once at the top of the panel,
        # shown when the first output file lands.
        self._folder_btn = QPushButton("Open output folder")
        self._folder_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; "
            f"color: {PRIMARY_BLUE}; border: 1px solid {PRIMARY_BLUE}; "
            f"font-size: 12px; font-weight: bold; "
            f"border-radius: {RADIUS_SM}px; padding: 6px 12px; }} "
            f"QPushButton:hover {{ background-color: {PRIMARY_BLUE}; color: white; }}"
        )
        self._folder_btn.clicked.connect(self._on_open_folder)
        self._folder_btn.hide()
        self._layout.addWidget(self._folder_btn)

        self.hide()

    def add_file(self, path: Path) -> None:
        if self._output_dir is None:
            self._output_dir = path.parent
            self._folder_btn.show()

        row = QFrame()
        row.setStyleSheet(
            f"QFrame {{ background-color: {BACKGROUND_PANEL}; "
            f"border: 1px solid {BORDER_LIGHT}; "
            f"border-radius: {RADIUS_MD}px; padding: 4px; }}"
        )
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 4, 8, 4)
        row_layout.setSpacing(8)

        icon = _FILE_ICONS.get(path.suffix.lower(), "📄")
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("border: none;")
        row_layout.addWidget(icon_label)

        label = QLabel(path.name)
        label.setToolTip(str(path))
        label.setStyleSheet(f"color: {TEXT_PRIMARY}; border: none;")
        row_layout.addWidget(label, stretch=1)

        small_btn_style = (
            f"QPushButton {{ background-color: {PRIMARY_BLUE}; color: white; "
            f"font-size: 12px; border: none; "
            f"border-radius: {RADIUS_SM}px; padding: 4px 10px; }} "
            f"QPushButton:hover {{ background-color: {PRIMARY_BLUE_HOVER}; }}"
        )

        open_btn = QPushButton("Open")
        open_btn.setToolTip(f"Open {path.name}")
        open_btn.setStyleSheet(small_btn_style)
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        )
        row_layout.addWidget(open_btn)

        reveal_btn = QPushButton("Reveal")
        reveal_btn.setToolTip(_reveal_label())
        reveal_btn.setStyleSheet(small_btn_style)
        reveal_btn.clicked.connect(lambda: _reveal_in_finder(path))
        row_layout.addWidget(reveal_btn)

        self._layout.addWidget(row)
        self._entries.append(row)
        self.show()

    def clear(self) -> None:
        for entry in self._entries:
            self._layout.removeWidget(entry)
            entry.deleteLater()
        self._entries.clear()
        self._output_dir = None
        self._folder_btn.hide()
        self.hide()

    def _on_open_folder(self) -> None:
        if self._output_dir is not None:
            _open_dir(self._output_dir)
