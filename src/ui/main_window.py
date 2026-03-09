from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.ui.drop_zone import DropZone
from src.ui.file_list_panel import FileListPanel
from src.ui.output_panel import OutputPanel
from src.ui.progress_panel import ProgressPanel
from src.ui.settings_panel import SettingsPanel
from src.ui.styles import ERROR_RED, PRIMARY_BLUE, PRIMARY_BLUE_HOVER, button_style

# Amber color for pause button
AMBER = "#f59e0b"
AMBER_HOVER = "#d97706"


class MainWindow(QMainWindow):
    files_dropped = pyqtSignal(list)
    start_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    resume_clicked = pyqtSignal()
    discard_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autodact")
        self.setMinimumSize(700, 650)

        # Add menu bar
        self._create_menu_bar()

        # Scrollable central area prevents panels from squishing
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(12)

        # Drop zone
        self.drop_zone = DropZone()
        layout.addWidget(self.drop_zone)

        # File list panel
        self.file_list_panel = FileListPanel()
        self.file_list_panel.files_changed.connect(self._on_file_list_changed)
        layout.addWidget(self.file_list_panel)

        # Settings
        self.settings_panel = SettingsPanel()
        layout.addWidget(self.settings_panel)

        # Start button
        self.start_button = QPushButton("Start Anonymization")
        self.start_button.setMinimumHeight(44)
        self.start_button.setStyleSheet(button_style(PRIMARY_BLUE, PRIMARY_BLUE_HOVER))
        self.start_button.setEnabled(False)
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.setToolTip("Begin processing selected files (Ctrl+Enter)")
        self.start_button.setShortcut("Ctrl+Return")
        layout.addWidget(self.start_button)

        # Pause button (shown during processing)
        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.setMinimumHeight(44)
        self.pause_button.setStyleSheet(button_style(AMBER, AMBER_HOVER))
        self.pause_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_button.setToolTip("Pause processing (Ctrl+P)")
        self.pause_button.setShortcut("Ctrl+P")
        self.pause_button.hide()
        layout.addWidget(self.pause_button)

        # Resume button (shown when paused)
        self.resume_button = QPushButton("▶ Resume Processing")
        self.resume_button.setMinimumHeight(44)
        self.resume_button.setStyleSheet(button_style(PRIMARY_BLUE, PRIMARY_BLUE_HOVER))
        self.resume_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.resume_button.setToolTip("Resume from where you left off (Ctrl+Enter)")
        self.resume_button.setShortcut("Ctrl+R")
        self.resume_button.hide()
        layout.addWidget(self.resume_button)

        # Discard button (shown when paused)
        self.discard_button = QPushButton("✗ Discard Session")
        self.discard_button.setMinimumHeight(36)
        self.discard_button.setStyleSheet(button_style(ERROR_RED, "#dc2626", text_color="white"))
        self.discard_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.discard_button.setToolTip("Delete saved session and start fresh")
        self.discard_button.hide()
        layout.addWidget(self.discard_button)

        # Progress
        self.progress_panel = ProgressPanel()
        layout.addWidget(self.progress_panel)

        # Output
        self.output_panel = OutputPanel()
        layout.addWidget(self.output_panel)

        layout.addStretch()
        scroll.setWidget(inner)
        self.setCentralWidget(scroll)

        # Wire internal signals
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        self.start_button.clicked.connect(self.start_clicked)
        self.pause_button.clicked.connect(self.pause_clicked)
        self.resume_button.clicked.connect(self.resume_clicked)
        self.discard_button.clicked.connect(self.discard_clicked)

    def _on_files_dropped(self, paths: list[Path]):
        # Append to existing selection
        existing = self.file_list_panel.get_files()
        existing_set = {p.resolve() for p in existing}
        new_files = [p for p in paths if p.resolve() not in existing_set]
        combined = existing + new_files
        self.file_list_panel.set_files(combined)
        self.start_button.setEnabled(bool(combined))
        self.files_dropped.emit(combined)

    def _on_file_list_changed(self, files: list[Path]):
        self.start_button.setEnabled(bool(files))
        self.files_dropped.emit(files)

    def set_file_list(self, files: list[Path]) -> None:
        self.file_list_panel.set_files(files)
        self.start_button.setEnabled(bool(files))

    def set_processing_state(self, is_processing: bool) -> None:
        # Hide all action buttons first
        self.start_button.hide()
        self.pause_button.hide()
        self.resume_button.hide()
        self.discard_button.hide()

        if is_processing:
            # Show pause button during processing
            self.pause_button.show()
            self.drop_zone.setEnabled(False)
            self.settings_panel.setEnabled(False)
            self.file_list_panel.setEnabled(False)
            self.progress_panel.reset()
            self.progress_panel.show()
            self.output_panel.clear()
        else:
            # Show start button when idle
            self.start_button.show()
            self.start_button.setEnabled(bool(self.file_list_panel.get_files()))
            self.drop_zone.setEnabled(True)
            self.settings_panel.setEnabled(True)
            self.file_list_panel.setEnabled(True)

    def set_progress(self, phase: str, current: int, total: int) -> None:
        self.progress_panel.set_progress(phase, current, total)

    def set_stats(self, stats: dict[str, int]) -> None:
        self.progress_panel.set_stats(stats)

    def set_complete(self, message: str = "Processing complete.") -> None:
        self.progress_panel.set_complete(message)

    def add_output_file(self, path: Path) -> None:
        self.output_panel.add_file(path)

    def get_settings(self) -> dict:
        return self.settings_panel.get_settings()

    def _create_menu_bar(self) -> None:
        """Create the application menu bar."""
        menubar = self.menuBar()

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = help_menu.addAction("About Autodact")
        about_action.triggered.connect(self._show_about_dialog)

    def _show_about_dialog(self) -> None:
        """Show the About/Disclaimer dialog."""
        from src.ui.about_dialog import AboutDialog

        dialog = AboutDialog(self)
        dialog.exec()

    def show_resume_option(self, session_id: str) -> None:
        """Show resume button if a saved session exists."""
        self.start_button.hide()
        self.resume_button.show()
        self.discard_button.show()
        self.resume_button.setEnabled(True)

    def set_paused_state(self) -> None:
        """Set UI to paused state."""
        self.pause_button.hide()
        self.resume_button.show()
        self.discard_button.show()
        self.drop_zone.setEnabled(False)
        self.settings_panel.setEnabled(False)
        self.file_list_panel.setEnabled(False)

    def hide_resume_option(self) -> None:
        """Hide resume/discard buttons and show start button."""
        self.resume_button.hide()
        self.discard_button.hide()
        self.start_button.show()
        self.start_button.setEnabled(bool(self.file_list_panel.get_files()))
