from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from src.config import AVAILABLE_MODELS, get_models_dir
from src.ui.styles import panel_style


class SettingsPanel(QGroupBox):
    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Settings", parent)
        self.setStyleSheet(panel_style())

        layout = QVBoxLayout(self)

        # Output format
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Output format:"))
        self._fmt_preserve = QRadioButton("Preserve original")
        self._fmt_txt = QRadioButton("Plain text (.txt)")
        self._fmt_preserve.setChecked(True)
        fmt_group = QButtonGroup(self)
        fmt_group.addButton(self._fmt_preserve)
        fmt_group.addButton(self._fmt_txt)
        fmt_row.addWidget(self._fmt_preserve)
        fmt_row.addWidget(self._fmt_txt)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        # Lookup table mode
        lookup_row = QHBoxLayout()
        lookup_row.addWidget(QLabel("Lookup table:"))
        self._lookup_persist = QRadioButton("Persist across files")
        self._lookup_per_file = QRadioButton("Reset per file")
        self._lookup_per_file.setChecked(True)
        lookup_group = QButtonGroup(self)
        lookup_group.addButton(self._lookup_persist)
        lookup_group.addButton(self._lookup_per_file)
        lookup_row.addWidget(self._lookup_persist)
        lookup_row.addWidget(self._lookup_per_file)
        lookup_row.addStretch()
        layout.addLayout(lookup_row)

        # Review step toggle
        review_row = QHBoxLayout()
        self._review_check = QCheckBox("Enable human review step")
        self._review_check.setChecked(True)
        self._review_check.setToolTip("When enabled, you can review and edit detected PII before saving")
        review_row.addWidget(self._review_check)
        review_row.addStretch()
        layout.addLayout(review_row)

        # Model selection — radio buttons per registered model + custom
        model_label_row = QHBoxLayout()
        model_label_row.addWidget(QLabel("LLM model:"))
        model_label_row.addStretch()
        layout.addLayout(model_label_row)

        self._model_group = QButtonGroup(self)
        self._model_radios: dict[str, QRadioButton] = {}
        models_dir = get_models_dir()

        for model in AVAILABLE_MODELS:
            downloaded = (models_dir / model.local_name).exists()
            suffix = "" if downloaded else "  (not downloaded)"
            radio = QRadioButton(f"{model.name}{suffix}")
            self._model_group.addButton(radio)
            self._model_radios[model.id] = radio
            layout.addWidget(radio)

        # Custom file option
        self._model_custom_radio = QRadioButton("Custom file…")
        self._model_group.addButton(self._model_custom_radio)
        layout.addWidget(self._model_custom_radio)

        # Custom file path row (hidden unless Custom is selected)
        self._custom_row = QHBoxLayout()
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("Path to .gguf file")
        self._custom_row.addWidget(self._model_path)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._browse_model)
        self._custom_row.addWidget(self._browse_btn)
        layout.addLayout(self._custom_row)

        # Default: Standard selected, custom path row hidden
        if "standard" in self._model_radios:
            self._model_radios["standard"].setChecked(True)
        self._set_custom_visible(False)

        self._model_custom_radio.toggled.connect(self._on_custom_toggled)

        # LLM detection categories
        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("Detect:"))
        self._cat_name = QCheckBox("Names")
        self._cat_org = QCheckBox("Organisations")
        self._cat_location = QCheckBox("Locations")
        self._cat_jobtitle = QCheckBox("Job titles")
        self._cat_name.setChecked(True)
        self._cat_org.setChecked(True)
        self._cat_location.setChecked(True)
        self._cat_jobtitle.setChecked(False)
        cat_row.addWidget(self._cat_name)
        cat_row.addWidget(self._cat_org)
        cat_row.addWidget(self._cat_location)
        cat_row.addWidget(self._cat_jobtitle)
        cat_row.addStretch()
        layout.addLayout(cat_row)

        # Window size
        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("Context window:"))
        self._window_size = QSpinBox()
        self._window_size.setRange(0, 10)
        self._window_size.setValue(2)
        self._window_size.setSuffix(" lines above/below")
        self._window_size.setToolTip("Number of surrounding lines provided to the AI for context")
        window_row.addWidget(self._window_size)
        window_row.addStretch()
        layout.addLayout(window_row)

        # Connect signals
        self._fmt_preserve.toggled.connect(lambda: self.settings_changed.emit())
        self._lookup_persist.toggled.connect(lambda: self.settings_changed.emit())
        self._review_check.toggled.connect(lambda: self.settings_changed.emit())
        self._model_group.buttonToggled.connect(lambda: self.settings_changed.emit())
        self._model_path.textChanged.connect(lambda: self.settings_changed.emit())
        self._cat_name.toggled.connect(lambda: self.settings_changed.emit())
        self._cat_org.toggled.connect(lambda: self.settings_changed.emit())
        self._cat_location.toggled.connect(lambda: self.settings_changed.emit())
        self._cat_jobtitle.toggled.connect(lambda: self.settings_changed.emit())
        self._window_size.valueChanged.connect(lambda: self.settings_changed.emit())

    # ------------------------------------------------------------------
    # Custom model path helpers
    # ------------------------------------------------------------------

    def _set_custom_visible(self, visible: bool) -> None:
        self._model_path.setVisible(visible)
        self._browse_btn.setVisible(visible)

    def _on_custom_toggled(self, checked: bool) -> None:
        self._set_custom_visible(checked)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF model file", "", "GGUF files (*.gguf);;All files (*)"
        )
        if path:
            self._model_path.setText(path)
            self.settings_changed.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_settings(self) -> dict:
        cats: list[str] = []
        if self._cat_name.isChecked():
            cats.append("NAME")
        if self._cat_org.isChecked():
            cats.append("ORG")
        if self._cat_location.isChecked():
            cats.append("LOCATION")
        if self._cat_jobtitle.isChecked():
            cats.append("JOBTITLE")

        # Determine selected_model and model_path
        selected_model = "standard"
        model_path = ""
        if self._model_custom_radio.isChecked():
            model_path = self._model_path.text()
        else:
            for model_id, radio in self._model_radios.items():
                if radio.isChecked():
                    selected_model = model_id
                    break

        return {
            "output_format": "preserve" if self._fmt_preserve.isChecked() else "txt",
            "lookup_mode": "persist" if self._lookup_persist.isChecked() else "per_file",
            "review_enabled": self._review_check.isChecked(),
            "selected_model": selected_model,
            "model_path": model_path,
            "window_size": self._window_size.value(),
            "enabled_categories": cats,
        }

    def apply_config(self, config) -> None:
        self._fmt_preserve.setChecked(config.output_format == "preserve")
        self._fmt_txt.setChecked(config.output_format == "txt")
        self._lookup_persist.setChecked(config.lookup_mode == "persist")
        self._lookup_per_file.setChecked(config.lookup_mode == "per_file")
        self._review_check.setChecked(config.review_enabled)
        self._window_size.setValue(config.window_size)

        # Model selection
        if config.model_path:
            self._model_custom_radio.setChecked(True)
            self._model_path.setText(config.model_path)
        elif config.selected_model in self._model_radios:
            self._model_radios[config.selected_model].setChecked(True)
        else:
            if "standard" in self._model_radios:
                self._model_radios["standard"].setChecked(True)

        # Categories
        enabled = set(config.enabled_categories)
        self._cat_name.setChecked("NAME" in enabled)
        self._cat_org.setChecked("ORG" in enabled)
        self._cat_location.setChecked("LOCATION" in enabled)
        self._cat_jobtitle.setChecked("JOBTITLE" in enabled)
