from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from src.config import AVAILABLE_MODELS, get_custom_lists_path, model_is_cached
from src.pipeline.custom_list_detector import load_custom_lists
from src.ui.styles import (
    BORDER_LIGHT,
    RADIUS_MD,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    panel_style,
)


# Categories the user can toggle.  These map 1:1 to lookup categories that
# can show up in the final output.  Layered detection means the same category
# may come from multiple detectors (e.g. NAME from Presidio + Names dict +
# DeBERTa); unticking the box drops the entire category from the output.
_DETECTION_CATEGORIES: list[tuple[str, str]] = [
    ("NAME", "Names"),
    ("ORG", "Organisations"),
    ("LOCATION", "Locations"),
    ("JOBTITLE", "Job titles"),
    ("EMAIL", "Emails"),
    ("PHONE", "Phone numbers"),
    ("ID", "IDs (SSN, account #)"),
    ("DOB", "Dates of birth"),
    ("POSTCODE", "Postcodes / ZIPs"),
    ("IP", "IP addresses"),
    ("URL", "URLs"),
]

_DEFAULT_ENABLED = {"NAME", "ORG", "LOCATION", "EMAIL", "PHONE", "ID",
                    "DOB", "POSTCODE", "IP", "URL"}

_DEVICE_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Auto-detect"),
    ("cpu", "CPU"),
    ("mps", "Apple Metal (MPS)"),
    ("cuda", "NVIDIA CUDA"),
]


def _subgroup(title: str) -> QGroupBox:
    """A nested groupbox styled for sub-sections of a panel."""
    box = QGroupBox(title)
    box.setStyleSheet(
        f"QGroupBox {{ "
        f"border: 1px solid {BORDER_LIGHT}; border-radius: {RADIUS_MD}px; "
        f"margin-top: 14px; padding-top: 12px; background-color: transparent; }} "
        f"QGroupBox::title {{ "
        f"subcontrol-origin: margin; subcontrol-position: top left; "
        f"left: 10px; padding: 0 4px; "
        f"color: {TEXT_PRIMARY}; font-weight: bold; font-size: 12px; }}"
    )
    return box


class SettingsPanel(QGroupBox):
    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Settings", parent)
        self.setStyleSheet(panel_style())

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_detection_group())
        layout.addWidget(self._build_model_group())

        self._wire_signals()

    # ------------------------------------------------------------------
    # Output group: format, lookup mode, review toggle
    # ------------------------------------------------------------------

    def _build_output_group(self) -> QGroupBox:
        group = _subgroup("Output")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # File format
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("File format:"))
        self._fmt_preserve = QRadioButton("Same as input")
        self._fmt_txt = QRadioButton("Plain text (.txt)")
        self._fmt_preserve.setChecked(True)
        self._fmt_preserve.setToolTip(
            "DOCX → DOCX, XLSX → XLSX, etc. Preserves formatting where possible."
        )
        self._fmt_txt.setToolTip("Convert all output to plain text.")
        fmt_group = QButtonGroup(self)
        fmt_group.addButton(self._fmt_preserve)
        fmt_group.addButton(self._fmt_txt)
        fmt_row.addWidget(self._fmt_preserve)
        fmt_row.addWidget(self._fmt_txt)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        # Tag numbering / lookup table mode
        lookup_row = QHBoxLayout()
        lookup_row.addWidget(QLabel("Tag numbering:"))
        self._lookup_per_file = QRadioButton("Restart per file")
        self._lookup_persist = QRadioButton("Continue across files")
        self._lookup_per_file.setChecked(True)
        self._lookup_per_file.setToolTip(
            "Each file gets its own [NAME 1], [NAME 2]… counter."
        )
        self._lookup_persist.setToolTip(
            "Numbering carries over so the same person gets the same tag in every file."
        )
        lookup_group = QButtonGroup(self)
        lookup_group.addButton(self._lookup_per_file)
        lookup_group.addButton(self._lookup_persist)
        lookup_row.addWidget(self._lookup_per_file)
        lookup_row.addWidget(self._lookup_persist)
        lookup_row.addStretch()
        layout.addLayout(lookup_row)

        # Review step
        self._review_check = QCheckBox("Show review dialog before saving")
        self._review_check.setChecked(True)
        self._review_check.setToolTip(
            "Pause after detection so you can accept, reject, or recategorise findings."
        )
        layout.addWidget(self._review_check)

        return group

    # ------------------------------------------------------------------
    # Detection group: categories, custom lists, fuzzy
    # ------------------------------------------------------------------

    def _build_detection_group(self) -> QGroupBox:
        group = _subgroup("Detection")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Category checkboxes — 4 columns
        cat_label = QLabel("Redact:")
        layout.addWidget(cat_label)

        cat_grid = QGridLayout()
        cat_grid.setSpacing(4)
        cat_grid.setContentsMargins(8, 0, 0, 0)
        self._cat_checks: dict[str, QCheckBox] = {}
        for i, (key, label) in enumerate(_DETECTION_CATEGORIES):
            cb = QCheckBox(label)
            cb.setChecked(key in _DEFAULT_ENABLED)
            self._cat_checks[key] = cb
            cat_grid.addWidget(cb, i // 4, i % 4)
        layout.addLayout(cat_grid)

        # Custom word lists
        custom_row = QHBoxLayout()
        self._custom_lists_btn = QPushButton("Custom word lists…")
        self._custom_lists_btn.setToolTip(
            "Add your own word lists to always redact specific terms."
        )
        self._custom_lists_btn.clicked.connect(self._on_custom_lists)
        self._custom_lists_label = QLabel("")
        self._custom_lists_label.setStyleSheet(
            f"color: {TEXT_TERTIARY}; font-size: 11px;"
        )
        custom_row.addWidget(self._custom_lists_btn)
        custom_row.addWidget(self._custom_lists_label, stretch=1)
        layout.addLayout(custom_row)
        self._refresh_custom_lists_label()

        # Fuzzy matching
        self._fuzzy_check = QCheckBox("Catch misspellings (fuzzy + phonetic match)")
        self._fuzzy_check.setChecked(False)
        self._fuzzy_check.setToolTip(
            "Find variants of detected PII (e.g. 'Johnsen' for 'Johnson'). "
            "Slower; can over-match in noisy data."
        )
        layout.addWidget(self._fuzzy_check)

        return group

    # ------------------------------------------------------------------
    # Model group: which encoder + device
    # ------------------------------------------------------------------

    def _build_model_group(self) -> QGroupBox:
        group = _subgroup("Model")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Model radio (collapsed when only one option)
        self._model_group = QButtonGroup(self)
        self._model_radios: dict[str, QRadioButton] = {}
        for model in AVAILABLE_MODELS:
            downloaded = model_is_cached(model.repo)
            suffix = "" if downloaded else "  · not downloaded"
            radio = QRadioButton(f"{model.name}{suffix}")
            radio.setStyleSheet(f"color: {TEXT_PRIMARY};")
            radio.setToolTip(model.description)
            self._model_group.addButton(radio)
            self._model_radios[model.id] = radio
            layout.addWidget(radio)

        self._model_custom_radio = QRadioButton("Custom local snapshot…")
        self._model_custom_radio.setToolTip(
            "Point at a directory containing a HuggingFace model snapshot, "
            "or paste an HF repo id."
        )
        self._model_group.addButton(self._model_custom_radio)
        layout.addWidget(self._model_custom_radio)

        custom_row = QHBoxLayout()
        custom_row.setContentsMargins(20, 0, 0, 0)
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText(
            "/path/to/snapshot  or  org/repo-name"
        )
        custom_row.addWidget(self._model_path)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._browse_model)
        custom_row.addWidget(self._browse_btn)
        layout.addLayout(custom_row)

        if AVAILABLE_MODELS and AVAILABLE_MODELS[0].id in self._model_radios:
            self._model_radios[AVAILABLE_MODELS[0].id].setChecked(True)
        self._set_custom_visible(False)
        self._model_custom_radio.toggled.connect(self._on_custom_toggled)

        # Device selector
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Run on:"))
        self._device_combo = QComboBox()
        for value, label in _DEVICE_OPTIONS:
            self._device_combo.addItem(label, value)
        self._device_combo.setCurrentIndex(0)
        self._device_combo.setToolTip(
            "Auto-detect picks Apple Metal on Mac, CUDA on NVIDIA, CPU otherwise. "
            "Override if you hit hardware issues."
        )
        device_row.addWidget(self._device_combo)
        device_row.addWidget(QLabel(""), stretch=1)
        layout.addLayout(device_row)

        return group

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        emit = lambda *_: self.settings_changed.emit()
        for w in (self._fmt_preserve, self._lookup_per_file, self._review_check,
                  self._fuzzy_check, self._model_custom_radio):
            w.toggled.connect(emit)
        self._model_group.buttonToggled.connect(emit)
        self._model_path.textChanged.connect(emit)
        self._device_combo.currentIndexChanged.connect(emit)
        for cb in self._cat_checks.values():
            cb.toggled.connect(emit)

    # ------------------------------------------------------------------
    # Custom model path helpers
    # ------------------------------------------------------------------

    def _set_custom_visible(self, visible: bool) -> None:
        self._model_path.setVisible(visible)
        self._browse_btn.setVisible(visible)

    def _on_custom_toggled(self, checked: bool) -> None:
        self._set_custom_visible(checked)

    def _browse_model(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select model snapshot directory", "",
        )
        if path:
            self._model_path.setText(path)
            self.settings_changed.emit()

    # ------------------------------------------------------------------
    # Custom lists
    # ------------------------------------------------------------------

    def _on_custom_lists(self) -> None:
        from src.pipeline.custom_list_detector import save_custom_lists
        from src.ui.custom_lists_dialog import CustomListsDialog

        path = get_custom_lists_path()
        lists = load_custom_lists(path)
        dialog = CustomListsDialog(lists, parent=self)
        if dialog.exec():
            save_custom_lists(path, dialog.get_lists())
            self._refresh_custom_lists_label()
            self.settings_changed.emit()

    def _refresh_custom_lists_label(self) -> None:
        try:
            lists = load_custom_lists(get_custom_lists_path())
        except Exception:
            lists = []
        if not lists:
            self._custom_lists_label.setText("(none configured)")
        else:
            total = sum(len(lst.get("words", [])) for lst in lists)
            self._custom_lists_label.setText(
                f"{len(lists)} list(s), {total} word(s)"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_settings(self) -> dict:
        cats = [k for k, cb in self._cat_checks.items() if cb.isChecked()]

        selected_model = AVAILABLE_MODELS[0].id if AVAILABLE_MODELS else ""
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
            "device": self._device_combo.currentData() or "auto",
            "enabled_categories": cats,
            "fuzzy_matching_enabled": self._fuzzy_check.isChecked(),
        }

    def apply_config(self, config) -> None:
        self._fmt_preserve.setChecked(config.output_format == "preserve")
        self._fmt_txt.setChecked(config.output_format == "txt")
        self._lookup_persist.setChecked(config.lookup_mode == "persist")
        self._lookup_per_file.setChecked(config.lookup_mode == "per_file")
        self._review_check.setChecked(config.review_enabled)

        # Model
        if config.model_path:
            self._model_custom_radio.setChecked(True)
            self._model_path.setText(config.model_path)
        elif config.selected_model in self._model_radios:
            self._model_radios[config.selected_model].setChecked(True)
        elif AVAILABLE_MODELS and AVAILABLE_MODELS[0].id in self._model_radios:
            self._model_radios[AVAILABLE_MODELS[0].id].setChecked(True)

        # Device
        device_value = getattr(config, "device", "auto")
        for i in range(self._device_combo.count()):
            if self._device_combo.itemData(i) == device_value:
                self._device_combo.setCurrentIndex(i)
                break

        # Categories
        enabled = set(config.enabled_categories)
        for key, cb in self._cat_checks.items():
            cb.setChecked(key in enabled)

        # Fuzzy
        self._fuzzy_check.setChecked(config.fuzzy_matching_enabled)
