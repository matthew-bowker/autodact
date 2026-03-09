from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.pipeline.lookup_table import LookupEntry
from src.ui.styles import (
    ERROR_RED,
    ERROR_RED_HOVER,
    SUCCESS_GREEN,
    SUCCESS_GREEN_HOVER,
    button_style,
)

CATEGORIES = ["NAME", "ORG", "LOCATION", "JOBTITLE", "EMAIL", "PHONE",
              "ID", "DOB", "POSTCODE", "IP", "URL", "OTHER"]

ACTION_ACCEPT = "accept"
ACTION_REJECT = "reject"


class ReviewDialog(QDialog):
    def __init__(self, entries: list[LookupEntry], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review PII Detections")
        self.setMinimumSize(900, 550)

        self._entries = entries
        self._decisions: list[tuple[str, str, str]] = []

        layout = QVBoxLayout(self)

        # Header
        header = QLabel(
            f"Review {len(entries)} detected PII entities. "
            "Accept, reject, or change the category for each."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Bulk action bar
        bulk_row = QHBoxLayout()
        accept_all_btn = QPushButton("✓ Accept All")
        accept_all_btn.setStyleSheet(button_style(SUCCESS_GREEN, SUCCESS_GREEN_HOVER))
        accept_all_btn.clicked.connect(self._accept_all)
        bulk_row.addWidget(accept_all_btn)

        reject_all_btn = QPushButton("✗ Reject All")
        reject_all_btn.setStyleSheet(button_style(ERROR_RED, ERROR_RED_HOVER))
        reject_all_btn.clicked.connect(self._reject_all)
        bulk_row.addWidget(reject_all_btn)

        bulk_row.addStretch()
        layout.addLayout(bulk_row)

        # Table
        self._table = QTableWidget(len(entries), 5)
        self._table.setHorizontalHeaderLabels([
            "Original", "Replacement", "Category", "Source", "Action"
        ])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )

        self._action_combos: list[QComboBox] = []
        self._category_combos: list[QComboBox] = []

        for row, entry in enumerate(entries):
            # Original
            item = QTableWidgetItem(entry.original_term)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, item)

            # Replacement tag
            item = QTableWidgetItem(entry.anonymised_term)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 1, item)

            # Category (editable combo)
            cat_combo = QComboBox()
            cat_combo.addItems(CATEGORIES)
            idx = CATEGORIES.index(entry.pii_category) if entry.pii_category in CATEGORIES else 0
            cat_combo.setCurrentIndex(idx)
            self._table.setCellWidget(row, 2, cat_combo)
            self._category_combos.append(cat_combo)

            # Source
            source_text = f"{entry.source_file}:{entry.first_seen_line}"
            item = QTableWidgetItem(source_text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 3, item)

            # Action combo
            action_combo = QComboBox()
            action_combo.addItems(["Accept", "Reject"])
            action_combo.setCurrentIndex(0)
            self._table.setCellWidget(row, 4, action_combo)
            self._action_combos.append(action_combo)

        layout.addWidget(self._table)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_all(self) -> None:
        for combo in self._action_combos:
            combo.setCurrentIndex(0)

    def _reject_all(self) -> None:
        for combo in self._action_combos:
            combo.setCurrentIndex(1)

    def _on_accept(self) -> None:
        self._decisions = []
        for i, entry in enumerate(self._entries):
            action_text = self._action_combos[i].currentText().lower()
            new_category = self._category_combos[i].currentText()
            self._decisions.append((entry.original_term, action_text, new_category))
        self.accept()

    def get_decisions(self) -> list[tuple[str, str, str]]:
        return self._decisions
