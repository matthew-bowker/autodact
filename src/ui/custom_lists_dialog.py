"""Dialog for managing custom word lists used by CustomListDetector."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

# Categories available for custom word lists (excludes Skip/Freetext).
_LIST_CATEGORIES = [
    "NAME",
    "EMAIL",
    "PHONE",
    "LOCATION",
    "ORG",
    "JOBTITLE",
    "DOB",
    "POSTCODE",
    "ID",
]


class CustomListsDialog(QDialog):
    """Manage custom word lists: add, edit, and remove lists."""

    def __init__(self, lists: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom Word Lists")
        self.setMinimumSize(500, 400)
        self._lists = [dict(lst) for lst in lists]  # deep copy

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Add word lists to always redact specific terms. "
            "Each list is tied to a PII category."
        ))

        # List widget showing existing lists.
        self._list_widget = QListWidget()
        layout.addWidget(self._list_widget)

        # Buttons row.
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add List...")
        self._edit_btn = QPushButton("Edit...")
        self._remove_btn = QPushButton("Remove")
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._edit_btn)
        btn_row.addWidget(self._remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Dialog buttons.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Signals.
        self._add_btn.clicked.connect(self._on_add)
        self._edit_btn.clicked.connect(self._on_edit)
        self._remove_btn.clicked.connect(self._on_remove)

        self._refresh_list()

    def _refresh_list(self) -> None:
        self._list_widget.clear()
        for lst in self._lists:
            name = lst.get("name", "Untitled")
            category = lst.get("category", "?")
            count = len(lst.get("words", []))
            item = QListWidgetItem(f"{name}  [{category}]  ({count} words)")
            self._list_widget.addItem(item)

    def _on_add(self) -> None:
        dialog = _EditListDialog(parent=self)
        if dialog.exec():
            self._lists.append(dialog.get_list())
            self._refresh_list()

    def _on_edit(self) -> None:
        row = self._list_widget.currentRow()
        if row < 0:
            return
        dialog = _EditListDialog(existing=self._lists[row], parent=self)
        if dialog.exec():
            self._lists[row] = dialog.get_list()
            self._refresh_list()

    def _on_remove(self) -> None:
        row = self._list_widget.currentRow()
        if row < 0:
            return
        name = self._lists[row].get("name", "this list")
        reply = QMessageBox.question(
            self, "Remove List",
            f"Remove \"{name}\"?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            del self._lists[row]
            self._refresh_list()

    def get_lists(self) -> list[dict]:
        return self._lists


class _EditListDialog(QDialog):
    """Sub-dialog for adding or editing a single word list."""

    def __init__(self, existing: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Word List" if existing else "Add Word List")
        self.setMinimumSize(450, 400)

        layout = QVBoxLayout(self)

        # Name field.
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. UK Place Names")
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        # Category dropdown.
        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("Category:"))
        self._category_combo = QComboBox()
        self._category_combo.addItems(_LIST_CATEGORIES)
        cat_row.addWidget(self._category_combo)
        cat_row.addStretch()
        layout.addLayout(cat_row)

        # Words text area.
        layout.addWidget(QLabel("Words (one per line):"))
        self._words_edit = QTextEdit()
        self._words_edit.setPlaceholderText("Belfast\nEdinburgh\nCardiff\n...")
        layout.addWidget(self._words_edit)

        # Import from file button.
        import_row = QHBoxLayout()
        import_btn = QPushButton("Import from .txt file...")
        import_btn.clicked.connect(self._on_import)
        import_row.addWidget(import_btn)
        import_row.addStretch()
        layout.addLayout(import_row)

        # Dialog buttons.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Populate if editing.
        if existing:
            self._name_edit.setText(existing.get("name", ""))
            cat = existing.get("category", "")
            idx = self._category_combo.findText(cat)
            if idx >= 0:
                self._category_combo.setCurrentIndex(idx)
            words = existing.get("words", [])
            self._words_edit.setPlainText("\n".join(words))

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Word List",
            "", "Text files (*.txt);;All files (*)",
        )
        if path:
            try:
                text = Path(path).read_text(encoding="utf-8")
                # Append to existing content.
                current = self._words_edit.toPlainText()
                if current.strip():
                    self._words_edit.setPlainText(current.rstrip("\n") + "\n" + text)
                else:
                    self._words_edit.setPlainText(text)
            except Exception as e:
                QMessageBox.warning(self, "Import Error", str(e))

    def _on_ok(self) -> None:
        if not self._name_edit.text().strip():
            QMessageBox.warning(self, "Validation", "Please enter a list name.")
            return
        words = self._parse_words()
        if not words:
            QMessageBox.warning(self, "Validation", "Please enter at least one word.")
            return
        self.accept()

    def _parse_words(self) -> list[str]:
        text = self._words_edit.toPlainText()
        return [w.strip() for w in text.splitlines() if w.strip()]

    def get_list(self) -> dict:
        return {
            "name": self._name_edit.text().strip(),
            "category": self._category_combo.currentText(),
            "words": self._parse_words(),
        }
