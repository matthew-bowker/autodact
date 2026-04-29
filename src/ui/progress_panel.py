from __future__ import annotations

import html
import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from src.ui.styles import (
    BACKGROUND_LIGHT,
    BORDER_LIGHT,
    PRIMARY_BLUE,
    RADIUS_MD,
    RADIUS_SM,
    TEXT_SECONDARY,
)


# Per-category accent colors for stat chips.  Anything not listed falls back
# to the default neutral.
_CATEGORY_COLORS: dict[str, str] = {
    "NAME": "#2563eb",       # blue
    "ORG": "#7c3aed",        # purple
    "LOCATION": "#0891b2",   # cyan
    "JOBTITLE": "#9333ea",   # violet
    "EMAIL": "#0d9488",      # teal
    "PHONE": "#059669",      # emerald
    "DOB": "#d97706",        # amber
    "POSTCODE": "#0284c7",   # sky
    "ID": "#dc2626",         # red
    "IP": "#475569",         # slate
    "URL": "#0369a1",        # blue-600
}
_CATEGORY_DEFAULT = "#6b7280"  # neutral grey


class ProgressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setMinimumHeight(24)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ "
            f"border: 1px solid {BORDER_LIGHT}; border-radius: {RADIUS_MD}px; "
            f"text-align: center; background-color: {BACKGROUND_LIGHT}; }} "
            f"QProgressBar::chunk {{ "
            f"background-color: {PRIMARY_BLUE}; border-radius: {RADIUS_MD - 1}px; }}"
        )
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; margin-top: 4px;")
        layout.addWidget(self._status_label)

        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px; margin-top: 2px;"
        )
        self._stats_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._stats_label)

        self._current_phase: str = ""
        self._phase_start: float = 0.0

        self.hide()

    def set_progress(self, phase: str, current: int, total: int) -> None:
        if phase != self._current_phase:
            self._current_phase = phase
            self._phase_start = time.monotonic()

        if total > 0:
            pct = int(current / total * 100)
            self._progress_bar.setValue(pct)

        eta_text = self._estimate_eta(current, total)
        status = f"{phase}: line {current}/{total}"
        if eta_text:
            status += f"  ({eta_text})"
        self._status_label.setText(status)

    def _estimate_eta(self, current: int, total: int) -> str:
        if current < 2 or total <= 0 or current >= total:
            return ""
        elapsed = time.monotonic() - self._phase_start
        if elapsed < 1.0:
            return ""
        remaining = elapsed / current * (total - current)
        return _format_duration(remaining)

    def set_stats(self, stats: dict[str, int]) -> None:
        if not stats:
            self._stats_label.setText("")
            return
        chips: list[str] = []
        for cat, count in sorted(stats.items(), key=lambda kv: -kv[1]):
            color = _CATEGORY_COLORS.get(cat, _CATEGORY_DEFAULT)
            chips.append(
                f"<span style='background-color:{color}; color:white; "
                f"padding:1px 6px; border-radius:{RADIUS_SM}px; "
                f"font-weight:600;'>"
                f"{html.escape(cat)} {count}"
                f"</span>"
            )
        self._stats_label.setText(" ".join(chips))

    def reset(self) -> None:
        self._progress_bar.setValue(0)
        self._status_label.setText("")
        self._stats_label.setText("")
        self._current_phase = ""
        self._phase_start = 0.0

    def set_complete(self, message: str = "Processing complete.") -> None:
        self._progress_bar.setValue(100)
        self._status_label.setText(message)


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"~{seconds}s remaining"
    minutes, secs = divmod(seconds, 60)
    if secs == 0:
        return f"~{minutes}m remaining"
    return f"~{minutes}m {secs}s remaining"
