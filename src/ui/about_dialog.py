from __future__ import annotations

from src import __version__

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
)


class AboutDialog(QDialog):
    """About dialog with app info and usage disclaimer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Autodact")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # App title and version
        title = QLabel("<h1>Autodact</h1>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        version = QLabel(f"v{__version__}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: #999; font-size: 12px;")
        layout.addWidget(version)

        subtitle = QLabel("Privacy-First Data Anonymization")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 13px; margin-bottom: 16px;")
        layout.addWidget(subtitle)

        # Disclaimer content (scrollable)
        disclaimer_text = QTextBrowser()
        disclaimer_text.setOpenExternalLinks(True)
        disclaimer_text.setHtml(self._get_disclaimer_html())
        disclaimer_text.setStyleSheet(
            "QTextBrowser { border: 1px solid #ddd; border-radius: 4px; "
            "padding: 12px; background-color: #fafafa; }"
        )
        layout.addWidget(disclaimer_text)

        # OK button
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _get_disclaimer_html(self) -> str:
        return """
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 13px; line-height: 1.6;">
            <h3 style="color: #333; margin-top: 0;">About This Tool</h3>
            <p>
                Autodact is a desktop application designed to help researchers and data handlers
                anonymize personally identifiable information (PII) in documents using a combination
                of pattern matching and local AI models.
            </p>

            <h3 style="color: #333; margin-top: 20px;">Privacy & Data Security</h3>
            <p>
                <strong>Fully Offline:</strong> All processing happens locally on your computer.
                No data is transmitted to external servers or cloud services.
            </p>

            <h3 style="color: #d9534f; margin-top: 20px;">⚠️ Important Disclaimer</h3>
            <p style="background-color: #fff3cd; padding: 12px; border-left: 4px solid #ffc107; margin: 8px 0;">
                <strong>This tool provides automated assistance with PII detection and anonymization,
                but it is NOT a guarantee of complete anonymization.</strong>
            </p>

            <h4 style="color: #333; margin-top: 16px;">Limitations & Responsible Use</h4>
            <ul style="margin-top: 8px; padding-left: 20px;">
                <li><strong>Not 100% Accurate:</strong> The tool may miss some PII or incorrectly flag non-PII content.</li>
                <li><strong>Human Review Required:</strong> Always review the output before sharing or publishing anonymized data.</li>
                <li><strong>Context Matters:</strong> Anonymization effectiveness depends on document type, language, and content structure.</li>
                <li><strong>Indirect Identifiers:</strong> The tool may not detect combinations of non-PII that could re-identify individuals.</li>
                <li><strong>No Legal Guarantee:</strong> This software is provided as-is without warranties. Users are responsible for ensuring compliance with applicable privacy laws (GDPR, HIPAA, etc.).</li>
            </ul>

            <h4 style="color: #333; margin-top: 16px;">Best Practices</h4>
            <ul style="margin-top: 8px; padding-left: 20px;">
                <li>Enable the <strong>human review step</strong> in settings to verify detections</li>
                <li>Test the tool on sample data before processing sensitive documents</li>
                <li>Keep the lookup tables secure (they contain the original PII)</li>
                <li>Combine with other anonymization techniques for high-risk data</li>
                <li>Consult with privacy/compliance professionals for regulated data</li>
            </ul>

            <h3 style="color: #333; margin-top: 20px;">Technology</h3>
            <p style="font-size: 12px; color: #666;">
                Autodact uses PyQt6, transformers + PyTorch for local DeBERTa inference,
                and Microsoft Presidio for structured PII detection. All models run entirely
                on your device.
            </p>

            <p style="margin-top: 20px; font-size: 12px; color: #888; text-align: center;">
                By using this software, you acknowledge and accept these limitations and agree
                to use the tool responsibly.
            </p>
        </div>
        """
