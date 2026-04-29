from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class ModelLoadWorker(QObject):
    """Load the DeBERTa PII detector on a background thread.

    Emits ``finished(detector)`` with the loaded ``DebertaDetector`` instance,
    or ``error(message)`` if loading fails.
    """

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, model_source: str, device: str = "auto") -> None:
        super().__init__()
        self._model_source = model_source
        self._device = device

    @pyqtSlot()
    def run(self) -> None:
        try:
            from src.pipeline.deberta_detector import DebertaDetector

            device = None if self._device == "auto" else self._device
            detector = DebertaDetector(
                model_name=self._model_source,
                device=device,
            )
            self.finished.emit(detector)
        except Exception as e:
            self.error.emit(str(e))
