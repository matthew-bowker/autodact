from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.pipeline.llm_engine import LLMEngine


class ModelLoadWorker(QObject):
    finished = pyqtSignal(object)  # LLMEngine instance
    error = pyqtSignal(str)

    def __init__(self, model_path: str, n_threads: int = 8) -> None:
        super().__init__()
        self._model_path = model_path
        self._n_threads = n_threads

    @pyqtSlot()
    def run(self) -> None:
        try:
            engine = LLMEngine(
                self._model_path,
                n_threads=self._n_threads,
            )
            self.finished.emit(engine)
        except Exception as e:
            self.error.emit(str(e))
