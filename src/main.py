import logging
import logging.handlers
import multiprocessing
import sys
from pathlib import Path

import traceback

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.config import AppConfig, get_app_data_dir
from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def _install_exception_hook() -> None:
    """Prevent PyQt6 from calling abort() on unhandled Python exceptions."""
    def _hook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.error("Unhandled exception:\n%s", msg)
        QMessageBox.critical(None, "Error", f"An unexpected error occurred:\n\n{msg}")
    sys.excepthook = _hook


def _configure_logging() -> None:
    logs_dir = get_app_data_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        logs_dir / "autodact.log",
        maxBytes=5_000_000,
        backupCount=3,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    ))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def _get_icon_path() -> Path:
    """Resolve the app icon, handling PyInstaller frozen bundles."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent.parent
    return base / "assets" / "icon.png"


def main():
    multiprocessing.freeze_support()
    _configure_logging()
    _install_exception_hook()
    app = QApplication(sys.argv)
    app.setApplicationName("Autodact")

    icon_path = _get_icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    config = AppConfig.load()
    window = MainWindow()
    window.settings_panel.apply_config(config)

    # Controller import deferred to avoid circular imports at module level
    from src.app_controller import AppController

    controller = AppController(window, config)
    window.show()
    controller.initialize()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
