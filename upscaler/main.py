"""Application entry point."""

import sys
from upscaler.config import ensure_dirs


def main():
    from upscaler.config import setup_logging
    setup_logging()

    from PySide6.QtWidgets import QApplication
    from upscaler.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Upscaler")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
