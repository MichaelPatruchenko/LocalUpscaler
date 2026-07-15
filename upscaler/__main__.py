"""Allow running as `python -m upscaler`."""
import sys


def main():
    # When launched with --worker flag, run as headless worker process
    if "--worker" in sys.argv:
        from upscaler.engine.worker import main as worker_main
        worker_main()
        return

    from upscaler.config import setup_logging, ensure_dirs
    from PySide6.QtWidgets import QApplication
    from upscaler.ui.main_window import MainWindow

    ensure_dirs()
    setup_logging()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
