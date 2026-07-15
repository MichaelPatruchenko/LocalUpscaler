import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from upscaler.ui.history_panel import HistoryPanel

_app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolate_i18n(tmp_path, monkeypatch):
    """Изолировать настройки языка и синглтон переводчика для каждого теста."""
    from upscaler import config
    from upscaler.ui import i18n
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(i18n, "_translator", None)
    yield
    i18n.set_language("ru")
    monkeypatch.setattr(i18n, "_translator", None)


def _panel_with(versions):
    panel = HistoryPanel()
    for v in versions:
        panel.add_version(v, None, {})
    return panel


def test_add_version_auto_selects_primary():
    panel = _panel_with([1])
    assert panel._primary == 1
    assert panel._secondary is None


def test_left_click_sets_primary():
    panel = _panel_with([1, 2])
    panel._on_primary_clicked(1)
    assert panel._primary == 1
    assert panel._secondary is None


def test_right_click_sets_secondary():
    panel = _panel_with([1, 2])
    panel._on_primary_clicked(1)
    panel._on_secondary_clicked(2)
    assert panel._primary == 1
    assert panel._secondary == 2


def test_left_click_on_primary_deselects_and_promotes_to_effective():
    # From compare (1=primary, 2=secondary), left-clicking the primary
    # deselects it. The raw primary slot is cleared and 2 remains in the
    # secondary slot, but 2 is now the *effective* primary (shown / working).
    panel = _panel_with([1, 2])
    panel._on_primary_clicked(1)
    panel._on_secondary_clicked(2)
    panel._on_primary_clicked(1)  # deselect primary
    assert panel._primary is None
    assert panel._secondary == 2
    assert panel.selected() == (2, None)  # compare off, 2 displayed as primary


def test_right_click_on_secondary_clears_it():
    panel = _panel_with([1, 2])
    panel._on_primary_clicked(1)
    panel._on_secondary_clicked(2)
    panel._on_secondary_clicked(2)  # toggle secondary off
    assert panel._primary == 1
    assert panel._secondary is None


def test_lmb_primary_then_rmb_secondary_clears_all():
    # The user's required scenario: from compare (1, 2), left-click the first
    # AND right-click the second -> nothing selected (empty).
    panel = _panel_with([1, 2])
    panel._on_primary_clicked(1)
    panel._on_secondary_clicked(2)   # (1, 2) compare
    panel._on_primary_clicked(1)     # LMB first -> deselect (2 now effective)
    panel._on_secondary_clicked(2)   # RMB second(2) -> clear it -> empty
    assert panel._primary is None
    assert panel._secondary is None
    assert panel.selected() == (None, None)


def test_lone_secondary_is_effective_primary_and_badge():
    # After deselecting the primary, the remaining selection (held in the raw
    # secondary slot) is the effective primary: shown with the ① badge and
    # emitted as the primary so it becomes the working image.
    panel = _panel_with([1, 2])
    received = []
    panel.selection_changed.connect(lambda p, s: received.append((p, s)))
    panel._on_primary_clicked(1)
    panel._on_secondary_clicked(2)
    panel._on_primary_clicked(1)     # (None, 2) -> effective (2, None)
    assert received[-1] == (2, -1)
    texts = {panel.list_widget.item(i).text()
             for i in range(panel.list_widget.count())}
    assert "① v2" in texts          # lone selection shown as primary ①
    assert all("②" not in t for t in texts)


def test_right_click_on_primary_is_noop():
    panel = _panel_with([1, 2])
    panel._on_primary_clicked(1)
    panel._on_secondary_clicked(1)  # same image cannot be both
    assert panel._primary == 1
    assert panel._secondary is None


def test_left_click_on_secondary_makes_it_primary():
    panel = _panel_with([1, 2])
    panel._on_primary_clicked(1)
    panel._on_secondary_clicked(2)
    panel._on_primary_clicked(2)  # 2 was secondary -> becomes primary
    assert panel._primary == 2
    assert panel._secondary is None


def test_selection_changed_emitted_with_minus_one_for_none():
    panel = _panel_with([1, 2])
    received = []
    panel.selection_changed.connect(lambda p, s: received.append((p, s)))
    panel._on_primary_clicked(1)
    assert received[-1] == (1, -1)
    panel._on_secondary_clicked(2)
    assert received[-1] == (1, 2)


def test_badge_text_reflects_slots():
    panel = _panel_with([1, 2])
    panel._on_primary_clicked(1)
    panel._on_secondary_clicked(2)
    texts = {panel.list_widget.item(i).text()
             for i in range(panel.list_widget.count())}
    assert "① v1" in texts  # ① v1
    assert "② v2" in texts  # ② v2


# --- Этап D: ретрансляция панели ---------------------------------------------

def test_history_panel_retranslate():
    from upscaler.ui import i18n
    panel = HistoryPanel()
    i18n.set_language("en")
    panel.retranslate()
    assert panel.revert_btn.text() == "Revert"
    i18n.set_language("ru")
    panel.retranslate()
    assert panel.revert_btn.text() == "Откат"
