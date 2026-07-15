"""Background QThread that runs the LLM advisor off the GUI thread.

Loading a multi-GB GGUF model and running inference can take several seconds;
doing that on the GUI thread would freeze the window. This worker runs the
refinement in a separate thread and emits the resulting config back to the GUI
thread via a queued signal.
"""
import logging

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)


class LLMRefineThread(QThread):
    """Runs ``advisor.refine`` in a worker thread and emits the refined config.

    The result is always a usable config dict: on any failure the advisor itself
    returns the unchanged ``base_config``, and this thread adds a final guard.
    """

    done = Signal(dict)

    def __init__(self, advisor, image, analysis: dict, base_config: dict,
                 parent=None, allow_deblur: bool = True,
                 allow_icedit: bool = True, allow_face: bool = True,
                 blend_enabled: bool = False):
        super().__init__(parent)
        self._advisor = advisor
        self._image = image
        self._analysis = analysis
        self._base_config = base_config
        self._allow_deblur = allow_deblur
        self._allow_icedit = allow_icedit
        self._allow_face = allow_face
        self._blend_enabled = blend_enabled

    def run(self):
        try:
            result = self._advisor.refine(
                self._image, self._analysis, self._base_config,
                allow_deblur=self._allow_deblur,
                allow_icedit=self._allow_icedit,
                allow_face=self._allow_face,
                blend_enabled=self._blend_enabled)
        except Exception as exc:  # advisor guards internally, but be safe
            log.warning("LLM refine thread failed (%s); using base config", exc)
            result = self._base_config
        self.done.emit(result)


class LLMEvaluateThread(QThread):
    """Runs ``advisor.evaluate`` off the GUI thread and emits the verdict.

    On any failure it emits a stop verdict so the refinement loop ends safely.
    """

    done = Signal(dict)

    def __init__(self, advisor, image, analysis: dict, parent=None,
                 allow_deblur: bool = True, allow_icedit: bool = True,
                 allow_face: bool = True):
        super().__init__(parent)
        self._advisor = advisor
        self._image = image
        self._analysis = analysis
        self._allow_deblur = allow_deblur
        self._allow_icedit = allow_icedit
        self._allow_face = allow_face

    def run(self):
        try:
            verdict = self._advisor.evaluate(
                self._image, self._analysis,
                allow_deblur=self._allow_deblur,
                allow_icedit=self._allow_icedit,
                allow_face=self._allow_face)
        except Exception as exc:
            log.warning("LLM evaluate thread failed (%s); stopping loop", exc)
            verdict = {"satisfied": True, "config": None}
        self.done.emit(verdict)
