from tts_audiobook_tool.app_types import ModelWarmUpResult
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationPhase
from tts_audiobook_tool.model_runtime import require_model_owner
from tts_audiobook_tool.sound.lava_sr_util import LavaSrUtil
from tts_audiobook_tool.sound.yamnet_detector import YamnetDetector
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *


class ModelManager:
    """
    Multiple-model management utils.

    Including YamnetDetector and LavaSR v2 (which it holds static instances of)
    """

    lava_sr_upsampler: LavaSrUtil | None = None
    yamnet_detector: YamnetDetector | None = None


    @staticmethod
    def warm_up_models(state: State, skip_yamnet: bool=False) -> ModelWarmUpResult:
        require_model_owner("model registry")
        """Reconcile process-local models with the models required by state.

        Presence and desired state are deliberately separate: an already-loaded
        model is retained when it is still wanted and cleared when it is not.
        In the interactive application this method is called only inside the
        model worker process.
        """

        want_stt = not bool(Stt.should_skip(state))
        need_tts = not Tts.instance_exists()
        need_stt = want_stt and not Stt.has_instance()

        if not want_stt:
            Stt.clear_stt_model()
        if skip_yamnet:
            ModelManager.clear_yamnet_detector()

        if need_tts and need_stt:
            print_init("Warming up models...")

        Interrupts().set("model init")

        # Init or retain TTS.
        if need_tts:
            GenerationEvents.emit(GenerationPhase("Loading text-to-speech model"))
        try:
            tts_instance = Tts.get_instance()
        except Exception as e:
            Interrupts().clear()
            app_memory.gc_ram_vram()
            return ModelWarmUpResult(error=str(e))

        if Interrupts().did_interrupt:
            Interrupts().clear()
            return ModelWarmUpResult(did_interrupt=True)

        # Init STT only when desired. An existing desired instance is retained.
        if need_stt:
            GenerationEvents.emit(GenerationPhase("Loading speech-to-text model"))
            try:
                Stt.eager_warm_up_for_inference()
            except Exception as e:
                Interrupts().clear()
                app_memory.gc_ram_vram()
                return ModelWarmUpResult(error=str(e))

        if Interrupts().did_interrupt:
            Interrupts().clear()
            return ModelWarmUpResult(did_interrupt=True)

        # Reconcile YAMNet after TTS exists because capability detection may
        # depend on the concrete TTS instance.
        want_yamnet = (
            not skip_yamnet
            and Tts.get_class().can_hallucinate_music(state.project, tts_instance)
        )
        if want_yamnet and not ModelManager.has_yamnet_detector():
            GenerationEvents.emit(GenerationPhase("Loading audio validation model"))
            try:
                _ = ModelManager.get_yamnet_detector()
            except Exception as e:
                Interrupts().clear()
                app_memory.gc_ram_vram()
                return ModelWarmUpResult(error=str(e))
        elif not want_yamnet:
            ModelManager.clear_yamnet_detector()

        if Interrupts().did_interrupt:
            Interrupts().clear()
            return ModelWarmUpResult(did_interrupt=True)

        Interrupts().clear()
        return ModelWarmUpResult()

    @staticmethod
    def clear_all_models(except_lava_sr: bool = False) -> None:
        require_model_owner("model registry")
        """Best-effort release of every process-local model."""
        errors: list[str] = []
        clearers = [
            ("STT", Stt.clear_stt_model),
            ("TTS", Tts.clear_tts_model),
            ("YAMNet", ModelManager.clear_yamnet_detector),
        ]
        if not except_lava_sr:
            clearers.append(("LavaSR", ModelManager.clear_lava_sr_upsampler))

        try:
            for label, clear in clearers:
                try:
                    clear()
                except Exception as exception:
                    errors.append(f"{label}: {type(exception).__name__}: {exception}")
        finally:
            app_memory.gc_ram_vram()

        if errors:
            raise RuntimeError("; ".join(errors))

    @staticmethod
    def is_any_model_loaded() -> bool:
        require_model_owner("model registry")
        return Stt.has_instance() or \
            Tts.instance_exists() or \
            ModelManager.has_yamnet_detector() or \
            ModelManager.lava_sr_upsampler is not None

    @staticmethod
    def get_yamnet_detector() -> YamnetDetector:
        require_model_owner("YAMNet")
        if ModelManager.yamnet_detector is None:
            print_init("Initializing YAMNet...")
            ModelManager.yamnet_detector = YamnetDetector()
        return ModelManager.yamnet_detector

    @staticmethod
    def has_yamnet_detector() -> bool:
        require_model_owner("model registry")
        return ModelManager.yamnet_detector is not None

    @staticmethod
    def clear_yamnet_detector() -> None:
        require_model_owner("model registry")
        detector = ModelManager.yamnet_detector
        ModelManager.yamnet_detector = None
        if detector is not None:
            try:
                detector.kill()
            finally:
                app_memory.gc_ram_vram()

    @staticmethod
    def get_lava_sr_upsampler(
        *, isolate_cuda: bool = True
    ) -> LavaSrUtil | None:
        require_model_owner("LavaSR")
        if not LavaSrUtil.has_lava_sr():
            return None
        if ModelManager.lava_sr_upsampler is None:
            print_init("Initializing LavaSR v2 upsampler...")
            ModelManager.lava_sr_upsampler = LavaSrUtil(
                isolate_cuda=isolate_cuda
            )
        return ModelManager.lava_sr_upsampler

    @staticmethod
    def clear_lava_sr_upsampler() -> None:
        require_model_owner("model registry")
        upsampler = ModelManager.lava_sr_upsampler
        ModelManager.lava_sr_upsampler = None
        if upsampler is not None:
            upsampler.kill()
