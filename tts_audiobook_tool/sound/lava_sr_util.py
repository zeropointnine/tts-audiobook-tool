from __future__ import annotations

import gc
import importlib
import multiprocessing
import warnings
from multiprocessing.connection import Connection
from typing import Any, Protocol

import numpy as np
import torch

from tts_audiobook_tool.app_types import DeviceType, Sound
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.util import printt


MODEL_PATH = "YatharthS/LavaSR"
INPUT_SR = 16_000
OUTPUT_SR = 48_000
CHUNK_DURATION = 120.0
OVERLAP_DURATION = 0.2
CHUNK_SAMPLES = int(CHUNK_DURATION * INPUT_SR)
OVERLAP_SAMPLES = int(OVERLAP_DURATION * INPUT_SR)
SR_RATIO = OUTPUT_SR // INPUT_SR
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10.0

# The v2 BWE model's feature extractor (Vocos MelSpectrogramFeatures,
# n_fft=2048 / hop=512 / padding="same") reflect-pads 768 samples on each side
# of the 48 kHz waveform, and PyTorch rejects padding that is not smaller than
# the dimension. 16 kHz audio resamples exactly 3x, so any single-chunk clip
# of at most 256 samples (16 ms) crashes with "padding (768, 768) at dimension
# 1 of input [1, 480]". Multi-chunk (> 120 s) audio never hits this: its last
# chunk is at least OVERLAP_SAMPLES + 1 samples long. This guard (16 kHz
# domain) covers very short clips with comfortable margin over the strict
# 257-sample minimum.
MIN_ENHANCE_SAMPLES = 512

# LavaSR's BWE enhancer still builds the deprecated torch.cuda.amp.autocast
# context; suppress that one FutureWarning so it does not pollute app logs.
# (The message starts with a backtick; the match is anchored.)
warnings.filterwarnings(
    "ignore",
    message=r"`?torch\.cuda\.amp\.autocast",
    category=FutureWarning,
)


class WorkerProcess(Protocol):
    def is_alive(self) -> bool: ...
    def join(self, timeout: float | None = None) -> None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


# CUDA isolation intentionally uses a spawned process rather than loading the
# model in the application process. The worker exclusively owns the model and
# CUDA context; requests and NumPy audio cross the Pipe, but CUDA objects never
# do. Consequently, stopping the worker releases all of its driver allocations
# when the process exits, including memory that PyTorch's cache APIs may retain.
# The worker constructs LavaSrUtil with isolation disabled to avoid recursively
# spawning another worker. Shutdown first asks it to exit cleanly, then escalates
# to terminate/kill so a stuck inference cannot leave its CUDA context alive.
def lava_sr_cuda_worker(connection: Connection, model_path: str) -> None:
    """Own the CUDA context in a process whose exit guarantees driver cleanup."""
    util: LavaSrUtil | None = None
    try:
        util = LavaSrUtil(
            device=DeviceType.CUDA,
            model_path=model_path,
            isolate_cuda=False,
        )
        connection.send(("ready", None))

        while True:
            command, payload = connection.recv()
            if command == "process":
                data, sample_rate, denoise = payload
                result = util.process(Sound(data, sample_rate), denoise=denoise)
                if isinstance(result, str):
                    connection.send(("error", result))
                else:
                    connection.send(("result", (result.data, result.sr)))
            elif command == "kill":
                connection.send(("stopped", None))
                break
            else:
                connection.send(("error", f"Unknown LavaSR worker command: {command}"))
    except EOFError:
        pass
    except BaseException as e:
        try:
            connection.send(("error", str(e)))
        except Exception:
            pass
    finally:
        connection.close()


class LavaSrUtil:
    """In-memory adapter for LavaSR v2 speech restoration.

    Importing this module is safe when LavaSR is not installed. Model-specific
    imports and the Hugging Face model download occur only when an instance is
    constructed.
    """

    @staticmethod
    def has_lava_sr() -> bool:
        try:
            model_module = importlib.import_module("LavaSR.model")
            return hasattr(model_module, "LavaEnhance2") #LavaEnhance2 == version 2
        except (ImportError, ModuleNotFoundError):
            return False

    @staticmethod
    def resolve_device(device: DeviceType | str | None = None) -> str:
        if isinstance(device, DeviceType):
            device = device.value

        if device is not None:
            normalized = device.lower().strip()
            supported = {item.value for item in DeviceType}
            if normalized not in supported:
                raise ValueError(
                    f"Unsupported LavaSR device '{device}'; expected one of: "
                    f"{', '.join(sorted(supported))}"
                )
            if normalized == DeviceType.CUDA.value and not torch.cuda.is_available():
                raise ValueError("LavaSR CUDA device requested, but CUDA is not available")
            if normalized == DeviceType.MPS.value and not torch.backends.mps.is_available():
                raise ValueError("LavaSR MPS device requested, but MPS is not available")
            return normalized

        if torch.cuda.is_available():
            return DeviceType.CUDA.value
        if torch.backends.mps.is_available():
            return DeviceType.MPS.value
        return DeviceType.CPU.value

    def __init__(
        self,
        device: DeviceType | str | None = None,
        model_path: str = MODEL_PATH,
        isolate_cuda: bool = True,
    ):
        self.model_path = model_path
        self.device = self.resolve_device(device)
        self.model: Any | None = None
        self.worker_process: WorkerProcess | None = None
        self.worker_connection: Connection | None = None
        self.killed = False

        try:
            if self.device == DeviceType.CUDA.value and isolate_cuda:
                self.start_cuda_worker()
            else:
                self.load_model()
        except Exception as e:
            if self.device != DeviceType.MPS.value:
                raise
            self.fallback_to_cpu(e, "initialization")

    def start_cuda_worker(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        process = context.Process(
            target=lava_sr_cuda_worker,
            args=(child_connection, self.model_path),
            name="lava-sr-cuda",
            daemon=True,
        )
        process.start()
        child_connection.close()
        self.worker_process = process
        self.worker_connection = parent_connection

        try:
            status, payload = parent_connection.recv()
        except EOFError as e:
            self.stop_cuda_worker(force=True)
            raise RuntimeError("LavaSR CUDA worker exited during initialization") from e
        if status != "ready":
            self.stop_cuda_worker(force=True)
            raise RuntimeError(f"LavaSR CUDA worker initialization failed: {payload}")

    def load_model(self) -> None:
        model_module = importlib.import_module("LavaSR.model")
        lava_enhance_type = getattr(model_module, "LavaEnhance2")
        self.model = lava_enhance_type(self.model_path, device=self.device)

    def configure_refiner(self, input_sr: int) -> None:
        """Mirror LavaSR.load_audio() while keeping this adapter in memory.

        LavaSR's direct enhance() call consumes a 16 kHz tensor, but its
        load_audio() helper also preserves the source bandwidth as the
        low/high-frequency merge cutoff. Recreate that second behavior before
        processing each Sound.
        """
        if self.model is None:
            raise RuntimeError("LavaSrUtil: model not loaded")
        if input_sr <= 0:
            raise ValueError("LavaSrUtil: sound sample rate must be positive")

        merge_module = importlib.import_module("LavaSR.enhancer.linkwitz_merge")
        fast_lr_merge_type = getattr(merge_module, "FastLRMerge")
        self.model.bwe_model.lr_refiner = fast_lr_merge_type(
            device=self.device,
            cutoff=input_sr // 2,
            transition_bins=1024,
        )

    def release_model(self) -> None:
        if self.worker_process is not None or self.worker_connection is not None:
            self.stop_cuda_worker()
            return

        self.model = None
        gc.collect()

        if self.device == DeviceType.MPS.value and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def stop_cuda_worker(self, force: bool = False) -> None:
        process = self.worker_process
        connection = self.worker_connection
        self.worker_process = None
        self.worker_connection = None

        if process is None:
            if connection is not None:
                connection.close()
            return

        if connection is not None and process.is_alive() and not force:
            try:
                connection.send(("kill", None))
                if connection.poll(WORKER_SHUTDOWN_TIMEOUT_SECONDS):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass

        process.join(WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join()
        if connection is not None:
            connection.close()

    def fallback_to_cpu(self, error: Exception, stage: str) -> None:
        printt(
            "Warning: LavaSR failed during "
            f"{stage} on MPS ({error}). Retrying on CPU."
        )
        self.release_model()
        self.device = DeviceType.CPU.value
        self.load_model()

    def kill(self) -> None:
        if self.killed:
            return
        self.release_model()
        self.killed = True

    def process(self, sound: Sound, denoise: bool = False) -> Sound | str:
        if self.killed:
            return "LavaSrUtil: model not loaded (kill() was called)"

        if self.worker_connection is not None:
            return self.process_in_cuda_worker(sound, denoise)
        if self.model is None:
            return "LavaSrUtil: model not loaded"

        try:
            return self.process_on_current_device(sound, denoise)
        except Exception as e:
            if self.device != DeviceType.MPS.value:
                return str(e)

            try:
                self.fallback_to_cpu(e, "inference")
                return self.process_on_current_device(sound, denoise)
            except Exception as cpu_error:
                return str(cpu_error)

    def process_in_cuda_worker(self, sound: Sound, denoise: bool) -> Sound | str:
        connection = self.worker_connection
        process = self.worker_process
        if connection is None or process is None or not process.is_alive():
            return "LavaSR CUDA worker is not running"

        try:
            connection.send(("process", (sound.data, sound.sr, denoise)))
            status, payload = connection.recv()
        except (BrokenPipeError, EOFError, OSError) as e:
            return f"LavaSR CUDA worker communication failed: {e}"

        if status == "result":
            data, sample_rate = payload
            return Sound(data, sample_rate)
        return str(payload)

    def process_on_current_device(self, sound: Sound, denoise: bool) -> Sound:
        if self.model is None:
            raise RuntimeError("LavaSrUtil: model not loaded")
        if not isinstance(sound.data, np.ndarray):
            raise ValueError("LavaSrUtil: sound data must be a NumPy array")
        if sound.data.ndim != 1:
            raise ValueError("LavaSrUtil: only mono, one-dimensional audio is supported")
        if sound.data.size == 0:
            raise ValueError("LavaSrUtil: sound data is empty")

        self.configure_refiner(sound.sr)
        sound_16k = SoundUtil.resample_if_necessary(sound, INPUT_SR)
        waveform = np.asarray(sound_16k.data, dtype=np.float32)
        sample_count = waveform.shape[-1]
        output = np.empty(sample_count * SR_RATIO, dtype=np.float32)

        stride = CHUNK_SAMPLES - OVERLAP_SAMPLES
        start = 0
        previous_end_out = 0

        while start < sample_count:
            end = min(start + CHUNK_SAMPLES, sample_count)
            chunk = waveform[start:end]
            expected_chunk_length = (end - start) * SR_RATIO
            if chunk.shape[-1] < MIN_ENHANCE_SAMPLES:
                # Only very short single-chunk clips reach this branch (see
                # MIN_ENHANCE_SAMPLES). Extend them with the final sample so
                # the BWE feature extractor's reflect padding fits;
                # process_chunk trims the output back to the true length.
                # (NumPy padding, since older torch builds reject 1D
                # non-constant F.pad.)
                chunk = np.pad(
                    chunk,
                    (0, MIN_ENHANCE_SAMPLES - chunk.shape[-1]),
                    mode="edge",
                )
            processed = self.process_chunk(
                torch.as_tensor(chunk, dtype=torch.float32).unsqueeze(0),
                denoise,
                expected_chunk_length,
            )

            start_out = start * SR_RATIO
            end_out = end * SR_RATIO
            if start == 0:
                output[start_out:end_out] = processed
            else:
                overlap_length = previous_end_out - start_out
                if overlap_length > 0:
                    fade_out = np.linspace(
                        1.0,
                        0.0,
                        overlap_length,
                        dtype=np.float32,
                    )
                    fade_in = 1.0 - fade_out
                    output[start_out:previous_end_out] = (
                        output[start_out:previous_end_out] * fade_out
                        + processed[:overlap_length] * fade_in
                    )
                output[previous_end_out:end_out] = processed[overlap_length:]

            previous_end_out = end_out
            if end >= sample_count:
                break
            start += stride

        return Sound(output, OUTPUT_SR)

    def process_chunk(
        self,
        chunk: torch.Tensor,
        denoise: bool,
        expected_length: int,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LavaSrUtil: model not loaded")

        enhanced = self.model.enhance(
            chunk,
            enhance=True,
            denoise=denoise,
            batch=False,
        )
        data = enhanced.detach().float().cpu().numpy().reshape(-1).astype(np.float32)

        if len(data) >= expected_length:
            return data[:expected_length]
        if len(data) == 0:
            return np.zeros(expected_length, dtype=np.float32)
        return np.pad(data, (0, expected_length - len(data)), mode="edge")
