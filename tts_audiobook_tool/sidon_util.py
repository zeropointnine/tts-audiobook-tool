import numpy as np
import torch
from huggingface_hub import hf_hub_download # type: ignore

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound_util import SoundUtil

_HF_REPO_ID       = "sarulab-speech/sidon-v0.1"
_INPUT_SR         = 16_000
_OUTPUT_SR        = 48_000
_MODEL_SAMPLES    = 30 * _INPUT_SR                        # 480,000 — fixed model input window
_CHUNK_DURATION   = 29.8
_OVERLAP_DURATION = 0.2
_CHUNK_SAMPLES    = int(_CHUNK_DURATION   * _INPUT_SR)    # 476,800 — content per chunk at 16kHz
_OVERLAP_SAMPLES  = int(_OVERLAP_DURATION * _INPUT_SR)    # 3,200   — overlap at 16kHz
_OVERLAP_OUT      = int(_OVERLAP_DURATION * _OUTPUT_SR)   # 9,600   — overlap at 48kHz


class SidonUtil:

    @staticmethod
    def has_sidon() -> bool:
        try:
            import sidon # type: ignore
            return True
        except ImportError:
            return False

    def __init__(self):

        # Is cuda only
        # (CPU model files exist on project hf repo but apparently have hardcoded CUDA ops)
        device_str = "cuda"
        device = "cuda"

        feature_extractor_path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=f"feature_extractor_{device_str}.pt",
        )
        decoder_path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=f"decoder_{device_str}.pt",
        )
        self.feature_extractor = torch.jit.load(feature_extractor_path, map_location=device)
        self.decoder = torch.jit.load(decoder_path, map_location=device)
        self.feature_extractor.eval()
        self.decoder.eval()
        self.device = device

    def kill(self) -> None:
        self.feature_extractor = None
        self.decoder = None

    def process(self, sound: Sound) -> Sound | str:
        # Runs Sidon noise reduction on a Sound. Long samples are split into
        # ~29.8s chunks (at 16kHz), each processed in memory as a float32
        # tensor, then crossfaded back together. The full resampled waveform
        # and all chunk tensors live in memory simultaneously, so peak usage
        # scales with input duration.
        if self.feature_extractor is None or self.decoder is None:
            return "SidonUtil: models not loaded (kill() was called)"
        try:
            sound_16k = SoundUtil.resample_if_necessary(sound, _INPUT_SR)
            waveform = torch.tensor(sound_16k.data, dtype=torch.float32)
            n = waveform.shape[-1]

            if n <= _CHUNK_SAMPLES:
                return Sound(self.process_chunk(waveform), _OUTPUT_SR)

            chunks = []
            start = 0
            stride = _CHUNK_SAMPLES - _OVERLAP_SAMPLES
            while start < n:
                end = min(start + _CHUNK_SAMPLES, n)
                chunks.append(waveform[start:end])
                if end >= n:
                    break
                start += stride

            processed = [self.process_chunk(c) for c in chunks]

            result = processed[0]
            for curr in processed[1:]:
                overlap_len = min(_OVERLAP_OUT, len(result), len(curr))
                fade_out = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32)
                fade_in  = np.linspace(0.0, 1.0, overlap_len, dtype=np.float32)
                crossfaded = result[-overlap_len:] * fade_out + curr[:overlap_len] * fade_in
                result = np.concatenate([result[:-overlap_len], crossfaded, curr[overlap_len:]])

            return Sound(result, _OUTPUT_SR)
        except Exception as e:
            return str(e)

    def process_chunk(self, chunk_waveform: torch.Tensor) -> np.ndarray:
        from sidon.cleansing.audio import extract_seamless_m4t_features # type: ignore

        assert self.feature_extractor is not None
        assert self.decoder is not None

        n = chunk_waveform.shape[-1]
        expected_length = int(round(n / _INPUT_SR * _OUTPUT_SR))

        if n < _MODEL_SAMPLES:
            chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, _MODEL_SAMPLES - n))

        features = extract_seamless_m4t_features(
            [chunk_waveform],
            return_tensors="pt",
            padding_value=1.0,
            device=self.device,
        )

        input_features = features["input_features"]
        assert isinstance(input_features, torch.Tensor)

        with torch.no_grad():
            hidden = self.feature_extractor(
                input_features.to(self.device)
            )["last_hidden_state"]
            output = self.decoder(hidden.transpose(1, 2)).cpu()

        data = output.view(-1).numpy().astype("float32")
        return data[:expected_length]

