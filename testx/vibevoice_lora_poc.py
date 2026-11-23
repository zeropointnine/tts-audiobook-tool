"""Per-voice VibeVoice LoRAs through the app's model wrapper, on real weights.

Loads `VibeVoiceModel` (1.5B by default) and generates one line per voice of a project
whose voices use: the adapter, no adapter, the adapter at strength 0.5, then the adapter
again (so every kind of switch happens: merge only, restore only, restore + merge). Prints
each switch's cost (the switcher logs it) and each line's speed, and checks that switching
back to no LoRA restores the checkpoint's weights exactly.

For reference it then loads the same adapter the way the app used to (unmerged, PEFT
`load_adapter` on the language model) and times one line: on a 3080 Ti that was ~76 ms
per generated frame versus ~55 ms merged (two language model passes per frame, CFG).

Run: ./venv-vv/bin/python -m testx.vibevoice_lora_poc [adapter] [base model]
Adapter: a local dir (the adapter itself or a trainer output dir with lora/), an hf repo
id, or `owner/repo/sub/dir`. Default: vibevoice-community/klett.
Outputs go to /tmp/vibevoice_lora_poc_*.wav.
"""

import sys
import time

import soundfile as sf
import torch

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.lora_switch import SafetensorsBaseWeights
from tts_audiobook_tool.tts_models.vibe_voice_model import VibeVoiceModel
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.tts_models import vibevoice_lora

DEFAULT_ADAPTER = "vibevoice-community/klett"
TEXT = ("The old lighthouse keeper climbed the stairs each evening, counting every step, "
        "and wondering whether the ships still needed his light at all.")
MODEL_ID = "vibevoice_local"
FRAME_RATE = 7.5


def generate(model: VibeVoiceModel, project: Project, index: int, label: str) -> None:
    prompt = VibeVoiceModel.massage_for_inference(TEXT)
    torch.cuda.synchronize()
    start = time.perf_counter()
    sounds = model.generate_using_project(project, [prompt], voice_selection_index=index)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert not isinstance(sounds, str), sounds
    seconds = len(sounds[0].data) / sounds[0].sr
    path = f"/tmp/vibevoice_lora_poc_{index}_{label}.wav"
    sf.write(path, sounds[0].data, sounds[0].sr)
    print(f"voice {index + 1} ({label}): {seconds:.1f} s audio in {elapsed:.2f} s "
          f"(incl. any switch), {1000 * elapsed / (seconds * FRAME_RATE):.0f} ms/frame -> {path}")


def main() -> None:
    adapter = ModelTarget.infer(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ADAPTER)
    base_target = ModelTarget.infer(sys.argv[2]) if len(sys.argv) > 2 else VibeVoiceBaseModel.DEFAULT_TARGET
    model = VibeVoiceModel(DeviceType.CUDA, model_target=base_target, max_new_tokens=VibeVoiceBaseModel.MAX_TOKENS)
    assert model.model is not None and model._loras is not None
    project = Project(tts_model_type=MODEL_ID, voice_references=[
        Voice(loras={MODEL_ID: VoiceLora(adapter)}),
        Voice(),
        Voice(loras={MODEL_ID: VoiceLora(adapter, 0.5)}),
    ])
    project.set_model_setting(MODEL_ID, "seed", 1234)
    print("preflight:", model.prepare_voice_loras(project, [0, 1, 2]) or "ok")

    # Warm-up, so the first timed line isn't paying for kernel compilation
    generate(model, project, 1, "warmup")
    for index, label in ((0, "lora"), (1, "base"), (2, "lora-0.5"), (0, "lora-again")):
        generate(model, project, index, label)

    params = dict(model.model.named_parameters())
    names = model._loras._prepared[(adapter, 1.0)].plan.target_names
    gib = sum(params[name].numel() * params[name].element_size() for name in names) / 1024 ** 3
    print(f"adapter targets: {len(names)} tensors, {gib:.2f} GiB; base copy: {model._loras.base.mode_reason}")
    assert model._loras.activate(None) == ""
    model_dir = base_target.resolve_local_dir()
    source = SafetensorsBaseWeights(model_dir)
    bad = [n for n, t in source.read(params, sorted(names)) if not torch.equal(params[n].detach().cpu(), t)]
    print(f"restored weights differing from the checkpoint: {len(bad)}")
    assert not bad
    model.end_generation_run()

    # The earlier, unmerged path, for comparison
    resolved = vibevoice_lora.resolve_adapter(adapter)
    model.model.model.language_model.load_adapter(resolved.directory)
    generate(model, Project(tts_model_type=MODEL_ID, voice_references=[Voice()]), 0, "peft-unmerged")


if __name__ == "__main__":
    main()
