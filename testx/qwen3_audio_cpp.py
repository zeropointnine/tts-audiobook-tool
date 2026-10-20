"""Live Qwen3-TTS audio.cpp adapter smoke check (requires a running server).

Run with venv-base/bin/python -m testx.qwen3_audio_cpp --help.
The model type is supplied manually, just as in the app. This script never
changes server configuration; a request may load/evict models on that server.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf

from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport

MODEL_ID = "qwen3tts_audiocpp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model-id", required=True, help="Exact configured audio.cpp server ID")
    parser.add_argument("--model-type", choices=("base", "custom_voice", "voice_design"), default="base")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--speaker", default="")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--text", default="The morning sun shone through the window.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    Tts.init_local_model_type()
    RemoteTtsDiscovery.set_base_url(args.url)
    snapshot = RemoteTtsDiscovery.refresh(force=True)
    if snapshot.issue:
        raise SystemExit(snapshot.issue.message)
    if not any(handle.id == MODEL_ID and server_id == args.model_id for handle, server_id in snapshot.candidates):
        raise SystemExit("The exact server ID is not an available offline Qwen3-TTS entry")
    AudioCppUtil.set_base_url(args.url)
    definition = Tts._audio_cpp_definitions[MODEL_ID]
    support = AudioCppModelSupport(definition)
    with tempfile.TemporaryDirectory(prefix="qwen3-audiocpp-") as directory:
        if args.reference:
            shutil.copyfile(args.reference, Path(directory) / args.reference.name)
        voice = Voice(
            file_name=args.reference.name if args.reference else "",
            transcript=args.transcript,
            presets={MODEL_ID: args.speaker},
            instructions={MODEL_ID: args.instructions},
        )
        project = Project(dir_path=directory, tts_model_type=MODEL_ID,
                          audio_cpp_model_id=args.model_id, voice_references=[voice],
                          language_code=args.language)
        for name, value in (("model_type", args.model_type), ("seed", args.seed)):
            project.set_model_setting(MODEL_ID, name, value)
        result = AudioCppBackendAdapter(definition, support, args.model_id).generate_using_project(
            project, [args.text], print_generation_request=True)
        if isinstance(result, str):
            raise SystemExit(result)
        sound = result[0]
        if not sound.data.size or not np.isfinite(sound.data).all():
            raise SystemExit("Server returned empty or non-finite audio")
        sf.write(args.output, sound.data, sound.sr)
        print(json.dumps({"model_id": args.model_id, "model_type": args.model_type,
                          "sample_rate": sound.sr,
                          "seconds": len(sound.data) / sound.sr,
                          "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
