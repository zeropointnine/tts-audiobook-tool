"""
Compares how the TTS model renders merged multi-paragraph segments, depending on
how the paragraph boundaries appear in the inference prompt:

- "space":   paragraphs joined with a space (current behavior, line breaks are
             removed by prompt normalization)
- "newline": paragraphs joined with a line break
- "period":  paragraphs joined with a space, adding a period to paragraphs that
             end without punctuation (eg headings)

Each variant is generated with the same seeds, so that takes are comparable.
Saves the takes as FLAC files and prints the duration and the internal silences
(pauses) of each take.

Usage:
    ./venv-qwen3tts/Scripts/python -m testx.merge_prompt_variants <project_dir> [num_takes] [out_dir]
"""

import os
import sys

from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.sound.silence_util import SilenceUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts import Tts

CASES: list[tuple[str, list[str]]] = [
    ("heading_kansas", ["Kapitel 1", "Kansas"]),
    ("heading_gewinne", ["Kapitel 3", "Gewinne und Verluste"]),
    ("heading_spur", ["Kapitel 9", "Auf der Spur der Entführer"]),
    ("heading_question", ["Kapitel 14", "Was können wir tun?"]),
    ("dialog_short", ["„Und?“", "„Klar.“", "„Dann los.“"]),
    ("dialog_mixed", ["„Wie lange ist es her?“, fragte Jason.", "„Fünf Wochen.“", "„Mensch, das ergibt keinen Sinn.“"]),
]

SEEDS = [11, 22, 33, 44, 55]

# Characters which already end a paragraph audibly
END_PUNCTUATION = set(".!?…:;,\"'“”„»«’‘‚›‹–—-")

# Silences at the very start or end of a take are not pauses between paragraphs
EDGE_MARGIN_SECONDS = 0.05


def make_variants(prepared: list[str]) -> dict[str, str]:
    with_periods = [
        text if text.rstrip()[-1:] in END_PUNCTUATION else text.rstrip() + "."
        for text in prepared
    ]
    return {
        "space": " ".join(prepared),
        "newline": "\n".join(prepared),
        "period": " ".join(with_periods),
    }


def internal_silences(sound) -> list[tuple[float, float]]:
    return [
        (start, end)
        for start, end in SilenceUtil.detect_silences(sound)
        if start > EDGE_MARGIN_SECONDS and end < sound.duration - EDGE_MARGIN_SECONDS
    ]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    project_dir = sys.argv[1]
    num_takes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(project_dir, "prompt_variant_test")
    os.makedirs(out_dir, exist_ok=True)

    Tts.init_local_model_type()
    project = ProjectLoadUtil.load_using_dir_path(project_dir, prompt_on_warnings=False)
    if isinstance(project, str):
        print(project)
        return

    Tts.set_model_params_using_project(project)
    instance = Tts.get_instance()
    blocking_issues = Tts.get_class().get_blocking_issues(project, instance)
    if blocking_issues:
        print("Errors:", *blocking_issues, sep="\n  - ")
        return

    for case_name, paragraphs in CASES:
        prepared = [instance.prepare_text_for_inference(project, paragraph) for paragraph in paragraphs]
        variants = make_variants(prepared)
        print(f"\n=== {case_name}")
        for variant_name, prompt in variants.items():
            if variant_name == "period" and prompt == variants["space"]:
                print(f"  {variant_name:8} (same prompt as space, skipped)")
                continue
            print(f"  {variant_name:8} prompt: {prompt!r}")
            for seed in SEEDS[:num_takes]:
                project.qwen3_seed = seed
                instance.clear_continuation()
                result = instance.generate_using_project(project, [prompt], voice_selection_index=0)
                if isinstance(result, str):
                    print(f"    seed {seed}: error: {result}")
                    continue
                sound = result[0]
                path = os.path.join(out_dir, f"{case_name}__{variant_name}__seed{seed}.flac")
                err = SoundFileUtil.save_flac(sound, path)
                if err:
                    print(f"    seed {seed}: couldn't save: {err}")
                pauses = ", ".join(f"{end - start:.2f}s@{start:.2f}" for start, end in internal_silences(sound))
                print(f"    seed {seed}: {sound.duration:5.2f}s  pauses: {pauses or '-'}")

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
