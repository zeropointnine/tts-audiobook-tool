from tts_audiobook_tool.constants import VOICE_ADVANCED_SUPERLABEL
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.voice import VoiceMenuShared
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.dots_base_model import (
    DotsBaseModel,
    DotsCompileMode,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_DEFAULT, COL_DIM, make_menu_label, print_feedback


class VoiceDotsMenu:
    @staticmethod
    def menu(state: State) -> None:
        def make_target_label(_: State) -> str:
            return VoiceMenuShared.make_target_label(
                label_prefix="Select dots.tts model",
                target=state.project.dots_target,
                default_target=DotsBaseModel.DEFAULT_REPO_ID,
                remove_prefixes=["dots-studio/"],
            )

        def make_compile_label(_: State) -> str:
            mode = DotsCompileMode.get_by_enabled(state.project.dots_compile)
            default = DotsCompileMode.default()
            return make_menu_label("Compile", mode.label, default.label)

        def make_items(_: State) -> list[MenuItem]:
            items = VoiceMenuShared.make_voice_sample_items(
                state, TtsModelType.DOTS
            )
            items.append(
                MenuItem(
                    make_target_label,
                    lambda _, __: VoiceDotsMenu.target_submenu(state),
                    superlabel=VOICE_ADVANCED_SUPERLABEL,
                )
            )
            items.append(
                MenuItem(
                    make_compile_label,
                    lambda _, __: VoiceDotsMenu.compile_submenu(state),
                )
            )

            if not DotsBaseModel.is_sampling_locked_target(
                state.project.dots_target
            ):
                attr, variant, default_steps, min_steps, max_steps = (
                    DotsBaseModel.get_num_steps_config(state.project.dots_target)
                )
                items.append(
                    MenuUtil.make_number_item(
                        state=state,
                        attr=attr,
                        base_label=f"Num steps ({variant})",
                        default_value=default_steps,
                        is_minus_one_default=True,
                        num_decimals=0,
                        prompt=make_ranged_prompt(
                            f"number of sampling steps ({variant})",
                            min_steps,
                            max_steps,
                        ),
                        min_value=min_steps,
                        max_value=max_steps,
                    )
                )

            items.append(
                MenuUtil.make_number_item(
                    state=state,
                    attr="dots_speaker_scale",
                    base_label="Speaker scale",
                    default_value=DotsBaseModel.SPEAKER_SCALE_DEFAULT,
                    is_minus_one_default=True,
                    num_decimals=1,
                    prompt=make_ranged_prompt(
                        "speaker scale",
                        DotsBaseModel.SPEAKER_SCALE_MIN,
                        DotsBaseModel.SPEAKER_SCALE_MAX,
                    ),
                    min_value=DotsBaseModel.SPEAKER_SCALE_MIN,
                    max_value=DotsBaseModel.SPEAKER_SCALE_MAX,
                )
            )

            if DotsBaseModel.is_cfg_configurable_target(
                state.project.dots_target
            ):
                items.append(
                    MenuUtil.make_number_item(
                        state=state,
                        attr="dots_guidance_scale",
                        base_label="CFG",
                        default_value=DotsBaseModel.GUIDANCE_SCALE_DEFAULT,
                        is_minus_one_default=True,
                        num_decimals=1,
                        prompt=make_ranged_prompt(
                            "CFG",
                            DotsBaseModel.GUIDANCE_SCALE_MIN,
                            DotsBaseModel.GUIDANCE_SCALE_MAX,
                        ),
                        min_value=DotsBaseModel.GUIDANCE_SCALE_MIN,
                        max_value=DotsBaseModel.GUIDANCE_SCALE_MAX,
                    )
                )

            items.append(VoiceMenuShared.make_seed_item(state, "dots_seed"))
            return items

        VoiceMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def target_submenu(state: State) -> None:
        targets = DotsBaseModel.PRESET_REPO_IDS
        sublabels = [
            "Full flow-matching (post-trained); configurable NFE (default 10) and CFG 1.2.",
            "MeanFlow-distilled student of SOAR; few-step (NFE 4), lower latency; CFG fused.",
            "Distilled, fixed two-step sCM sampling (NFE 2, CFG 0).",
            "Distilled, fixed one-step sampling (NFE 1, CFG 0).",
        ]

        def on_select(target: str) -> None:
            state.project.dots_target = (
                "" if target == DotsBaseModel.DEFAULT_REPO_ID else target
            )
            state.project.save()
            _ = ModelWorker.clear_models_if_running_blocking()
            print_feedback("Model set:", target)

        MenuUtil.options_menu(
            state=state,
            heading_text="Select dots.tts model",
            labels=targets,
            values=targets,
            current_value=DotsBaseModel.resolve_target(state.project.dots_target),
            default_value=DotsBaseModel.DEFAULT_REPO_ID,
            on_select=on_select,
            sublabels=sublabels,
            breadcrumb="dots.tts model",
        )

    @staticmethod
    def compile_submenu(state: State) -> None:
        modes = list(DotsCompileMode)
        subheading = (
            "Whether dots.tts compiles its inference kernels. Compiling is much\n"
            "faster to generate but slows model load (a one-time warm-up) and\n"
            "uses more working memory. Disabling skips compilation for a faster\n"
            "load and lower memory at the cost of slower generation.\n"
        )

        def on_select(mode: DotsCompileMode) -> None:
            state.project.dots_compile = mode.enabled
            state.project.save()
            _ = ModelWorker.clear_models_if_running_blocking()
            print_feedback("Compile set to:", mode.label)

        current = DotsCompileMode.get_by_enabled(state.project.dots_compile)
        MenuUtil.options_menu(
            state=state,
            heading_text="Compile",
            labels=[mode.label for mode in modes],
            values=modes,
            current_value=current,
            default_value=DotsCompileMode.default(),
            on_select=on_select,
            sublabels=[mode.description for mode in modes],
            subheading=subheading,
            breadcrumb="dots.tts compile",
        )

def make_ranged_prompt(what: str, min_value: float, max_value: float) -> str:
    return f"Enter {what} {COL_DIM}({min_value} to {max_value}){COL_DEFAULT}:"
