from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.dots_base_model import (
    DotsBaseModel,
    DotsCompileMode,
)
from tts_audiobook_tool.util import COL_DEFAULT, COL_DIM, make_menu_label, print_feedback

class ModelDotsMenu:
    @staticmethod
    def menu(state: State) -> None:
        def make_target_label(_: State) -> str:
            return ModelMenuShared.make_target_label(
                label_prefix="Select dots.tts model",
                target=state.project.get_model_setting('dots_local', 'target'),
                default_target=DotsBaseModel.DEFAULT_REPO_ID,
                remove_prefixes=["dots-studio/"],
            )

        def make_compile_label(_: State) -> str:
            mode = DotsCompileMode.get_by_enabled(state.project.get_model_setting('dots_local', 'compile'))
            default = DotsCompileMode.default()
            return make_menu_label("Compile", mode.label, default.label)

        def make_items(_: State) -> list[MenuItem]:
            items = []
            items.append(
                MenuItem(
                    make_target_label,
                    lambda _, __: ModelDotsMenu.target_submenu(state),
                )
            )
            items.append(
                MenuItem(
                    make_compile_label,
                    lambda _, __: ModelDotsMenu.compile_submenu(state),
                )
            )

            if not DotsBaseModel.is_sampling_locked_target(
                state.project.get_model_setting('dots_local', 'target')
            ):
                attr, variant, default_steps, min_steps, max_steps = (
                    DotsBaseModel.get_num_steps_config(state.project.get_model_setting('dots_local', 'target'))
                )
                items.append(
                    MenuUtil.make_number_item(
                        state=state,
                        target=attr,
                        base_label=f"Steps ({variant})",
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
                    target=SettingRef("dots_local", "speaker_scale"),
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
                state.project.get_model_setting('dots_local', 'target')
            ):
                items.append(
                    MenuUtil.make_number_item(
                        state=state,
                        target=SettingRef("dots_local", "guidance_scale"),
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

            items.append(ModelMenuShared.make_seed_item(state, SettingRef("dots_local", "seed")))
            return items

        ModelMenuShared.menu_wrapper(state, make_items)

    @staticmethod
    def target_submenu(state: State) -> None:
        targets = DotsBaseModel.PRESET_REPO_IDS
        sublabels = [
            "Full flow-matching (post-trained); configurable NFE and CFG 1.2",
            "MeanFlow-distilled student of SOAR; 4 steps, no CFG",
            "Distilled, fixed two-step sCM sampling; 2 steps, no CFG",
            "Distilled, fixed one-step sampling; 1 step, no CFG",
        ]

        def on_select(target: str) -> None:
            state.project.set_model_setting('dots_local', 'target', "" if target == DotsBaseModel.DEFAULT_REPO_ID else target)
            state.project.save()
            _ = ModelWorker.clear_models_if_running_blocking()
            print_feedback("Model set:", target)

        MenuUtil.options_menu(
            state=state,
            heading_text="Select dots.tts model",
            labels=targets,
            values=targets,
            current_value=DotsBaseModel.resolve_target(state.project.get_model_setting('dots_local', 'target')),
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
            "load and lower memory at the cost of slower generation."
        )

        def on_select(mode: DotsCompileMode) -> None:
            state.project.set_model_setting('dots_local', 'compile', mode.enabled)
            state.project.save()
            _ = ModelWorker.clear_models_if_running_blocking()
            print_feedback("Compile set to:", mode.label)

        current = DotsCompileMode.get_by_enabled(state.project.get_model_setting('dots_local', 'compile'))
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
