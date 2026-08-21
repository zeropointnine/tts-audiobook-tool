from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from tts_audiobook_tool.app_types.phrase import Reason


@dataclass(frozen=True)
class ReasonPauses:
    """A named pause-duration policy for phrase segmentation reasons."""

    id: str
    label: str
    menu_label: str
    description: str
    pauses: Mapping[Reason, float]

    def get_pause_for(self, reason: Reason) -> float:
        """Return the pause duration in seconds for ``reason``."""
        return self.pauses[reason]

    def _clone(self) -> "ReasonPauses":
        # The `pauses` mapping is a MappingProxyType, which copy.copy/deepcopy
        # cannot handle (TypeError: cannot pickle 'mappingproxy' object).
        # Rebuild the dataclass with a plain dict copy instead.
        return ReasonPauses(
            id=self.id,
            label=self.label,
            menu_label=self.menu_label,
            description=self.description,
            pauses=dict(self.pauses),
        )

    def __copy__(self) -> "ReasonPauses":
        return self._clone()

    def __deepcopy__(self, memo) -> "ReasonPauses":
        return self._clone()


NORMAL_REASON_PAUSES = MappingProxyType(
    {
        Reason.UNDEFINED: 1.0,
        Reason.WORD: 0.1,
        Reason.PHRASE_QUOTE_END: 0.1,
        Reason.PHRASE: 0.5,
        Reason.SENTENCE: 0.9,
        Reason.PARAGRAPH: 1.2,
        Reason.SPACE_BREAK: 2.0,
        Reason.SECTION_BREAK: 2.5,
    }
)

SHORTER_REASON_PAUSES = MappingProxyType(
    {
        Reason.UNDEFINED: 1.0,
        Reason.WORD: 0.1,
        Reason.PHRASE_QUOTE_END: 0.1,
        Reason.PHRASE: 0.3,
        Reason.SENTENCE: 0.6,
        Reason.PARAGRAPH: 0.9,
        Reason.SPACE_BREAK: 1.5,
        Reason.SECTION_BREAK: 2.0,
    }
)


class ReasonPauseTypes(Enum):
    """Enumerates the available pause-duration policies."""

    NORMAL = ReasonPauses(
        id="normal",
        label="Normal",
        menu_label="normal",
        description="",
        pauses=NORMAL_REASON_PAUSES,
    )
    SHORTER = ReasonPauses(
        id="shorter",
        label="Shorter",
        menu_label="shorter",
        description='Enabling "Generate > Limit silence gaps" works well with this setting',
        pauses=SHORTER_REASON_PAUSES,
    )

    @staticmethod
    def default() -> ReasonPauseTypes:
        return ReasonPauseTypes.NORMAL

    @staticmethod
    def get_by_id(id: str) -> ReasonPauseTypes | None:
        for item in ReasonPauseTypes:
            if item.value.id == id:
                return item
        return None
