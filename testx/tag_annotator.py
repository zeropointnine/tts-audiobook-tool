"""
WIP

Splits source text by line feed, similar to text segmenter logic.
Feeds LLM three lines at a time, with 3 lines before and after as 'context' (Uses current Prefs' LLM settings).
LLM returns annotated text based on system prompt instructions (using Higgs V3 tags).
Retries up to 2x if result lines minus tags does not equal source lines (Using Deepseek v4 Flash with thinking on, this has been fairly reliable).
Finally, combines cumulative result and saves parallel text file.

Results on typical novel:
    Overall, mildly interesting, at least for dialog.
    Emotion tags do make a different usually. Oftentimes much too dramatic, especially in narration.
    Prosody tags don't make much difference.
    Sfx hardly gets applied by DS hardly at all, at least on the source text

    TODO: Strip tags out of text in the reader web app.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tts_audiobook_tool import text_util
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer
from tts_audiobook_tool.validator import Validator

INPUT_PATH = r"/d/w/text/tag_annotator_input.txt"

MAX_PARAGRAPHS_PER_WINDOW = 3
MAX_WORDS_PER_WINDOW = 160
MAX_ATTEMPTS = 3
PRINT_FULL_LLM_DEBUG = False

# When True, if all attempts have errors, chooses and keeps attempt with the lowest word error count
ALLOW_WORD_ERRORS = True

HIGGS_TAG_RE = re.compile(
    r"<\|(?:"
    r"emotion:(?:elation|amusement|enthusiasm|determination|pride|contentment|affection|relief|contemplation|confusion|surprise|awe|longing|arousal|anger|fear|disgust|bitterness|sadness|shame|helplessness)|"
    r"style:(?:singing|shouting|whispering)|"
    r"prosody:(?:speed_very_slow|speed_slow|speed_fast|speed_very_fast|pitch_low|pitch_high|expressive_high|expressive_low|pause|long_pause)|"
    r"sfx:(?:cough|laughter|crying|screaming|burping|humming|sigh|sniff|sneeze)"
    r")\|>"
)

SYSTEM_PROMPT = """You annotate audiobook text for Higgs Audio V3 TTS.

You receive previous_text_items, current_text_items, and upcoming_text_items. Annotate only current_text_items. The previous and upcoming items are for context only.

Input/output contract:
- The current_text_items are consecutive parts of one continuous audiobook passage.
- Use the whole list as continuous narrative context, but keep item boundaries exactly as given.
- Return a JSON array of strings.
- Return exactly the same number of items as current_text_items.
- Do not merge, split, reorder, rewrite, normalize, trim, add, or remove original text inside any item.
- After supported Higgs tags are stripped from each returned item, it must exactly equal the corresponding current_text_items item.
- Do not include markdown, commentary, labels, code fences, or any text outside the JSON array.

Use only these supported tags:
- Emotion: <|emotion:elation|>, <|emotion:amusement|>, <|emotion:enthusiasm|>, <|emotion:determination|>, <|emotion:pride|>, <|emotion:contentment|>, <|emotion:affection|>, <|emotion:relief|>, <|emotion:contemplation|>, <|emotion:confusion|>, <|emotion:surprise|>, <|emotion:awe|>, <|emotion:longing|>, <|emotion:arousal|>, <|emotion:anger|>, <|emotion:fear|>, <|emotion:disgust|>, <|emotion:bitterness|>, <|emotion:sadness|>, <|emotion:shame|>, <|emotion:helplessness|>.
- Style: <|style:singing|>, <|style:shouting|>, <|style:whispering|>.
- Prosody: <|prosody:pitch_low|>, <|prosody:pitch_high|>, <|prosody:expressive_high|>, <|prosody:expressive_low|>.
- Sound effects: <|sfx:cough|>, <|sfx:laughter|>, <|sfx:crying|>, <|sfx:screaming|>, <|sfx:burping|>, <|sfx:humming|>, <|sfx:sigh|>, <|sfx:sniff|>, <|sfx:sneeze|>.

Put broad delivery tags near the beginning of a sentence. Tags do not carry over to subsequent sentences, so feel free to repeat the same tag on consecutive sentences to apply a continuous effect.

Apply tags only where they seem appropriate. Don't "force" them. Also, be more conservative with sentences which are pure narration. 

Do, however, find opportunities for using the sound effect ("sfx") tags...

"""
# Removed prosody:pause and prosody:speed-related (ineffective)


CURRENT_TEXT_START = "<<<CURRENT_TEXT_START>>>"
CURRENT_TEXT_END = "<<<CURRENT_TEXT_END>>>"
PREVIOUS_TEXT_START = "<<<PREVIOUS_TEXT_START>>>"
PREVIOUS_TEXT_END = "<<<PREVIOUS_TEXT_END>>>"
UPCOMING_TEXT_START = "<<<UPCOMING_TEXT_START>>>"
UPCOMING_TEXT_END = "<<<UPCOMING_TEXT_END>>>"


@dataclass
class ParagraphItem:
    original: str
    leading_whitespace: str
    core_text: str
    trailing_whitespace: str

    @property
    def annotated_fallback(self) -> str:
        return self.original

    def with_annotated_core(self, annotated_core_text: str) -> str:
        return f"{self.leading_whitespace}{annotated_core_text}{self.trailing_whitespace}"


@dataclass
class AnnotationValidationResult:
    is_valid: bool
    message: str
    annotated_items: list[str]
    word_error_count: int


def split_paragraphs_preserve_whitespace(text: str) -> list[str]:
    """Split after each line-feed run while preserving all original whitespace."""
    if text == "":
        return []

    paragraphs: list[str] = []
    start = 0
    for match in re.finditer(r"\n+", text):
        end = match.end()
        paragraphs.append(text[start:end])
        start = end
    if start < len(text):
        paragraphs.append(text[start:])
    return paragraphs


def split_outer_whitespace(paragraph: str) -> ParagraphItem:
    match = re.fullmatch(r"(\s*)(.*?)(\s*)", paragraph, flags=re.DOTALL)
    if match is None:
        return ParagraphItem(paragraph, "", paragraph, "")
    return ParagraphItem(
        original=paragraph,
        leading_whitespace=match.group(1),
        core_text=match.group(2),
        trailing_whitespace=match.group(3),
    )


def make_paragraph_items(paragraphs: list[str]) -> list[ParagraphItem]:
    return [split_outer_whitespace(paragraph) for paragraph in paragraphs]


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def take_forward_window(paragraphs: list[ParagraphItem], start_index: int) -> tuple[list[ParagraphItem], int]:
    selected: list[ParagraphItem] = []
    total_words = 0
    index = start_index

    while index < len(paragraphs) and len(selected) < MAX_PARAGRAPHS_PER_WINDOW:
        paragraph = paragraphs[index]
        paragraph_words = word_count(paragraph.core_text)
        if selected and total_words + paragraph_words > MAX_WORDS_PER_WINDOW:
            break
        selected.append(paragraph)
        total_words += paragraph_words
        index += 1

    if not selected and start_index < len(paragraphs):
        selected.append(paragraphs[start_index])
        index = start_index + 1

    return selected, index


def take_previous_window(paragraphs: list[ParagraphItem], current_index: int) -> list[ParagraphItem]:
    selected_reversed: list[ParagraphItem] = []
    total_words = 0
    index = current_index - 1

    while index >= 0 and len(selected_reversed) < MAX_PARAGRAPHS_PER_WINDOW:
        paragraph = paragraphs[index]
        paragraph_words = word_count(paragraph.core_text)
        if selected_reversed and total_words + paragraph_words > MAX_WORDS_PER_WINDOW:
            break
        selected_reversed.append(paragraph)
        total_words += paragraph_words
        index -= 1

    if not selected_reversed and current_index > 0:
        selected_reversed.append(paragraphs[current_index - 1])

    return list(reversed(selected_reversed))


def strip_higgs_tags(text: str) -> str:
    return HIGGS_TAG_RE.sub("", text)


def remove_common_response_wrapper(text: str) -> tuple[str, str]:
    """Remove only obvious LLM answer wrappers, without general whitespace normalization."""
    original = text
    cleaned = text
    notes: list[str] = []

    code_fence_match = re.fullmatch(r"\s*```(?:text)?\n(?P<body>.*?)\n```\s*", cleaned, flags=re.DOTALL)
    if code_fence_match:
        cleaned = code_fence_match.group("body")
        notes.append("removed markdown code fence wrapper")

    for prefix in ("Annotated current_text:", "Annotated text:", "current_text:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            if cleaned.startswith("\n"):
                cleaned = cleaned[1:]
            notes.append(f"removed leading label {prefix!r}")
            break

    if cleaned != original and not notes:
        notes.append("removed response wrapper")

    return cleaned, "; ".join(notes)


def find_first_difference(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def make_mismatch_report(original_text: str, stripped_text: str, context_chars: int = 35) -> str:
    diff_index = find_first_difference(original_text, stripped_text)
    start = max(0, diff_index - context_chars)
    end_original = min(len(original_text), diff_index + context_chars)
    end_stripped = min(len(stripped_text), diff_index + context_chars)

    normalized_original = re.sub(r"\s+", " ", original_text).strip()
    normalized_stripped = re.sub(r"\s+", " ", stripped_text).strip()
    whitespace_only_note = ""
    if normalized_original == normalized_stripped:
        whitespace_only_note = " semantic-ish whitespace-collapsed text matches; likely whitespace drift."

    return (
        f"first diff at index {diff_index}; "
        f"original length {len(original_text)}, stripped length {len(stripped_text)}; "
        f"original[{start}:{end_original}]={original_text[start:end_original]!r}; "
        f"stripped[{start}:{end_stripped}]={stripped_text[start:end_stripped]!r};"
        f"{whitespace_only_note}"
    )


def count_word_errors(source_text: str, candidate_text: str) -> int:
    normalized_source, normalized_candidate = TextNormalizer.normalize_source_and_transcript(source_text, candidate_text)
    return len(Validator.get_word_errors(normalized_source, normalized_candidate))


def count_batch_word_errors(current_text: list[ParagraphItem], annotated_items: list[str]) -> int:
    return sum(
        count_word_errors(paragraph_item.core_text, strip_higgs_tags(annotated_item))
        for paragraph_item, annotated_item in zip(current_text, annotated_items)
    )


def make_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem} [ANNOTATED]{input_path.suffix}")


def print_debug_block(title: str, text: str) -> None:
    if not PRINT_FULL_LLM_DEBUG:
        return
    separator = "=" * 80
    print(f"\n{separator}")
    print(title)
    print(separator)
    print(text)
    print(f"{separator}\n")


def make_llm(prefs: Prefs) -> LlmSession:
    return LlmSession(
        api_endpoint_url=prefs.llm_url,
        token=prefs.llm_api_key,
        model=prefs.llm_model,
        system_prompt=SYSTEM_PROMPT,
        extra_params=prefs.llm_extra_params,
        verbose=False,
    )


def make_annotation_prompt(previous_text: list[ParagraphItem], current_text: list[ParagraphItem], upcoming_text: list[ParagraphItem]) -> str:
    previous_text_items = [item.core_text for item in previous_text]
    current_text_items = [item.core_text for item in current_text]
    upcoming_text_items = [item.core_text for item in upcoming_text]
    return (
        "Annotate current_text_items using previous_text_items and upcoming_text_items as context. Return only the JSON array.\n\n"
        f"previous_text_items = {json.dumps(previous_text_items, ensure_ascii=False, indent=2)}\n\n"
        f"current_text_items = {json.dumps(current_text_items, ensure_ascii=False, indent=2)}\n\n"
        f"upcoming_text_items = {json.dumps(upcoming_text_items, ensure_ascii=False, indent=2)}"
    )


def make_retry_prompt(current_text_items: list[str], annotated_text: str, validation_message: str = "") -> str:
    return (
        "The previous response failed validation. Fix the response for current_text_items. Return only the JSON array.\n"
        f"Validation detail: {validation_message}\n\n"
        f"current_text_items = {json.dumps(current_text_items, ensure_ascii=False, indent=2)}\n\n"
        "Previous failed response follows for reference only.\n"
        f"<<<FAILED_RESPONSE_START>>>\n{annotated_text}<<<FAILED_RESPONSE_END>>>"
    )


def parse_annotated_items(annotated_text: str, expected_count: int) -> tuple[list[str] | None, str]:
    cleaned_text, wrapper_note = remove_common_response_wrapper(annotated_text)
    try:
        data = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        return None, f"response is not valid JSON: {e}"
    if not isinstance(data, list):
        return None, f"response JSON is {type(data).__name__}, expected list"
    if len(data) != expected_count:
        return None, f"response item count {len(data)} != expected count {expected_count}"
    if not all(isinstance(item, str) for item in data):
        return None, "response JSON array contains non-string item(s)"
    if wrapper_note:
        print(f"  validation cleanup: {wrapper_note}")
    return data, ""

def validate_annotation(current_text: list[ParagraphItem], annotated_text: str) -> AnnotationValidationResult:
    annotated_items, parse_error = parse_annotated_items(annotated_text, len(current_text))
    if annotated_items is None:
        return AnnotationValidationResult(False, parse_error, [], 0)

    word_error_count = count_batch_word_errors(current_text, annotated_items)

    for index, (paragraph_item, annotated_item) in enumerate(zip(current_text, annotated_items), start=1):
        stripped_text = strip_higgs_tags(annotated_item)
        if stripped_text != paragraph_item.core_text:
            mismatch_report = make_mismatch_report(paragraph_item.core_text, stripped_text)
            return AnnotationValidationResult(False, f"item {index}: {mismatch_report}", annotated_items, word_error_count)
    return AnnotationValidationResult(True, "", annotated_items, word_error_count)


def annotate_current_text(
    prefs: Prefs,
    previous_text: list[ParagraphItem],
    current_text: list[ParagraphItem],
    upcoming_text: list[ParagraphItem],
    chunk_number: int,
    total_paragraphs: int,
) -> str:
    current_text_items = [item.core_text for item in current_text]
    llm = make_llm(prefs)
    prompt = make_annotation_prompt(previous_text, current_text, upcoming_text)
    best_word_error_count: int | None = None
    best_word_error_items: list[str] = []

    for attempt_index in range(MAX_ATTEMPTS):
        attempt_number = attempt_index + 1
        print(f"  attempt {attempt_number}/{MAX_ATTEMPTS}: requesting annotation...")
        try:
            print_debug_block(f"LLM SYSTEM PROMPT | chunk {chunk_number} attempt {attempt_number}", SYSTEM_PROMPT)
            print_debug_block(f"LLM USER PROMPT | chunk {chunk_number} attempt {attempt_number}", prompt)
            annotated_text = llm.send(prompt)
            print_debug_block(f"LLM RAW RESPONSE | chunk {chunk_number} attempt {attempt_number}", annotated_text)
        except Exception as e:
            print(f"  attempt {attempt_number}/{MAX_ATTEMPTS}: LLM error: {e}")
            if attempt_number == MAX_ATTEMPTS:
                break
            prompt = make_retry_prompt(current_text_items, "", str(e))
            continue

        validation_result = validate_annotation(current_text, annotated_text)
        if validation_result.annotated_items and (
            best_word_error_count is None or validation_result.word_error_count < best_word_error_count
        ):
            best_word_error_count = validation_result.word_error_count
            best_word_error_items = validation_result.annotated_items

        if validation_result.is_valid:
            tag_count = sum(len(HIGGS_TAG_RE.findall(item)) for item in validation_result.annotated_items)
            print(f"  accepted chunk {chunk_number}; inserted {tag_count} supported tag(s).")
            return "".join(
                paragraph_item.with_annotated_core(annotated_item)
                for paragraph_item, annotated_item in zip(current_text, validation_result.annotated_items)
            )

        word_error_detail = ""
        if validation_result.annotated_items:
            word_error_detail = f"; cumulative word errors for batch: {validation_result.word_error_count}"
        print(f"  attempt {attempt_number}/{MAX_ATTEMPTS}: validation failed: {validation_result.message}{word_error_detail}")
        prompt = make_retry_prompt(current_text_items, annotated_text, validation_result.message)

    if ALLOW_WORD_ERRORS and best_word_error_items:
        tag_count = sum(len(HIGGS_TAG_RE.findall(item)) for item in best_word_error_items)
        print(
            f"  accepted chunk {chunk_number} with word errors allowed; "
            f"kept best response with {best_word_error_count} cumulative word error(s) "
            f"and {tag_count} supported tag(s)."
        )
        return "".join(
            paragraph_item.with_annotated_core(annotated_item)
            for paragraph_item, annotated_item in zip(current_text, best_word_error_items)
        )

    print(
        f"  fallback for chunk {chunk_number}: using original text "
        f"({len(current_text)} paragraph item(s), {total_paragraphs} total paragraph item(s))."
    )
    return "".join(item.annotated_fallback for item in current_text)


def annotate_paragraphs(paragraphs: list[ParagraphItem], prefs: Prefs) -> list[str]:
    annotated_paragraphs: list[str] = []
    index = 0
    chunk_number = 1

    while index < len(paragraphs):
        previous_text = take_previous_window(paragraphs, index)
        current_text, next_index = take_forward_window(paragraphs, index)
        upcoming_text, _ = take_forward_window(paragraphs, next_index)

        print(
            f"\nchunk {chunk_number}: paragraph items {index + 1}-{next_index} of {len(paragraphs)} "
            f"({word_count(''.join(item.core_text for item in current_text))} word(s))"
        )
        print(f"  context: previous={len(previous_text)} current={len(current_text)} upcoming={len(upcoming_text)}")

        annotated_text = annotate_current_text(
            prefs=prefs,
            previous_text=previous_text,
            current_text=current_text,
            upcoming_text=upcoming_text,
            chunk_number=chunk_number,
            total_paragraphs=len(paragraphs),
        )
        annotated_paragraphs.append(annotated_text)

        index = next_index
        chunk_number += 1

    return annotated_paragraphs


def main() -> int:
    input_path = Path(INPUT_PATH)
    print(f"Loading input: {input_path}")

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}")
        return 1
    if not input_path.is_file():
        print(f"Input path is not a file: {input_path}")
        return 1

    text = text_util.load_text_file(str(input_path))
    if text == "":
        print("No text was loaded. The file may be empty or unreadable.")
        return 1

    paragraphs = split_paragraphs_preserve_whitespace(text)
    paragraph_items = make_paragraph_items(paragraphs)
    print(f"Loaded {len(text)} character(s), {word_count(text)} word(s), {len(paragraph_items)} paragraph item(s).")

    prefs = Prefs.load(save_if_dirty=False)
    missing_settings = []
    if not prefs.llm_url:
        missing_settings.append("llm_url")
    if not prefs.llm_model:
        missing_settings.append("llm_model")
    if missing_settings:
        print(f"Missing LLM preference(s): {', '.join(missing_settings)}")
        return 1

    print(f"Using LLM model: {prefs.llm_model}")
    print(f"Using LLM endpoint: {prefs.llm_url}")

    annotated_paragraphs = annotate_paragraphs(paragraph_items, prefs)
    annotated_text = "".join(annotated_paragraphs)

    output_path = make_output_path(input_path)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(annotated_text)

    print(f"\nSaved annotated text: {output_path}")
    print(f"Output length: {len(annotated_text)} character(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
