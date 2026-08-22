import pytest

from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer, normalize_spacing_en


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Excess  white   space  ", "excess white space"),
        (
            "Single quote: Here's Johnny and 'single scare quote phrase'",
            "single quote heres johnny and single scare quote phrase",
        ),
        ("Weird punctuation: . ... x.x...,!a", "weird punctuation x x a"),
        (
            'Underscore: filenames should start with "test_" or end with _test',
            "underscore filenames should start with test or end with test",
        ),
        (
            "Dashes: dashed-word emdash——emdash ... –endash––endash–",
            "dashes dashed word emdash emdash endash endash",
        ),
        ("“This is too much, my love!”", "this is too much my love"),
        ("Random emojis: 😉 in the Read😉Me😉? Well... 🙂‍↔️, why not?",
         "random emojis in the readme well why not"),
        ("Café au lait", "café au lait"),
        # ensure non-roman characters don't get stripped
        ("Регулярные выражения", "регулярные выражения"),
    ],
)
def test_normalize_common(source: str, expected: str) -> None:
    assert TextNormalizer.normalize_common(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Hello 19", "hello 19"),
        ("hello nineteen", "hello 19"),
        ("ninety", "90"),
        ("twenty one", "21"),
        ("twenty-one", "21"),
        ("99% sure", "99 sure"),
        ("ninety-nine percent sure", "99 sure"),
        ("ninety nine percent sure", "99 sure"),
    ],
)
def test_normalize_common_numbers_en(source: str, expected: str) -> None:
    assert TextNormalizer.normalize_common(source, language_code="en") == expected


# Word-number forms the Spanish normalizer converts to digits.
SPANISH_NUMBER_CONVERSIONS = [
    ("veintidós", "22"),
    ("diecisiete", "17"),
    ("treinta y cinco", "35"),
    ("ciento cuarenta y dos", "142"),
    ("novecientos noventa y nueve", "999"),
    ("mil doscientos", "1200"),
    ("dos mil quinientas", "2500"),
    ("nueve mil novecientos noventa y nueve", "9999"),
    ("dos mil veintiséis", "2026"),
    ("ochocientas setenta y cinco", "875"),
    (
        "Llegaron 12 alumnos por la mañana y otros quince por la tarde.",
        "llegaron 12 alumnos por la mañana y otros 15 por la tarde",
    ),
    (
        "El capítulo cubre las páginas cuarenta y dos a cincuenta.",
        "el capitulo cubre las paginas 42 a 50",
    ),
    (
        "El informe mencionaba 1.200.000 personas afectadas.",
        "el informe mencionaba 1 200 000 personas afectadas",
    ),
    (
        "El veintidós de abril de dos mil veintiséis, a las ocho y media "
        "de la mañana, llegaron treinta y cinco cajas al almacén. Cada caja "
        "pesaba dos coma cinco kilos, aunque el recibo decía 2,4. En total, "
        "el encargado contó ochocientas setenta y cinco piezas, pero solo "
        "registró 870 porque cinco estaban rotas. “Un error así no importa”, "
        "dijo uno de los ayudantes, pero la jefa respondió que un error "
        "pequeño puede costar mil euros.",
        "el 22 de abril de 2026 a las ocho y media de la mañana llegaron "
        "35 cajas al almacen cada caja pesaba dos coma cinco kilos aunque el "
        "recibo decia 2 4 en total el encargado conto 875 piezas pero solo "
        "registro 870 porque 5 estaban rotas un error asi no importa dijo "
        "uno de los ayudantes pero la jefa respondio que un error pequeño "
        "puede costar 1000 euros",
    ),
]


@pytest.mark.parametrize(("source", "expected"), SPANISH_NUMBER_CONVERSIONS)
def test_normalize_common_numbers_es_converts_word_numbers(
    source: str, expected: str
) -> None:
    assert TextNormalizer.normalize_common(source, language_code="es") == expected


# Forms that must never be turned into digits: articles/indefinites,
# fractional "medio/tercer/primera" usages, ambiguous decimals ("coma"/"punto"),
# large numbers the normalizer conservatively leaves alone, and roman-numeral
# names (case-folded, but the letters are not digits).
#
# Note: the language-specific normalization paths drop the trailing period of
# the final token (eg, "kilometros." -> "kilometros"); the expectations below
# pin that current behavior.
SPANISH_NUMBER_NON_CONVERSIONS = [
    ("un", "un"),
    ("una", "una"),
    ("uno de ellos", "uno de ellos"),
    ("una vez", "una vez"),
    ("medio vaso", "medio vaso"),
    ("una taza y media", "una taza y media"),
    ("un cuarto de hora", "un cuarto de hora"),
    ("tercer piso", "tercer piso"),
    ("primera vez", "primera vez"),
    ("doce con cincuenta", "doce con cincuenta"),
    ("tres coma cinco", "tres coma cinco"),
    ("doce punto siete", "doce punto siete"),
    (
        "La distancia era de trescientos mil kilómetros.",
        "la distancia era de trescientos mil kilometros",
    ),
    (
        "El presupuesto superaba los dos millones de euros.",
        "el presupuesto superaba los dos millones de euros",
    ),
    ("Carlos V", "carlos v"),
    ("Felipe II", "felipe ii"),
]


@pytest.mark.parametrize(("source", "expected"), SPANISH_NUMBER_NON_CONVERSIONS)
def test_normalize_common_numbers_es_leaves_safe_forms_alone(
    source: str, expected: str
) -> None:
    result = TextNormalizer.normalize_common(source, language_code="es")

    assert result == expected
    assert not any(ch.isdigit() for ch in result)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("color", "colour", True),
        ("analyze", "analyse", True),
        ("been", "bean", True),
        ("one", "Juan", False),
        ("also", "Oslo", False),
        ("apple", "orange", False),
        ("embassy town", "embassytown", True),
    ],
)
def test_sounds_the_same(a: str, b: str, expected: bool) -> None:
    assert TextNormalizer.sounds_the_same_en(a, b) == expected


@pytest.mark.parametrize(
    ("source", "transcript", "expected"),
    [
        # STT breaks a compound word ("fire fly" -> "firefly")
        (
            "Look at that firefly glow.",
            "Look at that fire fly glow.",
            "Look at that firefly glow.",
        ),
        # STT merges two words ("highschool" -> "high school")
        (
            "I went to high school yesterday.",
            "I went to highschool yesterday.",
            "I went to high school yesterday.",
        ),
        # Non-space-related difference (should not be fixed)
        (
            "The quick brown fox.",
            "The quick fox.",
            "The quick fox.",
        ),
    ],
)
def test_normalize_spacing(source: str, transcript: str, expected: str) -> None:
    assert normalize_spacing_en(source=source, transcript=transcript) == expected