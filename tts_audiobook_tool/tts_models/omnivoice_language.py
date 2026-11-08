"""
OmniVoice language ids, shared by the local model and the audio.cpp behavior.

OmniVoice (local library and audio.cpp alike) accepts an exact id from its own
table — ISO 639-1 where one exists, otherwise ISO 639-3 (eg "yue", "nan", "nn")
— or an English language name. It has no region/script subtags, so "en-US" is
unknown to it: the audio.cpp session rejects the request and the local library
silently drops to language-agnostic mode.

`resolve_omnivoice_language` normalizes the project code to its base code,
corrects the few base codes OmniVoice spells differently, and checks the result
against `OMNIVOICE_LANGUAGE_IDS`. A code with no OmniVoice id sends no language
(OmniVoice's automatic language selection) rather than failing.
"""

from tts_audiobook_tool.text_ops.language_util import normalize_language_code
from tts_audiobook_tool.tts_models.language_resolution import LanguageResolution, resolve_from_table

# Base codes OmniVoice spells differently (its table has no "ar" or "tl").
_OMNIVOICE_ID_BY_BASE: dict[str, str] = {
    "ar": "arb",  # Standard Arabic
    "tl": "fil",  # Filipino
}

# Codes the shared alias table folds into "no", but which OmniVoice keeps
# distinct as Norwegian Bokmål.
_BOKMAL_CODES = frozenset({"nb", "nob"})

# OmniVoice's supported language ids (omnivoice 0.1.5 `omnivoice.utils.lang_map.LANG_IDS`,
# identical to audio.cpp src/models/omnivoice/language_map.inc). Neither the
# library nor the server can be queried from the main process.
OMNIVOICE_LANGUAGE_IDS: frozenset[str] = frozenset({
    "aae", "aal", "aao", "ab", "abb", "abn", "abr", "abs", "abv", "acm", "acw", "acx",
    "adf", "adx", "ady", "aeb", "aec", "af", "afb", "afo", "ahl", "ahs", "ajg", "aju",
    "ala", "aln", "alo", "am", "amu", "an", "anc", "ank", "anp", "anw", "aom", "apc",
    "apd", "arb", "arq", "ars", "ary", "arz", "as", "ast", "avl", "awo", "ayl", "ayp",
    "az", "ba", "bag", "bas", "bax", "bba", "bbj", "bbl", "bbu", "bce", "bci", "bcs",
    "bcy", "bda", "bde", "bdm", "be", "beb", "bew", "bfd", "bft", "bg", "bgp", "bhb",
    "bhh", "bho", "bhp", "bhr", "bjj", "bjk", "bjn", "bjt", "bkh", "bkm", "bky", "bmm",
    "bmq", "bn", "bnm", "bnn", "bns", "bo", "bou", "bqg", "br", "bra", "brh", "bri",
    "brx", "bs", "bsh", "bsj", "bsk", "btm", "btv", "bug", "bum", "buo", "bux", "bwr",
    "bxf", "byc", "bys", "byv", "byx", "bzc", "bzw", "ca", "ccg", "ceb", "cen", "cfa",
    "cgg", "chq", "cjk", "ckb", "ckl", "ckr", "cky", "cnh", "cpy", "cs", "cte", "ctl",
    "cut", "cux", "cv", "cy", "da", "dag", "dar", "dav", "dbd", "dcc", "de", "deg",
    "dgh", "dgo", "dje", "dmk", "dml", "dru", "dty", "dua", "dv", "dyu", "dzg", "ebr",
    "ebu", "ego", "eiv", "eko", "ekr", "el", "elm", "en", "eo", "es", "esu", "et",
    "eto", "ets", "etu", "eu", "ewo", "ext", "eyo", "fa", "fan", "fat", "ff", "ffm",
    "fi", "fia", "fil", "fip", "fkk", "fmp", "fr", "fub", "fuc", "fue", "fuf", "fuh",
    "fui", "fuq", "fuv", "fy", "ga", "gbm", "gbr", "gby", "gcc", "gdf", "gej", "ges",
    "ggg", "gid", "gig", "giz", "gjk", "gju", "gl", "glw", "gn", "gol", "gom", "gsl",
    "gu", "gui", "gur", "guz", "gv", "gwc", "gwe", "gwt", "gya", "gyz", "ha", "hah",
    "hao", "haw", "haz", "hbb", "he", "hem", "hi", "hia", "hkk", "hla", "hno", "hoj",
    "hr", "hsb", "ht", "hu", "hue", "hul", "hux", "hwo", "hy", "hz", "ia", "ibb", "id",
    "ida", "idu", "ig", "ijc", "ijn", "ik", "ikw", "is", "ish", "iso", "it", "its",
    "itw", "itz", "ja", "jal", "jax", "jgo", "jmx", "jns", "jqr", "juk", "juo", "jv",
    "ka", "kab", "kai", "kaj", "kam", "kbd", "kbl", "kbt", "kcq", "kdh", "kea", "keu",
    "kfe", "kfk", "kfp", "khg", "khw", "kj", "kjc", "kjk", "kk", "kln", "kls", "km",
    "kmr", "kmy", "kn", "kna", "knn", "ko", "kol", "koo", "kpo", "kqo", "ks", "ksd",
    "ksf", "kto", "kuh", "kvx", "kw", "kwm", "kxp", "ky", "kyx", "lag", "lb", "lcm",
    "ldb", "lg", "lij", "lir", "lkb", "lla", "ln", "lnu", "lo", "loa", "lrk", "lss",
    "lt", "ltg", "lto", "lua", "luo", "lus", "lv", "lwg", "mab", "maf", "mai", "mau",
    "max", "mbo", "mcf", "mcn", "mcx", "mdd", "mde", "mdf", "mek", "mer", "meu", "mfm",
    "mfn", "mfo", "mfv", "mgg", "mgi", "mhk", "mhr", "mi", "mig", "miu", "mk", "mkf",
    "mki", "ml", "mlq", "mn", "mne", "mni", "mqy", "mr", "mrj", "mrr", "mrt", "ms",
    "mse", "msh", "msw", "mt", "mtr", "mtu", "mtx", "mua", "mug", "mui", "mve", "mvy",
    "mxs", "mxu", "mxy", "my", "myv", "mzl", "nal", "nan", "nap", "nb", "nbh", "ncf",
    "nco", "ncx", "ndi", "ng", "ngi", "nhg", "nhi", "nhn", "nhq", "nja", "nl", "nla",
    "nlv", "nmg", "nmz", "nn", "nnh", "no", "noe", "npi", "nso", "ny", "nyu", "oc",
    "odk", "odu", "ogo", "om", "orc", "oru", "ory", "os", "pa", "pbs", "pbt", "pbu",
    "pcm", "pex", "phl", "phr", "pip", "piy", "pko", "pl", "plk", "plt", "pmq", "pms",
    "pmy", "pnb", "poc", "poe", "pow", "prq", "ps", "pst", "pt", "pua", "pwn", "qug",
    "qum", "qup", "qur", "qus", "quv", "qux", "quy", "qva", "qvi", "qvj", "qvl", "qwa",
    "qws", "qxa", "qxp", "qxt", "qxu", "qxw", "rag", "rm", "ro", "rob", "rof", "roo",
    "rth", "ru", "rup", "rw", "sa", "sah", "sat", "sau", "say", "sbn", "sc", "scl",
    "scn", "sd", "sei", "shu", "si", "sip", "siw", "sjr", "sk", "skg", "skr", "sl",
    "sn", "snc", "snk", "so", "sol", "sps", "sq", "sr", "src", "sro", "ssi", "ste",
    "sua", "sv", "sva", "sw", "szy", "ta", "tan", "tar", "tay", "tbf", "tcf", "tcy",
    "tdn", "tdx", "te", "tg", "tgc", "th", "the", "thq", "thr", "thv", "ti", "tig",
    "tio", "tk", "tkg", "tkt", "tli", "tlp", "tn", "tok", "tpl", "tpz", "tqp", "tr",
    "trp", "trq", "trv", "trw", "tt", "ttj", "ttr", "ttu", "tui", "tul", "tuq", "tuv",
    "tuy", "tvo", "tvu", "tw", "twu", "txs", "txy", "udl", "ug", "uk", "uki", "umb",
    "ur", "ush", "uz", "uzn", "vai", "var", "ver", "vi", "vmc", "vmj", "vmm", "vmp",
    "vmz", "vot", "vro", "wbl", "wci", "weo", "wes", "wja", "wji", "wo", "wof", "xh",
    "xhe", "xka", "xmf", "xmv", "xmw", "xpe", "xti", "xtu", "yaq", "yav", "yay", "ydd",
    "ydg", "yer", "yes", "yi", "yo", "yue", "zga", "zgh", "zh", "zoc", "zoh", "zor",
    "zpv", "zpy", "ztg", "ztn", "ztp", "zts", "ztu", "zu", "zza",
})


def get_omnivoice_language_id(language_code: str) -> str | None:
    """OmniVoice language id for a free-form project code, or None when empty or unknown."""
    subtag = (language_code or "").strip().lower().replace("_", "-").split("-", 1)[0]
    if subtag in _BOKMAL_CODES:
        return "nb"
    base = normalize_language_code(language_code or "")
    language_id = _OMNIVOICE_ID_BY_BASE.get(base, base)
    return language_id if language_id in OMNIVOICE_LANGUAGE_IDS else None


def resolve_omnivoice_language(language_code: str) -> LanguageResolution:
    """MAPPED to an OmniVoice id, else NOT_SET / NO_MATCH sending no language."""
    return resolve_from_table(get_omnivoice_language_id(language_code), language_code)
