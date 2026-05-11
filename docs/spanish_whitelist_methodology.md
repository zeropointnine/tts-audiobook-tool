# Spanish Whitelist Generation for Whisper/TTS Validation

## Overview

This document describes the final approach for generating a Spanish word whitelist
used during Whisper-based STT validation of TTS-generated audio.

The goal is to build a practical list of conventional Spanish word forms that can
be used to decide which words are reliable validation targets.

The current implementation uses the Python `wordfreq` package as the primary source
for Spanish frequency-ranked word forms.

## Use Case

The validation workflow is:

1. Start with known Spanish source text.
2. Generate TTS audio from that text.
3. Transcribe the audio with Whisper.
4. Normalize both the source text and the Whisper transcript.
5. Compare the normalized word sequences.
6. Use a whitelist to ignore or downweight words that are uncommon, unstable, or
   likely to produce unreliable validation results.

The whitelist is not intended to be a perfect Spanish dictionary. It is a practical
filter for evaluation stability.

Words outside the whitelist are treated as less reliable validation targets, not
necessarily as invalid Spanish.

## Final Source Choice

The final approach uses `wordfreq`.

`wordfreq` provides frequency-ranked word lists for many languages, including
Spanish. It is a better fit for this project than a raw corpus scrape because it is
already processed into a practical frequency resource and exposes Spanish words in
frequency order.

However, without using an excessively long list, it may not capture some expected words.
Eg, it may not include some legitimate noun variants for -o/-a/-os/-as
(eg, "sugestivo" exists but -a/-os/-as are missing). So we add a programmatic layer for that case.
Ideally, we should source a better list for our use case but yea.

The script uses:

```python
from wordfreq import iter_wordlist, zipf_frequency
```

The relevant behavior is:

- iterate Spanish words in frequency order
- normalize each word into the same form used at runtime
- filter malformed tokens
- deduplicate normalized forms
- rank by frequency
- keep the top N
- alphabetize the final output

The recommended starting cutoff is:

```bash
--limit 75000
```

A 75,000-word Spanish whitelist appears to provide good long-tail coverage while
remaining much cleaner than the earlier OpenSLR-based attempt.

## Why Not OpenSLR

An earlier attempt used OpenSLR SLR21, a Spanish word-frequency list derived from
Spanish Gigaword. It initially seemed attractive because it was large and frequency
ranked.

However, inspection showed that the list was too noisy for this purpose.

Examples of observed problems included:

- broken accented forms, such as `automtico`
- damaged `ñ` forms, such as `aadir`, `aadiendo`, `aadieron`
- flattened URL artifacts, such as `httpwwwnhcnoaagov`
- glued headline/category tokens, such as `mexicocrimendrogas`
- glued CamelCase entities, such as `CanadaCuba`, `BryanMike`, `BMWWilliams`
- random opaque junk strings, such as `axmjco`, `blre`, `misdk`, `misyow`

Some of these artifacts still appeared around the 75,000-word cutoff after several
filters were added. This made OpenSLR unsuitable as the primary whitelist source.

The conclusion was:

```text
Do not use OpenSLR as the primary Spanish whitelist source.
```

OpenSLR may still be useful someday as a heavily filtered supplemental source for
proper names or entities, but it is disregarded for the current implementation.

## Proper Names

Proper names are intentionally allowed when they appear in the frequency list.

This is useful for TTS/STT validation because ordinary Spanish text may contain:

- first names
- surnames
- countries
- cities
- regions
- demonyms
- common institutions or entities

Examples of useful proper-name-like normalized forms:

```text
maria
garcia
jose
juan
ana
luis
martinez
rodriguez
mexico
argentina
madrid
colombia
```

The whitelist is not restricted to dictionary-only Spanish words.

The intended category is:

```text
Spanish-ish surface forms likely to appear in ordinary source text or Whisper output.
```

## Normalization Policy

The whitelist is pre-normalized at build time.

The same normalization policy should be used for:

- whitelist generation
- source text
- Whisper transcript
- any future supplement files
- any stoplist files

The policy is:

1. Unicode normalize.
2. Casefold/lowercase.
3. Strip Spanish vowel accents and diaeresis.
4. Preserve `ñ`.
5. Remove punctuation.
6. Collapse whitespace.
7. Tokenize into words.

Examples:

```text
¿Qué está haciendo María?  -> que esta haciendo maria
acción                     -> accion
pingüino                   -> pinguino
dámelo                    -> damelo
vámonos                   -> vamonos
año                        -> año
señor                      -> señor
```

Important distinction:

```text
Strip vowel accents: yes
Strip ü diaeresis: yes
Preserve ñ: yes
Lowercase/casefold: yes
Remove punctuation: yes
```

`ñ` is preserved because it is phonemic:

```text
año != ano
señor != senor
niño != nino
```

## Build Process

The script performs the following steps:

1. Read Spanish words from `wordfreq` in frequency order.
2. Compute each raw word's Zipf frequency using `zipf_frequency`.
3. Normalize each word:
   - casefold/lowercase
   - strip vowel accents
   - preserve `ñ`
4. Reject malformed tokens.
5. Deduplicate normalized forms.
6. Rank retained forms by frequency.
7. Keep the top N forms.
8. Alphabetize the final retained words.
9. Write the final whitelist as one word per line.
10. Write audit files for inspection.

The output whitelist is already normalized and should be used directly at runtime.

## Output Files

### `spanish_whitelist.txt`

The final runtime whitelist.

Format:

```text
one normalized token per line
alphabetically sorted
UTF-8
```

Example:

```text
accion
ano
año
buenisimo
damelo
esta
maria
señor
vamos
vamonos
```

### `spanish_whitelist_ranked.tsv`

A ranked audit file.

Format:

```text
rank    word    zipf_frequency    raw_examples
```

This file is useful for inspecting the cutoff boundary.

For example, to inspect the 75,000-word cutoff:

```bash
awk -F '\t' 'NR == 1 || ($1 >= 74500 && $1 <= 75500)' spanish_whitelist_ranked.tsv > cutoff_75k_sample.tsv
```

### `spanish_whitelist_rejected.tsv`

A rejected-token audit file.

Format:

```text
raw_word    normalized    zipf_frequency    reason
```

This is useful for confirming that the filters are rejecting junk rather than
legitimate Spanish words.

## Recommended Command

Install dependency:

```bash
python -m pip install wordfreq
```

Generate the whitelist:

```bash
python build_wordfreq_spanish_whitelist.py --limit 75000
```

This creates:

```text
spanish_whitelist.txt
spanish_whitelist_ranked.tsv
spanish_whitelist_rejected.tsv
```

## Cutoff Rationale

The English whitelist used by the project is around 60,000 words.

Spanish may benefit from a larger surface-form list because of:

- verb conjugations
- gender agreement
- plural agreement
- clitic attachment
- accent normalization
- proper names and place names

The tested `wordfreq` 75,000-word boundary looked substantially cleaner than the
OpenSLR equivalent. It contained many useful long-tail Spanish forms such as:

```text
destilados
destilerias
devolviendole
dictarse
dificulte
digerible
disgustaba
diversificando
documentaron
dulceria
ecocidio
ejecutamos
ejidales
embarazosas
embarcamos
emplazadas
empleadora
empoderado
encasillar
encendera
```

There were still a few questionable entries, such as loanwords, product names,
acronyms, or English bleed, but this was considered acceptable for the intended
whitelist use case.

The current recommendation is:

```text
Use 75,000 as the initial Spanish cutoff.
Revisit only if real validation results show coverage or noise problems.
```

## Runtime Usage

At runtime:

1. Load `spanish_whitelist.txt` into a set.
2. Normalize the TTS source text using the same normalization policy.
3. Normalize the Whisper transcript using the same normalization policy.
4. Tokenize both normalized strings.
5. During comparison, use the whitelist to decide whether a word is a reliable
   validation target.

Example:

```python
with open("spanish_whitelist.txt", encoding="utf-8") as f:
    spanish_whitelist = {line.strip() for line in f if line.strip()}

if normalized_word in spanish_whitelist:
    # treat as conventional / reliable enough for validation
    ...
else:
    # treat as uncommon, noisy, or unreliable for this validation pass
    ...
```

## Supplement Files

A supplement file is optional.

The current plan is to start without one and simply use the 75,000-word `wordfreq`
whitelist.

If needed later, a small project-specific supplement file could be added for:

- book-specific character names
- author names
- place names
- recurring technical/domain terms
- intentional foreign words
- recurring proper nouns not covered by `wordfreq`

The supplement file should also be normalized by the build script before inclusion.

For now, no supplement file is required.

## Expected Limitations

The `wordfreq` whitelist should not have the severe surface-form holes typical of a
lemma-only dictionary list. It should cover many conjugations, plural/gender forms,
proper names, and clitic-attached forms.

Remaining holes may include:

- rare clitic-heavy forms
- uncommon names or surnames
- book-specific proper nouns
- technical/domain terms
- archaic or literary words
- intentionally foreign words
- stylized spellings

These are acceptable limitations because such words are often exactly the words that
are less reliable as STT/TTS validation targets.

## Summary

The final selected approach is:

```text
Use wordfreq Spanish as the primary source.
Generate a 75,000-word normalized whitelist.
Pre-normalize the whitelist.
Lowercase/casefold everything.
Strip vowel accents.
Preserve ñ.
Keep useful proper names.
Reject malformed tokens.
Ignore OpenSLR for now.
Use the final whitelist directly at runtime.
```

This provides a practical Spanish equivalent of the English frequency-list whitelist
approach without requiring a full Spanish morphology-generation project.
