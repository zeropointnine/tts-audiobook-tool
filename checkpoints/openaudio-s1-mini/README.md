---
tags:
- text-to-speech
license: cc-by-nc-sa-4.0
language:
- zh
- en
- de
- ja
- fr
- es
- ko
- ar
- nl
- ru
- it
- pl
- pt
pipeline_tag: text-to-speech
inference: false
extra_gated_prompt: >-
  You agree to not use the model to generate contents that violate DMCA or local
  laws.
extra_gated_fields:
  Country: country
  Specific date: date_picker
  I agree to use this model for non-commercial use ONLY: checkbox
---


# OpenAudio S1

**OpenAudio S1** is a leading text-to-speech (TTS) model trained on more than 2 million hours of audio data in multiple languages.

Supported languages:
- English (en)
- Chinese (zh)
- Japanese (ja)
- German (de)
- French (fr)
- Spanish (es)
- Korean (ko)
- Arabic (ar)
- Russian (ru)
- Dutch (nl)
- Italian (it)
- Polish (pl)
- Portuguese (pt)

Please refer to [Fish Speech Github](https://github.com/fishaudio/fish-speech) for more info.
Demo available at [Fish Audio Playground](https://fish.audio).
Visit the [OpenAudio website](https://openaudio.com) for blog & tech report.

## Emotion and Tone Support

OpenAudio S1 supports a variety of emotional, tone, and special markers to enhance speech synthesis:

**1. Emotional markers:**
(angry) (sad) (disdainful) (excited) (surprised) (satisfied) (unhappy) (anxious) (hysterical) (delighted) (scared) (worried) (indifferent) (upset) (impatient) (nervous) (guilty) (scornful) (frustrated) (depressed) (panicked) (furious) (empathetic) (embarrassed) (reluctant) (disgusted) (keen) (moved) (proud) (relaxed) (grateful) (confident) (interested) (curious) (confused) (joyful) (disapproving) (negative) (denying) (astonished) (serious) (sarcastic) (conciliative) (comforting) (sincere) (sneering) (hesitating) (yielding) (painful) (awkward) (amused)

**2. Tone markers:**
(in a hurry tone) (shouting) (screaming) (whispering) (soft tone)

**3. Special markers:**
(laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting) (groaning) (crowd laughing) (background laughter) (audience laughing)

**Special markers with corresponding onomatopoeia:**
- Laughing: Ha,ha,ha
- Chuckling: Hmm,hmm

## Model Variants and Performance

OpenAudio S1 includes the following models:
-   **S1 (4B, proprietary):** The full-sized model.
-   **S1-mini (0.5B):** A distilled version of S1.

Both S1 and S1-mini incorporate online Reinforcement Learning from Human Feedback (RLHF).

**Seed TTS Eval Metrics (English, auto eval, based on OpenAI gpt-4o-transcribe, speaker distance using Revai/pyannote-wespeaker-voxceleb-resnet34-LM):**

-   **S1:**
    -   WER (Word Error Rate): **0.008**
    -   CER (Character Error Rate): **0.004**
    -   Distance: **0.332**
-   **S1-mini:**
    -   WER (Word Error Rate): **0.011**
    -   CER (Character Error Rate): **0.005**
    -   Distance: **0.380**

## License

This model is permissively licensed under the CC-BY-NC-SA-4.0 license.
