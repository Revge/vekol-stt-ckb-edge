# Model Card — Vekol STT (ckb) Edge

Author: Darvan Shvan

## Overview

A set of three small Whisper models fine-tuned for Central Kurdish (Sorani, `ckb`)
speech-to-text, intended for on-device and CPU use: `whisper-tiny`, `whisper-base`,
and `whisper-small`.

- Task: automatic speech recognition (transcription)
- Language: Central Kurdish / Sorani (`ckb`), Arabic script
- Architecture: Whisper (encoder-decoder, seq2seq)
- Input: 16 kHz mono audio
- Output: normalized Sorani text

## Sizes and accuracy

| Model | Params | int8 / int4 | WER | CER (spacing-free) |
|-------|--------|-------------|-----|--------------------|
| whisper-tiny  | 39M  | 37 / 18 MB  | 35.0% | 9.85% |
| whisper-base  | 74M  | 72 / 36 MB  | 29.0% | 7.95% |
| whisper-small | 244M | 241 / 121 MB | 15.5% | ~3% |

Evaluated on the official, speaker-disjoint Common Voice 25.0 (ckb) test split. CER is
reported with spaces removed, because Kurdish has no standard word-spacing and WER
over-penalizes valid spelling variants.

## Training data

- Common Voice 25.0 (ckb) and FLEURS (ckb_iq).
- ~102k training clips after quality filtering (low-SNR, empty, and over-long clips
  removed; quiet FLEURS clips kept and peak-normalized).
- Text normalized to Kurdish orthography (Arabic-to-Kurdish letter and digit mapping,
  diacritic and zero-width stripping).

## Training procedure

- Fine-tuned from `openai/whisper-{tiny,base,small}`.
- 10 epochs, AdamW at 1e-5, linear schedule, 500 warmup steps, fp16, gradient checkpointing.
- Decoded and trained with the Persian (`fa`) language token to anchor the Arabic script.
- Waveform augmentation (time-stretch, pitch shift, Gaussian noise, gain, low-pass,
  time-mask, clipping, MP3 compression) plus encoder SpecAugment.

## Intended use

- On-device or offline Kurdish transcription, quick drafts, voice input, captioning aids.
- `small` for the best quality in this set; `tiny`/`base` for the smallest footprint.

## Limitations and risks

- Trained on read speech; accuracy degrades on spontaneous, far-field, or noisy audio.
- May mis-transcribe names, numbers, and code-switched words.
- Sorani only; not suitable for Kurmanji or other variants.
- Not intended for high-stakes use without human review.

## License

CC-BY-NC 4.0 (non-commercial). Base Whisper models are MIT (OpenAI). Commercial use is
available through the hosted service at [vekol.krd](https://vekol.krd). See `NOTICE`.
