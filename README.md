# Vekol · STT

Kurdish Sorani speech to text, on every device.

![quality](https://img.shields.io/badge/CER-1.9%25%20%E2%86%92%209.9%25-informational)
![license](https://img.shields.io/badge/license-CC--BY--NC%204.0-lightgrey)
![base](https://img.shields.io/badge/base-OpenAI%20Whisper-blue)
![language](https://img.shields.io/badge/language-ckb%20(Sorani)-green)
![runtime](https://img.shields.io/badge/runtime-CPU%20%2F%20edge-orange)

> These are the free, on-device edge models. For the large, more accurate models
> (down to ~1.9% CER) and real-time streaming, use the hosted service at
> [vekol.krd](https://vekol.krd).

## What it is

Three small Central Kurdish (Sorani) speech-to-text models you can run on a laptop,
phone, or small server with no GPU. They are Whisper models fine-tuned on Common Voice
and FLEURS, picked so you can trade size for accuracy:

| Model | Params | int8 / int4 size | CER (spacing-free) | Latency (CPU) |
|-------|--------|------------------|--------------------|---------------|
| `whisper-tiny`  | 39M  | 37 / 18 MB  | 9.85% | ~0.4 s/clip |
| `whisper-base`  | 74M  | 72 / 36 MB  | 7.95% | ~0.5 s/clip |
| `whisper-small` | 244M | 241 / 121 MB | ~3%  | ~0.9 s/clip |

CER is the spacing-free character error rate on the official, speaker-disjoint Common
Voice 25 test split (Kurdish has no standard word-spacing, so character error is the
fair measure). For higher accuracy (down to ~1.9% CER) and real-time streaming, see the
hosted service at [vekol.krd](https://vekol.krd).

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python vekol_stt.py audio.wav                 # base model
python vekol_stt.py audio.wav --model small   # tiny | base | small
python vekol_stt.py audio.wav --quant int8    # fp32 | fp16 | int8 | int4
```

Or from Python:

```python
from vekol_stt import transcribe
print(transcribe("audio.wav", model="base", quant="int8"))
```

Weights are pulled from Hugging Face on first run (one repo per size:
`RevgeAI/vekol-stt-ckb-tiny`, `-base`, `-small`), or pass a local path in `vekol_stt.py`.

## Models

Each size is fine-tuned with the Persian (`fa`) language token, which anchors the model
to the Arabic script (Whisper has no Sorani token). Pick `tiny` for the smallest
footprint, `small` for the best accuracy, `base` for the middle.

## How it was built

- Base: `openai/whisper-{tiny,base,small}`.
- Data: Common Voice 25.0 (ckb) + FLEURS (ckb_iq), normalized to Kurdish orthography.
- Training: 10 epochs on a single GPU, AdamW at 1e-5, fp16, with waveform augmentation
  (time-stretch, pitch, noise, gain, low-pass, time-mask, clipping, MP3) and SpecAugment.
- Augmentation was the largest single gain; see the research notes for the full record.

## Text and script handling

Output is normalized to Sorani: Arabic letters and digits are mapped to their Kurdish
forms, and diacritics, tatweel, and zero-width marks are stripped. The models predict
unspaced or differently-spaced text in places, which is expected for Kurdish and does
not affect the character-level reading.

## Running it elsewhere

The models load through `transformers`, so they run anywhere PyTorch does. For lower-level
deployment, export to ONNX or convert to a `whisper.cpp` / GGML build for the tiny and
base sizes; int8 keeps the accuracy and quarters the size.

## Limitations

- Trained on read speech (Common Voice, FLEURS); accuracy drops on far-field, heavy
  noise, or strong dialectal variation.
- `tiny` and `base` are meant for on-device and quick drafts, not transcription you ship
  unchecked. Use `small`, or the hosted service, when accuracy matters.
- Sorani only (ckb). Not for Kurmanji or other Kurdish variants.

## License and credits

Weights and code are released under CC-BY-NC 4.0 (non-commercial). The base Whisper
models are MIT (OpenAI). For commercial use, use the hosted service at
[vekol.krd](https://vekol.krd) or contact us for a license. See `NOTICE` for data and
model attribution.

```bibtex
@misc{vekol_stt_ckb_edge,
  title  = {Vekol STT (ckb) Edge: small Whisper models for Kurdish Sorani speech-to-text},
  author = {Revge},
  year   = {2026},
  url    = {https://github.com/Revge/vekol-stt-ckb-edge}
}
```
