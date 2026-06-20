<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vekol-white.svg">
  <img src="assets/vekol-black.svg" alt="Vekol" width="300">
</picture>

# Vekol&nbsp;·&nbsp;STT

### Sorani speech to text, on every device.

Offline **Central Kurdish (Sorani)** speech-to-text that runs on plain **CPU** —
no GPU, no internet. Small Whisper models (down to ~18 MB) that transcribe Sorani in real time.

[![Higher accuracy](https://img.shields.io/badge/higher%20accuracy-vekol.krd-6f42c1)](https://vekol.krd)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC--BY--NC--4.0-555)](LICENSE)
[![Models](https://img.shields.io/badge/🤗-RevgeAI%2Fvekol--stt--ckb--edge-yellow)](https://huggingface.co/RevgeAI)
![Language](https://img.shields.io/badge/lang-ckb%20(Sorani)-1f6feb)
![Runtime](https://img.shields.io/badge/Whisper-CPU-2ea043)

This is the free, open-source **edge** set. For higher-accuracy Sorani transcription, try the
hosted version at **[vekol.krd](https://vekol.krd)** · part of **Vekol**, Revge's Kurdish AI hub.

</div>

---

> **Non-commercial (CC-BY-NC 4.0).** Released for non-commercial use to keep the hosted
> service ([vekol.krd](https://vekol.krd)) sustainable. The base Whisper models are MIT
> (OpenAI); the fine-tuned weights and the code here are CC-BY-NC. Commercial use needs a
> license — use the hosted API or get in touch. See [`NOTICE`](NOTICE).

## What it is

| | |
|---|---|
| Models | `vekol-stt-ckb-edge` (tiny, base, small) |
| Language | Central Kurdish / Sorani (`ckb`), Arabic script |
| Task | speech-to-text (transcription) |
| Input | 16 kHz mono audio |
| Architecture | Whisper (seq2seq), fine-tuned with the `fa` script anchor, quantizable to int8/int4 |
| Size | 18–241 MB, by model and precision |
| Weights | [RevgeAI/vekol-stt-ckb-{tiny,base,small}](https://huggingface.co/RevgeAI) |

## Models

Three sizes, so you can trade footprint for accuracy:

| Model | Params | int8 / int4 | CER (spacing-free) | CPU latency |
|-------|--------|-------------|--------------------|-------------|
| `whisper-tiny`  | 39M  | 37 / 18 MB  | 9.85% | ~0.25 s |
| `whisper-base`  | 74M  | 72 / 36 MB  | 7.95% | ~0.45 s |
| `whisper-small` | 244M | 241 / 121 MB | ~3% | ~1.3 s |

CER is the spacing-free character error rate on the official, speaker-disjoint Common
Voice 25 test split (Kurdish has no standard word-spacing, so character error is the fair
measure). Latency is for a ~7 s clip on a 4-core CPU (fp32, greedy) — all three run
several times faster than real time. For the large models (down to ~1.9% CER) and
real-time streaming, see [vekol.krd](https://vekol.krd).

## Install

```bash
pip install -r requirements.txt
```

The weights live on Hugging Face. The script downloads them on first run, or grab one
yourself:

```bash
huggingface-cli download RevgeAI/vekol-stt-ckb-base --local-dir ./whisper-base
```

## Usage

```bash
# one file
python3 vekol_stt.py audio.wav

# choose size and precision
python3 vekol_stt.py audio.wav --model small   # tiny | base | small
python3 vekol_stt.py audio.wav --quant int8    # fp32 | fp16 | int8 | int4
```

From Python:

```python
from vekol_stt import transcribe
print(transcribe("audio.wav", model="base", quant="int8"))
```

## Samples

In [`samples/`](samples/) — each clip has the **edge** transcription (this repo) and a
**hosted, higher-accuracy** version (from [vekol.krd](https://vekol.krd)), so you can see
the difference:

| Audio | Edge (this model) | Hosted — higher accuracy |
|-------|-------------------|--------------------------|
| `sample1.wav` | `sample1.txt` | `sample1-hosted.txt` |
| `sample2.wav` | `sample2.txt` | `sample2-hosted.txt` |

The edge files run fully offline on CPU. The hosted output is more accurate — try your own
audio at **[vekol.krd](https://vekol.krd)**.

## How it was built

Fine-tuned from `openai/whisper-{tiny,base,small}` on about 100k clips of Central Kurdish
speech from Common Voice 25.0 (ckb) and FLEURS (ckb_iq), normalized to Sorani orthography.
Each size is trained with the Persian (`fa`) language token as a script anchor, since
Whisper has no Sorani token. Ten epochs, AdamW at 1e-5, fp16, with waveform augmentation
(time-stretch, pitch, noise, gain, low-pass, time-mask, clipping, MP3) and SpecAugment;
augmentation was the single biggest gain. The models quantize to int8/int4 with no
measurable accuracy loss.

## Letters & text handling

`vekol_stt.py` returns Sorani text normalized so nothing is dropped: Arabic letters and
digits are folded onto the Kurdish forms the model uses (Arabic kaf `ك`→`ک`, teh-marbuta
`ة`→`ە`, alef-maksura `ى`→`ی`, waw-hamza `ؤ`→`و`, Arabic/Farsi digits → ASCII), and
diacritics, tatweel, and zero-width marks are stripped (NFC). Kurdish has no standard
word-spacing, so the output may be unspaced or differently spaced in places; this is
expected and does not change the character-level reading.

## Running it elsewhere

The models load through `transformers`, so they run anywhere PyTorch does — Linux, macOS,
Windows, a small server. For lower-level deployment, export `tiny` and `base` to ONNX or a
`whisper.cpp` / GGML build; int8 keeps the accuracy at a quarter of the size.

## Limitations

Trained on read speech (Common Voice, FLEURS): clear and accurate there, but accuracy
drops on far-field, heavy noise, or strong dialectal variation. `tiny` and `base` are for
on-device drafts and quick input, not transcripts you ship unchecked — use `small`, or the
hosted service, when accuracy matters. Sorani only (`ckb`), not Kurmanji or other variants.

For the most accurate transcription and real-time streaming, use the hosted version at
[vekol.krd](https://vekol.krd).

## License & credits

CC-BY-NC 4.0 (non-commercial) — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Fine-tuned
from [OpenAI Whisper](https://github.com/openai/whisper) (MIT).

```bibtex
@software{vekol_stt_ckb_edge,
  title        = {Vekol-STT: Sorani (Central Kurdish) on-device STT},
  author       = {Shvan, Darvan},
  organization = {Revge},
  year         = {2026},
  url          = {https://github.com/Revge/vekol-stt-ckb-edge}
}
```

<div align="center">

Built by **Darvan Shvan** · **[Revge](https://github.com/Revge)** · part of the **Vekol** hub

</div>
