<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vekol-white.svg">
  <img src="assets/vekol-black.svg" alt="Vekol" width="300">
</picture>

# Vekol&nbsp;·&nbsp;STT

### Sorani speech to text, on every device.

Offline **Central Kurdish (Sorani)** speech-to-text on plain **CPU** — no GPU, **no PyTorch**,
no internet. Three Whisper models exported to **ONNX**, running on `onnxruntime` + numpy,
**several times faster than real time**, down to **~3% character error** on an honest,
speaker-disjoint test.

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
| Models | `vekol-stt-ckb-edge` — three sizes: tiny, base, small |
| Language | Central Kurdish / Sorani (`ckb`), Arabic script |
| Task | speech-to-text (transcription) |
| Input | any audio (loaded at 16 kHz mono) |
| Architecture | Whisper (seq2seq), fine-tuned with the `fa` script anchor, exported to ONNX |
| Runtime | ONNX Runtime + numpy (CPU) — no PyTorch, no transformers |
| Weights | [RevgeAI/vekol-stt-ckb-{tiny,base,small}](https://huggingface.co/RevgeAI) |

## Models

Pick by footprint vs accuracy — `tiny` for the smallest device, `small` for the best
transcripts, `base` for the balance.

| Model | Params | ONNX size | CER (spacing-free) | CPU latency |
|-------|--------|-----------|--------------------|-------------|
| `whisper-tiny`  | 39M  | ~150 MB | 9.85% | ~0.3 s |
| `whisper-base`  | 74M  | ~240 MB | 7.95% | ~0.5 s |
| `whisper-small` | 244M | ~650 MB | ~3% | ~1.5 s |

Numbers are on the **official, speaker-disjoint** Common Voice 25 test split — no speaker
overlap between train and test, so they reflect real generalization, not memorized voices
(a common way Kurdish ASR scores get inflated). CER is the **spacing-free** character error
rate, because Kurdish has no standard word-spacing and word error would over-penalize valid
spelling. Latency is one ~7 s clip on a 4-core CPU (fp32, greedy); all three beat real time.
For the large models (down to ~1.9% CER) and live streaming, see [vekol.krd](https://vekol.krd).

## Install

```bash
pip install -r requirements.txt
```

Just `onnxruntime`, `numpy`, `soundfile`, and `huggingface_hub` — no PyTorch. The ONNX
weights download from Hugging Face on first run.

## Usage

```bash
python3 vekol_stt.py audio.wav
# چاری دەردی کوردەواری خوێندنە، هەر خوێندنە

python3 vekol_stt.py audio.wav --model small   # tiny | base | small
```

From Python:

```python
from vekol_stt import transcribe
print(transcribe("audio.wav", model="base"))
```

## How it was built

Fine-tuned from `openai/whisper-{tiny,base,small}` on ~100k clips of Central Kurdish speech
from Common Voice 25.0 (ckb) and FLEURS (ckb_iq), normalized to Sorani orthography. Whisper
has no Sorani token, so each model is trained with the Persian (`fa`) token as a script
anchor — without it, the small models drift into English or Russian. Ten epochs, AdamW at
1e-5, fp16, with eight waveform augmentations (time-stretch, pitch, noise, gain, low-pass,
time-mask, clipping, MP3) plus SpecAugment; the augmentation was the single biggest gain.
Evaluation is on the official speaker-disjoint test split. The fine-tuned models are then
exported to ONNX (fp32, merged decoder with KV-cache) so they run on `onnxruntime` with no
PyTorch — the output is identical to the original.

## Letters & text handling

`vekol_stt.py` returns Sorani normalized so nothing is dropped. It folds the common Arabic
variants onto the Kurdish letters the model uses — kaf `ك`→`ک`, ya `ي`/`ى`→`ی`, teh-marbuta
`ة`→`ە`, waw-hamza `ؤ`→`و`, the alef variants → `ا` — maps Arabic and Farsi digits to ASCII,
and strips diacritics, tatweel, and zero-width marks (NFC throughout). Kurdish has no fixed
word-spacing, so output may be spaced differently from a given reference; this is expected
and does not change the reading, which is why accuracy is measured spacing-free.

## Running it elsewhere

The models are standard ONNX (encoder + merged decoder with KV-cache), so ONNX Runtime can
run them with no Python at all — C++, Rust, Go, Java, C#, JavaScript/Web, Android, iOS, or a
Raspberry Pi. `vekol_stt.py` is the reference CPU implementation (numpy log-mel + greedy
decode loop + byte-level token decode); port that small amount of glue to your target.

## Limitations

Trained on read speech (Common Voice, FLEURS): accurate there, weaker on far-field audio,
heavy noise, or strong dialect. `tiny` and `base` are for on-device drafts and quick input,
not transcripts you ship unchecked — use `small`, or the hosted service, when it has to be
right. Sorani only (`ckb`), not Kurmanji or other variants. Whisper transcribes a clip at a
time; for continuous live captioning use the streaming model at [vekol.krd](https://vekol.krd).

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
