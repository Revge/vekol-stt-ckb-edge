"""Vekol STT (ckb) edge — transcribe Kurdish Sorani audio with a small Whisper model.

Usage:
  python vekol_stt.py audio.wav                 # base model, fp32
  python vekol_stt.py audio.wav --model small   # tiny | base | small
  python vekol_stt.py audio.wav --quant int8    # fp32 | fp16 | int8 | int4
"""
import argparse, re, unicodedata
import librosa, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Hugging Face weights, one repo per size.
WEIGHTS = {"tiny": "RevgeAI/vekol-stt-ckb-tiny",
           "base": "RevgeAI/vekol-stt-ckb-base",
           "small": "RevgeAI/vekol-stt-ckb-small"}
BASE = {"tiny": "openai/whisper-tiny", "base": "openai/whisper-base", "small": "openai/whisper-small"}

# Sorani normalization: Arabic letters/digits -> Kurdish, strip diacritics/ZWNJ.
_MAP = str.maketrans({"ك":"ک","ي":"ی","ى":"ی","أ":"ا","إ":"ا","آ":"ا","ٱ":"ا","ة":"ە","ؤ":"و",
 "٠":"0","١":"1","٢":"2","٣":"3","٤":"4","٥":"5","٦":"6","٧":"7","٨":"8","٩":"9",
 "۰":"0","۱":"1","۲":"2","۳":"3","۴":"4","۵":"5","۶":"6","۷":"7","۸":"8","۹":"9"})
_STRIP = re.compile(r"[‌‍​‎‏﻿ً-ْٰﹰ-ﹿـ]")

def normalize(t):
    t = unicodedata.normalize("NFC", t).translate(_MAP)
    return re.sub(r"\s+", " ", _STRIP.sub("", t)).strip()

def load(model, quant):
    src = WEIGHTS[model]   # or pass a local path here
    proc = WhisperProcessor.from_pretrained(BASE[model])
    m = WhisperForConditionalGeneration.from_pretrained(src, torch_dtype=torch.float32).eval()
    m.generation_config.forced_decoder_ids = None
    if quant in ("int8", "int4"):
        from optimum.quanto import quantize, freeze, qint8, qint4
        quantize(m, weights=qint8 if quant == "int8" else qint4); freeze(m)
    elif quant == "fp16":
        m = m.half()
    return proc, m

def transcribe(path, model="base", quant="fp32"):
    proc, m = load(model, quant)
    audio, _ = librosa.load(path, sr=16000)
    feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    if quant == "fp16":
        feats = feats.half()
    with torch.no_grad():
        # 'fa' (Persian) anchors the small models to the Arabic script; Sorani has no Whisper token.
        ids = m.generate(feats, task="transcribe", language="fa", max_new_tokens=225, num_beams=1)
    return normalize(proc.tokenizer.decode(ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="path to a WAV/MP3/FLAC file")
    ap.add_argument("--model", default="base", choices=["tiny", "base", "small"])
    ap.add_argument("--quant", default="fp32", choices=["fp32", "fp16", "int8", "int4"])
    a = ap.parse_args()
    print(transcribe(a.audio, a.model, a.quant))
