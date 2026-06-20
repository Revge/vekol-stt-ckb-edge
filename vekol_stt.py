#!/usr/bin/env python3
"""Vekol-STT — Sorani (Central Kurdish, ckb) speech-to-text, edge build.

Runs offline on CPU with no PyTorch: ONNX Runtime + numpy only. Whisper models
(tiny/base/small) fine-tuned for Sorani; weights hosted on Hugging Face and
downloaded on first run. Part of the Vekol hub by Revge.

Usage:
    python3 vekol_stt.py audio.wav                 # base model
    python3 vekol_stt.py audio.wav --model small   # tiny | base | small

Deps:  pip install onnxruntime numpy soundfile huggingface_hub
"""
import os, sys, json, argparse, unicodedata, re
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = {"tiny": "RevgeAI/vekol-stt-ckb-tiny",
           "base": "RevgeAI/vekol-stt-ckb-base",
           "small": "RevgeAI/vekol-stt-ckb-small"}
# multilingual Whisper special-token ids (identical for tiny/base/small)
SOT, FA, TRN, NOTS, EOT = 50258, 50300, 50359, 50363, 50257
N_FFT, HOP, N_SAMPLES = 400, 160, 480000

# ---- asset loading: local bundle first, else download from Hugging Face ----
def _asset(model, name):
    local = os.path.join(HERE, f"whisper-{model}", name)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=WEIGHTS[model], filename=name)

# ---- log-mel (Whisper), pure numpy ----
_MELF = np.load(os.path.join(HERE, "mel_filters.npy"))   # (80, 201)
def _logmel(audio):
    audio = np.asarray(audio, np.float32)
    audio = np.pad(audio, (0, N_SAMPLES - len(audio))) if len(audio) < N_SAMPLES else audio[:N_SAMPLES]
    win = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    a = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(a) - N_FFT) // HOP
    fr = np.lib.stride_tricks.as_strided(a, (n, N_FFT), (a.strides[0] * HOP, a.strides[0])).copy()
    mag = (np.abs(np.fft.rfft(fr * win, axis=1)) ** 2).T[:, :-1]
    lm = np.log10(np.maximum(_MELF @ mag, 1e-10))
    lm = np.maximum(lm, lm.max() - 8.0)
    return ((lm + 4.0) / 4.0).astype(np.float32)[None]

# ---- byte-level token decode (GPT-2 style), pure python ----
def _byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
_BD = _byte_decoder()

_MAP = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی", "ة": "ە", "ؤ": "و",
                      "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
                      "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5",
                      "٦": "6", "٧": "7", "٨": "8", "٩": "9",
                      "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5",
                      "۶": "6", "۷": "7", "۸": "8", "۹": "9"})
_STRIP = re.compile(r"[‌‍ـً-ْ]")
def _norm(t):
    t = unicodedata.normalize("NFC", t).translate(_MAP)
    return re.sub(r"\s+", " ", _STRIP.sub("", t)).strip()

_CACHE = {}
def _load(model):
    if model in _CACHE:
        return _CACHE[model]
    so = ort.SessionOptions()
    enc = ort.InferenceSession(_asset(model, "encoder_model.onnx"), so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(_asset(model, "decoder_model_merged.onnx"), so, providers=["CPUExecutionProvider"])
    vocab = json.load(open(_asset(model, "vocab.json"), encoding="utf-8"))
    id2tok = {i: t for t, i in vocab.items()}
    # heads/head_dim from the decoder's past_key_values input shape: (batch, H, seq, HD)
    sh = next(i.shape for i in dec.get_inputs() if i.name.startswith("past_key_values"))
    H, HD = int(sh[1]), int(sh[3])
    pin = [i.name for i in dec.get_inputs() if i.name.startswith("past_key_values")]
    onames = [o.name for o in dec.get_outputs()]
    _CACHE[model] = (enc, dec, id2tok, H, HD, pin, onames)
    return _CACHE[model]

def transcribe(path, model="base", max_new_tokens=440):
    import soundfile as sf
    enc, dec, id2tok, H, HD, pin, onames = _load(model)
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        x = np.linspace(0, len(audio) / sr, int(len(audio) * 16000 / sr), endpoint=False)
        audio = np.interp(x, np.arange(len(audio)) / sr, audio).astype(np.float32)
    h = enc.run(None, {"input_features": _logmel(audio)})[0]
    empty = np.zeros((1, H, 0, HD), np.float32)
    feed = {"input_ids": np.array([[SOT, FA, TRN, NOTS]], np.int64), "encoder_hidden_states": h,
            "cache_position": np.arange(4, dtype=np.int64), "use_cache_branch": np.array([False])}
    for n in pin:
        feed[n] = empty
    out = dec.run(None, feed)
    logits = out[0]
    cache = {n.replace("present", "past_key_values"): v for n, v in zip(onames, out) if n != "logits"}
    nxt = int(logits[0, -1].argmax()); gen = [nxt]; pos = 4
    for _ in range(max_new_tokens):
        if nxt == EOT:
            break
        feed = {"input_ids": np.array([[nxt]], np.int64), "encoder_hidden_states": h,
                "cache_position": np.array([pos], np.int64), "use_cache_branch": np.array([True])}
        for k, v in cache.items():
            feed[k] = v
        out = dec.run(None, feed); logits = out[0]
        for n, v in zip(onames, out):
            if n != "logits" and ".decoder." in n:
                cache[n.replace("present", "past_key_values")] = v
        nxt = int(logits[0, -1].argmax()); gen.append(nxt); pos += 1
    s = "".join(id2tok[i] for i in gen if i < 50257 and i in id2tok)
    return _norm(bytes([_BD[c] for c in s]).decode("utf-8", errors="replace"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="base", choices=["tiny", "base", "small"])
    a = ap.parse_args()
    print(transcribe(a.audio, a.model))
