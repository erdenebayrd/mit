"""Pi Step 3 check.
1) every file arrived intact (size + md5 against MANIFEST.json from Colab)
2) the test set and vocabulary load
3) the KenLM decoder loads (same settings as the thesis)
4) every ONNX model runs on one utterance and gives a sensible sentence
5) every PyTorch Gaddy model loads and runs on one utterance
The times printed here are single first runs, not the benchmark (that is Pi Step 5).
Run:  cd ~/emg/pi5_bundle && python check_bundle.py
"""
import os, sys, json, time, glob, hashlib
import numpy as np

B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


print("== 1) files ==")
man = json.load(open(os.path.join(B, "MANIFEST.json")))
bad = 0
for rel, info in man.items():
    p = os.path.join(B, rel)
    if not os.path.exists(p):
        print(f"   MISSING  {rel}")
        bad += 1
        continue
    ok = os.path.getsize(p) == info["bytes"] and md5(p) == info["md5"]
    bad += 0 if ok else 1
    print(f"   {'ok ' if ok else 'BAD'}  {info['bytes'] / 1e6:9.2f} MB  {rel}")
print("   all files intact" if bad == 0 else f"   {bad} file(s) missing or damaged -> copy them again")

print("\n== 2) test set ==")
d = np.load(os.path.join(B, "test_set.npz"))
emg, off = d["emg"], d["offsets"]
refs = [str(r) for r in d["refs"]]
utts = [emg[off[i]:off[i + 1]] for i in range(len(refs))]
vocab = json.load(open(os.path.join(B, "vocab.json")))
fs = vocab.get("emg_rate_hz")
print(f"   {len(utts)} utterances, {sum(len(r.split()) for r in refs)} reference words, "
      f"EMG rate {fs} Hz, first utterance shape {utts[0].shape}")

print("\n== 3) decoder ==")
import torch
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

t0 = time.time()
dec = ctc_decoder(lexicon=os.path.join(B, "decoder", "gaddy_derived_lexicon.txt"),
                  tokens=list(vocab["chars"]) + ["_"],
                  lm=os.path.join(B, "decoder", "lm.bin"),
                  blank_token="_", sil_token="|", nbest=1, lm_weight=4.0,
                  word_score=-1.75, sil_score=0.0, beam_size=1500)
print(f"   KenLM decoder loaded in {time.time() - t0:.1f} s (beam 1500, lm_weight 4.0, word_score -1.75)")


def decode(lg):
    logp = F.log_softmax(torch.as_tensor(lg, dtype=torch.float32), dim=-1)
    return " ".join(dec(logp)[0][0].words).strip()


x0 = utts[0].astype(np.float32)
print(f'   reference of utterance 0: "{refs[0]}"')

print("\n== 4) ONNX models (utterance 0) ==")
import onnxruntime as ort

for p in sorted(glob.glob(os.path.join(B, "models", "*.onnx"))):
    name = os.path.basename(p)
    try:
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0]
        print(f"   {name:22s} input '{i.name}' {i.shape} {i.type}")
        t0 = time.time()
        y = s.run(None, {i.name: x0[None]})[0]
        dt = time.time() - t0
        print(f"   {'':22s} output {tuple(y.shape)}, first run {dt * 1000:.0f} ms")
        print(f'   {"":22s} "{decode(y)}"')
    except Exception as e:
        print(f"   {name:22s} FAILED: {type(e).__name__}: {str(e)[:200]}")

print("\n== 5) PyTorch Gaddy models (utterance 0) ==")
import gaddy_model as gm

print("   torch threads:", torch.get_num_threads(), "| quantized engines:", torch.backends.quantized.supported_engines)
loaders = [("gaddy_fp32.pt", gm.load_fp32, False),
           ("gaddy_fp16.pt", gm.load_fp16, True),
           ("gaddy_dynint8.pt", lambda q: gm.load_int8_saved(q, engine="qnnpack"), False),
           ("gaddy_kd_student.pt", gm.load_kd_student, False)]
for fn, loader, half in loaders:
    p = os.path.join(B, "models", fn)
    if not os.path.exists(p):
        print(f"   {fn:22s} not in the bundle")
        continue
    try:
        t0 = time.time()
        m = loader(p)
        lt = time.time() - t0
        t0 = time.time()
        y = gm.logits(m, x0, half=half)
        dt = time.time() - t0
        print(f"   {fn:22s} loaded in {lt:.1f} s, output {tuple(y.shape)}, first run {dt * 1000:.0f} ms")
        print(f'   {"":22s} "{decode(y)}"')
    except Exception as e:
        print(f"   {fn:22s} FAILED: {type(e).__name__}: {str(e)[:200]}")

print("\nDone. Paste everything above into the chat.")
