"""Pi Step 4: WER of every model ON THE RASPBERRY PI 5.
Same 98 test utterances, same KenLM decoder and settings as the thesis (beam 1500, lm_weight 4.0,
word_score -1.75). For each utterance it also records the model time and the decoder time as a
first look; the clean speed benchmark (warm-up, repeats, threads) is Pi Step 5.

Run (inside tmux, takes about 1 to 1.5 hours for all models):
    cd ~/emg/pi5_bundle && python wer_pi.py 2>&1 | tee ~/emg/results/wer_pi_log.txt
The script loads the 3.1 GB language model into memory by itself at the start (no manual step).
Only some models:
    python wer_pi.py --only l4_int8 tinymyo_int8
Gaddy FP16 is stored in 16 bit and computed in 32 bit by default, because PyTorch has no fast
half-precision path on the Pi CPU (one utterance took about 31 s natively). --fp16-native forces
true half-precision compute (very slow).

Outputs in ~/emg/results/:
    pi_wer.csv                 one row per model (WER Pi vs Colab, times, real-time factor, memory)
    pi_wer_<model>.csv         one row per utterance (EMG seconds, model ms, decoder ms)
    pi_pred_<model>.txt        the recognised sentence for every utterance
"""
import os, sys, csv, json, time, argparse, subprocess, gc
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import jiwer
import onnxruntime as ort
from torchaudio.models.decoder import ctc_decoder

B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B)
import gaddy_model as gm

OUT = os.path.expanduser("~/emg/results")
os.makedirs(OUT, exist_ok=True)
THREADS = 4

# WER measured in Colab = the thesis values (Tables 4.3 and 4.6)
COLAB = {"l4_int8": 45.69, "l4_fp32": 40.22, "tinymyo_int8": 36.51, "tinymyo_fp32": 34.69,
         "gaddy_kd_student": 63.30, "gaddy_fp32": 21.63, "gaddy_fp16": 21.51, "gaddy_dynint8": 21.69,
         "gaddy_fp32_onnx": 21.51, "gaddy_int8dyn_onnx": 22.30}
# name, file, how to run it  (smallest first, so useful results arrive early)
MODELS = [("l4_int8", "l4_int8.onnx", "onnx"), ("l4_fp32", "l4_fp32.onnx", "onnx"),
          ("tinymyo_int8", "tinymyo_int8.onnx", "onnx"), ("tinymyo_fp32", "tinymyo_fp32.onnx", "onnx"),
          ("gaddy_kd_student", "gaddy_kd_student.pt", "kd"), ("gaddy_fp32", "gaddy_fp32.pt", "fp32"),
          ("gaddy_fp16", "gaddy_fp16.pt", "fp16"), ("gaddy_dynint8", "gaddy_dynint8.pt", "int8"),
          ("gaddy_fp32_onnx", "gaddy_fp32.onnx", "onnx"), ("gaddy_int8dyn_onnx", "gaddy_int8dyn.onnx", "onnx")]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def warm_lm(path, chunk=64 << 20):
    """Read the 3.1 GB language model once, so it sits in memory (page cache) and decoding is not
    slowed by SD-card reads. About 1 minute after a reboot, about 1 second when already cached."""
    t0, n = time.perf_counter(), 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            n += len(b)
    dt = time.perf_counter() - t0
    print(f"language model warmed: {n / 1e9:.2f} GB in {dt:.1f} s ({n / 1e6 / max(dt, 1e-6):.0f} MB/s)", flush=True)


def make_runner(path, kind, fp16_native):
    """Return (function emg->logits, note)."""
    if kind == "onnx":
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        return (lambda x: s.run(None, {name: x[None]})[0]), "ONNX Runtime"
    if kind == "fp32":
        m = gm.load_fp32(path)
        return (lambda x: gm.logits(m, x)), "PyTorch FP32"
    if kind == "kd":
        m = gm.load_kd_student(path)
        return (lambda x: gm.logits(m, x)), "PyTorch FP32"
    if kind == "int8":
        m = gm.load_int8_saved(path, engine="qnnpack")
        return (lambda x: gm.logits(m, x)), "PyTorch dynamic INT8 (qnnpack)"
    if kind == "fp16":
        m = gm.load_fp16(path)
        if fp16_native:
            return (lambda x: gm.logits(m, x, half=True)), "PyTorch FP16 compute"
        m = m.float()      # 16-bit file, 32-bit compute
        return (lambda x: gm.logits(m, x)), "FP16 file, FP32 compute"
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="model names to run (default: all)")
    ap.add_argument("--fp16-native", action="store_true", help="compute Gaddy FP16 in half precision (very slow)")
    ap.add_argument("--beam", type=int, default=1500)
    args = ap.parse_args()

    torch.set_num_threads(THREADS)
    d = np.load(os.path.join(B, "test_set.npz"))
    refs = [str(r) for r in d["refs"]]
    utts = [d["emg"][d["offsets"][i]:d["offsets"][i + 1]].astype(np.float32) for i in range(len(refs))]
    vocab = json.load(open(os.path.join(B, "vocab.json")))
    fs = float(vocab["emg_rate_hz"])
    secs = [len(u) / fs for u in utts]
    print(f"{len(utts)} utterances, {sum(secs):.1f} s of EMG at {fs} Hz | torch {torch.__version__} "
          f"threads {torch.get_num_threads()} | onnxruntime {ort.__version__} | beam {args.beam}")
    print("governor:", sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
          "|", sh("vcgencmd measure_temp"), "|", sh("vcgencmd get_throttled"), flush=True)

    warm_lm(os.path.join(B, "decoder", "lm.bin"))
    t0 = time.perf_counter()
    dec = ctc_decoder(lexicon=os.path.join(B, "decoder", "gaddy_derived_lexicon.txt"),
                      tokens=list(vocab["chars"]) + ["_"], lm=os.path.join(B, "decoder", "lm.bin"),
                      blank_token="_", sil_token="|", nbest=1, lm_weight=4.0, word_score=-1.75,
                      sil_score=0.0, beam_size=args.beam)
    print(f"decoder loaded in {time.perf_counter() - t0:.1f} s, process memory {rss_mb():.0f} MB", flush=True)

    todo = [m for m in MODELS if not args.only or m[0] in args.only]
    summary_path = os.path.join(OUT, "pi_wer.csv")
    new_file = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as fsum:
        wsum = csv.writer(fsum)
        if new_file:
            wsum.writerow(["model", "how_run", "file_MB", "WER_pi", "WER_colab", "diff_words", "load_s",
                           "model_s_total", "decoder_s_total", "rtf_model", "rtf_total",
                           "mean_latency_s", "rss_after_load_MB", "rss_peak_MB", "temp_end", "throttled_end",
                           "beam", "date"])
        for name, fn, kind in todo:
            path = os.path.join(B, "models", fn)
            gc.collect()
            rss0 = rss_mb()
            t0 = time.perf_counter()
            run, how = make_runner(path, kind, args.fp16_native)
            load_s = time.perf_counter() - t0
            rss_load = rss_mb()
            run(utts[0])                                   # one warm-up run, not timed
            preds, rows, peak = [], [], rss_load
            for i, x in enumerate(utts):
                t1 = time.perf_counter()
                lg = run(x)
                t2 = time.perf_counter()
                logp = F.log_softmax(torch.as_tensor(lg, dtype=torch.float32), dim=-1)
                words = dec(logp)[0][0].words
                t3 = time.perf_counter()
                preds.append(" ".join(words).strip())
                rows.append([i, round(secs[i], 3), round((t2 - t1) * 1000, 1), round((t3 - t2) * 1000, 1)])
                peak = max(peak, rss_mb())
            w = jiwer.wer(refs, preds) * 100
            ms = sum(r[2] for r in rows) / 1000
            ds = sum(r[3] for r in rows) / 1000
            n_words = sum(len(r.split()) for r in refs)
            diff_words = round((w - COLAB[name]) / 100 * n_words)
            with open(os.path.join(OUT, f"pi_wer_{name}.csv"), "w", newline="") as fu:
                cw = csv.writer(fu)
                cw.writerow(["utt", "emg_s", "model_ms", "decoder_ms"])
                cw.writerows(rows)
            with open(os.path.join(OUT, f"pi_pred_{name}.txt"), "w") as fp:
                fp.write("\n".join(preds) + "\n")
            temp, thr = sh("vcgencmd measure_temp"), sh("vcgencmd get_throttled")
            wsum.writerow([name, how, round(os.path.getsize(path) / 1e6, 2), round(w, 2), COLAB[name], diff_words,
                           round(load_s, 2), round(ms, 1), round(ds, 1), round(ms / sum(secs), 3),
                           round((ms + ds) / sum(secs), 3), round((ms + ds) / len(utts), 2),
                           round(rss_load - rss0), round(peak), temp, thr, args.beam,
                           time.strftime("%Y-%m-%d %H:%M")])
            fsum.flush()
            print(f"{name:20s} WER {w:6.2f}% (Colab {COLAB[name]:6.2f}%, {diff_words:+d} words) | "
                  f"model {ms:6.1f} s + decoder {ds:6.1f} s for {sum(secs):.0f} s of EMG | "
                  f"RTF {ms / sum(secs):.3f} model, {(ms + ds) / sum(secs):.3f} total | "
                  f"+{rss_load - rss0:.0f} MB | {how} | {temp} {thr}", flush=True)
            del run
    print("\nsaved:", summary_path)


if __name__ == "__main__":
    main()
