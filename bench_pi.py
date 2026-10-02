"""Pi Step 5: speed details on the Raspberry Pi 5 (about 1 hour, runs by itself).

Every part runs in a fresh Python process, so the memory numbers are clean.
  Part A  network only, no language model in memory (one process per model):
          memory used by the model, network time with 4 threads and with 1 thread.
          The 4 candidate models run all 98 sentences and save their output for Part B;
          the other 6 models run 21 sentences (every 5th + the longest) for memory and speed.
  Part B  decoder only, on the saved network outputs (one process per setting):
          KenLM beam 1500 (thesis), 500, 150, 50, and greedy with no language model.
          WER, decoder time and memory for each.
  Part C  Gaddy FP16 really computed in 16 bit on 3 sentences (very slow on this CPU).

Run inside tmux:
    mkdir -p ~/emg/results/step5
    cd ~/emg/pi5_bundle && python bench_pi.py 2>&1 | tee ~/emg/results/step5/log.txt
One part again (for example only Part B):   python bench_pi.py --part B
Outputs in ~/emg/results/step5/:
    Part A  net_summary.csv, net_<model>.csv, logits_<model>.npz
    Part B  decoder_summary.csv, decoder_per_utt.csv, pred_<setting>_<model>.txt
    Part C  fp16_native.csv
"""
import os, sys, csv, json, time, argparse, subprocess, resource, gc
import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
# name, file, how it runs (the same as wer_pi.py); the 4 candidates first
MODELS = [("gaddy_fp32", "gaddy_fp32.pt", "fp32"), ("gaddy_dynint8", "gaddy_dynint8.pt", "int8"),
          ("tinymyo_int8", "tinymyo_int8.onnx", "onnx"), ("l4_fp32", "l4_fp32.onnx", "onnx"),
          ("gaddy_fp16", "gaddy_fp16.pt", "fp16"), ("gaddy_kd_student", "gaddy_kd_student.pt", "kd"),
          ("tinymyo_fp32", "tinymyo_fp32.onnx", "onnx"), ("l4_int8", "l4_int8.onnx", "onnx"),
          ("gaddy_fp32_onnx", "gaddy_fp32.onnx", "onnx"), ("gaddy_int8dyn_onnx", "gaddy_int8dyn.onnx", "onnx")]
CANDIDATES = ["gaddy_fp32", "gaddy_dynint8", "tinymyo_int8", "l4_fp32"]
SETTINGS = ["1500", "500", "150", "50", "greedy"]
STEP4_WER = {"gaddy_fp32": 21.51, "gaddy_dynint8": 21.63, "tinymyo_int8": 36.51, "l4_fp32": 40.22}

NET_HEADER = ["model", "how_run", "file_MB", "n_sentences", "emg_s", "net_s_4threads", "rtf_net_4threads",
              "mean_ms", "median_ms", "p95_ms", "max_ms", "subset_s_4threads", "subset_s_1thread",
              "slowdown_1thread", "rtf_subset_4threads", "load_s", "rss_base_MB", "ram_after_load_MB",
              "ram_peak_MB", "process_peak_MB", "temp_throttled", "date"]
DEC_HEADER = ["setting", "model", "WER", "WER_step4_beam1500", "dec_s_total", "rtf_dec", "mean_ms", "median_ms",
              "p95_ms", "max_ms", "lm_load_s", "rss_base_MB", "ram_decoder_loaded_MB", "ram_peak_MB",
              "process_peak_MB", "temp_throttled", "date"]
FP16_HEADER = ["utt", "emg_s", "fp16_compute_ms", "fp32_compute_ms", "fp16_over_fp32", "max_abs_diff",
               "same_symbol_pct"]


# ----------------------------------------------------------------- helpers
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def state():
    """CPU temperature and throttling flags (0x0 = never slowed down)."""
    return f"{sh('vcgencmd measure_temp')} {sh('vcgencmd get_throttled')}".strip()


def rss_mb():
    """Memory this process uses now (MB)."""
    return psutil.Process().memory_info().rss / 1e6


def peak_mb():
    """Highest memory this process has used so far (MB). Linux reports it in kB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6


def pct(v, p):
    return float(np.percentile(v, p))


def trim():
    """Free unused memory and give it back to the system, so the RAM numbers show live memory only."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def greedy_text(lg, chars):
    """Greedy CTC decoding without any language model: best symbol per frame, merge repeats, drop blanks."""
    x = np.asarray(lg)
    ids = x.reshape(-1, x.shape[-1]).argmax(-1)
    keep = np.ones(len(ids), dtype=bool)
    keep[1:] = ids[1:] != ids[:-1]                   # merge repeated symbols
    ids = ids[keep]
    ids = ids[ids != len(chars)]                     # drop the CTC blank (the last token)
    return " ".join("".join(chars[k] for k in ids).replace("|", " ").split())


def load_test(bundle):
    d = np.load(os.path.join(bundle, "test_set.npz"))
    refs = [str(r) for r in d["refs"]]
    off = d["offsets"]
    utts = [d["emg"][off[i]:off[i + 1]].astype(np.float32) for i in range(len(refs))]
    vocab = json.load(open(os.path.join(bundle, "vocab.json")))
    fs = float(vocab["emg_rate_hz"])
    secs = [len(u) / fs for u in utts]
    return refs, utts, secs, vocab


def subset(secs):
    """Every 5th sentence plus the longest one (21 sentences of the 98)."""
    return sorted(set(range(0, len(secs), 5)) | {int(np.argmax(secs))})


def append_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def warm_lm(path, chunk=64 << 20):
    """Read the 3.1 GB language model once so it sits in memory (fast when it is already cached)."""
    t0, n = time.perf_counter(), 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            n += len(b)
    return time.perf_counter() - t0


# ----------------------------------------------------------------- Part A: network only
def net_worker(a, name):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import onnxruntime as ort
    sys.path.insert(0, a.bundle)
    import gaddy_model as gm

    refs, utts, secs, vocab = load_test(a.bundle)
    fn, kind = {m[0]: (m[1], m[2]) for m in MODELS}[name]
    path = os.path.join(a.bundle, "models", fn)
    full = name in CANDIDATES
    sub = subset(secs)
    idx = list(range(len(utts))) if full else sub
    torch.set_num_threads(4)
    trim()
    rss0 = rss_mb()

    def make(threads):
        if kind == "onnx":
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
            nm = s.get_inputs()[0].name
            return (lambda x: s.run(None, {nm: x[None]})[0]), "ONNX Runtime"
        if kind == "fp32":
            m, how = gm.load_fp32(path), "PyTorch FP32"
        elif kind == "kd":
            m, how = gm.load_kd_student(path), "PyTorch FP32"
        elif kind == "int8":
            m, how = gm.load_int8_saved(path, engine="qnnpack"), "PyTorch dynamic INT8 (qnnpack)"
        elif kind == "fp16":
            m, how = gm.load_fp16(path).float(), "FP16 file, FP32 compute"
        else:
            raise ValueError(kind)
        return (lambda x: gm.logits(m, x)), how

    t0 = time.perf_counter()
    run, how = make(4)
    load_s = time.perf_counter() - t0
    trim()
    rss_load = rss_mb()
    run(utts[0])                                     # warm-up, not timed (as in Step 4)
    t4, out = {}, {}
    for i in idx:
        t1 = time.perf_counter()
        lg = run(utts[i])
        t4[i] = (time.perf_counter() - t1) * 1000
        if full:
            out[f"u{i}"] = np.asarray(lg, dtype=np.float32)
    peak4 = peak_mb()

    t1s = {}
    if full:                                         # the same 21 sentences again with 1 thread
        if kind == "onnx":
            del run
            gc.collect()
            run, _ = make(1)
        else:
            torch.set_num_threads(1)
        run(utts[0])
        for i in sub:
            t1 = time.perf_counter()
            run(utts[i])
            t1s[i] = (time.perf_counter() - t1) * 1000

    with open(os.path.join(a.out, f"net_{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt", "emg_s", "net_ms_4threads", "net_ms_1thread"])
        for i in idx:
            w.writerow([i, round(secs[i], 3), round(t4[i], 1), round(t1s[i], 1) if i in t1s else ""])
    if full:
        np.savez(os.path.join(a.out, f"logits_{name}.npz"), **out)

    v4 = [t4[i] for i in idx]
    emg = sum(secs[i] for i in idx)
    sub4 = sum(t4[i] for i in sub)
    emg_sub = sum(secs[i] for i in sub)
    sub1 = sum(t1s.values())
    append_row(os.path.join(a.out, "net_summary.csv"), NET_HEADER,
               [name, how, round(os.path.getsize(path) / 1e6, 2), len(idx), round(emg, 1),
                round(sum(v4) / 1000, 1), round(sum(v4) / 1000 / emg, 3), round(np.mean(v4)),
                round(np.median(v4)), round(pct(v4, 95)), round(max(v4)), round(sub4 / 1000, 1),
                round(sub1 / 1000, 1) if t1s else "", round(sub1 / sub4, 2) if t1s else "",
                round(sub4 / 1000 / emg_sub, 3), round(load_s, 2), round(rss0), round(rss_load - rss0),
                round(peak4 - rss0), round(peak4), state(), time.strftime("%Y-%m-%d %H:%M")])
    msg = (f"  {name:19s} {how:30s} RAM +{rss_load - rss0:5.0f} MB after load, peak +{peak4 - rss0:5.0f} MB"
           f" | 4 threads {sum(v4) / 1000:6.1f} s for {emg:5.1f} s of EMG ({len(idx)} sentences, RTF {sum(v4) / 1000 / emg:.3f})")
    if t1s:
        msg += f" | 1 thread {sub1 / 1000:5.1f} s vs 4 threads {sub4 / 1000:5.1f} s (x{sub1 / sub4:.2f})"
    print(msg + f" | {state()}", flush=True)


# ----------------------------------------------------------------- Part B: decoder only
def dec_worker(a, setting):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import torch.nn.functional as F
    import jiwer
    from torchaudio.models.decoder import ctc_decoder

    torch.set_num_threads(4)
    refs, utts, secs, vocab = load_test(a.bundle)
    chars = vocab["chars"]
    trim()
    rss0 = rss_mb()
    load_s, rss_dec, label = 0.0, rss0, "greedy, no LM"
    if setting == "greedy":
        def decode(lg):
            return greedy_text(lg, chars)
    else:
        label = f"KenLM beam {setting}"
        lm = os.path.join(a.bundle, "decoder", "lm.bin")
        warm_s = warm_lm(lm)
        t0 = time.perf_counter()
        dec = ctc_decoder(lexicon=os.path.join(a.bundle, "decoder", "gaddy_derived_lexicon.txt"),
                          tokens=list(chars) + ["_"], lm=lm, blank_token="_", sil_token="|", nbest=1,
                          lm_weight=4.0, word_score=-1.75, sil_score=0.0, beam_size=int(setting))
        load_s = time.perf_counter() - t0
        rss_dec = rss_mb()
        print(f"  {label}: language model read in {warm_s:.1f} s, decoder loaded in {load_s:.1f} s, "
              f"RAM +{rss_dec - rss0:.0f} MB", flush=True)

        def decode(lg):
            logp = F.log_softmax(torch.as_tensor(lg, dtype=torch.float32), dim=-1)
            return " ".join(dec(logp)[0][0].words).strip()

    short = int(np.argmin(secs))
    for name in CANDIDATES:
        p = os.path.join(a.out, f"logits_{name}.npz")
        if not os.path.exists(p):
            print(f"  {label:15s} {name:14s} skipped: no saved network output from Part A", flush=True)
            continue
        z = np.load(p)
        lgs = [z[f"u{i}"] for i in range(len(utts))]
        decode(lgs[short])                           # warm-up, not timed
        preds, ms = [], []
        for lg in lgs:
            t1 = time.perf_counter()
            preds.append(decode(lg))
            ms.append((time.perf_counter() - t1) * 1000)
        w = jiwer.wer(refs, preds) * 100
        peak = peak_mb()
        with open(os.path.join(a.out, f"pred_{setting}_{name}.txt"), "w") as f:
            f.write("\n".join(preds) + "\n")
        per = os.path.join(a.out, "decoder_per_utt.csv")
        new = not os.path.exists(per)
        with open(per, "a", newline="") as f:
            cw = csv.writer(f)
            if new:
                cw.writerow(["setting", "model", "utt", "emg_s", "decoder_ms"])
            cw.writerows([[setting, name, i, round(secs[i], 3), round(ms[i], 1)] for i in range(len(ms))])
        append_row(os.path.join(a.out, "decoder_summary.csv"), DEC_HEADER,
                   [setting, name, round(w, 2), STEP4_WER[name], round(sum(ms) / 1000, 1),
                    round(sum(ms) / 1000 / sum(secs), 3), round(np.mean(ms)), round(np.median(ms)),
                    round(pct(ms, 95)), round(max(ms)), round(load_s, 2), round(rss0), round(rss_dec - rss0),
                    round(peak - rss0), round(peak), state(), time.strftime("%Y-%m-%d %H:%M")])
        print(f"  {label:15s} {name:14s} WER {w:6.2f}% (Step 4 beam 1500: {STEP4_WER[name]:.2f}%) | decoder "
              f"{sum(ms) / 1000:6.1f} s total, mean {np.mean(ms):5.0f} ms, median {np.median(ms):5.0f} ms, "
              f"p95 {pct(ms, 95):5.0f} ms | RAM peak +{peak - rss0:.0f} MB | {state()}", flush=True)


# ----------------------------------------------------------------- Part C: true FP16 compute
def fp16_worker(a):
    import warnings
    warnings.filterwarnings("ignore")
    import copy
    import torch
    sys.path.insert(0, a.bundle)
    import gaddy_model as gm

    torch.set_num_threads(4)
    refs, utts, secs, vocab = load_test(a.bundle)
    med = float(np.median(secs))
    pick = sorted(sorted(range(len(secs)), key=lambda i: abs(secs[i] - med))[:3])
    m16 = gm.load_fp16(os.path.join(a.bundle, "models", "gaddy_fp16.pt"))
    m32 = copy.deepcopy(m16).float()
    short = int(np.argmin(secs))
    gm.logits(m16, utts[short], half=True)           # warm-up, not timed
    gm.logits(m32, utts[short])
    for i in pick:
        t1 = time.perf_counter()
        y16 = gm.logits(m16, utts[i], half=True)
        t16 = (time.perf_counter() - t1) * 1000
        t1 = time.perf_counter()
        y32 = gm.logits(m32, utts[i])
        t32 = (time.perf_counter() - t1) * 1000
        diff = (y16 - y32).abs().max().item()
        same = (y16.argmax(-1) == y32.argmax(-1)).float().mean().item() * 100
        append_row(os.path.join(a.out, "fp16_native.csv"), FP16_HEADER,
                   [i, round(secs[i], 3), round(t16, 1), round(t32, 1), round(t16 / t32, 1), round(diff, 4),
                    round(same, 1)])
        print(f"  sentence {i:2d} ({secs[i]:.2f} s): FP16 compute {t16 / 1000:6.1f} s, FP32 compute "
              f"{t32 / 1000:5.2f} s (FP16 takes {t16 / t32:.1f} times as long) | output max difference {diff:.4f}, "
              f"same symbol in {same:.1f}% of frames", flush=True)


# ----------------------------------------------------------------- driver
def child(a, *extra):
    cmd = [sys.executable, os.path.abspath(__file__), "--bundle", a.bundle, "--out", a.out, *extra]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    if r.returncode != 0:
        print(f"  !! {' '.join(extra)} stopped with an error (code {r.returncode}), the rest continues", flush=True)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def show_a(a):
    rows = read_csv(os.path.join(a.out, "net_summary.csv"))
    if not rows:
        return
    print("\n== Part A: network only, no language model (RAM = extra memory for this model) ==")
    print(f"{'model':19s} {'runs in':30s} {'file MB':>8s} {'RAM load':>9s} {'RAM peak':>9s}"
          f" {'RTF (21 sentences, 4 threads)':>30s} {'1 thread':>9s}")
    for r in rows:
        slow = f"x{r['slowdown_1thread']}" if r["slowdown_1thread"] else ""
        print(f"{r['model']:19s} {r['how_run']:30s} {float(r['file_MB']):8.1f} {float(r['ram_after_load_MB']):9.0f}"
              f" {float(r['ram_peak_MB']):9.0f} {float(r['rtf_subset_4threads']):30.3f} {slow:>9s}")


def show_b(a):
    rows = read_csv(os.path.join(a.out, "decoder_summary.csv"))
    if not rows:
        return
    print("\n== Part B: decoder only (seconds per sentence; RAM = extra memory incl. language model) ==")
    print(f"{'setting':8s} {'model':14s} {'WER %':>7s} {'mean s':>7s} {'median s':>9s} {'p95 s':>7s} {'RAM peak':>9s}")
    for r in rows:
        print(f"{r['setting']:8s} {r['model']:14s} {float(r['WER']):7.2f} {float(r['mean_ms']) / 1000:7.2f}"
              f" {float(r['median_ms']) / 1000:9.2f} {float(r['p95_ms']) / 1000:7.2f} {float(r['ram_peak_MB']):9.0f}")


def show_c(a):
    rows = read_csv(os.path.join(a.out, "fp16_native.csv"))
    if not rows:
        return
    print("\n== Part C: Gaddy FP16 computed in 16 bit vs 32 bit ==")
    for r in rows:
        print(f"sentence {r['utt']:>2s} ({float(r['emg_s']):.2f} s): FP16 {float(r['fp16_compute_ms']) / 1000:.1f} s, "
              f"FP32 {float(r['fp32_compute_ms']) / 1000:.2f} s, FP16 takes {r['fp16_over_fp32']} times as long, "
              f"same symbol {r['same_symbol_pct']}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "A", "B", "C"])
    ap.add_argument("--bundle", default=HERE)
    ap.add_argument("--out", default=os.path.expanduser("~/emg/results/step5"))
    ap.add_argument("--net", help=argparse.SUPPRESS)
    ap.add_argument("--dec", help=argparse.SUPPRESS)
    ap.add_argument("--fp16", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.bundle, a.out = os.path.abspath(a.bundle), os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    if a.net:
        return net_worker(a, a.net)
    if a.dec:
        return dec_worker(a, a.dec)
    if a.fp16:
        return fp16_worker(a)

    t_start = time.time()
    gov = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    print(f"Step 5 started {time.strftime('%Y-%m-%d %H:%M')} | governor: {gov} | {state()} | "
          f"free memory {psutil.virtual_memory().available / 1e9:.1f} GB", flush=True)
    if gov and gov != "performance":
        print("  note: the governor is not 'performance', so the times are not comparable with Step 4", flush=True)
    print("This takes about 1 hour. You can detach from tmux now (Ctrl+b, then d).", flush=True)

    if a.part in ("all", "A"):
        print(f"\n== Part A: network only, one fresh process per model ({time.strftime('%H:%M')}) ==", flush=True)
        for f in ["net_summary.csv"]:
            if os.path.exists(os.path.join(a.out, f)):
                os.remove(os.path.join(a.out, f))
        for name, fn, kind in MODELS:
            if not os.path.exists(os.path.join(a.bundle, "models", fn)):
                print(f"  {name}: {fn} not found in {a.bundle}/models, skipped", flush=True)
                continue
            child(a, "--net", name)
    if a.part in ("all", "B"):
        print(f"\n== Part B: decoder only, one fresh process per setting ({time.strftime('%H:%M')}) ==", flush=True)
        for f in ["decoder_summary.csv", "decoder_per_utt.csv"]:
            if os.path.exists(os.path.join(a.out, f)):
                os.remove(os.path.join(a.out, f))
        for s in SETTINGS:
            child(a, "--dec", s)
    if a.part in ("all", "C"):
        print(f"\n== Part C: Gaddy FP16 in true 16 bit ({time.strftime('%H:%M')}) ==", flush=True)
        if os.path.exists(os.path.join(a.out, "fp16_native.csv")):
            os.remove(os.path.join(a.out, "fp16_native.csv"))
        child(a, "--fp16")

    show_a(a)
    show_b(a)
    show_c(a)
    print(f"\nStep 5 finished in {(time.time() - t_start) / 60:.0f} min | {state()} | results in {a.out}", flush=True)


if __name__ == "__main__":
    main()
