"""Pi Step 6: power, energy per sentence and temperature on the Raspberry Pi 5 (about 50 minutes).

Power comes from the Pi 5's own power chip (PMIC), read with `vcgencmd pmic_read_adc`: the script adds
current x voltage for every rail that reports both. This covers the main rails on the board (CPU core,
memory, input/output). It leaves out the USB ports, the fan and the power chip's own losses, so the
real input power is somewhat higher, but it is good for comparing settings with each other.

A separate sampler process reads power, temperature, CPU clock and fan speed every 0.5 s.
Phases (each recognition run goes through all 98 test sentences, one after another, like a device):
    idle_start     2 min, nothing running
    7 runs         model + decoder beam + threads (list RUNS below), 30 s rest between runs
    sustained      10 min non-stop with the best candidate (temperature, clock, throttling)
    idle_end       1 min

Run inside tmux:
    mkdir -p ~/emg/results/step6
    cd ~/emg/pi5_bundle && python power_pi.py 2>&1 | tee ~/emg/results/step6/log.txt
Outputs in ~/emg/results/step6/:
    power_summary.csv   one row per phase: power, energy per sentence, temperature, clock
    runs.csv            one row per run: time per sentence, real-time factor, WER
    run_<label>.csv     one row per sentence: network ms, decoder ms
    sustained_minutes.csv, power_samples.csv (all samples), markers.csv (phase start/end)
"""
import os, sys, csv, json, time, argparse, subprocess, re, glob, gc
import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {"gaddy_dynint8": ("gaddy_dynint8.pt", "int8"), "gaddy_fp32": ("gaddy_fp32.pt", "fp32"),
               "tinymyo_int8": ("tinymyo_int8.onnx", "onnx"), "l4_fp32": ("l4_fp32.onnx", "onnx")}
# label, model, decoder beam, threads
RUNS = [("gaddy_int8_b150_4t", "gaddy_dynint8", 150, 4),
        ("gaddy_int8_b150_1t", "gaddy_dynint8", 150, 1),
        ("gaddy_int8_b1500_4t", "gaddy_dynint8", 1500, 4),
        ("gaddy_fp32_b150_4t", "gaddy_fp32", 150, 4),
        ("tinymyo_int8_b500_4t", "tinymyo_int8", 500, 4),
        ("tinymyo_int8_b500_1t", "tinymyo_int8", 500, 1),
        ("l4_fp32_b150_4t", "l4_fp32", 150, 4)]
SUSTAINED = ("sustained_gaddy_int8_b150_4t", "gaddy_dynint8", 150, 4)
STEP5_WER = {("gaddy_dynint8", 150): 23.27, ("gaddy_dynint8", 1500): 21.63, ("gaddy_fp32", 150): 22.78,
             ("tinymyo_int8", 500): 36.88, ("l4_fp32", 150): 44.23}
PMIC_RX = re.compile(r"(\w+)_([AV])\s+(?:current|volt)\(\d+\)=\s*([-+0-9.eE]+)")


# ----------------------------------------------------------------- helpers
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def state():
    return f"{sh('vcgencmd measure_temp')} {sh('vcgencmd get_throttled')}".strip()


def read_pmic():
    """Board power from the PMIC: sum of current x voltage over the rails that report both."""
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=5).stdout
    cur, vol = {}, {}
    for name, kind, val in PMIC_RX.findall(out):
        (cur if kind == "A" else vol)[name] = float(val)
    rails = [r for r in cur if r in vol]
    total = sum(cur[r] * vol[r] for r in rails)
    core = cur.get("VDD_CORE", 0.0) * vol.get("VDD_CORE", 0.0)
    return total, core, rails, out


def read_num(path, scale=1.0):
    try:
        with open(path) as f:
            return float(f.read().strip()) / scale
    except Exception:
        return float("nan")


def load_test(bundle):
    d = np.load(os.path.join(bundle, "test_set.npz"))
    refs = [str(r) for r in d["refs"]]
    off = d["offsets"]
    utts = [d["emg"][off[i]:off[i + 1]].astype(np.float32) for i in range(len(refs))]
    vocab = json.load(open(os.path.join(bundle, "vocab.json")))
    secs = [len(u) / float(vocab["emg_rate_hz"]) for u in utts]
    return refs, utts, secs, vocab


def warm_lm(path, chunk=64 << 20):
    with open(path, "rb") as f:
        while f.read(chunk):
            pass


def append_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------- sampler process
def sampler(path):
    fan = sorted(glob.glob("/sys/devices/platform/cooling_fan/hwmon/hwmon*/fan1_input"))
    with open(path, "w") as f:
        f.write("t,board_W,core_W,temp_C,clock_MHz,fan_rpm\n")
        f.flush()
        while True:
            t = time.time()
            try:
                total, core, _, _ = read_pmic()
            except Exception:
                total, core = float("nan"), float("nan")
            temp = read_num("/sys/class/thermal/thermal_zone0/temp", 1000)
            clock = read_num("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000)
            rpm = read_num(fan[0]) if fan else float("nan")
            f.write(f"{t:.3f},{total:.4f},{core:.4f},{temp:.1f},{clock:.0f},{rpm:.0f}\n")
            f.flush()
            time.sleep(max(0.0, 0.5 - (time.time() - t)))


# ----------------------------------------------------------------- one recognition run
def run_worker(a, label):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import torch.nn.functional as F
    import jiwer
    import onnxruntime as ort
    from torchaudio.models.decoder import ctc_decoder
    sys.path.insert(0, a.bundle)
    import gaddy_model as gm

    _, name, beam, threads = {r[0]: r for r in RUNS + [SUSTAINED]}[label]
    torch.set_num_threads(threads)
    refs, utts, secs, vocab = load_test(a.bundle)
    lm = os.path.join(a.bundle, "decoder", "lm.bin")
    warm_lm(lm)
    dec = ctc_decoder(lexicon=os.path.join(a.bundle, "decoder", "gaddy_derived_lexicon.txt"),
                      tokens=list(vocab["chars"]) + ["_"], lm=lm, blank_token="_", sil_token="|", nbest=1,
                      lm_weight=4.0, word_score=-1.75, sil_score=0.0, beam_size=beam)
    fn, kind = MODEL_FILES[name]
    path = os.path.join(a.bundle, "models", fn)
    if kind == "onnx":
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        nm = sess.get_inputs()[0].name
        run, how = (lambda x: sess.run(None, {nm: x[None]})[0]), "ONNX Runtime"
    else:
        m = gm.load_fp32(path) if kind == "fp32" else gm.load_int8_saved(path, engine="qnnpack")
        how = "PyTorch FP32" if kind == "fp32" else "PyTorch dynamic INT8 (qnnpack)"
        run = lambda x: gm.logits(m, x)

    def decode(lg):
        logp = F.log_softmax(torch.as_tensor(lg, dtype=torch.float32), dim=-1)
        return " ".join(dec(logp)[0][0].words).strip()

    decode(run(utts[0]))                             # warm-up, not measured
    gc.collect()
    rows, preds, rep = [], [], 0
    t_start = time.time()
    stop_at = t_start + a.loop if a.loop else None
    while True:
        for i, x in enumerate(utts):
            t1 = time.perf_counter()
            lg = run(x)
            t2 = time.perf_counter()
            text = decode(lg)
            t3 = time.perf_counter()
            rows.append([rep, i, round(secs[i], 3), round((t2 - t1) * 1000, 1), round((t3 - t2) * 1000, 1),
                         round(time.time(), 3)])
            if rep == 0:
                preds.append(text)
            if stop_at and time.time() >= stop_at:
                break
        rep += 1
        if not stop_at or time.time() >= stop_at:
            break
    t_end = time.time()

    append_row(os.path.join(a.out, "markers.csv"), ["phase", "start", "end"], [label, round(t_start, 3), round(t_end, 3)])
    with open(os.path.join(a.out, f"run_{label}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pass", "utt", "emg_s", "net_ms", "dec_ms", "t_end"])
        w.writerows(rows)
    n = len(rows)
    emg = sum(r[2] for r in rows)
    busy = sum(r[3] + r[4] for r in rows) / 1000
    wer = round(jiwer.wer(refs, preds) * 100, 2) if len(preds) == len(refs) else ""
    exp = STEP5_WER.get((name, beam), "")
    append_row(os.path.join(a.out, "runs.csv"),
               ["label", "model", "how_run", "beam", "threads", "n_sentences", "emg_s", "run_s", "mean_s_per_sentence",
                "rtf", "WER", "WER_step5", "state_end"],
               [label, name, how, beam, threads, n, round(emg, 1), round(t_end - t_start, 1), round(busy / n, 3),
                round(busy / emg, 3), wer, exp, state()])
    msg = (f"  {label:30s} {n:3d} sentences in {t_end - t_start:6.1f} s | {busy / n:5.2f} s per sentence, "
           f"RTF {busy / emg:.3f}")
    if wer != "":
        msg += f" | WER {wer:.2f}% (Step 5: {exp})"
    print(msg + f" | {state()}", flush=True)


# ----------------------------------------------------------------- summary
def fnum(f, x):
    """f over the finite values of x, or nan when there are none."""
    x = x[np.isfinite(x)]
    return float(f(x)) if x.size else float("nan")


def cell(x, nd=0):
    """Value for the CSV: rounded number, or empty when missing."""
    if x is None or not np.isfinite(x):
        return ""
    return round(x, nd) if nd else int(round(x))


def txt(x, w, nd):
    """Value for the printed table: fixed width, or blanks when missing."""
    if x is None or not np.isfinite(x):
        return " " * w
    return f"{x:{w}.{nd}f}"


def summarize(a):
    S = read_csv(os.path.join(a.out, "power_samples.csv"))
    M = read_csv(os.path.join(a.out, "markers.csv"))
    R = {r["label"]: r for r in read_csv(os.path.join(a.out, "runs.csv"))}
    if not S or not M:
        print("no samples or markers, nothing to summarise", flush=True)
        return
    t = np.array([float(x["t"]) for x in S])
    col = {k: np.array([float(x[k]) for x in S]) for k in ["board_W", "core_W", "temp_C", "clock_MHz", "fan_rpm"]}
    win = {m["phase"]: (float(m["start"]), float(m["end"])) for m in M}

    def part(s, e):
        k = (t >= s) & (t <= e)
        return int(np.isfinite(col["board_W"][k]).sum()), {c: col[c][k] for c in col}

    idle_W = fnum(np.mean, part(*win["idle_start"])[1]["board_W"]) if "idle_start" in win else float("nan")
    out = os.path.join(a.out, "power_summary.csv")
    if os.path.exists(out):
        os.remove(out)
    print("\n== Step 6 summary (board power from the PMIC; energy per sentence = mean power x run time / sentences) ==")
    print(f"{'phase':30s} {'W mean':>7s} {'W max':>6s} {'core W':>7s} {'s/sent':>7s} {'J/sent':>7s} "
          f"{'J above idle':>12s} {'temp max':>8s} {'clock MHz':>9s} {'fan rpm':>8s}")
    for m in M:
        ph = m["phase"]
        s, e = win[ph]
        n_samp, v = part(s, e)
        if n_samp == 0:
            continue
        mean_W, dur = fnum(np.mean, v["board_W"]), e - s
        r = R.get(ph)
        n = int(r["n_sentences"]) if r else 0
        per = mean_W * dur / n if n else float("nan")
        above = (mean_W - idle_W) * dur / n if n else float("nan")
        per_emg = mean_W * dur / float(r["emg_s"]) if r else float("nan")
        sps = float(r["mean_s_per_sentence"]) if r else float("nan")
        row = [ph, cell(mean_W, 3), cell(fnum(np.max, v["board_W"]), 3), cell(fnum(np.mean, v["core_W"]), 3),
               cell(dur, 1), n_samp, n, cell(per, 2), cell(above, 2), cell(per_emg, 3),
               cell(fnum(np.mean, v["temp_C"]), 1), cell(fnum(np.max, v["temp_C"]), 1),
               cell(fnum(np.mean, v["clock_MHz"])), cell(fnum(np.mean, v["fan_rpm"])), cell(sps, 3),
               r["WER"] if r else "", cell(idle_W, 3)]
        append_row(out, ["phase", "mean_W", "max_W", "core_W", "duration_s", "samples", "n_sentences",
                         "J_per_sentence", "J_above_idle_per_sentence", "J_per_emg_second", "temp_mean_C",
                         "temp_max_C", "clock_mean_MHz", "fan_mean_rpm", "s_per_sentence", "WER", "idle_W"], row)
        print(f"{ph:30s} {txt(mean_W, 7, 2)} {txt(fnum(np.max, v['board_W']), 6, 2)} "
              f"{txt(fnum(np.mean, v['core_W']), 7, 2)} {txt(sps, 7, 2)} {txt(per, 7, 2)} {txt(above, 12, 2)} "
              f"{txt(fnum(np.max, v['temp_C']), 8, 1)} {txt(fnum(np.mean, v['clock_MHz']), 9, 0)} "
              f"{txt(fnum(np.mean, v['fan_rpm']), 8, 0)}", flush=True)

    lab = SUSTAINED[0]
    if lab in win:
        s, e = win[lab]
        per = read_csv(os.path.join(a.out, f"run_{lab}.csv"))
        tend = np.array([float(p["t_end"]) for p in per])
        busy = np.array([(float(p["net_ms"]) + float(p["dec_ms"])) / 1000 for p in per])
        emg = np.array([float(p["emg_s"]) for p in per])
        path = os.path.join(a.out, "sustained_minutes.csv")
        if os.path.exists(path):
            os.remove(path)
        print(f"\n== Sustained run, minute by minute ({lab}) ==")
        print(f"{'minute':>6s} {'W mean':>7s} {'temp max':>8s} {'clock MHz':>9s} {'fan rpm':>8s} {'sentences':>9s} {'RTF':>6s}")
        for i in range(int(np.ceil((e - s) / 60))):
            a0, a1 = s + 60 * i, min(s + 60 * (i + 1), e)
            n_samp, v = part(a0, a1)
            j = (tend > a0) & (tend <= a1)
            if n_samp == 0:
                continue
            rtf = float(busy[j].sum() / emg[j].sum()) if j.any() else float("nan")
            vals = [fnum(np.mean, v["board_W"]), fnum(np.max, v["temp_C"]), fnum(np.mean, v["clock_MHz"]),
                    fnum(np.mean, v["fan_rpm"])]
            append_row(path, ["minute", "mean_W", "temp_max_C", "clock_mean_MHz", "fan_mean_rpm", "sentences", "rtf"],
                       [i + 1, cell(vals[0], 3), cell(vals[1], 1), cell(vals[2]), cell(vals[3]), int(j.sum()),
                        cell(rtf, 3)])
            print(f"{i + 1:6d} {txt(vals[0], 7, 2)} {txt(vals[1], 8, 1)} {txt(vals[2], 9, 0)} {txt(vals[3], 8, 0)} "
                  f"{int(j.sum()):9d} {txt(rtf, 6, 3)}", flush=True)


# ----------------------------------------------------------------- driver
def child(a, *extra):
    cmd = [sys.executable, os.path.abspath(__file__), "--bundle", a.bundle, "--out", a.out, *extra]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    if r.returncode != 0:
        print(f"  !! {' '.join(extra)} stopped with an error (code {r.returncode}), the rest continues", flush=True)


def rest(a, label, seconds):
    t0 = time.time()
    time.sleep(seconds)
    if label:
        append_row(os.path.join(a.out, "markers.csv"), ["phase", "start", "end"], [label, round(t0, 3), round(time.time(), 3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=HERE)
    ap.add_argument("--out", default=os.path.expanduser("~/emg/results/step6"))
    ap.add_argument("--idle", type=float, default=120)
    ap.add_argument("--gap", type=float, default=30)
    ap.add_argument("--sustain", type=float, default=600)
    ap.add_argument("--idle-end", type=float, default=60)
    ap.add_argument("--summary-only", action="store_true", help="only print the summary of an earlier run")
    ap.add_argument("--run", help=argparse.SUPPRESS)
    ap.add_argument("--loop", type=float, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--sampler", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.bundle, a.out = os.path.abspath(a.bundle), os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    if a.sampler:
        return sampler(a.sampler)
    if a.run:
        return run_worker(a, a.run)
    if a.summary_only:
        return summarize(a)

    t_start = time.time()
    gov = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    print(f"Step 6 started {time.strftime('%Y-%m-%d %H:%M')} | governor: {gov} | {state()} | "
          f"free memory {psutil.virtual_memory().available / 1e9:.1f} GB", flush=True)
    if gov and gov != "ondemand":
        print("  note: the governor is not the default 'ondemand'. To set it for this run:\n"
              "  echo ondemand | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor", flush=True)
    try:
        total, core, rails, raw = read_pmic()
    except Exception as e:
        total, core, rails, raw = 0.0, 0.0, [], f"{type(e).__name__}: {e}"
    if not rails:
        print("PMIC check FAILED: could not read the power rails. Paste this output to Claude:\n" + raw, flush=True)
        return
    print(f"PMIC check: {len(rails)} rails, board power now {total:.2f} W (CPU core {core:.2f} W)", flush=True)
    for name, (fn, _) in MODEL_FILES.items():
        if not os.path.exists(os.path.join(a.bundle, "models", fn)):
            print(f"missing model file {fn} in {a.bundle}/models, stopping", flush=True)
            return
    for f in ["power_samples.csv", "markers.csv", "runs.csv", "power_summary.csv", "sustained_minutes.csv"]:
        if os.path.exists(os.path.join(a.out, f)):
            os.remove(os.path.join(a.out, f))
    total_min = (a.idle + len(RUNS) * a.gap + a.sustain + a.gap + a.idle_end) / 60 + 30
    print(f"This takes about {total_min:.0f} minutes. You can detach from tmux now (Ctrl+b, then d).", flush=True)

    samp = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--sampler",
                             os.path.join(a.out, "power_samples.csv")])
    try:
        time.sleep(3)
        print(f"\n== idle, {a.idle:.0f} s ({time.strftime('%H:%M')}) ==", flush=True)
        rest(a, "idle_start", a.idle)
        print(f"\n== recognition runs, all test sentences each ({time.strftime('%H:%M')}) ==", flush=True)
        for label, _, _, _ in RUNS:
            child(a, "--run", label)
            rest(a, None, a.gap)
        print(f"\n== sustained run, {a.sustain:.0f} s non-stop ({time.strftime('%H:%M')}) ==", flush=True)
        child(a, "--run", SUSTAINED[0], "--loop", str(a.sustain))
        rest(a, None, a.gap)
        print(f"\n== idle again, {a.idle_end:.0f} s ({time.strftime('%H:%M')}) ==", flush=True)
        rest(a, "idle_end", a.idle_end)
    finally:
        samp.terminate()
        samp.wait()
    summarize(a)
    print(f"\nStep 6 finished in {(time.time() - t_start) / 60:.0f} min | {state()} | results in {a.out}", flush=True)


if __name__ == "__main__":
    main()
