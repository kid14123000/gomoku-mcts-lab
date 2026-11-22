"""最小对比 v2: 纯两字组测「没」的声调
A: 没落  (若按词读 → mòluò, 四声=降)
B: 没有  (按词组 → méiyǒu, 二声=升)
C/D: 整句新旧对比, 在「没」附近逐 50ms 看 F0
"""
import json, os, sys
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = os.path.expanduser("~/tts/index-tts")
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.expanduser("~/tts"))
from narrate import clean, trim_silence, PROFILES, SR
import soundfile as sf
from indextts.infer_v2_5 import IndexTTS2

OUT = "/Users/kid/tts/f0probe"
os.makedirs(OUT, exist_ok=True)
prof = PROFILES["science"]

CASES = [
    ("A_moluo",    "没落"),
    ("B_meiyou",   "没有"),
    ("E_meiyouluo", "没有落"),
    ("C_full_old", "第一次进去，一个点都还没落。算完先验就退出来。"),
    ("D_full_new", "第一次进去，一个点都还没有落。算完先验就退出来。"),
]

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                device="mps", use_qwen_emo=False)


def f0_track(x, sr, fmin=90, fmax=400, win=0.030, hop=0.005, thr=0.35):
    """自相关 + 取第一个过阈值的峰(避免倍频误判)"""
    w = int(win * sr); h = int(hop * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    ts, fs = [], []
    for i in range(0, len(x) - w, h):
        fr = x[i:i + w] * np.hanning(w)
        if np.sqrt(np.mean(fr ** 2)) < 1e-3:
            continue
        ac = np.correlate(fr, fr, "full")[w - 1:]
        ac /= (ac[0] + 1e-12)
        seg = ac[lo:hi]
        if len(seg) < 3:
            continue
        peaks = [j for j in range(1, len(seg) - 1)
                 if seg[j] > seg[j - 1] and seg[j] >= seg[j + 1] and seg[j] > thr]
        if not peaks:
            continue
        ts.append(i / sr + win / 2); fs.append(sr / (peaks[0] + lo))
    return np.array(ts), np.array(fs)


res = {}
for tag, text in CASES:
    t = clean(text, "ZH")
    tmp = f"{OUT}/_t.wav"
    tts.infer(spk_audio_prompt=os.path.expanduser("~/tts/ref_zf001.wav"), text=t, lang="ZH",
              output_path=tmp, emo_vector=prof["emo"], emo_alpha=1.0, duration_factor=prof["speed"])
    a, sr = sf.read(tmp, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = trim_silence(a)
    sf.write(f"{OUT}/{tag}.wav", a, sr)
    ts, fs = f0_track(a, sr)
    res[tag] = (len(a) / sr, ts, fs)
    print(f"{tag:12s} {len(a)/sr:5.2f}s  浊音帧 {len(ts)}  F0 中位 {np.median(fs) if len(fs) else 0:.0f}Hz")

print("\n=== 两字组的第一个音节 (前 50% 时长) ===")
for tag in ("A_moluo", "B_meiyou", "E_meiyouluo"):
    dur, ts, fs = res[tag]
    m = ts < dur * 0.50
    if m.sum() < 4:
        print(f"{tag}: 帧太少"); continue
    tt, ff = ts[m], fs[m]
    k = np.polyfit(tt, ff, 1)[0]
    shape = "升(二声)" if k > 40 else "降(四声)" if k < -40 else "平"
    print(f"{tag:10s} 起 {ff[:3].mean():5.0f} → 末 {ff[-3:].mean():5.0f}Hz  斜率 {k:+7.0f} Hz/s → {shape}")

print("\n=== 整句 F0 逐 0.05s (前缀相同, 『没』应在 1.9~2.2s 一带) ===")
for tag in ("C_full_old", "D_full_new"):
    dur, ts, fs = res[tag]
    print(f"\n{tag}  {dur:.2f}s")
    line = []
    for i in range(0, int(dur / 0.05)):
        m = (ts >= i * 0.05) & (ts < (i + 1) * 0.05)
        line.append(f"{fs[m].mean():.0f}" if m.sum() else "·")
    for r in range(0, len(line), 20):
        print(f"  {r*0.05:4.1f}s " + " ".join(f"{x:>4s}" for x in line[r:r+20]))
