"""决定性对比: 同一左文境, 只改「没」后面一个字
X: 点都还没落      Y: 点都还没有落
前缀「点都还」完全相同 → 用互相关对齐 → 比较「没」那个音节的 F0 斜率
"""
import os, sys
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = os.path.expanduser("~/tts/index-tts")
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.expanduser("~/tts"))
from narrate import clean, trim_silence, PROFILES, SR
import soundfile as sf
from indextts.infer_v2_5 import IndexTTS2

OUT = "/Users/kid/tts/xy"
os.makedirs(OUT, exist_ok=True)
prof = PROFILES["science"]
CASES = [("X_old", "点都还没落"), ("Y_new", "点都还没有落")]

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                device="mps", use_qwen_emo=False)


def f0_track(x, sr, fmin=90, fmax=400, win=0.030, hop=0.005, thr=0.35):
    w = int(win*sr); h = int(hop*sr); lo, hi = int(sr/fmax), int(sr/fmin)
    ts, fs = [], []
    for i in range(0, len(x)-w, h):
        fr = x[i:i+w]*np.hanning(w)
        if np.sqrt(np.mean(fr**2)) < 1e-3:
            continue
        ac = np.correlate(fr, fr, "full")[w-1:]; ac /= (ac[0]+1e-12)
        seg = ac[lo:hi]
        if len(seg) < 3:
            continue
        pk = [j for j in range(1, len(seg)-1)
              if seg[j] > seg[j-1] and seg[j] >= seg[j+1] and seg[j] > thr]
        if not pk:
            continue
        ts.append(i/sr + win/2); fs.append(sr/(pk[0]+lo))
    return np.array(ts), np.array(fs)


aud = {}
for tag, text in CASES:
    tmp = f"{OUT}/_t.wav"
    tts.infer(spk_audio_prompt=os.path.expanduser("~/tts/ref_zf001.wav"),
              text=clean(text, "ZH"), lang="ZH", output_path=tmp,
              emo_vector=prof["emo"], emo_alpha=1.0, duration_factor=prof["speed"])
    a, sr = sf.read(tmp, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = trim_silence(a)
    sf.write(f"{OUT}/{tag}.wav", a, sr)
    aud[tag] = (a, sr)
    print(f"{tag}: {len(a)/sr:.2f}s  「{text}」")

# ── 对齐: 取 X 前 0.55s 作模板, 在 Y 里找最佳位置 ──
(x, xsr), (y, ysr) = aud["X_old"], aud["Y_new"]
L = int(0.55 * xsr)
tpl = x[:L]
best = (1e9, 0)
for off in range(0, min(len(y)-L, int(0.5*ysr))):
    seg = y[off:off+L]
    d = np.mean((seg - tpl[np.argmax(np.correlate(tpl, tpl, "full")[L-1:])-0:0][:0].sum() if False else 0)**2) if False else np.mean((seg - tpl)**2)
    if d < best[0]:
        best = (d, off)
print(f"\n前缀对齐: Y 相对 X 偏移 {best[1]} 样本 ({best[1]/ysr*1000:.1f}ms), 残差 {best[0]:.2e}")

# ── 前缀结束后 0.05~0.30s = 「没」音节 ──
print("\n「没」音节 F0 (前缀结束后 0.06–0.28s):")
res = {}
for tag in ("X_old", "Y_new"):
    a, sr = aud[tag]
    ts, fs = f0_track(a, sr)
    t0 = 0.55 + 0.06
    m = (ts > t0) & (ts < 0.55 + 0.28)
    if m.sum() < 4:
        print(f"  {tag}: 浊音帧太少 ({m.sum()})"); continue
    tt, ff = ts[m], fs[m]
    k = np.polyfit(tt, ff, 1)[0]
    res[tag] = k
    print(f"  {tag:8s} {m.sum():2d}帧  {ff[:3].mean():5.0f} → {ff[-3:].mean():5.0f}Hz  "
          f"斜率 {k:+6.0f} Hz/s → {'升(二声 méi)' if k > 30 else '降(四声 mò)' if k < -30 else '平'}")

if len(res) == 2:
    dx = "降(四声)" if res["X_old"] < -30 else "升(二声)"
    dy = "降(四声)" if res["Y_new"] < -30 else "升(二声)"
    print(f"\n结论: 旧「还没落」开头 {dx}  →  新「还没有落」开头 {dy}")
