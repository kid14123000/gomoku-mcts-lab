"""主片最终混音: 音乐床(指定曲目) + 人声 + 轻 cue, 视频流 copy
用法: python film_final.py
     BED_DB=-29 CUE_GAIN=1.0 DUCK=0.55 TAG=music
"""
import json, os, subprocess, wave
import numpy as np
import soundfile as sf

SR = 48000
DUR = 223.0
N = int(SR * DUR)
OUT = "/Users/kid/.hermes/cache/scratch/film"
MUS = f"{OUT}/music"
VID = "/Users/kid/gomoku_mcts_lab/video/mcts_explained_720p.mp4"
FF = "/Users/kid/.hermes/tools/ffmpeg-9.0.1-darwin-arm64/ffmpeg"
SRC = os.environ.get("SRC", "/Users/kid/Downloads/诺贝尔化学奖の小曲 (科普小曲)_DJ白琉璃_高音质在线试听_诺贝尔化学奖の小曲 (科普小曲)歌词|歌曲下载_酷狗音乐.mp3")
BED_DB = float(os.environ.get("BED_DB", "-29"))
DUCK = float(os.environ.get("DUCK", "0.55"))
CUE_GAIN = float(os.environ.get("CUE_GAIN", "1.0"))
TAG = os.environ.get("TAG", "music")

def db(x): return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-12))

def readfit(path):
    """读任意采样率的 wav → 单声道 48k → 裁剪/补零到片长 (之前漏了这步, 导致旁白被提速 8.8%)"""
    x, r = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if r != SR:
        k = int(len(x) * SR / r)
        x = np.interp(np.linspace(0, len(x) - 1, k), np.arange(len(x)), x).astype(np.float32)
        print(f"  resample {os.path.basename(path)}: {r} → {SR} ({k/SR:.2f}s)")
    return x[:N] if len(x) >= N else np.pad(x, (0, N - len(x)))

# ── 1. 解码 + tempo 拉伸到片长(保音高) ──────────────
raw = f"{MUS}/_dec.wav"
subprocess.run([FF, "-y", "-v", "error", "-i", SRC, "-ac", "1", "-ar", str(SR),
                "-c:a", "pcm_f32le", raw], check=True)
m, sr = sf.read(raw, dtype="float32")
src_dur = len(m) / SR
tempo = src_dur / DUR
print(f"原曲 {src_dur:.2f}s → 片长 {DUR:.0f}s   atempo = {tempo:.4f} (+{(tempo-1)*100:.1f}% 速)")
fit = f"{MUS}/_fit.wav"
subprocess.run([FF, "-y", "-v", "error", "-i", raw, "-filter:a", f"atempo={tempo:.5f}",
                "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le", fit], check=True)
m, _ = sf.read(fit, dtype="float32")
print(f"拉伸后 {len(m)/SR:.2f}s  峰值 {20*np.log10(np.abs(m).max()):+.1f}dB")
if len(m) < N:
    m = np.pad(m, (0, N - len(m)))
m = m[:N]

# ── 2. 定标 + 首尾淡入淡出 ────────────────────────
m = m * 10 ** ((BED_DB - db(m)) / 20)
fi, fo = int(2.0 * SR), int(4.0 * SR)
m[:fi] *= np.linspace(0, 1, fi) ** 1.2
m[-fo:] *= np.linspace(1, 0, fo) ** 0.8
print(f"音乐床 RMS {db(m):.1f}dB (目标 {BED_DB})")

# ── 3. 旁白段 duck ───────────────────────────────
rows = json.load(open("/Users/kid/tts/film_audio/timing.json"))
FIX = {"pick1": 162.35, "pick2": 167.15}
# 合并靠得近的说话窗: 句间空隙只有 0.3~1.2s 时不要抬起再压下, 否则音乐一句一句地抽搐
MERGE = float(os.environ.get("MERGE", "1.6"))      # 间隔小于该值的窗口合并
wins = sorted((FIX.get(r["id"], r["at"]), FIX.get(r["id"], r["at"]) + r["dur"] + 0.25) for r in rows)
merged = []
for a, b in wins:
    if merged and a - merged[-1][1] < MERGE:
        merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    else:
        merged.append((a, b))

gain = np.ones(N)
rp_in, rp_out = int(0.50 * SR), int(0.90 * SR)      # 淡入快、淡出慢, 像手动推子
for a_s, b_s in merged:
    a, b = int(a_s * SR), int(b_s * SR)
    k = a - max(0, a - rp_in)                 # 开头那句贴着 0s, 斜坡要被裁短
    if k > 0:
        gain[a-k:a] = np.minimum(gain[a-k:a], np.linspace(0, 1, k))
    gain[a:b] = np.minimum(gain[a:b], DUCK)
    e = min(N, b + rp_out); k2 = e - b
    if k2 > 0:
        gain[b:e] = np.minimum(gain[b:e], np.linspace(1, 0, k2))
m *= gain
span = sum(b_s-a_s for a_s, b_s in merged)
print(f"duck: {len(rows)} 个说话窗 → 合并成 {len(merged)} 块, 覆盖 {span:.0f}s/{DUR:.0f}s "
      f"({span/DUR*100:.0f}%), ×{DUCK:.2f} ({20*np.log10(DUCK):.1f}dB)")

# ── 4. 人声 + cue ────────────────────────────────
vl = readfit(f"{OUT}/voice_only.wav")
cu = readfit(f"{OUT}/cues_only.wav")
sf.write(f"{OUT}/voice48.wav", vl, SR)
cu *= CUE_GAIN
print(f"人声 RMS {db(vl):.1f}dB | 音乐(已 duck) {db(m):.1f}dB | cue {db(cu):.1f}dB "
      f"(峰 {20*np.log10(np.abs(cu).max()):.1f})")

mix = vl + m + cu
print(f"混合 RMS {db(mix):.1f}dB  峰值 {20*np.log10(np.abs(mix).max()):+.2f}dBFS")
CEIL = 10 ** (-1.2 / 20)
mix = CEIL * np.tanh(mix / CEIL)
print(f"限幅后 峰值 {20*np.log10(np.abs(mix).max()):+.2f}dBFS  RMS {db(mix):.1f}dB")

pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
wv = f"{OUT}/mix_{TAG}.wav"
with wave.open(wv, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())

dst = f"{OUT}/mcts_explained_{TAG}.mp4"
r = subprocess.run([FF, "-y", "-i", VID, "-i", wv, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest",
                    "-movflags", "+faststart", dst], capture_output=True, text=True)
print("ffmpeg rc =", r.returncode, "" if r.returncode == 0 else r.stderr[-400:])
print("输出", dst, f"{os.path.getsize(dst)/1048576:.1f}MB" if r.returncode == 0 else "")
