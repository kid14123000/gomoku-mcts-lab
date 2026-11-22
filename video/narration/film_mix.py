"""把 29 句旁白 + 音效床混成一条音轨, 再嵌回 mp4 (视频流直接 copy)
用法: python film_mix.py
"""
import json, os, subprocess, wave
import numpy as np
import soundfile as sf

SR = 44100
DUR = 223.0
N = int(SR * DUR)
V = "/Users/kid/tts/film_audio"
OUT = "/Users/kid/.hermes/cache/scratch/film"
VID = "/Users/kid/gomoku_mcts_lab/video/mcts_explained_720p.mp4"
FF = "/Users/kid/.hermes/tools/ffmpeg-9.0.1-darwin-arm64/ffmpeg"
os.makedirs(OUT, exist_ok=True)

rows = json.load(open(f"{V}/timing.json"))
# 手工对位微调: 旁白要落在画面揭示之后, 不能提前剧透
AT_FIX = {"pick1": 162.35, "pick2": 167.15}
for r in rows:
    if r["id"] in AT_FIX:
        print(f"  对位 {r['id']}: {r['at']} → {AT_FIX[r['id']]}")
        r["at"] = AT_FIX[r["id"]]
print(f"旁白 {len(rows)} 句")

voice = np.zeros(N)
for r in rows:
    a, sr = sf.read(f"{V}/{r['id']}.wav", dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        k = int(len(a) * SR / sr)
        a = np.interp(np.linspace(0, len(a) - 1, k), np.arange(len(a)), a).astype(np.float32)
    # 旁白统一归一到峰值 -3dB, 避免某句特别响
    a = a / (np.abs(a).max() + 1e-9) * 10 ** (-3.0 / 20)
    i = int(r["at"] * SR); j = min(N, i + len(a))
    voice[i:j] += a[: j - i]

sfx, ssr = sf.read(f"{OUT}/sfx.wav", dtype="float32")
if ssr != SR:
    k = int(len(sfx) * SR / ssr)
    sfx = np.interp(np.linspace(0, len(sfx) - 1, k), np.arange(len(sfx)), sfx).astype(np.float32)
if len(sfx) < N:
    sfx = np.pad(sfx, (0, N - len(sfx)))

def db(x): return 20 * np.log10(max(float(np.sqrt(np.mean(x**2))), 1e-12))

print(f"旁白 RMS {db(voice):.1f} dB   音效 RMS {db(sfx):.1f} dB")

# 旁白单独落盘, 用于同步自检(音效床会把 silencedetect 糊掉)
pcmv = (np.clip(voice, -1, 1)*32767).astype("<i2")
with wave.open(f"{OUT}/voice_only.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcmv.tobytes())

mix = voice * 1.26 + sfx * 1.26
print(f"混合 RMS {db(mix):.1f} dB  峰值 {20*np.log10(np.abs(mix).max()):.2f} dBFS")
CEIL = 10 ** (-1.5 / 20)
mix = CEIL * np.tanh(mix / CEIL)
print(f"限幅后 峰值 {20*np.log10(np.abs(mix).max()):.2f} dBFS  RMS {db(mix):.1f} dB")

pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
with wave.open(f"{OUT}/mix.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())

dst = f"{OUT}/mcts_explained_voiced.mp4"
cmd = [FF, "-y", "-i", VID, "-i", f"{OUT}/mix.wav",
       "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
       "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "1",
       "-shortest", "-movflags", "+faststart", dst]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg rc =", r.returncode)
print(r.stderr.strip().splitlines()[-1] if r.returncode else "ok")
