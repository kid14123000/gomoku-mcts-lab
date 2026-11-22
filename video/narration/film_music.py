"""把一段音乐做成 223s 的背景床, 与人声混音 (视频流 copy)
用法: MUSIC=/path/to/track.mp3 python film_music.py
     可选 LOOP_START / LOOP_END 指定循环区段(秒), 否则自动找低能量点
"""
import json, os, subprocess, wave
import numpy as np

SR = 48000
DUR = 223.0
N = int(SR * DUR)
OUT = "/Users/kid/.hermes/cache/scratch/film"
VID = "/Users/kid/gomoku_mcts_lab/video/mcts_explained_720p.mp4"
FF = "/Users/kid/.hermes/tools/ffmpeg-9.0.1-darwin-arm64/ffmpeg"
MUSIC = os.environ["MUSIC"]
TAG = os.environ.get("TAG", "music")

def db(x): return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-12))

# ── 1. 解码成 48k 单声道 float ─────────────────────
raw = f"{OUT}/_music_raw.wav"
subprocess.run([FF, "-y", "-v", "error", "-i", MUSIC, "-ac", "1", "-ar", str(SR),
                "-c:a", "pcm_f32le", raw], check=True)
import soundfile as sf
m, sr = sf.read(raw, dtype="float32")
assert sr == SR
print(f"原曲 {len(m)/SR:.2f}s  峰值 {20*np.log10(np.abs(m).max()):.1f}dB  RMS {db(m):.1f}dB")

# ── 2. 能量包络, 找合适的循环接点 ──────────────────
win = int(SR * 0.05)
env = np.array([np.sqrt(np.mean(m[i:i+win]**2) + 1e-12) for i in range(0, len(m)-win, win)])
env = 20 * np.log10(env + 1e-12)
t_env = np.arange(len(env)) * 0.05

def pick_loop(total_needed):
    """在曲子里找一段足够长的、起止都是局部低能量的区间"""
    L = float(os.environ.get("LOOP_START", "-1"))
    E = float(os.environ.get("LOOP_END", "-1"))
    if L >= 0 and E > L:
        return L, E
    need = min(total_needed, len(m)/SR - 0.5)
    # 从中间往后找能量最低的 1s 窗口当循环结束点
    best, bi = 1e9, int(len(env)*0.85)
    for i in range(int(len(env)*0.4), len(env)-int(1.0/0.05)):
        seg = env[i:i+int(1.0/0.05)]
        if seg.max() < best:
            best, bi = seg.max(), i
    return 0.0, max(need, bi*0.05)

LS, LE = pick_loop(DUR)
seg = m[int(LS*SR):int(LE*SR)]
print(f"循环区段 {LS:.2f}–{LE:.2f}s ({len(seg)/SR:.1f}s) → 需要 {DUR:.0f}s, "
      f"循环 {DUR/(len(seg)/SR):.2f} 遍")

# ── 3. 交叉淡化拼接成 223s ────────────────────────
XF = float(os.environ.get("XFADE", "2.5"))
xf = int(XF * SR)
bed = seg.copy()
while len(bed) < N + xf:
    tail = bed[-xf:].copy()
    head = seg[:xf].copy()
    ramp = np.linspace(0, 1, xf) ** 1.5
    joint = tail * (1 - ramp) + head * ramp
    bed = np.concatenate([bed[:-xf], joint, seg[xf:]])
bed = bed[:N]
# 首尾淡入淡出
fade_in, fade_out = int(1.5*SR), int(4.0*SR)
bed[:fade_in] *= np.linspace(0, 1, fade_in)
bed[-fade_out:] *= np.linspace(1, 0, fade_out)
print(f"拼接后 {len(bed)/SR:.2f}s  接缝 {len(bed)//len(seg)} 处, 交叉淡化 {XF}s")

# 音乐床基准电平: 比人声低约 12dB
voice, vsr = sf.read(f"{OUT}/voice_ref.wav", dtype="float32") if os.path.exists(f"{OUT}/voice_ref.wav") else (None, None)
TARGET_BED_DB = float(os.environ.get("BED_DB", "-31"))
bed = bed * 10 ** ((TARGET_BED_DB - db(bed)) / 20)

# ── 4. 旁白段压低 ────────────────────────────────
rows = json.load(open("/Users/kid/tts/film_audio/timing.json"))
FIX = {"pick1": 162.35, "pick2": 167.15}
gain = np.ones(N)
rp = int(0.35 * SR)                      # 音乐 duck 要更缓, 否则像抽搐
up = np.linspace(0, 1, rp); dn = np.linspace(1, 0, rp)
DUCK = float(os.environ.get("DUCK", "0.50"))
for r in rows:
    at = FIX.get(r["id"], r["at"])
    a = int(at * SR); b = int((at + r["dur"] + 0.25) * SR)
    gain[max(0, a-rp):a] = np.minimum(gain[max(0, a-rp):a], up)
    gain[a:b] = np.minimum(gain[a:b], DUCK)
    gain[b:b+rp] = np.minimum(gain[b:b+rp], dn)
bed *= gain
print(f"duck {len(rows)} 个窗口 → {DUCK:.2f} ({20*np.log10(DUCK):.1f}dB)")

# ── 5. 与人声混 ──────────────────────────────────
vl, vsr2 = sf.read(f"{OUT}/voice_only.wav", dtype="float32")
if len(vl) < N:
    vl = np.pad(vl, (0, N - len(vl)))
vl = vl[:N]
print(f"人声 RMS {db(vl):.1f}dB   音乐床 RMS {db(bed):.1f}dB  (差 {db(vl)-db(bed):.1f}dB)")
mix = vl + bed
print(f"混合 RMS {db(mix):.1f}dB  峰值 {20*np.log10(np.abs(mix).max()):.2f}dBFS")
CEIL = 10 ** (-1.2/20)
mix = CEIL * np.tanh(mix / CEIL)
print(f"限幅后 峰值 {20*np.log10(np.abs(mix).max()):.2f}dBFS  RMS {db(mix):.1f}dB")

pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
with wave.open(f"{OUT}/mix_{TAG}.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())

dst = f"{OUT}/mcts_explained_{TAG}.mp4"
cmd = [FF, "-y", "-i", VID, "-i", f"{OUT}/mix_{TAG}.wav",
       "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
       "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", dst]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg rc =", r.returncode, "" if r.returncode == 0 else r.stderr[-500:])
print("输出", dst)
