"""主片音效床 (223s) —— cue 全部锚在 render.py 里确认过的时刻
场景边界: 0 6 22 32 40 54 66 78 90 148 160 170 181 191 209 223
"""
import json, os, wave, numpy as np

SR = 44100
DUR = 223.0
N = int(SR * DUR)
rng = np.random.default_rng(11)
OUT = "/Users/kid/.hermes/cache/scratch/film"

def t_axis(n): return np.arange(n) / SR
def rms(x): return float(np.sqrt(np.mean(np.square(x)) + 1e-20))
def db(x): return 20 * np.log10(max(x, 1e-12))
def set_rms(x, target_db): return x * (10 ** (target_db / 20) / (rms(x) + 1e-12))

def bandpass(x, lo, hi, soft=0.3):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    m = ((f >= lo) & (f <= hi)).astype(float)
    sw = max(1e-6, soft * (hi - lo))
    ls = (f >= lo - sw) & (f < lo); hs = (f > hi) & (f <= hi + sw)
    m[ls] = 0.5 * (1 - np.cos(np.pi * (f[ls] - (lo - sw)) / sw))
    m[hs] = 0.5 * (1 + np.cos(np.pi * (f[hs] - hi) / sw))
    return np.fft.irfft(X * m, len(x))

def place(buf, sig, at):
    i = int(round(at * SR))
    if 0 <= i < len(buf):
        j = min(len(buf), i + len(sig)); buf[i:j] += sig[: j - i]

def tilt_env(n, atk, dec, hold=0.0):
    t = t_axis(n)
    a = np.clip(t / max(atk, 1e-4), 0, 1)
    d = np.exp(-np.clip(t - atk - hold, 0, None) / max(dec, 1e-4))
    return a * d

out = np.zeros(N)
t = t_axis(N)
CUES = []

# ══════ 1. 底: 55Hz 双 drone + 气息, 全程 ══════
pad = (np.sin(2*np.pi*55.0*t) + 0.42*np.sin(2*np.pi*82.5*t + 0.9)
       + 0.22*np.sin(2*np.pi*110.0*t + 2.1))
pad *= 1 + 0.22*np.sin(2*np.pi*0.037*t)
pad *= np.minimum(1.0, t/3.0) * np.clip((DUR-t)/3.0, 0, 1)
pad = set_rms(bandpass(pad, 40, 280, 0.5), -37.0)
air = bandpass(rng.standard_normal(N), 400, 5200, 0.7)
air *= (1 + 0.45*np.sin(2*np.pi*0.023*t + 1.3))
air *= np.minimum(1.0, t/4.0) * np.clip((DUR-t)/4.0, 0, 1)
air = set_rms(air, -50.0)
out += pad + air
CUES.append((0, DUR, "底 drone+气息", f"{db(rms(pad)):.0f}/{db(rms(air)):.0f}dB"))

# ══════ 2. 转场 whoosh (14 个边界) ══════
def whoosh(dur, lo, hi, pw=1.5):
    n = int(SR*dur); tw = t_axis(n)
    a = bandpass(rng.standard_normal(n), lo, lo*2.2, 0.8)
    b = bandpass(rng.standard_normal(n), lo*2.0, hi, 0.8)
    x = a*(1-tw/tw[-1]) + b*(tw/tw[-1])
    x = x/(np.abs(x).max()+1e-9)
    return x * np.sin(np.pi*np.clip(tw/dur, 0, 1))**pw

for b in [6, 22, 32, 40, 54, 66, 78, 90, 148, 160, 170, 181, 191, 209]:
    place(out, 0.055*whoosh(0.55, 260, 2600), b - 0.22)
CUES.append(("6..209", "转场", "14 个边界各 0.55s 噪声扫频", "0.055"))

# 开场: 低频渐起
place(out, 0.10*(np.sin(2*np.pi*48*t_axis(int(SR*4.0)))*tilt_env(int(SR*4.0), 2.0, 1.6)), 0.2)
CUES.append((0.2, 4.2, "开场低频渐起", "48Hz", "0.10"))

# ══════ 3. ② 分支爆炸: 22s→29.3s 顶到 10^46 ══════
n_r = int(SR*7.3); tr = t_axis(n_r)
riser = np.sin(2*np.pi*(60 + 340*(tr/7.3)**2)*tr) * 0.6
riser += bandpass(rng.standard_normal(n_r), 200, 3000, 0.6) * 0.5
riser = riser/(np.abs(riser).max()+1e-9) * (tr/7.3)**2.2
place(out, 0.085*riser, 22.0)
# 爆炸点
n_b = int(SR*1.8); tb = t_axis(n_b)
boom = np.sin(2*np.pi*44*tb)*np.exp(-tb/0.55)*1.0
boom += 0.5*np.sin(2*np.pi*66*tb)*np.exp(-tb/0.30)
boom += 0.6*bandpass(rng.standard_normal(n_b), 60, 900, 0.7)*np.exp(-tb/0.18)
boom *= np.minimum(1.0, tb/0.004); boom /= np.abs(boom).max()+1e-9
place(out, 0.20*boom, 29.3)
CUES.append((22.0, 29.3, "爆炸 riser 60→400Hz", "顶到 10^46"))
CUES.append((29.3, 31.1, "爆炸低频 44Hz", "峰值最大的一处", "0.20"))

# ══════ 4. ④ 四阶段 40-54: 每阶段一个木质轻点 ══════
def wood(dur=0.13, f0=520):
    n = int(SR*dur); tw = t_axis(n)
    x = np.sin(2*np.pi*f0*tw)*np.exp(-tw/0.030) + 0.5*np.sin(2*np.pi*f0*2.7*tw)*np.exp(-tw/0.014)
    x *= np.minimum(1.0, tw/0.0015)
    return x/(np.abs(x).max()+1e-9)
for k in range(4):
    place(out, 0.075*wood(0.14, 470 + 60*k), 40.5 + k*3.3)
CUES.append((40.5, 50.5, "四阶段 4 个木点", "每 3.3s 一次", "0.075"))

# ══════ 5. ⑤ rollout 54-66: 抖动纹理, 越抽越稳 ══════
n_j = int(SR*11.0); tj = t_axis(n_j)
jit = bandpass(rng.standard_normal(n_j), 700, 4000, 0.6)
jit *= (0.35 + 0.65*np.clip(1 - tj/11.0, 0, 1))          # 逐渐变稳
jit *= (0.5 + 0.5*np.abs(np.sin(2*np.pi*3.1*tj)))
jit = jit/(np.abs(jit).max()+1e-9)
place(out, 0.030*jit, 54.5)
CUES.append((54.5, 65.5, "rollout 抖动纹理", "幅度随时间收敛", "0.030"))

# ══════ 6. ⑥ 探索 66-78: A/B 两次轻 ping ══════
def ping(f, dur=0.5):
    n = int(SR*dur); tp = t_axis(n)
    x = np.sin(2*np.pi*f*tp)*np.exp(-tp/0.16) + 0.25*np.sin(2*np.pi*f*2.01*tp)*np.exp(-tp/0.09)
    return x/(np.abs(x).max()+1e-9) * np.minimum(1.0, tp/0.003)
place(out, 0.070*ping(659.25), 69.2)
place(out, 0.070*ping(880.00), 73.0)
CUES.append((69.2, 73.5, "探索 A/B 两声 ping", "E5 / A5", "0.070"))

# ══════ 7. ⑦ 网络进场 78-90: 上扫 + 先验柱 shimmer ══════
n_u = int(SR*2.2); tu = t_axis(n_u)
up = np.sin(2*np.pi*(180 + 620*(tu/2.2)**1.5)*tu)*0.7
up += 0.5*bandpass(rng.standard_normal(n_u), 800, 6000, 0.6)
up = up/(np.abs(up).max()+1e-9) * np.sin(np.pi*np.clip(tu/2.2, 0, 1))**1.3
place(out, 0.075*up, 78.3)
for i in range(10):
    place(out, 0.030*ping(1200 + 180*i, 0.25), 83.0 + i*0.16)
CUES.append((78.3, 80.5, "网络进场 上扫 180→800Hz", None, "0.075"))
CUES.append((83.0, 84.6, "先验柱 10 个高音点", "1200→2820Hz 递增", "0.030"))

# ══════ 8. ⑧ s_walk 90-148: 滴答, 按 beat 加速 ══════
# beat 边界(绝对): 90 95 101.5 108.5 118 127.5 135.5 143.5
def tick(dur, f0):
    n = int(SR*dur); tt = t_axis(n)
    x = bandpass(rng.standard_normal(n), f0, f0*2.2, 0.5)
    x = x/(np.abs(x).max()+1e-9)*np.exp(-tt/(dur*0.32))
    x += 0.30*np.sin(2*np.pi*f0*0.42*tt)*np.exp(-tt/(dur*0.32))
    x *= np.minimum(1.0, tt/0.0015)
    return x/(np.abs(x).max()+1e-9)

# 前段 90-118: 稀疏 (0.95s); 118-143.5: 加速 0.95→0.13
tt_, gap = 90.35, 0.95
cnt = 0
while tt_ < 143.3:
    frac = max(0.0, (tt_ - 118.0) / (143.3 - 118.0))
    f0 = 1500 + 3000*frac
    place(out, 0.052*(0.55+0.45*frac)*tick(0.020, f0), tt_)
    cnt += 1
    if tt_ >= 118.0:
        gap = 0.95 * (0.13/0.95) ** ((tt_ - 118.0) / (143.3 - 118.0))
    tt_ += gap
CUES.append((90.35, 143.3, f"walk 滴答 {cnt} 枚", "0.95s 匀速 → 末段 0.13s", "0.052"))

# walk 结束: 800 次收束
place(out, 0.085*whoosh(1.1, 300, 4200, 1.0), 143.5)
CUES.append((143.5, 144.6, "收束 whoosh", "滴答停止处", "0.085"))

# ══════ 9. ⑨ 树 148-160: boom + 生长 sparkle ══════
place(out, 0.14*boom[:int(SR*1.5)], 148.15)
CUES.append((148.15, 149.7, "树生长 低频 boom", None, "0.14"))
for k in range(26):
    place(out, 0.020*ping(rng.uniform(1800, 4600), 0.18), 148.6 + k*0.28 + rng.uniform(-0.05, 0.05))
place(out, 0.055*ping(1046.5, 0.6), 155.8)
place(out, 0.050*ping(783.99, 0.7), 156.2)
CUES.append((148.6, 156, "树节点 26 个 sparkle", "1.8–4.6kHz", "0.020"))

# ══════ 10. ⑩ 选点 160-170: 定音在 162.6 ══════
place(out, 0.105*ping(1046.5, 1.4), 162.6)
place(out, 0.045*ping(1567.98, 1.1), 162.6)
CUES.append((162.6, 164.0, "选定 (8,8) 定音", "C6 + G6", "0.105"))
place(out, 0.045*ping(880.0, 0.6), 166.2)
CUES.append((166.2, 166.8, "「不用懂棋」点缀", "A5", "0.045"))

# ══════ 11. ⑪ 自对弈 170-181: 落子脉冲 ══════
for k in range(20):
    place(out, 0.040*wood(0.10, 380 + rng.uniform(-40, 40)), 170.4 + k*0.48)
CUES.append((170.4, 180.0, "自对弈 20 声落子", "0.48s 一次", "0.040"))

# ══════ 12. ⑫ 训练 181-191: 下降扫频 (梯度下降) ══════
n_d = int(SR*7.5); td = t_axis(n_d)
down = np.sin(2*np.pi*(420*np.exp(-td/2.6) + 70)*td)
down *= np.sin(np.pi*np.clip(td/7.5, 0, 1))**0.8
down /= np.abs(down).max()+1e-9
place(out, 0.055*down, 181.4)
CUES.append((181.4, 188.9, "训练 下降扫频 420→70Hz", "梯度下降的听感", "0.055"))

# ══════ 13. ⑬ 两层循环 191-209: 双频脉冲 ══════
n_l = int(SR*17.5); tlp = t_axis(n_l)
inner = (np.abs(np.sin(2*np.pi*0.85*tlp))**6) * np.sin(2*np.pi*220*tlp)
outer = (np.abs(np.sin(2*np.pi*0.19*tlp))**4) * np.sin(2*np.pi*110*tlp)
layer = inner*0.7 + outer*1.0
layer /= np.abs(layer).max()+1e-9
place(out, 0.048*layer, 191.4)
CUES.append((191.4, 208.9, "两层循环 双频脉冲", "内 0.85Hz / 外 0.19Hz", "0.048"))

# ══════ 14. ⑭ 滚雪球 209-223: 上升琶音 + 收尾和弦 ══════
for k, f in enumerate([523.25, 659.25, 783.99, 1046.50]):
    place(out, 0.055*ping(f, 0.9), 210.0 + k*0.35)
CUES.append((210.0, 212.1, "滚雪球 4 音上行琶音", "C5 E5 G5 C6", "0.055"))
# 收尾和弦 220.0
for f, a in [(220.0, 0.070), (329.63, 0.055), (440.0, 0.048), (554.37, 0.040), (880.0, 0.030)]:
    place(out, a*ping(f, 2.6), 220.0)
CUES.append((220.0, 222.6, "收尾和弦", "A3 E4 A4 C#5 A5 长衰减", "0.070"))

# ══════ 15. 旁白 ducking ══════
# ══════ 15b. 旁白 ducking ══════
tj_path = f"{OUT}/timing.json"
gain = np.ones(N)
if os.path.exists(tj_path):
    rows = json.load(open(tj_path))
    rp = int(0.180*SR)
    up = np.linspace(0, 1, rp); dn = np.linspace(1, 0, rp)
    DUCK = 0.38
    for r in rows:
        a = int(r["at"]*SR); b = int((r["at"] + r["dur"] + 0.10)*SR)
        gain[max(0, a-rp):a] = np.minimum(gain[max(0, a-rp):a], up)
        gain[a:b] = np.minimum(gain[a:b], DUCK)
        gain[b:b+rp] = np.minimum(gain[b:b+rp], dn)
    print(f"duck 窗口 {len(rows)} 个")
else:
    print("⚠ 没有 timing.json, 不做 ducking")

out *= gain
out *= 10**(4.0/20)                                   # 主推 4dB
CEIL = 10**(-4.0/20)
out = CEIL*np.tanh(out/CEIL)

print(f"音效 峰值 {db(np.abs(out).max()):.1f} dBFS  RMS {db(rms(out)):.1f} dB  长度 {len(out)/SR:.2f}s")
for c in CUES:
    print("  ", c)

pcm = (np.clip(out, -1, 1)*32767).astype("<i2")
os.makedirs(OUT, exist_ok=True)
with wave.open(f"{OUT}/sfx.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("写 sfx.wav ok")
