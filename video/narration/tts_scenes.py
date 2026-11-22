"""逐句生成旁白（每句一个 wav，便于精确对位到幕）
用法: cd ~/tts/index-tts && HF_ENDPOINT=https://hf-mirror.com \
        ./.venv/bin/python ~/tts/tts_scenes.py <script.json> <outdir> [--ref ...]
"""
import argparse, json, os, sys, time
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = os.path.expanduser("~/tts/index-tts")
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.expanduser("~/tts"))
from narrate import clean, trim_silence, PROFILES, SR   # 复用清洗/去静音/口径

import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("outdir")
ap.add_argument("--ref", default=os.path.expanduser("~/tts/ref_zf001.wav"))
ap.add_argument("--profile", default="science")
ap.add_argument("--device", default="mps")
ap.add_argument("--key", default="", help="只重跑 id 含该子串的句")
a = ap.parse_args()

prof = PROFILES[a.profile]
items = json.load(open(a.script))
if a.key:
    items = [x for x in items if a.key in x["id"]]
os.makedirs(a.outdir, exist_ok=True)

from indextts.infer_v2_5 import IndexTTS2
t0 = time.time()
tts = IndexTTS2(cfg_path=os.path.join("checkpoints", "config.yaml"),
                model_dir="checkpoints", device=a.device, use_qwen_emo=False)
print(f"[load] {time.time()-t0:.1f}s  共 {len(items)} 句  profile={a.profile}\n", flush=True)

rows, bad = [], []
for it in items:
    text = clean(it["text"], "ZH")
    speed = float(it.get("speed", prof["speed"]))
    tmp = os.path.join(a.outdir, "_tmp.wav")
    t1 = time.time()
    tts.infer(spk_audio_prompt=a.ref, text=text, lang="ZH", output_path=tmp,
              emo_vector=prof["emo"], emo_alpha=1.0, duration_factor=speed)
    audio, sr = sf.read(tmp, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        k = int(len(audio) * SR / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, k), np.arange(len(audio)), audio).astype(np.float32)
    audio = trim_silence(audio)
    dur = len(audio) / SR
    p = os.path.join(a.outdir, it["id"] + ".wav")
    sf.write(p, audio, SR)
    budget = float(it["b"])
    flag = "OK " if dur <= budget else "超!"
    if dur > budget:
        bad.append((it["id"], dur, budget))
    rows.append({"id": it["id"], "scene": it["scene"], "at": it["at"],
                 "dur": round(dur, 3), "budget": budget, "over": round(dur - budget, 3),
                 "speed": speed, "text": it["text"]})
    print(f"{flag} {it['id']:11s} {dur:5.2f}s / 预算 {budget:5.2f}s  (超 {dur-budget:+.2f}) "
          f"gen={time.time()-t1:4.1f}s  {it['text'][:26]}", flush=True)

json.dump(rows, open(os.path.join(a.outdir, "timing.json"), "w"), ensure_ascii=False, indent=1)
tot = sum(r["dur"] for r in rows)
print(f"\n合计语音 {tot:.1f}s / 片长 223s  = {tot/223*100:.0f}%")
print(f"超预算 {len(bad)} 句: {[b[0] for b in bad]}")
if bad:
    for i, d, b in bad:
        print(f"   {i}: {d:.2f}s > {b:.2f}s  需要 -{d-b:.2f}s "
              f"(speed 提到 {rows[[r['id'] for r in rows].index(i)]['speed'] * d / b:.2f})")
