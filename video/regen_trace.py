"""重新录制"逐 sim"的真实 trace —— 视频里⑧那一幕的每一个数字都来自这里。

用法（在仓库根目录）：
    MODEL_PY=/path/to/Gomoku.py MODEL_W=/path/to/weights.pt \
    POSITIONS=/path/to/positions.json \
    python video/regen_trace.py

不传环境变量时的默认值：
    MODEL_PY   = vendor/ 下第一个 Gomoku*.py
    MODEL_W    = weights/ 下第一个 *.pt
    POSITIONS  = ~/AlphaZero_Gomoku_15x15_v24/web/positions.json（本机旧看板），
                 找不到就退化为内置的 23 子局面（不挑局面，直接用它）

输出：video/trace_23stones.json —— 14 次模拟的每一层 PUCT 表、被选点、叶子估值、回传值。
"""
import importlib.util, json, os, glob, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 视频里用的就是这套子（局面名「看不见白棋活三」，23 子，轮到白）
FALLBACK_STONES = [(9,6,1),(8,7,-1),(9,8,1),(9,7,-1),(10,7,1),(8,5,-1),(8,9,1),(11,6,-1),
                   (7,10,1),(6,11,-1),(6,9,1),(5,8,-1),(7,9,1),(9,9,-1),(5,9,1),(4,9,-1),
                   (6,7,1),(7,8,-1),(6,8,1),(6,6,-1),(7,6,1),(7,7,-1),(5,7,1)]

def pick(env, pattern, default):
    v = os.environ.get(env)
    if v and os.path.exists(v):
        return v
    hits = sorted(glob.glob(os.path.join(ROOT, pattern)))
    return hits[0] if hits else default

MODEL_PY = pick("MODEL_PY", "vendor/Gomoku*.py", None)
MODEL_W  = pick("MODEL_W", "weights/*.pt", None)
POS_FILE = os.environ.get("POSITIONS") or os.path.expanduser(
    "~/AlphaZero_Gomoku_15x15_v24/web/positions.json")

if not MODEL_PY or not MODEL_W:
    raise SystemExit("找不到被追踪脚本或权重；请用 MODEL_PY / MODEL_W 指定")

import torch, gym
spec = importlib.util.spec_from_file_location("traced", MODEL_PY)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.device = torch.device("cpu")
env = gym.make("Gomuku-v0")

NET_KW = {"conv_filters": [128], "residual_filters": [[128, 128]] * 8, "policy_filters": [128]}
ag = mod.AlphaZeroAgent(env=env, kwargs=NET_KW, sim_count=800, batches=1,
                        batch_size=8, disable_compile=True)
ag.net.load_state_dict(torch.load(MODEL_W, map_location="cpu"))
ag.net.to(mod.device); ag.net.eval()

# ── 选局面：优先用 positions.json 里那个 23 子的，否则用内置子力 ──
name, stones = "看不见白棋活三", FALLBACK_STONES
if os.path.exists(POS_FILE):
    try:
        for p in json.load(open(POS_FILE)):
            if p.get("name") == name and p.get("stones"):
                stones = [(int(s["r"]), int(s["c"]), int(s["v"])) for s in p["stones"]]
                break
    except Exception as e:
        print("positions.json 读取失败，用内置局面:", e)

board = np.zeros((15, 15), dtype=np.int8)
for r, c, v in stones:
    board[r, c] = v
nb, nw = int((board == 1).sum()), int((board == -1).sum())
player = 1 if nb == nw else -1
canon = (player * board).astype(np.int8)
last = (stones[-1][0], stones[-1][1])

pi, vs = ag.predict(canon[np.newaxis], last_moves=np.array([last]), players=player)
pr = np.exp(np.asarray(pi[0])).reshape(15, 15)
valid = (canon == 0).astype(np.float32)
pr = pr * valid; pr /= pr.sum()
v0 = float(np.asarray(vs[0]).reshape(-1)[0])
flat = sorted([(float(pr[r, c]), (int(r), int(c))) for r, c in np.argwhere(valid > 0)], reverse=True)
print(f"局面 {name}: {len(stones)} 子，轮到 {'黑' if player == 1 else '白'}，先验 v={v0:+.4f}")
print("先验 top5:", [(c, round(p, 4)) for p, c in flat[:5]])

TRACE, STACK, CUR = [], [], {}
ORIG = ag.search

def wrapped(bd, prior_noise=False, last_move=None, player=None):
    b = np.asarray(bd); s = b.tobytes(); d = len(STACK)
    rec = {"depth": d, "is_new": s not in ag.policy, "is_term": ag.winner.get(s) is not None}
    if d == 0:
        CUR["steps"] = []
        STACK.append({"s": s, "board": b, "depth": d})
    else:
        pb = STACK[-1]["board"]
        diff = np.argwhere((pb == 0) & (b != 0))
        a = tuple(int(x) for x in diff[0])
        par_s = STACK[-1]["s"]
        rec["a"] = a
        if par_s in ag.count:
            N = ag.count[par_s]; Q = ag.q[par_s]; Pp = ag.policy[par_s]; va = ag.valid[par_s]
            sN = float(N.sum())
            coef = (ag.c_init + math.log1p((1 + sN) / ag.c_base)) * math.sqrt(sN) / (1.0 + N)
            U = coef * Pp; PUCT = Q + U
            tbl = [{"a": [int(r), int(c)], "N": float(N[r, c]), "Q": float(Q[r, c]),
                    "P": float(Pp[r, c]), "U": float(U[r, c]), "PUCT": float(PUCT[r, c])}
                   for r, c in np.argwhere(va > 0)]
            tbl.sort(key=lambda x: -x["PUCT"])
            rec["sigmaN"] = sN
            rec["coef_at_chosen"] = float(coef[a])
            rec["table"] = tbl[:8]
            rec["rank_of_chosen"] = next((i + 1 for i, e in enumerate(tbl) if tuple(e["a"]) == a), None)
        STACK.append({"s": s, "board": b, "depth": d})
    try:
        rv = ORIG(bd, prior_noise, last_move, player)
    finally:
        if STACK and STACK[-1]["s"] == s:
            STACK.pop()
    rec["ret"] = float(rv)
    if d > 0:
        rec["v_edge"] = -float(rv)      # next_player 恒为 -1
        CUR["steps"].append(rec)
    else:
        CUR["v_root"] = float(rv)
    return rv

ag.search = wrapped
ag.reset_mcts()
for i in range(14):
    STACK.clear(); CUR["steps"] = []
    ag.search(canon, prior_noise=False, last_move=last, player=player)
    c = ag.count[canon.tobytes()].astype(float)
    q = np.array(ag.q[canon.tobytes()]).astype(float)
    tops = sorted([(int(c[r, cc]), float(q[r, cc]), (int(r), int(cc)))
                   for r, cc in np.argwhere(c > 0)], reverse=True)
    TRACE.append({"idx": i + 1, "steps": CUR["steps"], "v_root": CUR["v_root"],
                  "root_tops": [{"N": n, "Q": round(qv, 4), "a": list(a)} for n, qv, a in tops[:8]],
                  "root_sum": int(c.sum())})

def _j(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))

out = os.path.join(HERE, "trace_23stones.json")
with open(out, "w") as f:
    json.dump({"name": name, "stones": stones, "player": player, "prior_v": v0,
               "prior_top": [(round(p, 5), list(c)) for p, c in flat[:10]], "trace": TRACE},
              f, ensure_ascii=False, indent=1, default=_j)
print("已写", out)
for t in TRACE[:4]:
    path = " -> ".join(str(tuple(s["a"])) for s in reversed(t["steps"]) if "a" in s)
    print(f"  sim{t['idx']}: {path or '(只展开根)'}  v_root={t['v_root']:+.4f}")
