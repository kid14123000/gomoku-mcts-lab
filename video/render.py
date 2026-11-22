"""MCTS 科普短片 —— matplotlib 逐帧渲染 → ffmpeg 合成 mp4
60s @ 1280x720 @ 30fps, 6 幕
所有数字为实测（权重 it176 / 题目:黑棋反杀点）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np, os, sys

avail = {f.name for f in fm.fontManager.ttflist}
for c in ["PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti"]:
    if c in avail:
        plt.rcParams["font.family"] = c
        break
plt.rcParams["axes.unicode_minus"] = False

W, H, FPS = 1280, 720, 30
BG     = "#0F172A"
PRIOR  = "#58C4DD"
SEARCH = "#83C167"
DANGER = "#F43F5E"
ACCENT = "#FFFF00"
GREY   = "#94A3B8"
WHITE  = "#F8FAFC"

OUT = os.environ.get("FRAMES_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")
os.makedirs(OUT, exist_ok=True)

def ease(p):
    p = min(max(p, 0.0), 1.0)
    return 1 - (1 - p) ** 3

def seg(t, a, b):
    """t 在 [a,b] 内的归一化进度"""
    if b <= a: return 1.0
    return min(max((t - a) / (b - a), 0.0), 1.0)

def fade(t, a, b, c=None, d=None):
    """a..b 淡入, c..d 淡出"""
    al = ease(seg(t, a, b))
    if c is not None:
        al *= (1 - ease(seg(t, c, d)))
    return al

class Frame:
    def __init__(self):
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        self.fig.patch.set_facecolor(BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(0, H); self.ax.axis("off")

    def txt(self, x, y, s, size=26, color=WHITE, al=1.0, ha="center", va="center", weight="normal"):
        if al <= 0.01: return
        self.ax.text(x, y, s, fontsize=size, color=color, alpha=al, ha=ha, va=va,
                     fontweight=weight, zorder=10)

    def line(self, x1, y1, x2, y2, color=GREY, lw=1.2, al=1.0, ls="-"):
        if al <= 0.01: return
        self.ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=al, ls=ls,
                     solid_capstyle="round", zorder=5)

    def dot(self, x, y, r, color=PRIOR, al=1.0, ec="none", lw=0.0):
        if al <= 0.01: return
        self.ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor=ec, linewidth=lw,
                                 alpha=al, zorder=8))

    def rect(self, x, y, w, h, color, al=1.0, ec="none", lw=0, z=6):
        if al <= 0.01 or w <= 0 or h <= 0: return
        self.ax.add_patch(Rectangle((x, y), w, h, color=color, alpha=al, ec=ec, lw=lw, zorder=z))

    def arrow(self, x1, y1, x2, y2, color=GREY, al=1.0, lw=2.0):
        if al <= 0.01: return
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                         mutation_scale=16, color=color, alpha=al, lw=lw, zorder=7))

    def board(self, cx, cy, n=15, cell=26, al=0.18, lw=1.0):
        h = (n - 1) * cell / 2
        for i in range(n):
            o = -h + i * cell
            self.line(cx - h, cy + o, cx + h, cy + o, GREY, lw, al)
            self.line(cx + o, cy - h, cx + o, cy + h, GREY, lw, al)

    def cell(self, r, c, cx, cy, n=15, cell=26):
        h = (n - 1) * cell / 2
        return cx - h + c * cell, cy + h - r * cell

    def ring(self, x, y, r, color=ACCENT, al=1.0, lw=1.6):
        if al <= 0.01 or r <= 0: return
        self.ax.add_patch(Circle((x, y), r, fill=False, ec=color, lw=lw, alpha=al, zorder=9))

    def growline(self, p, c, q, color=GREY, lw=1.3, al=1.0):
        if al <= 0.01 or q <= 0: return
        self.line(p[0], p[1], p[0] + (c[0] - p[0]) * q, p[1] + (c[1] - p[1]) * q, color, lw, al)

    def save(self, i):
        self.fig.savefig(f"{OUT}/f{i:05d}.png", dpi=100, facecolor=BG)
        plt.close(self.fig)

# ───────────────────────── 幕 1：标题 ─────────────────────────
def s1(t, fr):
    fr.board(W / 2, H / 2 + 10, al=0.10 * fade(t, 0, 1.2), cell=34, lw=1.1)
    a = fade(t, 0.6, 1.8, 4.6, 5.4)
    fr.txt(W/2, 430, "一次 MCTS 决策", 54, WHITE, a, weight="bold")
    b = fade(t, 1.4, 2.4, 4.6, 5.4)
    fr.txt(W/2, 348, "拆开来看", 54, PRIOR, b, weight="bold")
    c = fade(t, 2.6, 3.4, 4.6, 5.4)
    fr.txt(W/2, 262, "15×15 五子棋   ·   800 次模拟   ·   权重 it176", 22, GREY, c)

# ───────────────────────── 幕 2：先验 ─────────────────────────
def s2(t, fr):
    cx, cy = 300, 350
    fr.board(cx, cy, al=0.20 * fade(t, 0, 1.0, 8.6, 9.4), cell=30, lw=1.1)
    fr.txt(cx, 640, "① 网络先给一个先验", 30, PRIOR, fade(t, 0.2, 1.2, 8.6, 9.4), weight="bold")

    picks = [((6, 6), 0.62), ((4, 9), 0.44), ((9, 4), 0.35), ((7, 2), 0.27),
             ((11, 11), 0.22), ((2, 4), 0.19), ((12, 6), 0.16), ((5, 12), 0.13),
             ((8, 8), 0.11), ((3, 3), 0.09)]
    for k, ((r, c), p) in enumerate(picks):
        t0 = 1.2 + k * 0.28
        al = fade(t, t0, t0 + 0.55, 8.6, 9.4)
        if al <= 0.01: continue
        x, y = fr.cell(r, c, cx, cy, cell=30)
        fr.dot(x, y, 4 + 26 * p * ease(seg(t, t0, t0 + 0.55)), PRIOR, al * (0.3 + 0.7 * p))

    x0 = 660
    fr.txt(x0, 470, "合法点  205", 28, WHITE, fade(t, 3.6, 4.3, 8.6, 9.4), ha="left")
    fr.txt(x0, 408, "先验只对其中 10 个说了话", 26, PRIOR, fade(t, 4.0, 4.8, 8.6, 9.4), ha="left")
    fr.txt(x0, 330, "top10 质量 = 0.970", 24, SEARCH, fade(t, 4.6, 5.3, 8.6, 9.4), ha="left")
    fr.txt(x0, 282, "top1  P = 0.4863", 24, SEARCH, fade(t, 4.9, 5.6, 8.6, 9.4), ha="left")
    fr.txt(x0, 210, "点位示意，数值为实测", 15, GREY, fade(t, 5.2, 5.9, 8.6, 9.4), ha="left")
    fr.txt(W/2, 92, "先验决定：谁有资格被看", 27, ACCENT, fade(t, 5.8, 6.8, 8.6, 9.4))

# ───────────────────────── 幕 3：搜索 ─────────────────────────
def s3(t, fr):
    al = fade(t, 0, 0.8, 14.6, 15.4)
    fr.txt(W/2, 662, "④ 800 次模拟，一次次写进 Q", 30, SEARCH, al, weight="bold")

    root = (760, 545)
    kids = [(440, 425), (600, 425), (760, 425), (920, 425), (1080, 425)]
    gk   = [(365, 305), (440, 305), (515, 305)]

    for k in kids:
        fr.line(root[0], root[1], k[0], k[1], GREY, 1.4, 0.5 * al)
    for g in gk:
        fr.line(kids[0][0], kids[0][1], g[0], g[1], GREY, 1.2, 0.35 * al)

    grow = ease(seg(t, 0.8, 1.6))
    fr.dot(*root, 9 * grow, WHITE, al)
    for i, k in enumerate(kids):
        g = ease(seg(t, 1.0 + i * 0.12, 1.5 + i * 0.12))
        fr.dot(k[0], k[1], 7 * g, PRIOR, al)
    for i, g in enumerate(gk):
        gg = ease(seg(t, 2.0 + i * 0.1, 2.4 + i * 0.1))
        fr.dot(g[0], g[1], 5 * gg, GREY, al)

    widths = [230, 78, 52, 36, 28]
    bg_ = ease(seg(t, 2.6, 3.6))
    for k, w in zip(kids, widths):
        h = 9
        fr.rect(k[0] - w / 2 * bg_, k[1] + 14, w * bg_, h, SEARCH, 0.85 * al)

    # token: 下走 -> 叶子闪光 -> 回传（循环 3 次，越走越快）
    for cyc in range(3):
        t0 = 4.0 + cyc * 3.2
        p = seg(t, t0, t0 + 2.4)
        if p <= 0 or p >= 1: continue
        if p < 0.33:
            q = p / 0.33
            x = root[0] + (kids[0][0] - root[0]) * q
            y = root[1] + (kids[0][1] - root[1]) * q
        elif p < 0.66:
            q = (p - 0.33) / 0.33
            x = kids[0][0] + (gk[1][0] - kids[0][0]) * q
            y = kids[0][1] + (gk[1][1] - kids[0][1]) * q
        else:
            q = (p - 0.66) / 0.34
            x = gk[1][0] + (root[0] - gk[1][0]) * q
            y = gk[1][1] + (root[1] - gk[1][1]) * q
        fr.dot(x, y, 6, ACCENT, al)

    fr.txt(230, 210, "Q ← Q + (v − Q) / N", 26, ACCENT,
           fade(t, 6.0, 6.9, 14.6, 15.4), ha="center")
    fr.txt(230, 158, "coef = 36.5   (总访问 800 时)", 19, GREY,
           fade(t, 6.6, 7.4, 14.6, 15.4), ha="center")

    if t >= 7.6:
        tv = 800 if t >= 9.6 else (450 if t >= 8.6 else 200)
        fr.txt(1115, 610, f"sim  {tv}", 26, SEARCH, al, ha="center")

    fr.txt(W/2, 40, "200 个合法点里，只有 5 个真的被看过", 25, ACCENT,
           fade(t, 10.6, 11.8, 14.6, 15.4))

# ───────────────────────── 幕：逐步走一次 decide（真实 trace） ─────────────────────────
W_STONES = [(9,6,1),(8,7,-1),(9,8,1),(9,7,-1),(10,7,1),(8,5,-1),(8,9,1),(11,6,-1),(7,10,1),
            (6,11,-1),(6,9,1),(5,8,-1),(7,9,1),(9,9,-1),(5,9,1),(4,9,-1),(6,7,1),(7,8,-1),
            (6,8,1),(6,6,-1),(7,6,1),(7,7,-1),(5,7,1)]
PRIOR6 = [((8,8),0.2918), ((4,6),0.1755), ((8,10),0.1236), ((5,5),0.0666),
          ((8,6),0.0663), ((3,8),0.0201)]
BX, BY, BCELL = 250, 360, 27

# 每个 beat: (时长, 标题, 说明1, 说明2, 表格行, 棋盘高亮, 回传行)
BEATS = [
 (5.0, "⑧ 走一次 decide：从先验到落子",
  "23 子局面　合法点 202", "", None, None, None),
 (6.5, "第 0 步：网络前向一次 → 先验",
  "一次前向只给出概率和估值，不给答案",
  "v = −0.3000 （白棋视角）", "prior", None, None),
 (7.0, "sim #1：先算根的先验，什么都不落",
  "根节点还没有 count → 直接返回 v = −0.3000",
  "模拟次数 0 ／ 访问 0", "sim1", None, None),
 (9.5, "sim #2：所有点的 PUCT 都是 0",
  "ΣN = 0 → coef = 0 → PUCT = 0·P = 0（都一样）",
  "取扫描顺序第一个合法点 → A1 (0,0)", "sim2", "A1", "+0.9980  →  取反  →  −0.9980"),
 (9.5, "sim #3：先验终于起作用",
  "ΣN = 1 → coef = 1.25 → PUCT = coef × P",
  "先验最高的 (8,8) 胜出", "sim3", "H8", "−0.9614  →  取反  →  +0.9614"),
 (8.0, "sim #4：Q 高的点被继续加深",
  "根 : (8,8) PUCT = 1.2193 仍第一",
  "进子树后又遇到 ΣN=0 → 再次全 0 → 扫描取 (0,0)", "sim4", "H8", "+0.9981  →  取反  →  −0.9981"),
 (8.0, "14 次之后，访问长成这样",
  "π = N / ΣN　→　(8,8) 84.6%", "按 N 选点 → (8,8)", "agg", "H8", None),
 (4.5, "800 次，就是这个过程重复 800 遍",
  "树越深的地方，模拟越少；没被访问的点，Q 永远是 0", "", "agg2", "H8", None),
]

ROWS = [  # (label, N, Q, P, U, PUCT)
 ("(8,8)",  "0", "+0.0000", "0.2918", "0.3648", "+0.3648"),
 ("(4,6)",  "0", "+0.0000", "0.1755", "0.2194", "+0.2194"),
 ("(8,10)", "0", "+0.0000", "0.1236", "0.1545", "+0.1545"),
 ("(5,5)",  "0", "+0.0000", "0.0666", "0.0833", "+0.0833"),
 ("(8,6)",  "0", "+0.0000", "0.0663", "0.0828", "+0.0828"),
]
TAB5 = {  # beat -> 5 行内容
 "prior": [("(8,8)",  "", "", "0.2918", "", ""), ("(4,6)", "", "", "0.1755", "", ""),
           ("(8,10)", "", "", "0.1236", "", ""), ("(5,5)", "", "", "0.0666", "", ""),
           ("(8,6)",  "", "", "0.0663", "", "")],
 "sim1":  [("—", "", "", "", "", "")] * 5,
 "sim2":  [("(0,0)", "0", "+0.0000", "0.0007", "0.0000", "+0.0000"),
           ("(0,1)", "0", "+0.0000", "0.0007", "0.0000", "+0.0000"),
           ("(0,2)", "0", "+0.0000", "0.0006", "0.0000", "+0.0000"),
           ("(0,3)", "0", "+0.0000", "0.0010", "0.0000", "+0.0000"),
           ("...",   "",  "",        "",       "",       "")],
 "sim3":  [("(8,8)", "0", "+0.0000", "0.2918", "0.3648", "+0.3648"),
           ("(4,6)", "0", "+0.0000", "0.1755", "0.2194", "+0.2194"),
           ("(8,10)","0", "+0.0000", "0.1236", "0.1545", "+0.1545"),
           ("(5,5)", "0", "+0.0000", "0.0666", "0.0833", "+0.0833"),
           ("(8,6)", "0", "+0.0000", "0.0663", "0.0828", "+0.0828")],
 "sim4":  [("(8,8)", "1", "+0.9614", "0.2918", "0.2579", "+1.2193"),
           ("(4,6)", "0", "+0.0000", "0.1755", "0.3103", "+0.3103"),
           ("(8,10)","0", "+0.0000", "0.1236", "0.2185", "+0.2185"),
           ("(5,5)", "0", "+0.0000", "0.0666", "0.1177", "+0.1177"),
           ("(0,0)", "1", "−0.9980", "0.0007", "0.0009", "−0.9971")],
 "agg":   [("(8,8)", "11", "+0.6269", "0.2918", "", "84.6%"),
           ("(4,6)", "1",  "−0.9367", "0.1755", "", "7.7%"),
           ("(0,0)", "1",  "−0.9980", "0.0007", "", "7.7%"),
           ("(8,10)","0",  "—",       "0.1236", "", "0%"),
           ("(5,5)", "0",  "—",       "0.0666", "", "0%")],
 "agg2":  [("(8,8)", "11", "+0.6269", "0.2918", "", "84.6%"),
           ("(4,6)", "1",  "−0.9367", "0.1755", "", "7.7%"),
           ("(0,0)", "1",  "−0.9980", "0.0007", "", "7.7%"),
           ("(8,10)","0",  "—",       "0.1236", "", "0%"),
           ("(5,5)", "0",  "—",       "0.0666", "", "0%")],
}

FORMULA = {
 "prior": ("一次前向：P = softmax(输出)   __  v = −0.3000", PRIOR),
 "sim1":  ("根节点第一次展开：只算 P 和 v，不落子、不加 count", GREY),
 "sim2":  ("PUCT = Q + coef × P = 0.0000 + 0.00 × 0.0007 = 0.0000", ACCENT),
 "sim3":  ("PUCT = Q + coef × P = 0.0000 + 1.25 × 0.2918 = 0.3648", ACCENT),
 "sim4":  ("PUCT = Q + coef × P = 0.9614 + 0.88 × 0.2918 = 1.2193", ACCENT),
 "agg":   ("π(8,8) = N / ΣN = 11 / 13 = 84.6%", ACCENT),
 "agg2":  ("π(8,8) = N / ΣN = 11 / 13 = 84.6%", ACCENT),
}

def draw_board(fr, al, cells_hl=None, badges=None):
    fr.board(BX, BY, n=15, cell=BCELL, al=0.20 * al, lw=1.0)
    for (r, c, v) in W_STONES:
        x, y = fr.cell(r, c, BX, BY, n=15, cell=BCELL)
        if v == 1:
            fr.dot(x, y, BCELL * 0.42, "#131E2E", al, ec="#7C8DA6", lw=1.1)
        else:
            fr.dot(x, y, BCELL * 0.42, "#E2E8F0", al, ec="#FFFFFF", lw=0.8)
    if cells_hl:
        for (rr, cc, col, lab) in cells_hl:
            x, y = fr.cell(rr, cc, BX, BY, n=15, cell=BCELL)
            fr.rect(x - BCELL * 0.44, y - BCELL * 0.44, BCELL * 0.88, BCELL * 0.88,
                    "none", al, ec=col, lw=2.0, z=8)
            if lab:
                fr.txt(x, y, lab, 14, col, al, weight="bold")

def s_walk(t, fr):
    # 找到当前 beat
    acc = 0.0; idx = 0; tl = t
    for i, b in enumerate(BEATS):
        if tl < acc + b[0]:
            idx = i; break
        acc += b[0]; idx = i + 1
    idx = min(idx, len(BEATS) - 1)
    start = sum(b[0] for b in BEATS[:idx])
    lt = t - start
    dur, title, s1, s2, tab, hl, back = BEATS[idx]
    al = fade(lt, 0, 0.55)
    TOT = sum(b[0] for b in BEATS)
    al *= (1 - ease(seg(t, TOT - 0.9, TOT)))

    fr.txt(W / 2, 662, title, 27, WHITE, al, weight="bold")
    fr.txt(W / 2, 612, s1, 20, GREY, al)
    if s2:
        fr.txt(W / 2, 574, s2, 20, PRIOR, al)

    # 棋盘
    hl_cells = []
    if hl == "A1": hl_cells = [(0, 0, DANGER, "1")]
    if hl == "H8": hl_cells = [(8, 8, SEARCH, "")]
    if idx >= 4 or tab in ("agg", "agg2"):
        hl_cells = [(8, 8, SEARCH, "")]
        if tab in ("sim4", "agg", "agg2"):
            hl_cells = [(8, 8, SEARCH, ""), (0, 0, DANGER, "1")]
    draw_board(fr, al, hl_cells)

    # 表格（还没 count 的步不画）
    show_tab = tab not in (None, "sim1")
    HDR = ["点", "N", "Q", "P", "U", "PUCT"]
    CX = [560, 690, 790, 900, 1000, 1120]
    y0 = 470
    for j, h in enumerate(HDR):
        if not show_tab: break
        if tab in ("prior",) and j in (1, 2, 4, 5): continue
        fr.txt(CX[j], y0 + 40, h, 16, GREY, al * 0.9, ha="center")
    rows = TAB5.get(tab or "", TAB5["sim1"]) if show_tab else []
    for i, row in enumerate(rows):
        ry = y0 - i * 38
        is_win = (tab in ("sim3", "sim4", "sim2") and i == 0)
        col = SEARCH if is_win else WHITE
        g = ease(seg(lt, 0.5 + i * 0.09, 0.95 + i * 0.09)) if tab else 1
        for j, val in enumerate(row):
            if tab in ("prior",) and j in (1, 2, 4, 5): continue
            if tab in ("sim1",): continue
            if not val: continue
            fr.txt(CX[j], ry, val, 17, col if j != 0 else PRIOR, al * g, ha="center")
        if is_win and g > 0.05:
            fr.rect(CX[0] - 55, ry - 15, CX[5] - CX[0] + 110, 30, SEARCH, 0.10 * al * g, z=4)

    # 算式行
    if tab in FORMULA:
        f_txt, f_col = FORMULA[tab]
        ga = ease(seg(lt, 0.8, 1.4))
        fr.txt(520, 246, f_txt, 19, f_col, al * ga, ha="left")

    # 回传条
    if back:
        fr.rect(520, 118, 700, 46, "#1E293B", al, z=3)
        fr.txt(870, 141, "叶子返回 v（对手视角）    " + back, 19, ACCENT, al)

    # 尾声：访问数标注在棋盘
    if tab in ("agg", "agg2"):
        for (r, c), lab, col in [((8, 8), "N=11", SEARCH), ((0, 0), "N=1", DANGER), ((4, 6), "N=1", DANGER)]:
            x, y = fr.cell(r, c, BX, BY, n=15, cell=BCELL)
            fr.txt(x, y - 26, lab, 13, col, al, weight="bold")


# ══════════ 新增：人类读棋 / 分支爆炸 / 人机分屏 / 四阶段 / rollout / 探索-利用 / 网络替代 ══════════
def board2(ax_x, ax_y, cell=27, n=15, al=0.2):
    """独立棋盘(避免依赖 W_STONES 的全局坐标)"""
    return ax_x, ax_y, cell, n

def draw_stones(fr, cx, cy, cell, al, stones=W_STONES, r=0.42):
    for (rr, cc, v) in stones:
        x, y = fr.cell(rr, cc, cx, cy, n=15, cell=cell)
        if v == 1:
            fr.dot(x, y, cell * r, "#131E2E", al, ec="#7C8DA6", lw=1.1)
        else:
            fr.dot(x, y, cell * r, "#E2E8F0", al, ec="#FFFFFF", lw=0.8)

def mark(fr, cx, cy, cell, pt, color, lab, al, size=13):
    x, y = fr.cell(pt[0], pt[1], cx, cy, n=15, cell=cell)
    fr.rect(x - cell * 0.46, y - cell * 0.46, cell * 0.92, cell * 0.92, "none", al, ec=color, lw=2.0, z=9)
    if lab:
        fr.txt(x + cell * 0.52, y + cell * 0.5, lab, size, color, al, ha="left", weight="bold")

# ── ① 人类怎么读棋 ─────────────────────────────────────
HX, HY, HC = 250, 350, 30
H_CAND = [((8, 8), "我方能成四（活四）", SEARCH), ((6, 10), "对方活三 · 一端", DANGER),
          ((8, 10), "对方活三 · 另一端", DANGER), ((4, 6), "对方也能成四", DANGER)]

def s_human(t, fr):
    TOT = 16.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.8, TOT)))
    fr.txt(W / 2, 668, "① 人类怎么读棋：先看哪里，而不是全部算一遍", 26, WHITE, al, weight="bold")

    fr.board(HX, HY, n=15, cell=HC, al=0.20 * al, lw=1.0)
    draw_stones(fr, HX, HY, HC, al)

    # 4 个候选（右侧竖排，互不重叠）
    fr.txt(500, 585, "人类先看这 4 个点：", 18, GREY, al * ease(seg(t, 1.8, 2.4)), ha="left")
    for i, (pt, lab, col) in enumerate(H_CAND):
        g = ease(seg(t, 2.0 + i * 0.3, 2.6 + i * 0.3)) * al
        if g <= 0.02: continue
        x, y = fr.cell(pt[0], pt[1], HX, HY, n=15, cell=HC)
        fr.rect(x - HC * 0.46, y - HC * 0.46, HC * 0.92, HC * 0.92, "none", g, ec=col, lw=1.8, z=9)
        fr.txt(500, 548 - i * 32, f"{pt}   {lab}", 16, col, g, ha="left")

    # 试 A
    a1 = ease(seg(t, 5.2, 5.9)) * al
    if a1 > 0.02:
        x, y = fr.cell(6, 10, HX, HY, n=15, cell=HC)
        fr.dot(x, y, HC * 0.42, "#E2E8F0", a1, ec="#FFFFFF", lw=0.8)
        fr.txt(x, y, "1", 12, "#0F172A", a1, weight="bold")
    a2 = ease(seg(t, 6.4, 7.1)) * al
    if a2 > 0.02:
        x, y = fr.cell(8, 10, HX, HY, n=15, cell=HC)
        fr.dot(x, y, HC * 0.42, "#131E2E", a2, ec="#7C8DA6", lw=1.1)
        fr.txt(x, y, "2", 12, "#E2E8F0", a2, weight="bold")
        fr.txt(500, 380, "1  我先堵 (6,10)", 17, WHITE, a2, ha="left")
        fr.txt(500, 350, "2  对方 (8,10) 成活四 —— 我挡不住", 17, DANGER, a2, ha="left")
        fr.txt(500, 318, "这条不通 → 回退", 18, DANGER, a2 * ease(seg(t, 7.2, 7.7)), ha="left")

    # 回退提示
    back = 1 - ease(seg(t, 8.6, 9.6))
    if t < 9.8 and back > 0.02:
        for pt, lab in (((6, 10), "1"), ((8, 10), "2")):
            x, y = fr.cell(pt[0], pt[1], HX, HY, n=15, cell=HC)
            fr.txt(x, y, lab, 12, DANGER, al * back, weight="bold")

    # 试 D
    d1 = ease(seg(t, 10.0, 10.8)) * al
    if d1 > 0.02:
        x, y = fr.cell(8, 8, HX, HY, n=15, cell=HC)
        fr.dot(x, y, HC * 0.42, "#E2E8F0", d1, ec="#FFFFFF", lw=0.8)
        fr.txt(x, y, "3", 12, "#0F172A", d1, weight="bold")
        for rr in range(6, 10):
            x2, y2 = fr.cell(rr, rr, HX, HY, n=15, cell=HC)
            fr.rect(x2 - HC * 0.44, y2 - HC * 0.44, HC * 0.88, HC * 0.88, "none", d1, ec=SEARCH, lw=1.6, z=7)
        fr.txt(500, 250, "3  我走 (8,8) —— (6,6)(7,7)(8,8)(9,9) 成活四", 16, SEARCH, d1, ha="left")
        fr.txt(500, 218, "黑方没有一手连五 → 我先赢", 18, ACCENT, d1 * ease(seg(t, 11.6, 12.2)), ha="left")

    if t > 13.4:
        fr.txt(W / 2, 110, "人类只读了 3 手。剩下 198 个点，一个都没进脑子", 23, ACCENT,
               al * ease(seg(t, 13.4, 14.2)))

# ── ② 分支爆炸 ────────────────────────────────────────
def s_boom(t, fr):
    TOT = 10.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.8, TOT)))
    fr.txt(W / 2, 668, "② 计算机没有这个直觉：只能全试", 26, WHITE, al, weight="bold")

    counts = [1, 6, 34, 140, 380]
    labs = ["根", "第 1 层 6", "第 2 层 34", "第 3 层 140", "第 4 层 380+"]
    rng = np.random.default_rng(1)
    for lv, cnt in enumerate(counts):
        t0 = 0.8 + lv * 1.15
        g = ease(seg(t, t0, t0 + 0.9))
        if g <= 0.02: continue
        y = 560 - lv * 78
        spread = min(860, 64 * cnt ** 0.55)
        xs = np.linspace(770 - spread / 2, 770 + spread / 2, cnt)
        shown = max(1, int(cnt * g))
        for i in range(shown):
            fr.dot(xs[i] + rng.normal(0, 1.2), y + rng.normal(0, 7.0 if lv >= 3 else 3.0),
                   6.5 if lv < 3 else 3.8, [WHITE, PRIOR, PRIOR, GREY, GREY][lv], al * 0.85)
        fr.txt(70, y, labs[lv], 15, GREY, al * g, ha="left")

    for i, (txt, tt, col, fs) in enumerate([("200 个点 × 20 层", 6.4, WHITE, 28),
                                           ("≈ 10^46 种可能", 7.3, DANGER, 32)]):
        a2 = ease(seg(t, tt, tt + 0.7)) * al
        if a2 > 0.02:
            fr.txt(W / 2, 168 - i * 54, txt, fs, col, a2, weight="bold")
    if t > 8.4:
        fr.txt(W / 2, 66, "再快的机器，也不可能把这条路走完", 19, GREY, al * ease(seg(t, 8.4, 9.1)))

# ── ③ 人 vs 机器 ─────────────────────────────────────
def s_vs(t, fr):
    TOT = 8.0
    al = fade(t, 0, 0.6) * (1 - ease(seg(t, TOT - 0.7, TOT)))
    fr.txt(W / 2, 668, "③ 差别不在算得多深，在于看得准不准", 26, WHITE, al, weight="bold")

    lx, ly, lc = 320, 340, 24
    fr.board(lx, ly, n=15, cell=lc, al=0.16 * al, lw=0.9)
    draw_stones(fr, lx, ly, lc, al * 0.5)
    fr.txt(lx, 588, "人类", 22, SEARCH, al, weight="bold")
    for i, (pt, _, col) in enumerate(H_CAND):
        g = ease(seg(t, 0.8 + i * 0.2, 1.3 + i * 0.2))
        x, y = fr.cell(pt[0], pt[1], lx, ly, n=15, cell=lc)
        fr.rect(x - lc * 0.46, y - lc * 0.46, lc * 0.92, lc * 0.92, "none", al * g, ec=col, lw=2.0, z=9)
    fr.txt(lx, 108, "只看 4 个点", 20, SEARCH, al)

    rx, ry, rc = 950, 340, 24
    fr.board(rx, ry, n=15, cell=rc, al=0.16 * al, lw=0.9)
    draw_stones(fr, rx, ry, rc, al * 0.5)
    fr.txt(rx, 588, "计算机", 22, DANGER, al, weight="bold")
    empty = [(r, c) for r in range(15) for c in range(15)
             if not any((s0, s1) == (r, c) for s0, s1, _ in W_STONES)]
    for i, (r, c) in enumerate(empty):
        g = ease(seg(t, 1.0 + i * 0.004, 1.5 + i * 0.004))
        if g <= 0.02: continue
        x, y = fr.cell(r, c, rx, ry, n=15, cell=rc)
        fr.dot(x, y, 4.6 * g, DANGER, al * 0.45)
    fr.txt(rx, 108, "全部 200 个点", 20, DANGER, al)

    if t > 4.6:
        fr.txt(W / 2, 52, "它不知道哪一个值得看 —— 这才是真问题", 20, ACCENT, al * ease(seg(t, 4.6, 5.3)))

# ── ④ MCTS 四阶段 ────────────────────────────────────
def s_phase(t, fr):
    TOT = 14.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.8, TOT)))
    fr.txt(W / 2, 668, "④ 2006：MCTS —— 不评估局面，靠走到终局看谁赢", 24, WHITE, al, weight="bold")

    cx, cy, R = W / 2, 385, 165
    items = [("选择", "Q + 先验 挑一条路", cx, cy + R, PRIOR),
             ("扩展", "叶子第一次到达", cx + R, cy, SEARCH),
             ("模拟", "随机走到终局", cx, cy - R, ACCENT),
             ("回传", "逐层取反，更新 N / Q", cx - R, cy, DANGER)]
    beat = int(t / 2.6) % 4
    for i, (nm, sub, x, y, col) in enumerate(items):
        on = (i == beat)
        g = ease(seg(t, 0.5 + i * 0.3, 1.1 + i * 0.3))
        fr.txt(x, y + 10, nm, 23 if on else 19, col if on else GREY, al * g, weight="bold")
        fr.txt(x, y - 18, sub, 14, col if on else GREY, al * g * (1.0 if on else 0.7))

    for i in range(4):
        a = items[i]; b = items[(i + 1) % 4]
        g = ease(seg(t, 1.5 + i * 0.28, 2.0 + i * 0.28))
        if g <= 0.03: continue
        x1, y1 = a[2], a[3]; x2, y2 = b[2], b[3]
        fr.arrow(x1 + (x2 - x1) * 0.22 * g, y1 + (y2 - y1) * 0.22 * g,
                 x1 + (x2 - x1) * 0.74 * g, y1 + (y2 - y1) * 0.74 * g,
                 GREY if not (i == beat) else ACCENT, al * 0.85, 2.0)

    fr.txt(cx, cy + 2, "一次模拟\n= 绕一圈", 20, GREY, al * ease(seg(t, 2.4, 3.0)))
    if t > 11.0:
        fr.txt(W / 2, 60, "重复 N 次，访问次数 N(s,a) 就成了这一手的答案", 20, ACCENT,
               al * ease(seg(t, 11.0, 11.8)))

# ── ⑤ rollout 有多不靠谱 ──────────────────────────────
ROLL = [(5, 60.0), (10, 40.0), (20, 45.0), (50, 46.0), (100, 45.0),
        (200, 48.0), (500, 50.6), (1000, 52.8), (2000, 52.3)]

def s_rollout(t, fr):
    TOT = 12.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.8, TOT)))
    fr.txt(W / 2, 668, "⑤ rollout 有个致命问题：它测的是「双方乱下谁赢」", 23, WHITE, al, weight="bold")

    x0, x1 = 170, 690
    base, hmax = 180, 320
    fr.line(x0, base, x1, base, GREY, 1.2, al * 0.6)
    fr.line(x0, base, x0, base + hmax, GREY, 1.2, al * 0.6)
    yy = lambda pp: base + (pp - 20) / 50.0 * hmax
    for pct in (20, 35, 50, 70):
        fr.line(x0 - 6, yy(pct), x0, yy(pct), GREY, 1.0, al * 0.6)
        fr.txt(x0 - 14, yy(pct), f"{pct}%", 12, GREY, al, ha="right")
    yy35 = yy(35)
    fr.line(x0, yy35, x1, yy35, SEARCH, 1.5, al * 0.9, ls="--")
    fr.txt(x1 + 10, yy35, "网络估值 35%", 14, SEARCH, al, ha="left")

    n = len(ROLL)
    for i, (nx, est) in enumerate(ROLL):
        t0 = 1.3 + i * 0.55
        g = ease(seg(t, t0, t0 + 0.45))
        if g <= 0.02: continue
        x = x0 + 30 + i * (x1 - x0 - 60) / (n - 1)
        h = yy(est) - base
        if i > 0:
            xp = x0 + 30 + (i - 1) * (x1 - x0 - 60) / (n - 1)
            hp = yy(ROLL[i - 1][1]) - base
            fr.line(xp, base + hp, x, base + h * g, DANGER, 1.5, al * 0.75)
        fr.dot(x, base + h * g, 6.5, DANGER, al)
        fr.txt(x, base - 20, str(nx), 11, GREY, al * g)
    fr.txt((x0 + x1) / 2, base - 44, "rollout 次数", 14, GREY, al)

    fr.txt(880, 470, "5 次：60%　10 次：40%", 18, GREY, al * ease(seg(t, 2.2, 2.8)), ha="left")
    fr.txt(880, 424, "2000 次：稳定在 52.3%", 19, DANGER, al * ease(seg(t, 6.6, 7.2)), ha="left")
    fr.txt(880, 368, "而网络说白只有 35%", 20, SEARCH, al * ease(seg(t, 7.6, 8.2)), ha="left")
    if t > 8.8:
        fr.txt(W / 2, 84, "不是采样不够，是它算的根本不是「好好下谁赢」", 21, ACCENT,
               al * ease(seg(t, 8.8, 9.6)))

# ── ⑥ 探索 vs 利用 ────────────────────────────────────
def s_explore(t, fr):
    TOT = 12.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.8, TOT)))
    fr.txt(W / 2, 668, "⑥ 探索 vs 利用：试腻了的让位，没试过的给机会", 24, WHITE, al, weight="bold")
    fr.txt(W / 2, 612, "PUCT = Q + coef × P / (1 + N)", 23, ACCENT, al)

    base, hQ, hU = 235, 190, 640
    for i, (lab, Nv, Qv, Pv, col) in enumerate([("点 A", 8, 0.96, 0.2918, SEARCH),
                                                ("点 B", 0, 0.00, 0.1755, DANGER)]):
        x = 400 + i * 380
        gA = ease(seg(t, 1.0 + i * 0.3, 1.6 + i * 0.3))
        u = 1.25 * Pv / (1 + Nv)
        hq, hu = hQ * Qv, hU * u
        fr.rect(x - 55, base, 110, hq * gA, col, al * 0.30)
        fr.rect(x - 55, base + hq * gA, 110, hu * gA, PRIOR, al * 0.70)
        fr.txt(x, base + hq * gA + hu * gA + 30, f"Q {Qv:.2f} + U {u:.3f} = {Qv + u:.2f}", 16, ACCENT, al * gA)
        if Qv > 0:
            fr.txt(x, base + hq / 2, "Q", 15, WHITE, al * gA)
        if hu > 18:
            fr.txt(x, base + hq * gA + hu * gA / 2, "U", 14, "#0F172A", al * gA)
        fr.txt(x, base - 26, lab, 19, WHITE, al, weight="bold")
        fr.txt(x, base - 52, f"N={Nv}   P={Pv:.4f}", 14, GREY, al)

    if t > 6.8:
        fr.txt(W / 2, 150, "A 的 Q 高，但被访问 8 次后 U 只剩 0.040", 18, GREY, al * ease(seg(t, 6.8, 7.4)))
        fr.txt(W / 2, 116, "B 一次没试过，U = 0.219 —— 总分会反超", 18, ACCENT, al * ease(seg(t, 7.4, 8.0)))
    if t > 8.4:
        fr.txt(W / 2, 62, "所以它一定会去试没试过的点。这就是探索", 19, ACCENT, al * ease(seg(t, 8.4, 9.1)))

# ── ⑦ 网络替代 rollout + 先验 ─────────────────────────
def s_net(t, fr):
    TOT = 12.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.8, TOT)))
    fr.txt(W / 2, 668, "⑦ 神经网络替掉两件事：模拟到终局、以及「该看哪里」", 23, WHITE, al, weight="bold")

    fr.txt(300, 590, "以前：模拟走到终局", 19, GREY, al, weight="bold")
    fr.txt(300, 556, "随机 200 步 → 一个 ±1", 16, DANGER, al * ease(seg(t, 1.0, 1.6)))
    for i in range(6):
        g = ease(seg(t, 1.2 + i * 0.14, 1.5 + i * 0.14))
        fr.dot(190 + i * 44, 500, 10 * g, DANGER, al * 0.5)
    fr.arrow(150, 470, 470, 470, DANGER, al * 0.6, 1.5)

    fr.txt(960, 590, "现在：一次前向", 19, SEARCH, al, weight="bold")
    fr.txt(960, 556, "直接给出胜率估值", 16, SEARCH, al * ease(seg(t, 2.0, 2.6)))
    fr.arrow(760, 480, 1140, 480, SEARCH, al * ease(seg(t, 2.2, 2.8)), 2.0)
    fr.txt(960, 430, "v = −0.30", 30, ACCENT, al * ease(seg(t, 2.8, 3.4)), weight="bold")

    fr.txt(W / 2, 350, "第二次：同一个网络给出先验", 19, PRIOR,
           al * ease(seg(t, 4.2, 4.9)), weight="bold")
    for i, (pt, pv) in enumerate(PRIOR6):
        g = ease(seg(t, 5.0 + i * 0.16, 5.5 + i * 0.16))
        x = 400 + i * 100
        fr.rect(x - 30, 175, 60, 110 * pv / 0.30 * g, PRIOR, al * 0.55)
        fr.txt(x, 175 + 110 * pv / 0.30 * g + 16, f"{pv:.3f}", 12, WHITE, al * g)
        fr.txt(x, 152, f"({pt[0]},{pt[1]})", 11, GREY, al * g)
    fr.txt(W / 2, 108, "前 10 个点拿了 97% 的先验质量", 17, GREY, al * ease(seg(t, 7.6, 8.2)))
    if t > 8.8:
        fr.txt(W / 2, 60, "于是问题从「200 个点全试」变成「先看这 10 个」", 19, ACCENT,
               al * ease(seg(t, 8.8, 9.5)))


# ── ⑩ MCTS 算完：选出一招 ────────────────────────────
def s_pick(t, fr):
    TOT = 10.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.8, TOT)))
    fr.txt(W / 2, 668, "⑩ 算完了：访问最多的点，就是它的答案", 26, WHITE, al, weight="bold")

    bx, by, bc = 285, 340, 30
    fr.board(bx, by, n=15, cell=bc, al=0.20 * al, lw=1.0)
    draw_stones(fr, bx, by, bc, al)

    rows = [("(8,8)", "11", "84.6%", SEARCH), ("(4,6)", "1", "7.7%", DANGER),
            ("(0,0)", "1", "7.7%", DANGER)]
    fr.txt(580, 540, "点", 15, GREY, al, ha="left")
    fr.txt(760, 540, "N", 15, GREY, al, ha="left")
    fr.txt(880, 540, "π", 15, GREY, al, ha="left")
    for i, (pt, nv, pv, col) in enumerate(rows):
        g = ease(seg(t, 0.8 + i * 0.3, 1.4 + i * 0.3)) * al
        if g <= 0.02: continue
        y = 500 - i * 40
        fr.txt(580, y, pt, 19, col, g, ha="left", weight="bold")
        fr.txt(760, y, nv, 19, col, g, ha="left")
        fr.txt(880, y, pv, 19, col, g, ha="left")

    g2 = ease(seg(t, 2.6, 3.4)) * al
    if g2 > 0.02:
        x, y = fr.cell(8, 8, bx, by, n=15, cell=bc)
        pulse = 1 + 0.12 * np.sin((t - 2.6) * 5)
        fr.rect(x - bc * 0.5 * pulse, y - bc * 0.5 * pulse, bc * pulse, bc * pulse,
                "none", g2, ec=SEARCH, lw=2.6, z=9)
        fr.txt(580, 330, "它选了 (8,8)", 27, SEARCH, g2, ha="left", weight="bold")
        fr.txt(580, 286, "800 次模拟的预算，", 18, GREY, g2, ha="left")
        fr.txt(580, 254, "最后都压在访问次数上", 18, GREY, g2, ha="left")

    if t > 6.2:
        fr.txt(W / 2, 110, "这就是 MCTS 有效的地方：它不需要理解棋理，只需要把预算分配好", 20, ACCENT,
               al * ease(seg(t, 6.2, 7.0)))

# ── ⑪ 自对弈 ─────────────────────────────────────────
def s_selfplay(t, fr):
    TOT = 11.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.8, TOT)))
    fr.txt(W / 2, 668, "⑪ 然后它自己跟自己下成千上万局", 26, WHITE, al, weight="bold")

    rng = np.random.default_rng(7)
    x0, y0, step, sz = 100, 540, 76, 46
    for i in range(24):
        r, c = i // 6, i % 6
        x = x0 + c * step; y = y0 - r * step
        g = ease(seg(t, 0.4 + i * 0.14, 0.9 + i * 0.14))
        if g <= 0.02: continue
        fr.rect(x - sz / 2, y - sz / 2, sz, sz, "none", al * g * 0.8, ec=GREY, lw=0.9, z=4)
        for k in range(13):
            fr.dot(x + rng.uniform(-sz * 0.38, sz * 0.38), y + rng.uniform(-sz * 0.38, sz * 0.38),
                   2.4, "#E2E8F0" if k % 2 else "#131E2E", al * g * 0.9)

    cnt = int(min(10456, 10456 * ease(seg(t, 1.0, 8.0))))
    fr.txt(760, 540, f"第 {cnt:,} 局", 30, ACCENT, al, ha="left", weight="bold")
    for i, (lab, sub) in enumerate([("每一手都记下来：", ""), ("局面 s", "搜索时的棋盘"),
                                    ("搜索分布 π", "800 次模拟的结果"),
                                    ("终局胜负 z", "+1 / −1")]):
        g = ease(seg(t, 2.6 + i * 0.45, 3.2 + i * 0.45)) * al
        if g <= 0.02: continue
        fr.txt(760, 470 - i * 52, lab, 19, WHITE if i == 0 else PRIOR, g, ha="left",
               weight="bold" if i == 0 else "normal")
        if sub:
            fr.txt(960, 470 - i * 52, sub, 15, GREY, g, ha="left")

    if t > 7.6:
        fr.txt(W / 2, 150, "实测：108 代共 10,456 局，每代约 97 局、1,993 条样本", 19, GREY,
               al * ease(seg(t, 7.6, 8.3)))
        fr.txt(W / 2, 100, "它没有棋谱可学 —— 老师就是它自己的搜索", 21, ACCENT,
               al * ease(seg(t, 8.3, 9.1)))

# ── ⑫ 训练先验 ───────────────────────────────────────
def s_train(t, fr):
    TOT = 10.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.8, TOT)))
    fr.txt(W / 2, 668, "⑫ 这些数据反过来训练先验网络", 26, WHITE, al, weight="bold")

    # 左：数据堆
    for i in range(12):
        r, c = i // 4, i % 4
        g = ease(seg(t, 0.5 + i * 0.06, 0.9 + i * 0.06)) * al
        fr.rect(110 + c * 40, 330 + r * 40, 34, 34, SEARCH, g * 0.25, ec=SEARCH, lw=0.8, z=5)
    fr.txt(190, 288, "数据集", 17, SEARCH, al, ha="center")

    # 中：网络
    gN = ease(seg(t, 1.8, 2.5)) * al
    if gN > 0.02:
        fr.rect(500, 350, 240, 120, PRIOR, gN * 0.16, ec=PRIOR, lw=1.8, z=5)
        fr.txt(620, 430, "神经网络", 21, PRIOR, gN, weight="bold")
        fr.txt(620, 396, "策略头 · 价值头", 15, GREY, gN)

    # 右：新先验
    for i, (pt, pv) in enumerate(PRIOR6):
        g = ease(seg(t, 3.4 + i * 0.12, 3.9 + i * 0.12)) * al
        if g <= 0.02: continue
        x = 950 + i * 52
        fr.rect(x - 22, 330, 44, 160 * pv / 0.30 * g, PRIOR, g * 0.6, z=5)
        fr.txt(x, 330 + 160 * pv / 0.30 * g + 14, f"{pv:.2f}", 11, WHITE, g)
    fr.txt(1105, 292, "新的先验", 17, PRIOR, al * ease(seg(t, 3.4, 4.0)), ha="center")

    fr.arrow(400, 410, 480, 410, SEARCH, al * ease(seg(t, 1.2, 1.8)), 2.0)
    fr.arrow(760, 410, 900, 410, SEARCH, al * ease(seg(t, 3.0, 3.6)), 2.0)

    if t > 5.4:
        fr.txt(W / 2, 200, "搜索当老师，网络当学生", 22, ACCENT, al * ease(seg(t, 5.4, 6.1)))
        fr.txt(W / 2, 152, "把 800 次模拟的结论，学成一次前向就能给出的直觉", 18, GREY,
               al * ease(seg(t, 6.1, 6.8)))
        fr.txt(W / 2, 100, "实测：网络与搜索的距离 KL ≈ 0.49 ~ 1.41 —— 每代都要搬这一段", 16, GREY,
               al * ease(seg(t, 6.8, 7.5)))

# ── ⑬ 良性循环 ───────────────────────────────────────
def s_loop(t, fr):
    TOT = 14.0
    al = fade(t, 0, 0.7) * (1 - ease(seg(t, TOT - 0.9, TOT)))
    fr.txt(W / 2, 668, "⑬ 于是形成良性循环", 27, WHITE, al, weight="bold")

    cx, cy = W / 2, 380
    # 螺旋轨迹
    tp = ease(seg(t, 0.6, 9.6))
    N = 220
    th_max = tp * 6 * np.pi
    pts = []
    for i in range(N + 1):
        th = th_max * i / N
        r = 92 + 17 * th / (2 * np.pi)
        pts.append((cx + r * np.cos(th - np.pi / 2), cy + r * np.sin(th - np.pi / 2)))
    for i in range(len(pts) - 1):
        fr.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], PRIOR, 2.0, al * 0.75)
    if len(pts) > 2:
        fr.dot(pts[-1][0], pts[-1][1], 6, ACCENT, al)

    labels = [("先验更准", 0), ("搜索更准", 1), ("对局更强", 2), ("数据更好", 3)]
    for lab, k in labels:
        ang = -np.pi / 2 + k * np.pi / 2
        x = cx + 196 * np.cos(ang); y = cy + 196 * np.sin(ang)
        g = ease(seg(t, 1.0 + k * 0.35, 1.6 + k * 0.35)) * al
        fr.txt(x, y, lab, 20, PRIOR if k % 2 == 0 else SEARCH, g, weight="bold")

    if t > 10.2:
        fr.txt(W / 2, 128, "它每一代都在把搜索的结论变成直觉", 20, GREY, al * ease(seg(t, 10.2, 10.9)))
        fr.txt(W / 2, 74, "先验决定考场，价值决定胜负", 29, ACCENT, al * ease(seg(t, 10.9, 11.7)), weight="bold")

# ───────────────────────── 幕 3：树一层层长开 ─────────────────────────
NODES = [
    ("A", 640, 575, None), ("B", 430, 480, "A"), ("C", 535, 480, "A"),
    ("D", 640, 480, "A"), ("E", 745, 480, "A"), ("F", 850, 480, "A"),
    ("B1", 395, 385, "B"), ("B2", 465, 385, "B"), ("D1", 605, 385, "D"),
    ("D2", 675, 385, "D"), ("F1", 815, 385, "F"), ("F2", 885, 385, "F"),
    ("B1a", 370, 290, "B1"), ("B1b", 415, 290, "B1"), ("B2a", 460, 290, "B2"),
]
POS = {n: (x, y) for n, x, y, _ in NODES}
PAR = {n: p for n, _, _, p in NODES}
LVL = {"A": 0}
for n, _, _, p in NODES:
    if p: LVL[n] = LVL[p] + 1
BY_LVL = {k: [n for n in POS if LVL[n] == k] for k in range(4)}

def s_tree(t, fr):
    al = fade(t, 0, 0.8, 11.2, 12.0)
    fr.txt(W / 2, 662, "⑨ 但树不是均匀长的", 30, SEARCH, al, weight="bold")

    # 层标签
    for k, lab in enumerate(["根", "第 1 层", "第 2 层", "第 3 层"]):
        y = {0: 575, 1: 480, 2: 385, 3: 290}[k]
        a2 = fade(t, 0.4 + k * 0.2, 1.1 + k * 0.2, 11.2, 12.0)
        fr.txt(158, y, lab, 15, GREY, a2 * 0.9, ha="left")

    # 逐层展开：边线生长 + 节点绽放光环
    order = []
    for k in range(4):
        t0 = {0: 0.5, 1: 0.9, 2: 2.1, 3: 3.5}[k]
        for j, n in enumerate(BY_LVL[k]):
            order.append((k, j, n, t0 + j * 0.13))

    for k, j, n, t0 in order:
        x, y = POS[n]
        p = PAR[n]
        # 边
        if p:
            q = ease(seg(t, t0, t0 + 0.45))
            fr.growline(POS[p], (x, y), q, GREY, 1.3, 0.5 * al)
        # 节点
        g = ease(seg(t, t0 + 0.2, t0 + 0.55))
        if g > 0.02:
            r = {0: 9, 1: 7, 2: 6, 3: 5.5}[k]
            fr.dot(x, y, r * g, WHITE if k == 0 else PRIOR, al * (1.0 if k < 2 else 0.85))
            # 绽放光环
            ringp = seg(t, t0 + 0.2, t0 + 0.95)
            fr.ring(x, y, r + 22 * ease(ringp), PRIOR, al * (1 - ringp) * 0.85, 1.4)

    # 只往一条路加深：token 反复下走
    path = ["A", "B", "B1", "B1a"]
    for cyc in range(3):
        t0 = 4.5 + cyc * 1.05
        p = seg(t, t0, t0 + 0.95)
        if p <= 0 or p >= 1: continue
        seg_n = min(int(p * 3), 2)
        q = (p * 3) - seg_n
        a_, b_ = POS[path[seg_n]], POS[path[seg_n + 1]]
        x = a_[0] + (b_[0] - a_[0]) * q
        y = a_[1] + (b_[1] - a_[1]) * q
        fr.dot(x, y, 5.5, ACCENT, al)
        if p > 0.92:
            fr.ring(POS[path[-1]][0], POS[path[-1]][1], 10 + 30 * seg(t, t0 + 0.9, t0 + 1.05),
                    ACCENT, al * 0.9, 1.4)

    # 访问数落在节点旁（示意）
    nb = [("A", "800", 14, 12, 15), ("B", "375", 14, 12, 15), ("B1", "269", 14, 12, 15),
          ("C", "1", 15, 10, 13), ("E", "1", 15, 10, 13), ("F", "1", 15, 10, 13),
          ("D1", "1", 14, 9, 12), ("F1", "1", 14, 9, 12), ("B1b", "1", 12, 9, 12)]
    for i, (n, v, dx, dy, fs) in enumerate(nb):
        a2 = fade(t, 6.9 + i * 0.08, 7.5 + i * 0.08, 11.2, 12.0)
        if a2 <= 0.01: continue
        x, y = POS[n]
        col = SEARCH if v != "1" else DANGER
        fr.txt(x + dx, y + dy, v, fs, col, a2, ha="left", weight="bold" if v != "1" else "normal")

    # 其余分支暗淡（试过就弃）
    dim = ease(seg(t, 7.6, 8.6))
    if dim > 0.02:
        for n, x, y, p in NODES:
            if n in ("A", "B", "B1", "B1a"): continue
            fr.dot(x, y, 6.5, DANGER, al * dim * 0.35)

    fr.txt(W / 2, 196, "800 次模拟 → 790 个局面、最深 8 层", 22, WHITE,
           fade(t, 7.8, 8.5, 11.2, 12.0))
    fr.txt(W / 2, 146, "75% 的边，一辈子只走过一次", 25, DANGER,
           fade(t, 8.2, 8.9, 11.2, 12.0))
    fr.txt(W / 2, 86, "树形为示意；节点数 / 边数 / 深度为实测", 15, GREY,
           fade(t, 8.5, 9.1, 11.2, 12.0))

# ───────────────────────── 幕 4：分布 ─────────────────────────
def s4(t, fr):
    al = fade(t, 0, 0.8, 8.6, 9.4)
    fr.txt(W/2, 662, "⑩ 800 次访问，长成了这个分布", 30, SEARCH, al, weight="bold")
    fr.txt(W/2, 600, "π = N / ΣN", 26, ACCENT, fade(t, 0.6, 1.4, 8.6, 9.4))

    base = 190
    vals = [(470, 300, "399"), (620, 138, "172"), (740, 82, "103"),
            (845, 50, "61"), (945, 65, "65")]
    for k, (x, h, lab) in enumerate(vals):
        g = ease(seg(t, 1.2 + k * 0.16, 2.1 + k * 0.16))
        fr.rect(x - 22, base, 44, h * g, SEARCH, 0.9 * al)
        fr.txt(x, base + h * g + 22, lab, 19, WHITE, al * g)

    g0 = ease(seg(t, 2.6, 3.4))
    fr.rect(1080, base, 44 * g0, 6, DANGER, 0.9 * al)
    fr.txt(1102, base + 30, "200 个点", 17, DANGER, al * g0)
    fr.line(420, base, 1160, base, GREY, 1.2, 0.5 * al)

    fr.txt(215, 470, "冠军 (10,7)", 26, WHITE, fade(t, 3.6, 4.3, 8.6, 9.4))
    fr.txt(215, 415, "N = 399   Q = +0.973", 21, SEARCH, fade(t, 4.0, 4.7, 8.6, 9.4))
    fr.txt(215, 320, "访问过 / 合法点", 20, GREY, fade(t, 4.6, 5.3, 8.6, 9.4))
    fr.txt(215, 262, "5 / 205 = 2.4%", 29, ACCENT, fade(t, 4.9, 5.7, 8.6, 9.4))
    fr.txt(W/2, 96, "柱高为示意；冠军 N=399 与访问点数 5 为实测", 16, GREY,
           fade(t, 5.4, 6.2, 8.6, 9.4))

# ───────────────────────── 幕 5：饥饿 ─────────────────────────
def s5(t, fr):
    al = fade(t, 0, 0.8, 10.6, 11.4)
    fr.txt(W/2, 662, "⑪ 没被看过的那一个，会静默地死掉", 30, DANGER, al, weight="bold")

    cx, cy = 330, 375
    fr.board(cx, cy, n=7, cell=62, al=0.28 * al, lw=1.3)
    gx, gy = fr.cell(3, 3, cx, cy, n=7, cell=62)
    bx, by = fr.cell(3, 4, cx, cy, n=7, cell=62)

    g1 = ease(seg(t, 0.8, 1.5))
    fr.dot(gx, gy, 15 * g1, SEARCH, al)
    fr.dot(bx, by, 15 * g1, DANGER, al)
    fr.rect(gx - 26, gy - 26, 52, 52, "none", al * g1, ec=SEARCH, lw=2, z=7)
    fr.rect(bx - 26, by - 26, 52, 52, "none", al * g1, ec=DANGER, lw=2, z=7)

    g2 = ease(seg(t, 2.2, 3.0))
    if g2 > 0.05:
        fr.line(gx - 14 * g2, gy - 14 * g2, gx + 14 * g2, gy + 14 * g2, DANGER, 3.2, al)
        fr.line(gx - 14 * g2, gy + 14 * g2, gx + 14 * g2, gy - 14 * g2, DANGER, 3.2, al)

    x0 = 700
    fr.txt(x0, 480, "Q = +0.959    N = 0", 26, DANGER, fade(t, 3.2, 4.0, 10.6, 11.4), ha="left")
    fr.txt(x0, 428, "这一手是必胜的", 21, GREY, fade(t, 3.6, 4.3, 10.6, 11.4), ha="left")
    fr.txt(x0, 350, "Q = −0.087    N = 235", 26, SEARCH, fade(t, 4.4, 5.2, 10.6, 11.4), ha="left")
    fr.txt(x0, 298, "它只是被先验压了 98%", 21, GREY, fade(t, 4.8, 5.5, 10.6, 11.4), ha="left")

    fr.txt(W/2, 118, "未访问  →  Q 永远是 0  →  永远排不进 argmax", 22, ACCENT,
           fade(t, 6.0, 7.0, 10.6, 11.4))
    fr.txt(W/2, 46, "它从未上场，所以永远没机会证明自己", 25, ACCENT,
           fade(t, 7.2, 8.4, 10.6, 11.4))

# ───────────────────────── 幕 6：闭环 ─────────────────────────
def s6(t, fr):
    al = fade(t, 0, 0.7)
    items = [("先验 P", 380, 470, PRIOR), ("800 次搜索", 900, 470, SEARCH),
             ("π = N / ΣN", 900, 270, ACCENT), ("蒸馏回网络", 380, 270, PRIOR)]
    for k, (s, x, y, c) in enumerate(items):
        g = ease(seg(t, 0.3 + k * 0.22, 1.0 + k * 0.22))
        fr.txt(x, y, s, 26, c, al * g, weight="bold")

    gA = ease(seg(t, 1.4, 2.2))
    if gA > 0.03:
        fr.arrow(500, 470, 780, 470, GREY, al * gA, 2.0)
        fr.arrow(900, 420, 900, 320, GREY, al * gA, 2.0)
        fr.arrow(780, 270, 500, 270, GREY, al * gA, 2.0)
    gB = ease(seg(t, 2.2, 3.0))
    if gB > 0.03:
        fr.arrow(380, 320, 380, 420, DANGER, al * gB, 2.4)

    fr.txt(W/2, 195, "每一代，搜索的结论都变成下一代的直觉", 20, GREY,
           fade(t, 2.8, 3.6))
    fr.txt(380, 92, "先验决定考场", 32, PRIOR, fade(t, 4.2, 5.4), weight="bold")
    fr.txt(900, 92, "价值决定胜负", 32, SEARCH, fade(t, 5.0, 6.2), weight="bold")

SCENES = [(s1, 6.0), (s_human, 16.0), (s_boom, 10.0), (s_vs, 8.0), (s_phase, 14.0),
          (s_rollout, 12.0), (s_explore, 12.0), (s_net, 12.0), (s_walk, 58.0),
          (s_tree, 12.0), (s_pick, 10.0), (s_selfplay, 11.0), (s_train, 10.0), (s_loop, 14.0)]

def main():
    n = 0
    for fn, dur in SCENES:
        for k in range(int(dur * FPS)):
            fr = Frame()
            fn(k / FPS, fr)
            fr.save(n)
            n += 1
        print(f"{fn.__name__}: {int(dur*FPS)} 帧 → 累计 {n}", flush=True)
    print("TOTAL", n)

if __name__ == "__main__":
    main()
