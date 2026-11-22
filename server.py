#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gomoku v24 MCTS Lab — decide→search 过程可视化后端

设计原则（唯一真相源）:
  1. 直接 importlib 加载 Gomoku-v0_pt_v24.py **原文件**，不复制 search()/decide() 逻辑。
  2. 只在外层包一个 traced wrapper：每次 agent.search(...) 调用时记录
     (深度, 落点, 返回的 v, 节点类型 leaf/node/terminal)，递归调用也走同一个 wrapper。
  3. decide() 本身原样执行（噪声注入 / sim 循环 / prob / 温度采样），所以我们看到的
     就是训练时真实发生的过程。
  4. 自检: 用记录的轨迹在 Python 里重放 count 增量，必须等于 decide 后 count 的真实增量。
     不一致 → selfcheck=false（前端会红字报警），绝不静默。

启动: /Users/kid/AlphaZero_Gomoku_15x15_v24/.venv/bin/python server.py
"""
import os
import sys
import json
import math
import time
import threading
import warnings
import importlib.util

warnings.filterwarnings('ignore')

import numpy as np
import torch
import gym
from flask import Flask, jsonify, request, send_file

HERE = os.path.dirname(os.path.abspath(__file__))

# 优先用仓库自带副本(克隆即可跑); 找不到再回落到本机训练目录
_HERE_PY = os.path.join(HERE, 'vendor', 'Gomoku-v0_pt_v24.py')
_HERE_W = os.path.join(HERE, 'weights', 'weights_it176_v24.pt')
_TRAIN_PY = '/Users/kid/AlphaZero_Gomoku_15x15_v24/Gomoku-v0_pt_v24.py'
_TRAIN_W = '/Users/kid/AlphaZero_Gomoku_15x15_v24/gomoku_checkpoints_v24/weights_it176_v24.pt'

V24_PY = os.environ.get('V24_PY') or (_HERE_PY if os.path.exists(_HERE_PY) else _TRAIN_PY)
V24_W = os.environ.get('V24_W') or (_HERE_W if os.path.exists(_HERE_W) else _TRAIN_W)
PORT = int(os.environ.get('LAB_PORT', '5055'))
N = 15


def _load_v24():
    spec = importlib.util.spec_from_file_location('gomoku_v24_live', V24_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


log = lambda *a: print('[lab]', *a, flush=True)

t0 = time.time()
mod = _load_v24()
mod.device = torch.device('cpu')          # batch=1 时 CPU 比 MPS 快 (5.9 vs 13.9 ms/sim)
env = gym.make('Gomuku-v0')
AGENT = mod.AlphaZeroAgent(
    env=env,
    kwargs={'conv_filters': [128], 'residual_filters': [[128, 128]] * 8},
    sim_count=50, batches=1, batch_size=8, disable_compile=True)
AGENT.net.load_state_dict(torch.load(V24_W, map_location='cpu'))
AGENT.net.to(mod.device)
AGENT.net.eval()
log('v24 module:', V24_PY)
log('weights   :', V24_W)
log('loaded in %.2fs' % (time.time() - t0))

# ── 追踪器 ────────────────────────────────────────────────────────────
_ORIG_SEARCH = AGENT.search
ST = {'cur': None, 'sims': [], 'stack': [], 'paths': []}
LOCK = threading.Lock()
PATH_OF = {}          # s(bytes) -> 从根到该节点的落点序列 ((r,c), ...)
PATH_S = {}           # 落点序列 -> s(bytes), _tree_payload 取父边计数用
ROOT_STATE = {'stones': [], 'player': 1}   # 最近一次 analyze 的根局面(给 /api/node 重演用)


def _traced_search(board, prior_noise=False, last_move=None, player=None):
    """包装 v24 的 search(): 进入记录, 调用原函数, 退出时用状态差判类型。
    v24 的 search 内部递归也调 self.search → 同样走这里。"""
    b = np.asarray(board)
    s = b.tobytes()
    depth = len(ST['stack'])
    if depth == 0:
        ST['cur'] = {'arcs': {}, 'vs': {}, 'kinds': {}}
    lv = {
        'depth': depth,
        's': s,
        'has_pol': s in AGENT.policy,
        'n_before': float(AGENT.count[s].sum()) if s in AGENT.count else 0.0,
        'board': b,
    }
    if depth > 0:
        par = ST['stack'][-1]
        d = np.argwhere((par['board'] == 0) & (b != 0))
        lv['a'] = [int(d[0][0]), int(d[0][1])] if len(d) else None
    ST['stack'].append(lv)
    try:
        v = _ORIG_SEARCH(board, prior_noise, last_move, player)
    finally:
        # 路径 = 栈上每一层"父层选了哪一手"; 必须在 pop 前取(叶子先退出, 父层还没轮到退出)
        lv['path'] = tuple(tuple(e['a']) for e in ST['stack'][1:] if e.get('a'))
        ST['stack'].pop()
    # 退出后判类型: 终局 / 叶子(本次网络展开) / 内部节点
    w = AGENT.winner.get(s)
    if w is not None:
        kind = 'terminal'
    elif (not lv['has_pol']) and (s in AGENT.policy):
        kind = 'leaf'
    else:
        kind = 'node'
    cur = ST['cur']
    cur['vs'][depth] = round(float(v), 6)
    cur['kinds'][depth] = kind
    PATH_OF[s] = lv['path']      # 根=() ; 子=父路径+本层落点
    PATH_S[lv['path']] = s
    if depth > 0:
        cur['arcs'][depth - 1] = lv['a']
    if depth == 0:
        cur['leaf_kind'] = kind
        _mx = max(cur['vs'])
        cur['leaf_v'] = round(float(cur['vs'][_mx]), 6)   # 叶子自身视角(最深层), 不是根返回的 v
        cur['winner'] = int(w) if w is not None else None
        cur['depth'] = max(cur['vs'].keys()) + 1
        ST['sims'].append(cur)
        ST['cur'] = None
    return v


AGENT.search = _traced_search

# ── HTTP ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)


def _board_from_stones(stones):
    b = np.zeros((N, N), dtype=np.int8)
    for it in stones or []:
        try:
            r, c, v = int(it[0]), int(it[1]), int(it[2])
        except Exception:
            r, c, v = int(it['r']), int(it['c']), int(it['v'])
        if 0 <= r < N and 0 <= c < N:
            b[r, c] = v
    return b


def _five_in_row(b):
    for r in range(N):
        for c in range(N):
            v = b[r, c]
            if v == 0:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cnt = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < N and 0 <= cc < N and b[rr, cc] == v:
                    cnt += 1
                    rr += dr
                    cc += dc
                if cnt >= 5:
                    return int(v)
    return 0


@app.route('/')
def index():
    return send_file(os.path.join(HERE, 'static', 'index.html'))


def _tree_payload():
    """把当前 MCTS 树导出成扁平列表: 每个节点 = 从根出发的落点路径 + 它自己的统计 + 出边。
    边格式 [r, c, n, q, p]; 节点 path 用扁平 [r0,c0,r1,c1,...] 省体积。"""
    out = []
    for sk, path in PATH_OF.items():
        pol = AGENT.policy.get(sk)
        if pol is None:          # 没展开过的节点不可能是树节点
            continue
        # N = 进入该节点的次数 = 父节点该边计数(与 decide 的 prob 同一口径);
        # 根没有父边, 用 count 总和。⚠️ v24 只给"内部节点"建 count, 叶子只有 policy。
        n_in = 1
        if path:
            par_s = PATH_S.get(path[:-1])
            par_cnt = AGENT.count.get(par_s) if par_s else None
            if par_cnt is not None:
                a = path[-1]
                n_in = int(par_cnt[a[0], a[1]]) or 1
        else:
            n_in = int(AGENT.count[sk].sum()) if sk in AGENT.count else 1
        cnt = AGENT.count.get(sk)   # ⚠️ 叶子只有 policy, 没有 count
        if cnt is None:             # 叶子 = 被到达 1 次即展开
            out.append({'p': [int(x) for a in path for x in a], 'n': n_in, 'e': []})
            continue
        q = AGENT.q.get(sk)
        edges = []
        for idx in np.nonzero(cnt.reshape(-1) > 0)[0]:
            r, c = np.unravel_index(int(idx), (N, N))
            edges.append([int(r), int(c), int(cnt[r, c]),
                          round(float(q[r, c]), 4), round(float(pol[r, c]), 5)])
        edges.sort(key=lambda e: -e[2])
        out.append({'p': [int(x) for a in path for x in a], 'n': n_in, 'e': edges})
    out.sort(key=lambda o: (len(o['p']), o['p']))
    return out


def _replay(path):
    """从根局面按落点序列重演, 返回 (canonical, 绝对棋盘棋子列表, 轮到谁)。"""
    b = _board_from_stones(ROOT_STATE['stones'])
    player = int(ROOT_STATE['player'])
    cur = (player * b).astype(np.int8)
    p = player
    stones = [list(map(int, st)) for st in ROOT_STATE['stones']]
    for i in range(0, len(path), 2):
        r, c = path[i], path[i + 1]
        (nb, np_), _, _, _ = env.next_step((cur, mod.BLACK), np.array([r, c]))
        cur = (np_ * nb).astype(np.int8)
        stones.append([r, c, int(p)])
        p = -p
    return cur, stones, p


@app.route('/api/node', methods=['POST'])
def node_view():
    """按路径重演到树中任意节点, 返回该节点自己的 225 个候选(P/N/Q)。"""
    d = request.get_json(force=True, silent=True) or {}
    path = [int(x) for x in (d.get('path') or [])]
    with LOCK:
        cur, stones, p = _replay(path)
        s = cur.tobytes()
        if s not in AGENT.policy:
            return jsonify({'ok': False, 'error': '该节点不在当前树里(换局面或重置树后请重新分析)'})
        pol = AGENT.policy[s]
        cnt = AGENT.count.get(s)                    # 叶子无 count → 全 0
        if cnt is None:
            cnt = np.zeros((N, N), dtype=float)
        q = AGENT.q.get(s)
        if q is None:
            q = np.zeros((N, N), dtype=float)
        # N = 进入次数 = 父边计数(与树里、与 decide 的 prob 同口径)
        if path:
            par_c, _, _ = _replay(path[:-2])
            par_cnt = AGENT.count.get(par_c.tobytes())
            n_in = int(par_cnt[path[-2], path[-1]]) if par_cnt is not None else 1
        else:
            n_in = int(cnt.sum()) or 1
        occ = np.zeros((N, N), dtype=np.int8)
        for r, c, v in stones:
            occ[r, c] = v
        return jsonify({
            'ok': True, 'path': path, 'player': int(p), 'depth': len(path) // 2,
            'n_total': n_in,
            'stones': [[int(r), int(c), int(v)] for r, c, v in stones],
            'policy': [round(float(x), 6) for x in pol.reshape(-1)],
            'count': [int(x) for x in cnt.reshape(-1)],
            'q': [round(float(x), 5) for x in q.reshape(-1)],
            'legal': [0 if occ[r, c] else 1 for r in range(N) for c in range(N)],
        })


@app.route('/api/status')
def status():
    return jsonify({
        'ok': True,
        'v24_py': V24_PY,
        'weights': os.path.basename(V24_W),
        'device': str(mod.device),
        'tree_nodes': len(AGENT.policy),
        'board_cache': len(AGENT.count),
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    with LOCK:
        AGENT.reset_mcts()
        ST['sims'] = []
        ST['stack'] = []
        ST['paths'] = []
        ST['cur'] = None
        PATH_OF.clear()
        PATH_S.clear()
    return jsonify({'ok': True, 'tree_nodes': 0})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    d = request.get_json(force=True, silent=True) or {}
    stones = d.get('stones') or []
    sim = int(d.get('sim', 50))
    sim = max(1, min(sim, 3000))
    noise = bool(d.get('noise', False))
    reuse = bool(d.get('reuse', True))
    tau = float(d.get('tau', 0.0))
    seed = d.get('seed')

    b = _board_from_stones(stones)
    w5 = _five_in_row(b)
    if w5:
        return jsonify({'ok': False, 'error': '该局面已有五连(%s胜)，无搜索意义' % ('黑' if w5 == 1 else '白')})

    nb = int((b == 1).sum())
    nw = int((b == -1).sum())
    player = int(d.get('player') or (1 if nb == nw else -1))
    last_move = d.get('last_move')

    with LOCK:
        if not reuse:
            AGENT.reset_mcts()
        AGENT.sim_count = sim
        AGENT.c_init = float(d.get('c_init', 1.25))
        AGENT.c_base = float(d.get('c_base', 19652.))
        AGENT.prior_exploration_fraction = float(d.get('eps', 0.25))
        mod.DIRICHLET_ALPHA = float(d.get('alpha', 0.03))

        canon = (player * b).astype(np.int8)
        s = canon.tobytes()
        n_pre = float(AGENT.count[s].sum()) if s in AGENT.count else 0.0
        count_snap = (AGENT.count[s].copy() if s in AGENT.count
                      else np.zeros((N, N), dtype=float))
        q_snap = (AGENT.q[s].copy() if s in AGENT.q else np.zeros((N, N), dtype=float))

        # 原始先验(不含噪声/搜索), 给 policy 棋盘用 —— 3ms
        pis_raw, vs_raw = AGENT.predict(
            canon[None], last_moves=(np.array([last_move]) if last_move is not None else None),
            players=player)
        pi_raw = np.exp(np.asarray(pis_raw[0])).reshape(N, N)
        v_raw = float(np.asarray(vs_raw[0]).reshape(-1)[0])
        legal = (b == 0).astype(np.float32)
        pi_show = pi_raw * legal
        pi_show = pi_show / pi_show.sum() if pi_show.sum() > 0 else legal / legal.sum()

        ROOT_STATE['stones'] = [list(map(int, st)) for st in stones]
        ROOT_STATE['player'] = player
        if seed is not None:
            np.random.seed(int(seed))
        t1 = time.time()
        ST['sims'] = []
        ST['stack'] = []
        ST['paths'] = []
        ST['cur'] = None
        loc, prob = AGENT.decide((b, player), return_prob=True, noise=noise,
                                 temperature=tau, greedy=(tau <= 1e-2),
                                 last_move=(np.array(last_move) if last_move else None))
        dt = time.time() - t1
        sims = ST['sims']

        cnt = AGENT.count[s]
        q = AGENT.q[s]
        pol_now = AGENT.policy[s]
        n_post = float(cnt.sum())

        # ── 自检: 用轨迹重放 count 增量, 必须等于真实增量 ──
        # 只有每条模拟路径的第 0 条弧(根→子)才增加根节点的 count
        replay = np.zeros((N, N), dtype=float)
        n_visits = 0
        for sm in sims:
            a = sm['arcs'].get(0)
            if a is not None:
                replay[a[0], a[1]] += 1
                n_visits += 1
        truth = cnt - count_snap
        selfcheck = bool(np.array_equal(replay, truth) and n_visits == int(truth.sum()))

        # ── 根节点子节点树(浅层, 供下钻) ──
        kids = {}
        for idx in np.nonzero(cnt.reshape(-1) > 0)[0]:
            r, c = np.unravel_index(int(idx), (N, N))
            cs = AGENT.count[s]
            # 子局面的 canonical bytes: 用 env.next_step 推
            (nb_, np_), _, _, _ = env.next_step((canon, mod.BLACK), np.array([int(r), int(c)]))
            sk = (np_ * nb_).tobytes()
            kk = []
            if sk in AGENT.count:
                cc = AGENT.count[sk]
                for j in np.nonzero(cc.reshape(-1) > 0)[0]:
                    r2, c2 = np.unravel_index(int(j), (N, N))
                    kk.append([int(r2), int(c2), int(cc[r2, c2]), round(float(AGENT.q[sk][r2, c2]), 4),
                               round(float(AGENT.policy[sk][r2, c2]), 5)])
                kk.sort(key=lambda x: -x[2])
            kids['%d,%d' % (r, c)] = {
                'n': int(cs[r, c]), 'q': round(float(q[r, c]), 4),
                'p': round(float(pol_now[r, c]), 5),
                'total': int(cc.sum()) if sk in AGENT.count else 0,
                'kids': kk[:6],
            }

        out = {
            'ok': True,
            'player': player,
            'sim': sim,
            'noise': noise,
            'tau': tau,
            'chosen': [int(loc[0]), int(loc[1])],
            'prob': [round(float(x), 6) for x in prob.reshape(-1)],
            'policy_raw': [round(float(x), 6) for x in pi_show.reshape(-1)],
            'policy_now': [round(float(x), 6) for x in pol_now.reshape(-1)],
            'legal': [int(x) for x in legal.reshape(-1)],
            'count': [int(x) for x in cnt.reshape(-1)],
            'q': [round(float(x), 5) for x in q.reshape(-1)],
            'count_pre': [int(x) for x in count_snap.reshape(-1)],
            'q_pre': [round(float(x), 5) for x in q_snap.reshape(-1)],
            'root': {'n_pre': n_pre, 'n_post': n_post, 'v_raw': round(v_raw, 4),
                     'v_root': round(float(sims[0]['vs'][0]), 4) if sims else None},
            'sims': [{'a': [sm['arcs'][k] for k in sorted(sm['arcs'])],
                      'v': [sm['vs'][k] for k in sorted(sm['vs'])],
                      'k': [sm['kinds'][k] for k in sorted(sm['kinds'])],
                      'leaf_v': sm['leaf_v'], 'w': sm['winner']} for sm in sims],
            'kids': kids,
            'tree': _tree_payload(),
            'selfcheck': selfcheck,
            'replay_delta': float(replay.sum()),
            'truth_delta': float(truth.sum()),
            'ms': round(dt * 1000),
            'ms_per_sim': round(dt * 1000 / max(1, sim), 2),
            'c': {'c_init': AGENT.c_init, 'c_base': AGENT.c_base,
                  'eps': AGENT.prior_exploration_fraction, 'alpha': mod.DIRICHLET_ALPHA},
        }
    log('analyze stones=%d player=%+d sim=%d noise=%s reuse=%s -> %s  %dms (%.1f/sim) selfcheck=%s'
        % (len(stones), player, sim, noise, reuse, out['chosen'], out['ms'], out['ms_per_sim'],
           selfcheck))
    return jsonify(out)


if __name__ == '__main__':
    log('http://127.0.0.1:%d' % PORT)
    app.run(host='127.0.0.1', port=PORT, threaded=True, debug=False)
