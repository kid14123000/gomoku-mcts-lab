#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gomoku MCTS Lab — decide→search 过程可视化后端

设计原则（唯一真相源）:
  1. 直接 importlib 加载被追踪的训练脚本**原文件**，不复制 search()/decide() 逻辑。
  2. 只在外层包一个 traced wrapper：每次 agent.search(...) 调用时记录
     (深度, 落点, 返回的 v, 节点类型 leaf/node/terminal)，递归调用也走同一个 wrapper。
  3. decide() 本身原样执行（噪声注入 / sim 循环 / prob / 温度采样），所以我们看到的
     就是训练时真实发生的过程。
  4. 自检: 用记录的轨迹在 Python 里重放 count 增量，必须等于 decide 后 count 的真实增量。
     不一致 → selfcheck=false（前端会红字报警），绝不静默。

启动: ./run.sh   (或 <训练项目>/.venv/bin/python server.py)
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

# 优先用仓库自带副本(克隆即可跑); 找不到再回落到本机训练目录(~ 展开, 不写死用户名)
_HERE_PY = os.path.join(HERE, 'vendor', 'Gomoku-v0_pt_v24.py')
_HERE_W = os.path.join(HERE, 'weights', 'weights_it140_v24.pt')
_TRAIN_PY = os.path.expanduser('~/AlphaZero_Gomoku_15x15_v24/Gomoku-v0_pt_v24.py')
# 注意: 默认权重必须与仓库那份同一条训练线(v24c_run1)。旧的 gomoku_checkpoints_v24/ 是另一条线,
# 同名文件内容不同(实测 it140 两处 tensor 签名不同), 指错会静默换掉整条训练线的基线。
_TRAIN_W = os.path.expanduser(
    '~/AlphaZero_Gomoku_15x15_v24/gomoku_checkpoints_v24_v24c_run1/weights_it140_v24.pt')

# 棋谱/题目: 默认复用旧项目那份(读写同一个文件, 避免两套数据分叉), 可用 POSITIONS_FILE 覆盖
_OLD_POS = os.path.expanduser('~/AlphaZero_Gomoku_15x15_v24/web/positions.json')
POS_FILE = os.environ.get('POSITIONS_FILE') or (_OLD_POS if os.path.exists(_OLD_POS)
                                                else os.path.join(HERE, 'positions.json'))
_POS_LOCK = threading.Lock()
_POS_BAKED = [False]

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
NET_KW = {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8}
AGENT = mod.AlphaZeroAgent(
    env=env, kwargs=dict(NET_KW),
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
        # ── 选择快照: 此刻父节点的 count/q 仍是"选点之前"的状态(更新要等递归返回) ──
        sp = par['s']
        if lv['a'] is not None and sp in AGENT.count:
            cnt, qq, pol, val = AGENT.count[sp], AGENT.q[sp], AGENT.policy[sp], AGENT.valid[sp]
            cs = float(cnt.sum())
            coef = (AGENT.c_init + np.log1p((1 + cs) / AGENT.c_base)) * math.sqrt(cs) / (1. + cnt)
            ub = np.where(val > 0, qq + coef * pol, -np.inf)
            a = (lv['a'][0], lv['a'][1])
            pu = float(ub[a])
            legal_ub = ub[val > 0]
            tmp = ub.copy(); tmp[a] = -np.inf
            tf = tmp.reshape(-1)
            j = int(np.argmax(tf))
            lv['sel'] = {
                'cs': round(cs, 1), 'n': int(cnt[a]), 'q': round(float(qq[a]), 5),
                'p': round(float(pol[a]), 6), 'coef': round(float(coef[a]), 5),
                'u': round(float(coef[a] * pol[a]), 5), 'puct': round(pu, 5),
                'rank': int((legal_ub > pu).sum()) + 1, 'legal': int((val > 0).sum()),
                'second': [int(j // N), int(j % N), round(float(tf[j]), 5)] if np.isfinite(tf[j]) else None,
            }
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
    if 'sel' in lv:
        ST['cur'].setdefault('sel', {})[depth] = lv['sel']
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

# 对比用: 每个权重一个"不追踪"的 agent(避免污染主树), 按路径缓存
POOL = {}
W_DIRS = [os.path.join(HERE, 'weights'),
          os.path.expanduser('~/AlphaZero_Gomoku_15x15_v24/gomoku_checkpoints_v24')]


def _agent_for(path):
    a = POOL.get(path)
    if a is None:
        if not os.path.exists(path):
            raise ValueError('权重不存在: %s' % path)
        a = mod.AlphaZeroAgent(env=env, kwargs=dict(NET_KW), sim_count=50,
                               batches=1, batch_size=8, disable_compile=True)
        a.net.load_state_dict(torch.load(path, map_location='cpu'))
        a.net.to(mod.device)
        a.net.eval()
        POOL[path] = a
        log('pool + %s (共 %d 个)' % (os.path.basename(path), len(POOL)))
    return a


# ── HTTP ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)


@app.errorhandler(Exception)
def _json_err(e):
    import traceback
    traceback.print_exc()
    from werkzeug.exceptions import HTTPException
    code = e.code if isinstance(e, HTTPException) else 500
    return jsonify({'ok': False, 'error': '%s: %s' % (type(e).__name__, e)}), code


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


def _load_pos():
    if not os.path.exists(POS_FILE):
        return []
    try:
        with open(POS_FILE, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception as e:
        log('positions 读取失败: %s' % e)
        return []


def _save_pos(items):
    # 首次写入前留一份 .bak(这份文件属于训练项目, 别让它被一次手误覆盖掉)
    if not _POS_BAKED[0] and os.path.exists(POS_FILE):
        try:
            import shutil
            shutil.copy2(POS_FILE, POS_FILE + '.bak')
            _POS_BAKED[0] = True
            log('已备份 %s → %s.bak' % (POS_FILE, POS_FILE))
        except Exception as e:
            log('备份失败(继续): %s' % e)
    tmp = POS_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    os.replace(tmp, POS_FILE)


def _pos_brief(it):
    meta = it.get('meta') or {}
    st = it.get('stones') or []
    return {
        'name': it.get('name'), 'tag': it.get('tag') or '', 'created': it.get('created') or '',
        'n_stones': len(st), 'has_answer': bool(it.get('answer')), 'score': it.get('score'),
        'winner': meta.get('winner_text'), 'weights': [meta.get('weight_black'), meta.get('weight_white')],
        'sim': meta.get('sim_count'), 'moves': meta.get('moves'), 'version': meta.get('version'),
    }


@app.route('/api/positions')
def positions_list():
    items = _load_pos()
    q = (request.args.get('q') or '').strip().lower()
    tag = (request.args.get('tag') or '').strip()
    ha = request.args.get('has_answer') == '1'
    out = [_pos_brief(it) for it in reversed(items)]      # 新的在前
    if q:
        out = [o for o in out if q in (o['name'] or '').lower() or q in (o['created'] or '')]
    if tag:
        out = [o for o in out if (o['tag'] or '') == tag]
    if ha:
        out = [o for o in out if o['has_answer']]
    return jsonify({'ok': True, 'file': POS_FILE, 'total': len(items), 'items': out[:300]})


@app.route('/api/positions/get')
def positions_get():
    name = request.args.get('name') or ''
    it = next((x for x in _load_pos() if x.get('name') == name), None)
    if it is None:
        return jsonify({'ok': False, 'error': '找不到: %s' % name})
    return jsonify({'ok': True, 'item': it})


@app.route('/api/positions/save', methods=['POST'])
def positions_save():
    d = request.get_json(force=True, silent=True) or {}
    name = (d.get('name') or '').strip()
    stones = d.get('stones') or []
    if not name or not stones:
        return jsonify({'ok': False, 'error': '需要名称和棋子'})
    def _rcv(s):
        if isinstance(s, dict):
            return int(s['r']), int(s['c']), int(s.get('v', 0))
        return int(s[0]), int(s[1]), int(s[2])
    stones = [dict(zip('rcv', _rcv(s))) for s in stones if 0 <= _rcv(s)[0] < N and 0 <= _rcv(s)[1] < N]

    def _rc(a):
        if isinstance(a, dict):
            return int(a['r']), int(a['c'])
        return int(a[0]), int(a[1])
    answer = d.get('answer')
    if answer:
        answer = [dict(zip('rc', _rc(a))) for a in answer][:20]
    score = d.get('score')
    try:
        score = None if score is None else max(-1.0, min(1.0, float(score)))
    except Exception:
        score = None
    meta = d.get('meta') if isinstance(d.get('meta'), dict) else None
    if meta and len(json.dumps(meta, ensure_ascii=False)) > 800:
        meta = None
    with _POS_LOCK:
        items = _load_pos()
        items = [x for x in items if x.get('name') != name]      # 同名覆盖
        items.append({'name': name, 'tag': (d.get('tag') or '').strip(), 'stones': stones,
                      'answer': answer, 'score': score, 'meta': meta,
                      'created': time.strftime('%Y-%m-%d %H:%M')})
        _save_pos(items)
    log('positions 保存 %r (%d 子, tag=%s)' % (name, len(stones), d.get('tag') or ''))
    return jsonify({'ok': True, 'total': len(items), 'file': POS_FILE})


@app.route('/api/positions/delete', methods=['POST'])
def positions_delete():
    d = request.get_json(force=True, silent=True) or {}
    name = d.get('name') or ''
    with _POS_LOCK:
        items = _load_pos()
        before = len(items)
        items = [x for x in items if x.get('name') != name]
        if len(items) == before:
            return jsonify({'ok': False, 'error': '找不到: %s' % name})
        _save_pos(items)
    log('positions 删除 %r' % name)
    return jsonify({'ok': True, 'total': len(items)})


@app.route('/api/weights')
def weights_list():
    import glob
    out = []
    for d in W_DIRS:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, '*.pt')):
            st = os.stat(f)
            out.append({'path': f, 'name': os.path.basename(f),
                        'mb': round(st.st_size / 1048576, 1), 'mtime': int(st.st_mtime)})
    out.sort(key=lambda x: -x['mtime'])
    return jsonify({'ok': True, 'current': V24_W, 'weights': out[:60]})


@app.route('/api/compare', methods=['POST'])
def compare():
    """同一局面, 多副权重各自 decide 一次, 返回 policy/访问/选点 供并排对比。
    走独立的未追踪 agent + 每次 reset_mcts, 不碰主页面的树。"""
    d = request.get_json(force=True, silent=True) or {}
    stones = d.get('stones') or []
    sim = max(1, min(int(d.get('sim', 50)), 800))
    paths = [str(x) for x in (d.get('weights') or [])][:4]
    if len(paths) < 2:
        return jsonify({'ok': False, 'error': '至少选两副权重'})
    b = _board_from_stones(stones)
    nb, nw = int((b == 1).sum()), int((b == -1).sum())
    player = int(d.get('player') or (1 if nb == nw else -1))
    lm = d.get('last_move')
    # ⚠️ decide 内部会再包一层 np.array([last_move]) → 这里必须给 1 维 (2,), 给 (1,2) 会变 3 维
    lm_dec = np.array(lm) if lm else None
    lm_pred = np.array([lm]) if lm else None
    canon = (player * b).astype(np.int8)
    out = []
    with LOCK:
        for p in paths:
            try:
                a = _agent_for(p)
            except Exception as e:
                return jsonify({'ok': False, 'error': str(e)})
            a.sim_count = sim
            a.reset_mcts()
            t0 = time.time()
            loc, prob = a.decide((b, player), return_prob=True, noise=False,
                                 temperature=0.0, greedy=True, last_move=lm_dec)
            dt = (time.time() - t0) * 1000
            pis, vs = a.predict(canon[None], last_moves=lm_pred, players=player)
            pi = np.exp(np.asarray(pis[0])).reshape(N, N) * (b == 0)
            pi = pi / pi.sum() if pi.sum() > 0 else pi
            s = canon.tobytes()
            out.append({
                'name': os.path.basename(p), 'path': p,
                'v_root': round(float(np.asarray(vs[0]).reshape(-1)[0]), 4),
                'chosen': [int(loc[0]), int(loc[1])],
                'prob': [round(float(x), 6) for x in prob.reshape(-1)],
                'policy': [round(float(x), 6) for x in pi.reshape(-1)],
                'count': [int(x) for x in a.count[s].reshape(-1)] if s in a.count else [0] * (N * N),
                'q': [round(float(x), 5) for x in a.q[s].reshape(-1)] if s in a.q else [0.0] * (N * N),
                'ms': round(dt),
            })
    log('compare %s sim=%d -> %s' % ([o['name'] for o in out], sim,
                                     [o['chosen'] for o in out]))
    return jsonify({'ok': True, 'player': player, 'sim': sim, 'items': out})


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
                      'sel': [sm.get('sel', {}).get(k) for k in range(1, max(sm['vs']) + 1)],
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
