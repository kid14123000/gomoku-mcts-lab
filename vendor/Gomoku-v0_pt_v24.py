"""
五子棋 AlphaZero 训练 v18（v12 全量复制 + ch2 视角颜色通道修复）

════════════════════════════════════════════════════════════
v6 → v8 唯一改动（2026-08-07，基于色盲/重黑轻白诊断）：

[根因] v6 单通道输入（canonical: 1=当前玩家 -1=对手）导致颜色偏置:
       网络对"黑方威胁"敏感、对"白方威胁"钝感（颜色对称性测试实测误差 0.7-1.1）
       实战表现为: 活四不堵（白方威胁识别弱）、败局识别全红

[改动] 输入升维 1ch → 4ch（AlphaZero 式 feature planes）:
       ch0: 当前玩家棋子 | ch1: 对手棋子 | ch2: 常数1(当前视角) | ch3: 上一步位置
       颜色翻转 = 交换 ch0/ch1 → 颜色对称性结构性满足，消除"重黑轻白"

[保持不变] 网络主干(128ch×8res)/policy头/value头/训练配置/MCTS参数/增强/数据格式 全同 v6
[目录]    gomoku_checkpoints_v8（独立，避免覆盖 v6）

════════════════════════════════════════════════════════════
v1 → v2 关键改动（2026-08-01，基于一周诊断）：

[根因] v1 的 value 头对"连子=胜负"判断退化（活四/五连盲测全反、
       实战不拦截连五、开局 50% 下边角），loss 却持续下降。
       根因 = 数据多样性不足 + 探索参数错误。

[改动]
1. ✅ temperature 调度（v1 完全没有！）
   - 前 TEMP_HIGH_MOVES(=15) 手: temperature=1 按概率采样（鼓励探索）
   - 中段: temperature 线性下降到 0.25
   - 终局(最后 10 手): temperature→0 贪心（保证终局质量）
   - 原版 AlphaZero: 前30手 T=1, 之后 T→0
2. ✅ Dirichlet alpha 从 1/合法数(≈0.0045) 固定为 0.03
   - 标准 AlphaZero 用 0.03；v1 的 alpha 太小探索几乎无效
3. ✅ 动态模拟量重新设计（v1: 50/200/400/800 每代固定）
   - 前 30 代: 200 模拟（快、多样，快速积累棋谱）
   - 31-60 代: 400 模拟
   - 61+ 代: 800 模拟（精算收敛）
4. ✅ prior_exploration_fraction 保持 0.25（标准值）
5. ✅ c_init=1.25, c_base=19652 保持（标准值）
6. ✅ 经验回放池：保留最近 N=5 代数据混合训练（v1 只训当前代）
   - 增加数据多样性，防止 value 头在单一棋风里自嗨
7. ✅ 训练轮数自适应：数据多则多训（v1 固定 32 batches×512）
8. ✅ 开局强制多样性：前 3 手强制落子在距离天元 4 格内（含噪音）
   - 防止网络永远从边角开局（v1 的 50% 边角开局是病）
9. ✅ MCTS 根节点噪音始终开启（self-play 时 noise=True）

[保留] 网络结构(128ch×8res)、8方向增强、checkpoint 格式、
      工兵并行框架、FP16/CUDA Graph 优化全部保留。
════════════════════════════════════════════════════════════
"""
import sys, os, time, math, json, glob, pickle, itertools, random, hashlib
import logging
import collections
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('使用设备: {}'.format(device))

import boardgame2
from boardgame2 import BLACK, WHITE

# ─────────────────────────────────────────────
# 版本标识（v1 的存档/恢复不通用，避免误恢复）
# ─────────────────────────────────────────────
VERSION = "v24"

# ── v8 输入升维: 4 feature planes ──
# ch0: 当前玩家棋子 | ch1: 对手棋子 | ch2: 常数1(当前视角) | ch3: 上一步位置
INPUT_CHANNELS = 4

def make_input(canonical, last_move=None, player=None):
    """1ch canonical board (15,15) → 4ch (4,15,15) float32
    canonical: 1=当前玩家 -1=对手 0=空; last_move: (r,c) tuple/array 或 None
    player: 当前玩家颜色 (1=黑 / -1=白), 用于 ch2 视角平面; None 则默认黑 (兼容旧调用)
    """
    c = np.asarray(canonical)
    out = np.zeros((INPUT_CHANNELS,) + c.shape, dtype=np.float32)
    out[0] = (c > 0)
    out[1] = (c < 0)
    # v24 (2026-08-30, 从零+ch2恒1): ch2 恒定 1.0, 撤销 v18 执色编码
    # 理由: ch2 编码执色 → 空盘捷径(网络学"我是黑→赢/我是白→输")→ value 极端化
    # ch2 恒 1 = 无执色信息 → 黑白视角输入对称 → 捷径物理不存在, 网络从棋盘学
    out[2] = 1.0
    if last_move is not None:
        out[3][int(last_move[0]), int(last_move[1])] = 1.0
    return out


def _to_planes(batch_1ch, players=None):
    """(15,15) 或 (N,15,15) 1ch canonical → (N,4,15,15) tensor（诊断探针用）
    players: (N,) 或标量 当前玩家颜色(1=黑/-1=白), 用于 ch2; None 默认黑 (兼容旧调用)"""
    b = np.asarray(batch_1ch)
    single = b.ndim == 2
    if single:
        b = b[None]
    inp = np.zeros((b.shape[0], INPUT_CHANNELS) + b.shape[1:], dtype=np.float32)
    inp[:, 0] = b > 0
    inp[:, 1] = b < 0
    if players is None:
        inp[:, 2] = 1.0  # v24: ch2 恒定 1
    else:
        pl = np.asarray(players)
        if pl.ndim == 0:
            inp[:, 2] = 1.0  # v24: ch2 恒定 1, 无执色信息
        else:
            inp[:, 2] = 1.0  # v24: ch2 恒定 1, 无执色信息
    x = torch.as_tensor(inp, device=device)
    return x[0] if single else x

# ── v2 新参数 ──
TEMP_HIGH_MOVES = 16       # 前16手 temperature=1 (2026-08-23 用户拍板: 15x15推荐)
TEMP_LOW_MOVES = 10        # 保留定义(不再使用, 旧调度废弃)
DIRICHLET_ALPHA = 0.03     # v2: 固定0.03 (v1是 1/N ≈ 0.0045)
REPLAY_POOL_SIZE = 5       # 保留最近5代数据混合训练

# ─────────────────────────────────────────────
# 网络（与 v1 相同）
# ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AlphaZeroNet(nn.Module):
    """v6 网络: opt/v1 式极简头 (3×3 感受野直出), 主干 8×ResBlock
    policy: conv(128→1, k3) → bn → log_softmax 直出 (225)
    value:  conv(128→1, k3) → bn → fc(225→1) 单层
    ⚠️ policy 输出 log 概率 (v2 learn 需要), 结构与 v1/opt 相同
    """
    def __init__(self, conv_channels=128, residual_blocks=8, board_size=15):
        super().__init__()
        self.board_size = board_size
        self.conv_input = nn.Conv2d(INPUT_CHANNELS, conv_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(conv_channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(conv_channels, conv_channels, conv_channels)
            for _ in range(residual_blocks)
        ])
        # Policy head (v1式): 3×3 直出单通道
        # v20 (2026-08-27, 用户手术 v2): policy 头加隐藏层 — 保留式双分支
        # 旧识别模块(继承): conv(128→1, 3×3, pad=1) → BN(1) → ReLU  [it234 权重完整继承]
        # 新通道(随机):     conv(128→1, 1×1) → BN(1) → ReLU            [新增识别通道]
        # 组合层(随机):     cat 2通道 → flatten(450) → Linear(450→225) → log_softmax
        # 原 v19 极简头直出 (~1.2K 参, 无跨位置组合); 新头 ~103K 参, Linear 提供全局组合
        self.policy_conv = nn.Conv2d(conv_channels, 1, 3, padding=1, bias=False)   # 继承 (识别模块)
        self.policy_bn = nn.BatchNorm2d(1)                                        # 继承
        self.policy_conv2 = nn.Conv2d(conv_channels, 1, 1, padding=0, bias=False) # 随机 (新通道)
        self.policy_bn2 = nn.BatchNorm2d(1)                                       # 随机
        self.policy_fc = nn.Linear(board_size * board_size * 2, board_size * board_size)  # 随机 (组合层)
        # Value head (v19, 2026-08-26): conv(128→1,3×3) + 隐藏层 fc(225→128) → fc(128→1)
        # 主流 AlphaZero/Leela 式带隐藏层 value 头 (v18 为 225→1 单 FC 直出)
        # value_conv/value_bn 与 v18 同形状 → 继承权重直接复用, 仅 fc 层为新结构
        self.value_conv = nn.Conv2d(conv_channels, 1, 3, padding=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        # Policy head: 1通道 → (N, 1, 15, 15) → (N, 225) → log_softmax
        # policy (v20 保留式): 旧3×3识别分支 + 新1×1通道 → cat → flatten(450) → Linear(450→225) → log_softmax
        p1 = F.relu(self.policy_bn(self.policy_conv(x)))     # (N,1,15,15) 继承识别器
        p2 = F.relu(self.policy_bn2(self.policy_conv2(x)))   # (N,1,15,15) 新通道
        p = torch.cat([p1, p2], dim=1)                       # (N,2,15,15)
        p = p.reshape(p.shape[0], -1)                        # (N, 450)
        p = F.log_softmax(self.policy_fc(p), dim=1)
        # Value head (v19): conv → flatten(225) → fc1(225→128) → relu → fc2(128→1) → tanh
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.shape[0], -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v


def to_tensor(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).unsqueeze(1)


def to_numpy(x):
    return x.detach().cpu().numpy()


# ─────────────────────────────────────────────
# Agent（v2 新增 temperature / alpha / 开局约束）
# ─────────────────────────────────────────────
class AlphaZeroAgent:
    def __init__(self, env, batches=1, batch_size=4096,
                 kwargs=None, sim_count=200,
                 c_init=1.25, c_base=19652., prior_exploration_fraction=0.25,
                 lr=0.001, disable_compile=False):
        if kwargs is None:
            kwargs = {}
        self.env = env
        self.board = np.zeros_like(env.board)
        self.batches = batches
        self.batch_size = batch_size

        conv = kwargs.get('conv_filters', [128])[0]
        resid = len(kwargs.get('residual_filters', [[128, 128]] * 8))
        raw_net = AlphaZeroNet(conv_channels=conv, residual_blocks=resid).to(device)

        self.raw_net = raw_net
        # v6: 恢复 torch.compile 提速 (速度是 v2 的 3.5-4.5 倍差距主因)
        # 修复: learn 的 scaler 分支已补 zero_grad (原缺失导致 CUDAGraph 梯度冲突)
        if not disable_compile and device.type == 'cuda':
            try:
                self.net = torch.compile(raw_net, mode='reduce-overhead', fullgraph=False)
                logging.info('torch.compile 已启用 (reduce-overhead)')
            except Exception as e:
                logging.warning('torch.compile 失败: {}, 使用原始网络'.format(e))
                self.net = raw_net
        else:
            self.net = raw_net
            logging.info('torch.compile 禁用或不可用，使用原始网络')

        # v12: BN 参数(weight/bias)单独分组, weight_decay=0 (防 BN γ 收缩死亡螺旋);
        #      其余参数保持 v8 原样 weight_decay=1e-4。其余训练配置与 v8 完全一致。
        bn_params = []
        other_params = []
        for name, param in self.raw_net.named_parameters():
            if param.requires_grad:
                if 'bn' in name and ('weight' in name or 'bias' in name):
                    bn_params.append(param)
                else:
                    other_params.append(param)
        self.optimizer = optim.Adam([
            {'params': bn_params, 'weight_decay': 0.0},
            {'params': other_params, 'weight_decay': 1e-4},
        ], lr=lr)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        # v19 (2026-08-26, Kimi 三步走): 阶段1 冻结主干+Policy, 只训新 Value 头
        self.freeze_policy_only = False
        self.freeze_value_only = False

        self.reset_mcts()
        self.sim_count = sim_count
        self.c_init = c_init
        self.c_base = c_base
        self.prior_exploration_fraction = prior_exploration_fraction
        # v2: 开局中心约束（search 与 decide 共用）
        self.center_only = False
        n = self.board.shape[0]
        center = n // 2
        self.center_mask = np.zeros((n, n), dtype=bool)
        for r in range(n):
            for c in range(n):
                self.center_mask[r, c] = (abs(r - center) + abs(c - center) <= 4)

    def reset_mcts(self):
        def zero_board_factory():
            return np.zeros_like(self.board, dtype=float)
        self.q = collections.defaultdict(zero_board_factory)
        self.count = collections.defaultdict(zero_board_factory)
        self.policy = {}
        self.valid = {}
        self.winner = {}

    @torch.inference_mode()
    def predict_inference(self, board_batch, last_moves=None, players=None):
        """board_batch: (N,15,15) canonical; last_moves: (N,2) 或 None → 内部转 4-plane
        players: (N,) 或标量 当前玩家颜色(1=黑/-1=白), 用于 ch2; None 默认黑 (兼容旧调用)
        ⚠️ 推理固定 eval: 只在 training 模式时切一次 (省每次 predict 的 eval() 全模块遍历开销)"""
        if self.net.training:
            self.net.eval()
        b = np.asarray(board_batch)
        inp = np.zeros((b.shape[0], INPUT_CHANNELS) + b.shape[1:], dtype=np.float32)
        inp[:, 0] = b > 0
        inp[:, 1] = b < 0
        if players is None:
            inp[:, 2] = 1.0  # v24: ch2 恒定 1
        else:
            pl = np.asarray(players)
            if pl.ndim == 0:
                inp[:, 2] = 1.0  # v24: ch2 恒定 1, 无执色信息
            else:
                inp[:, 2] = 1.0  # v24: ch2 恒定 1, 无执色信息
        if last_moves is not None:
            for i, lm in enumerate(last_moves):
                if lm is not None:
                    inp[i, 3, int(lm[0]), int(lm[1])] = 1.0
        tensor = torch.as_tensor(inp, device=device)
        with torch.cuda.amp.autocast():
            p, v = self.net(tensor)
        return to_numpy(p.float()), to_numpy(v.float())

    def predict(self, board_batch, last_moves=None, players=None):
        return self.predict_inference(board_batch, last_moves, players)

    def decide(self, observation, greedy=False, return_prob=False, noise=True,
               temperature=1.0, force_center=False, move_no=0, last_move=None):
        """v2: 支持 temperature 采样 + 开局中心约束
        temperature=1.0: 按概率采样（探索）
        temperature→0:   贪心（接近 argmax）
        force_center: 开局强制落子在天元附近（训练早期防止边角开局）
        last_move: 本局面刚落的子 (r,c)，作为 v8 输入 plane 3（None 则全 0）
        """
        board, player = observation
        canonical_board = player * board
        s = canonical_board.tobytes()
        # ⚠️ 2026-08-18 count污染修复: 每步固定搜 sim_count 次 (论文式, 树可复用但搜索量不打折)
        # 原 while count[s].sum() < sim_count: 当 count 被父节点搜索灌满(>=800)时跳过搜索,
        # 直接用被污染的分布采样 -> 杀点不锁定/训练目标杀点信号弱 (白棋杀棋不精准)
        # ⚠️ 2026-08-29 noise 注入修正(论文式): 根先验加一次 noise, 800 sim 共用
        # 原实现每次 sim 都重新采样 noise (explore 放大 800 倍, value 极端化疑似病根)
        # 做法: noise 在 decide 层生成一次, 写入根先验 policy[s]; search 不再重新采样
        if noise:
            if s not in self.policy:
                # 首次需先推理缓存根节点先验 (search 内部会做, 但这里要拿到 policy 才能混 noise)
                self.search(canonical_board, prior_noise=False, last_move=last_move, player=player)
            alpha = DIRICHLET_ALPHA
            nv = np.random.gamma(alpha, 1., board.shape)
            nv *= self.valid[s]
            nv /= nv.sum()
            self.policy[s] = (1. - self.prior_exploration_fraction) * self.policy[s] \
                             + self.prior_exploration_fraction * nv
        for _ in range(self.sim_count):
            self.search(canonical_board, prior_noise=False, last_move=last_move, player=player)
        prob = self.count[s] / self.count[s].sum()

        # ── v2: 开局中心约束 ──
        if force_center:
            center = board.shape[0] // 2
            prob = prob.copy()
            for r in range(board.shape[0]):
                for c in range(board.shape[1]):
                    if abs(r - center) + abs(c - center) > 4:
                        prob[r, c] = 0.0
            if prob.sum() > 0:
                prob /= prob.sum()

        # ── v2: temperature 采样 ──
        # 2026-08-23 修复: temp<=1e-2 直接 argmax。
        # 原因: temp=1e-3 时 exp(log_p/temp)=exp(log_p*1000) 全下溢→soft全0→0/0=NaN 崩溃
        # 语义: 1e-3 本就是"接近贪心", argmax 等价且数值安全
        if temperature <= 1e-2 or greedy:
            location_index = np.argmax(prob.reshape(-1))
        else:
            # 按 temperature 软化后的分布采样 (temp 较大时 exp 不消失, 安全)
            log_p = np.log(prob + 1e-12)
            soft = np.exp(log_p / temperature)
            soft /= soft.sum()
            location_index = np.random.choice(prob.size, p=soft.reshape(-1))

        location = np.unravel_index(location_index, prob.shape)
        if return_prob:
            return location, prob
        return location

    def set_freeze_policy_only(self, freeze):
        """v20 (2026-08-27, 用户手术): freeze=True → 冻结主干+Value 头, 只训 policy 头参数
        freeze=False → 解冻全网络。只改 requires_grad, 不重建 optimizer
        (optimizer 参数引用固定, requires_grad=False 的参数梯度为 None 自动跳过;
         避免重建导致 GradScaler 状态丢失)。"""
        if freeze == self.freeze_policy_only:
            return
        net = self.raw_net if (hasattr(self, 'raw_net') and self.raw_net is not None) else self.net
        for name, param in net.named_parameters():
            if freeze:
                param.requires_grad = name.startswith('policy_')
            else:
                param.requires_grad = True
        self.freeze_policy_only = freeze
        logging.info('v20 冻结切换: {} (policy_only={})'.format(
            '冻结主干+Value, 只训 Policy 头' if freeze else '解冻全网络', freeze))

    def set_freeze_value_only(self, freeze):
        """v23 (2026-08-29, 重置value头后冻结期): freeze=True → 冻结主干+Policy 头, 只训 value 头参数
        freeze=False → 解冻全网络。只改 requires_grad, 不重建 optimizer
        (optimizer 参数引用固定, requires_grad=False 的参数梯度为 None 自动跳过;
         避免重建导致 GradScaler 状态丢失)。"""
        if freeze == self.freeze_value_only:
            return
        net = self.raw_net if (hasattr(self, 'raw_net') and self.raw_net is not None) else self.net
        for name, param in net.named_parameters():
            if freeze:
                param.requires_grad = name.startswith('value_')
            else:
                param.requires_grad = True
        self.freeze_value_only = freeze
        logging.info('v23 冻结切换: {} (value_only={})'.format(
            '冻结主干+Policy, 只训 Value 头' if freeze else '解冻全网络', freeze))

    def learn(self, dfs):
        self.net.train()
        # v8: 每个 df 补 last_move 列 (从相邻 board diff 推, 不改原数据格式)
        dfs_aug = []
        for d in dfs:
            d = d.reset_index(drop=True)
            last_moves = []
            boards = d['board'].values
            last_moves.append(None)
            for i in range(1, len(d)):
                prev = np.asarray(boards[i - 1])
                cur = np.asarray(boards[i])
                diff = np.argwhere(prev != cur)
                last_moves.append(tuple(diff[0]) if len(diff) else None)
            d['last_move'] = last_moves
            dfs_aug.append(d)
        df = pd.concat(dfs_aug).reset_index(drop=True)
        losses = {'policy_loss': [], 'value_loss': [], 'total_loss': []}
        for batch in range(self.batches):
            indices = np.random.choice(len(df), size=self.batch_size)
            players, boards, probs, winners = (
                np.stack(df.loc[indices, field])
                for field in ['player', 'board', 'prob', 'winner']
            )
            last_moves = df.loc[indices, 'last_move'].values
            canonical_boards = players[:, np.newaxis, np.newaxis] * boards
            vs = (players * winners)[:, np.newaxis]

            # v8: 1ch canonical (N,15,15) → 4-plane input (N,4,15,15)
            inp = np.zeros((len(players), INPUT_CHANNELS) + boards.shape[1:],
                           dtype=np.float32)
            inp[:, 0] = canonical_boards > 0
            inp[:, 1] = canonical_boards < 0
            # v24: ch2 恒定 1, 无执色信息 (撤销 v18 编码)
            inp[:, 2] = 1.0
            for i, lm in enumerate(last_moves):
                if lm is not None:
                    inp[i, 3, int(lm[0]), int(lm[1])] = 1.0

            # 8 方向对称增强 (v8 修正: v6 的 fliplr 对 4D 无效, 反射分支实际失效,
            # 这里用 np.flip(axis=-1) 正确实现水平翻转)
            boards_aug = [inp]
            probs_aug = [probs]
            vs_aug = [vs]
            for reflection in [False, True]:
                rot_start = 1 if not reflection else 0
                for rot_k in range(rot_start, 4):
                    b = np.flip(inp, axis=-1) if reflection else inp
                    p = np.flip(probs, axis=-1) if reflection else probs
                    boards_aug.append(np.rot90(b, rot_k, axes=(2, 3)))
                    probs_aug.append(np.rot90(p, rot_k, axes=(1, 2)))
                    vs_aug.append(vs)
            inp = np.concatenate(boards_aug)
            probs = np.concatenate(probs_aug)
            vs = np.concatenate(vs_aug)

            x = torch.as_tensor(inp, device=device)
            with torch.cuda.amp.autocast():
                pred_p, pred_v = self.net(x)
                # pred_p: (batch, 225) → reshape为 (batch, 15, 15) 与 probs 对齐
                bs = pred_p.shape[0]
                board_sz = int(math.sqrt(pred_p.shape[1]))
                pred_p_b = pred_p.reshape(bs, board_sz, board_sz)
                policy_loss = -(torch.as_tensor(probs, device=device) *
                                pred_p_b).sum(dim=(1, 2)).mean()
                value_loss = F.mse_loss(pred_v, torch.as_tensor(vs, device=device))
                total_loss = policy_loss + value_loss
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()   # v6 修复: scaler 分支也要清梯度 (原代码缺失, 梯度跨batch累积)
            else:
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses['policy_loss'].append(policy_loss.item())
            losses['value_loss'].append(value_loss.item())
            losses['total_loss'].append(total_loss.item())

        self.reset_mcts()
        result = {}
        if losses['policy_loss']:
            result['policy_loss'] = sum(losses['policy_loss']) / len(losses['policy_loss'])
        if losses['value_loss']:
            result['value_loss'] = sum(losses['value_loss']) / len(losses['value_loss'])
        if losses['total_loss']:
            result['total_loss'] = sum(losses['total_loss']) / len(losses['total_loss'])
        return result

    def search(self, board, prior_noise=False, last_move=None, player=None):
        s = board.tobytes()

        if s not in self.winner:
            # canonical board: 1=当前玩家, -1=对手
            # get_winner((board, BLACK)) 检测"值为1的棋子五连"→返回1 = 当前玩家胜
            # 恰好是"当前玩家视角"(与 value 标签 players*winners 一致) → 无需翻转
            self.winner[s] = self.env.get_winner((board, BLACK))
        if self.winner[s] is not None:
            return self.winner[s]

        if s not in self.policy:
            # v18: 若未传 player (外部直接调 search), 用 canonical 无法推颜色 → 默认黑(与旧行为一致)
            p_color = player if player is not None else 1
            pis, vs = self.predict(
                board[np.newaxis],
                last_moves=np.array([last_move]) if last_move is not None else None,
                players=p_color)
            pi = np.exp(np.asarray(pis[0]))  # ⚠️ 修复: net输出log概率, 转概率 (2026-08-10 发现search用log当概率致先验反转)
            if pi.shape != board.shape:
                pi = pi.reshape(board.shape)
            v = float(np.asarray(vs[0]).reshape(-1)[0])
            valid = (board == 0).astype(np.float32)
            masked_pi = pi * valid
            total_masked_pi = np.sum(masked_pi)
            if total_masked_pi <= 0:
                masked_pi = valid
                total_masked_pi = np.sum(masked_pi)
            self.policy[s] = masked_pi / total_masked_pi
            self.valid[s] = valid
            return v

        count_sum = self.count[s].sum()
        coef = (self.c_init + np.log1p((1 + count_sum) / self.c_base)) * \
               math.sqrt(count_sum) / (1. + self.count[s])
        if prior_noise:
            # v2: alpha 固定 0.03（标准 AlphaZero），v1 是 1/N ≈ 0.0045
            alpha = DIRICHLET_ALPHA
            noise = np.random.gamma(alpha, 1., board.shape)
            noise *= self.valid[s]
            noise /= noise.sum()
            prior = (1. - self.prior_exploration_fraction) * \
                    self.policy[s] + \
                    self.prior_exploration_fraction * noise
        else:
            prior = self.policy[s]
        ub = np.where(self.valid[s], self.q[s] + coef * prior, np.nan)
        location_index = np.nanargmax(ub)
        location = np.unravel_index(location_index, board.shape)

        (next_board, next_player), _, _, _ = self.env.next_step(
            (board, BLACK), np.array(location))
        next_canonical_board = next_player * next_board
        # v18 修复 (2026-08-22): next_player 是 boardgame2 的视角翻转标记(传入BLACK恒=-1),
        # 不是绝对颜色! ch2 需要每层翻转的绝对颜色 → 传 -player (不能传 next_player)
        # 否则黑视角层被当成白, 深层 MCTS 评估全错
        next_v = self.search(next_canonical_board, last_move=location, player=-player)
        # v1 原始写法 (server_sync/scripts/Gomoku-v0_pt_opt.py 一致): 已验证正确
        v = next_player * next_v

        self.count[s][location] += 1
        self.q[s][location] += (v - self.q[s][location]) / \
                               self.count[s][location]
        return v


# ─────────────────────────────────────────────
# 自我对弈（v2: temperature 调度 + 开局约束）
# ─────────────────────────────────────────────
def self_play(env, agent, return_trajectory=False, verbose=False,
              visualize=False, temperature_schedule=True, force_center=False):
    # ⚠️ 2026-08-15 修复: 每局开始必须 reset_mcts() (标准 AlphaZero 每局独立搜索树)
    # 之前(书版继承)每局不清理 → 空盘 key 跨局串扰: 局2-10 复用局1搜索分布 →
    # 首手定式 + 探索独立性被破坏 + 偶然错误被放大10倍 (v8 it99+ 每代90%同首手根因)
    agent.reset_mcts()
    if return_trajectory:
        trajectory = []
    observation, _ = env.reset()
    max_moves = env.board.size
    last_move = None
    for step in itertools.count():
        if step >= max_moves:
            break
        board, player = observation

        # ── v2: temperature 调度 ──
        # 2026-08-23 用户拍板修正: 旧调度 total_est=max_moves//2=112 远大于实际对局~45手
        # → 终局段(>=102)永远走不到, 中段线性在45手时仍≈0.74 → 实际全局≈1 随机采样
        # → 15x15推荐: 前16手 temp=1 探索, 之后 temp=1e-3 接近贪心 (棋谱质量↑ 信号↑)
        temp = 1.0
        if temperature_schedule:
            # v24 (2026-08-31 用户拍板): 温和指数衰减, 替代"前16手恒1+突变1e-3"
            # 白棋斜防守 π~30% 被 temp=1 采样压制(落地率仅30%), 信号被稀释;
            # 突变 1e-3 又让中盘过早贪心。指数衰减: 开局保留探索, 中盘渐锐。
            # step: 4→1.00  8→0.61  12→0.37  16→0.22  20→0.14  25→0.07  30→0.05
            if step < 4:
                temp = 1.0
            else:
                temp = max(0.05, math.exp(-(step - 4) / 8.0))

        # ── v2: 开局中心约束（前3手） ──
        fc = force_center and step < 3

        action, prob = agent.decide(observation, return_prob=True,
                                    temperature=temp, force_center=fc,
                                    move_no=step, last_move=last_move)
        if verbose:
            logging.info('第 {} 步：玩家 {}, 动作 {}, temp={:.2f}'.format(
                step, player, action, temp))
        observation, reward, terminated, truncated, info = env.step(action)
        last_move = np.array(action, dtype=int)
        winner = env.get_winner(observation)
        done = terminated or truncated
        if return_trajectory:
            trajectory.append((player, board, prob))
        if done:
            break
    if return_trajectory:
        df_trajectory = pd.DataFrame(trajectory,
                                     columns=['player', 'board', 'prob'])
        df_trajectory['winner'] = winner
        return df_trajectory
    else:
        return winner


# ─────────────────────────────────────────────
# 存档/读档（v2 独立目录，避免覆盖 v1）
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'gomoku_checkpoints_v24')
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, 'weights.pt')
METADATA_PATH = os.path.join(CHECKPOINT_DIR, 'metadata.json')
WORKER_DIR = os.path.join(CHECKPOINT_DIR, 'worker_data')
REPLAY_DIR = os.path.join(CHECKPOINT_DIR, 'replay_pool')  # v2: 经验回放池


def config_hash(train_iterations, train_episodes_per_iteration, batches, batch_size, sim_count, net_kwargs, **extra):
    raw = json.dumps({
        'train_iterations': train_iterations,
        'train_episodes_per_iteration': train_episodes_per_iteration,
        'batches': batches,
        'batch_size': batch_size,
        'sim_count': sim_count,
        'net_kwargs': net_kwargs,
        'version': VERSION,
        **extra,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def save_checkpoint(agent, iteration, config):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if hasattr(agent, 'raw_net') and agent.raw_net is not None:
        save_net = agent.raw_net
    else:
        save_net = agent.net
    torch.save(save_net.state_dict(), WEIGHTS_PATH)
    history_path = os.path.join(CHECKPOINT_DIR, f'weights_it{iteration}_v24.pt')  # v24 标识后缀
    torch.save(save_net.state_dict(), history_path)
    hash_config = {k: v for k, v in config.items() if k != 'losses'}
    metadata = {
        'iteration': iteration + 1,
        'config_hash': config_hash(**hash_config),
        'config': config,
        'version': VERSION,
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info('检查点已保存 (iteration {})'.format(iteration))


def load_checkpoint(agent, config):
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(METADATA_PATH):
        logging.info('未找到 v2 检查点，从头开始训练')
        return None
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    if metadata.get('version') != VERSION:
        logging.warning('检查点版本不匹配 ({} vs {}), 忽略旧版存档'.format(
            metadata.get('version'), VERSION))
        return None
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    # v19 (2026-08-26): 继承 v18 权重时, 旧 value_fc(225→1) 键与新 value_fc1/fc2 结构不匹配
    # pop 旧键 → strict=False 加载 (value_conv/value_bn 同形状继承, fc 新层随机初始化)
    if 'value_fc.weight' in state_dict:
        del state_dict['value_fc.weight']
    if 'value_fc.bias' in state_dict:
        del state_dict['value_fc.bias']
    # v20 (2026-08-27, 保留式 v2): 旧 policy_conv/policy_bn(128→1,3×3) 与新结构同形状 → 完整继承(识别模块)
    # 新 policy_conv2/policy_bn2/policy_fc 旧档无此键 → strict=False 自动随机初始化
    if hasattr(agent, 'raw_net') and agent.raw_net is not None:
        agent.raw_net.load_state_dict(state_dict, strict=False)
        agent.raw_net.to(device)
        agent.net = agent.raw_net
    else:
        agent.net.load_state_dict(state_dict, strict=False)
        agent.net.to(device)
    agent.reset_mcts()
    iteration = metadata.get('iteration', -1)
    logging.info('已从 v2 检查点恢复 (上次训练到 iteration {})'.format(iteration))
    return iteration


# ─────────────────────────────────────────────
# v2: 每代健康体检（价值盲测 + 激活率 + 死神经元）
# 训练完一代立即执行，结果写入 health_log.json
# ─────────────────────────────────────────────
HEALTH_LOG_PATH = os.path.join(CHECKPOINT_DIR, 'health_log.json')


def _probe_board(rows, cols, color):
    b = np.zeros((15, 15), dtype=np.float32)
    for r, c in zip(rows, cols):
        b[r, c] = color
    return b


def _probe_board_alt(rows, cols, color, opp_scatter=6, seed=None):
    """交替局面探针：主线(rows,cols)放 color 棋子，再加 opp_scatter 个对手散子(远离主线)。

    纯色探针(黑活四=棋盘仅4个黑子)是 OOD——训练自对弈从未出现单色棋盘，
    真实对局中黑方形成活四时白方已下过 N 手。本函数构造接近真实的交替局面。
    """
    b = np.zeros((15, 15), dtype=np.float32)
    for r, c in zip(rows, cols):
        b[r, c] = color
    if opp_scatter:
        rng = random.Random(seed)
        main_cells = set(zip(rows, cols))
        placed = 0
        guard = 0
        while placed < opp_scatter and guard < 2000:
            guard += 1
            r, c = rng.randrange(15), rng.randrange(15)
            if (r, c) in main_cells or b[r, c] != 0:
                continue
            if min(abs(r - rr) + abs(c - cc) for rr, cc in main_cells) < 3:
                continue
            b[r, c] = -color
            placed += 1
        if placed < opp_scatter:
            return None  # 无法放置足够散子（极端情况）
    return b


def _policy_to_prob(p_tensor, board_shape):
    """把 policy 头输出转成 (H,W) 概率分布。

    ⚠️ v1/v2 输出语义不同（2026-08-02 实踩，首次结论全错）：
    - v1 original forward 已做全局 225 softmax（输出就是概率）→ 只 reshape
    - v2 policy_fc 输出 logits → reshape(-1) 全局 softmax 再 reshape
    对 reshape 后 15×15 按行(dim=-1) softmax 是错的（每行归一化成 1/15≈0.067 假象）。
    """
    p = p_tensor.reshape(-1)
    # 检测是否已是概率: 和≈1 且全在 [0,1] → v1 已 softmax
    if bool((p >= 0).all()) and abs(float(p.sum()) - 1.0) < 0.05:
        return p.reshape(board_shape).cpu().numpy()
    p = torch.softmax(p, dim=-1)
    return p.reshape(board_shape).cpu().numpy()


def health_check(agent, iteration):
    """对当前网络跑价值盲测 + 激活率检查，返回dict"""
    net = agent.raw_net if (hasattr(agent, 'raw_net') and agent.raw_net is not None) else agent.net
    net.eval()
    result = {'iteration': iteration, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}

    # ── 1. 价值盲测（训练中真实出现的中间局面，非终局状态） ──
    # 注意: 训练数据里从未出现过"已成五连"的棋盘(成五即终局,不进trajectory)
    # 所以探针只用活三/活四/冲四等真实中间状态
    probes = {
        '黑活四': (_probe_board([7]*4, [5, 6, 7, 8], 1), 1.0),
        '白活四': (_probe_board([7]*4, [5, 6, 7, 8], -1), 1.0),
        '黑活三': (_probe_board([7]*3, [5, 6, 7], 1), 0.6),
        '白活三': (_probe_board([7]*3, [5, 6, 7], -1), 0.6),
        '黑冲四': (_probe_board([3, 4, 5, 7], [7, 7, 7, 7], 1), 0.8),  # 断一格的四连
        '白冲四': (_probe_board([3, 4, 5, 7], [7, 7, 7, 7], -1), 0.8),
        # ⚠️ 2026-08-01 补盲: 五连是最基础"已赢"形状, 纯色探针漏测导致"黑冲四准但黑五连判反"假象;
        # 注意: 活四探针(7,5-8)本身就是连续四连, 无需另设四连
        '黑五连': (_probe_board([7]*5, [5, 6, 7, 8, 9], 1), 1.0),  # 已赢, OOD但必须监控
        '白五连': (_probe_board([7]*5, [5, 6, 7, 8, 9], -1), 1.0),
        '空盘': (np.zeros((15, 15), dtype=np.float32), 0.0),
        # ⚠️ 2026-08-16 补盲: 原探针全是横/竖方向, 斜向棋型从未检测（用户发现白棋斜向识别差）
        # 主对角线 (r,r) 斜向变体: 活四(5-8)两端(4,4)/(9,9)空; 冲四(4,4-6,6)+(8,8)断(7,7); 五连(5-9)
        '黑活四斜': (_probe_board([5, 6, 7, 8], [5, 6, 7, 8], 1), 1.0),
        '白活四斜': (_probe_board([5, 6, 7, 8], [5, 6, 7, 8], -1), 1.0),
        '黑活三斜': (_probe_board([5, 6, 7], [5, 6, 7], 1), 0.6),
        '白活三斜': (_probe_board([5, 6, 7], [5, 6, 7], -1), 0.6),
        '黑冲四斜': (_probe_board([4, 5, 6, 8], [4, 5, 6, 8], 1), 0.8),
        '白冲四斜': (_probe_board([4, 5, 6, 8], [4, 5, 6, 8], -1), 0.8),
        '黑五连斜': (_probe_board([5, 6, 7, 8, 9], [5, 6, 7, 8, 9], 1), 1.0),
        '白五连斜': (_probe_board([5, 6, 7, 8, 9], [5, 6, 7, 8, 9], -1), 1.0),
    }
    # v18 修复 (2026-08-22): 探针传 players (黑棋型=1/白棋型=-1) → ch2 正确编码"当前玩家颜色"
    # 期望统一为"当前玩家视角" (持有棋型的一方优 → +) — 与训练标签 vs=players*winners 一致
    def _probe_players(name):
        if name.startswith('黑'):
            return 1
        if name.startswith('白'):
            return -1
        return None  # 空盘
    with torch.no_grad():
        for name, (b, expect) in probes.items():
            # 🚨 2026-08-23 探针视角修复: _to_planes 输入必须是 canonical(当前玩家=+1/对手=-1)
            # 旧代码直接传绝对棋盘(黑=1/白=-1) → 白视角探针 ch0/ch1 反了(白子进"对手"通道)
            # 修复: canonical = player * 绝对棋盘, 且 players 必须传 player (黑=1/白=-1)
            pl = _probe_players(name)
            if pl is None:
                p, v = net(_to_planes(b[None, :, :], players=None))  # 空盘全0, 无视角
            else:
                p, v = net(_to_planes((pl * b)[None, :, :], players=pl))
            pred = float(v.reshape(-1)[0])
            # 空盘: 期望|value|小; 其他: 期望符号正确
            if expect == 0:
                correct = abs(pred) < 0.3
            else:
                correct = (pred > 0) == (expect > 0)
            result['value_' + name] = round(pred, 4)
            result['value_' + name + '_ok'] = bool(correct)

    # ── 1b. 交替局面探针（2026-08-01 新增: 纯色探针 OOD 缺陷实证后）
    # 纯色盘(黑活四=仅4黑子)训练中从未出现 → 假信号; 真实对局黑白交替。
    # 主线 + 6 对手散子 × 10 组随机, 取均值/标准差; 通过 = 方向对 + |均值|>0.3
    ALT_TRIALS = 10
    ALT_SCATTER = 6
    for name, (b, expect) in probes.items():
        if expect == 0:  # 空盘无散子可加
            continue
        rows, cols, colors = [], [], []
        for r in range(15):
            for c in range(15):
                if b[r, c] != 0:
                    rows.append(r); cols.append(c); colors.append(b[r, c])
        if not rows:
            continue
        main_color = colors[0]  # 主线所有棋子同色
        pl = _probe_players(name)  # 视角: 黑=1/白=-1
        vals = []
        for s in range(ALT_TRIALS):
            ba = _probe_board_alt(rows, cols, main_color, opp_scatter=ALT_SCATTER, seed=1000 + s)
            if ba is None:
                continue
            # 🚨 2026-08-23 视角修复: canonical = pl * 绝对棋盘 (同纯色探针)
            _, va = net(_to_planes((pl * ba)[None, :, :], players=pl))
            vals.append(float(va.reshape(-1)[0]))
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            ok = (mean > 0) == (expect > 0) and abs(mean) > 0.3
            result['alt_' + name] = round(mean, 4)
            result['alt_' + name + '_std'] = round(std, 4)
            result['alt_' + name + '_ok'] = bool(ok)

    # ── 1c. Policy 头探针（2026-08-02 新增: 体检只看 value 头的盲区实锤后）
    # 背景: MCTS 实战里 policy 是"搜索向导"(PUCT 公式 Q + c·P·√N/(1+n) 的 P),
    #        policy 均匀则搜索无方向 → "体检数值高、实战一碰就碎"的技术根源。
    # 方法: 构造"必须应对"的局面, 测 policy 给关键应对点的概率 vs 随机基线(1/225≈0.0044)。
    # 四类: 黑攻(找成活四点)/白攻/黑守(堵白威胁)/白守(堵黑威胁), 各 4 局面 × 纯色/交替。
    # 通过: 关键点概率 > 8×随机基线(0.0355), 每类多数通过才算该类 ok。
    POLICY_THRESH = 8.0 / 225.0  # ≈0.0355
    POLICY_TRIALS = 5
    POLICY_SCATTER = 6
    policy_cats = {
        '黑攻': (1, [  # 黑有活三/冲四, 找成活四点(进攻要点), 主线=黑(+1)
            ([7, 7, 7], [5, 6, 7], [(7, 8)]),          # 活三横→成活四 [直向]
            ([5, 6, 7], [7, 7, 7], [(8, 7)]),          # 活三竖→成活四 [直向]
            ([3, 4, 5, 7], [7, 7, 7, 7], [(6, 7)]),    # 冲四→成五 [直向]
            ([5, 6, 7], [5, 6, 7], [(8, 8)]),          # 活三斜→成活四 [斜向]
            ([5, 5, 5, 5], [5, 6, 7, 8], [(4, 4), (9, 9)]),  # 斜活四→成五(两端) [斜向]
            ([4, 5, 6, 8], [4, 5, 6, 8], [(7, 7)]),          # 斜冲四→成五(补缺口) [斜向]
        ]),
        '白攻': (-1, [  # 白有活三/冲四, 找成活四点, 主线=白(-1)
            ([7, 7, 7], [5, 6, 7], [(7, 8)]),
            ([5, 6, 7], [7, 7, 7], [(8, 7)]),
            ([3, 4, 5, 7], [7, 7, 7, 7], [(6, 7)]),
            ([5, 6, 7], [5, 6, 7], [(8, 8)]),
            ([5, 5, 5, 5], [5, 6, 7, 8], [(4, 4), (9, 9)]),
            ([4, 5, 6, 8], [4, 5, 6, 8], [(7, 7)]),
        ]),
        '黑守': (-1, [  # 白有活三/冲四, 黑找堵点, 主线=白(-1)是威胁方
            ([7, 7, 7], [5, 6, 7], [(7, 4), (7, 8)]),      # 白活三横→堵两端 [直向]
            ([5, 6, 7], [7, 7, 7], [(4, 7), (8, 7)]),      # 白活三竖 [直向]
            ([3, 4, 5, 7], [7, 7, 7, 7], [(6, 7)]),        # 白冲四→堵缺口 [直向]
            ([5, 6, 7], [5, 6, 7], [(4, 4), (8, 8)]),      # 白活三斜 [斜向]
            ([5, 5, 5, 5], [5, 6, 7, 8], [(4, 4), (9, 9)]),      # 白斜活四→堵两端 [斜向]
            ([4, 5, 6, 8], [4, 5, 6, 8], [(7, 7)]),              # 白斜冲四→堵缺口 [斜向]
        ]),
        '白守': (1, [  # 黑有活三/冲四, 白找堵点, 主线=黑(+1)是威胁方
            ([7, 7, 7], [5, 6, 7], [(7, 4), (7, 8)]),
            ([5, 6, 7], [7, 7, 7], [(4, 7), (8, 7)]),
            ([3, 4, 5, 7], [7, 7, 7, 7], [(6, 7)]),
            ([5, 6, 7], [5, 6, 7], [(4, 4), (8, 8)]),
            ([5, 5, 5, 5], [5, 6, 7, 8], [(4, 4), (9, 9)]),
            ([4, 5, 6, 8], [4, 5, 6, 8], [(7, 7)]),
        ]),
    }
    with torch.no_grad():
        for cat_name, (cat_color, cat_probes) in policy_cats.items():
            pass_counts = {'纯色': 0, '交替': 0}
            straight_counts = {'纯色': 0, '交替': 0}   # 直向(横/竖)
            diag_counts = {'纯色': 0, '交替': 0}       # 斜向(对角线)
            key_means = []
            for pi, (rows, cols, key_cells) in enumerate(cat_probes):
                is_diag = pi >= 3  # 索引0-2直向(横活三/竖活三/冲四), 3-5斜向(斜活三/斜活四/斜冲四)
                main_color = cat_color
                # 🚨 2026-08-23 视角修复: 行动方视角 ≠ 主线颜色!
                # 攻类(黑攻/白攻): 行动方=持子方(主线) → player = cat_color
                # 守类(黑守/白守): 行动方=防守方(堵对方威胁) → player = -cat_color
                # 旧代码统一传 cat_color → 黑守(白威胁黑堵)传成了白视角, 白守(黑威胁白堵)传成黑视角 → 全反
                if cat_name in ('黑攻', '白攻'):
                    act_player = cat_color
                else:
                    act_player = -cat_color
                # 纯色版 (v18: 传 canonical=act_player*棋盘 + players=act_player → ch2 正确视角)
                b_p = _probe_board(rows, cols, main_color)
                p_p, _ = net(_to_planes((act_player * b_p)[None, :, :], players=act_player))
                # v2 输出 logits → softmax; 兼容 v1(已是概率, reshape 即可)
                prob_p = _policy_to_prob(p_p, b_p.shape)
                kp_pure = max(prob_p[r, c] for r, c in key_cells if b_p[r, c] == 0) if key_cells else 0.0
                pass_counts['纯色'] += int(kp_pure > POLICY_THRESH)
                (diag_counts if is_diag else straight_counts)['纯色'] += int(kp_pure > POLICY_THRESH)
                key_means.append(kp_pure)
                # 交替版: 加对手散子, 多组取均值
                kps_alt = []
                for s in range(POLICY_TRIALS):
                    b_a = _probe_board_alt(rows, cols, main_color, opp_scatter=POLICY_SCATTER, seed=2000 + pi * 10 + s)
                    if b_a is None:
                        continue
                    p_a, _ = net(_to_planes((act_player * b_a)[None, :, :], players=act_player))
                    prob_a = _policy_to_prob(p_a, b_a.shape)
                    kps_alt.append(max(prob_a[r, c] for r, c in key_cells if b_a[r, c] == 0) if key_cells else 0.0)
                if kps_alt:
                    kp_alt_mean = sum(kps_alt) / len(kps_alt)
                    pass_counts['交替'] += int(kp_alt_mean > POLICY_THRESH)
                    (diag_counts if is_diag else straight_counts)['交替'] += int(kp_alt_mean > POLICY_THRESH)
            n_probes = len(cat_probes)
            result[f'pol_{cat_name}_纯色通过'] = pass_counts['纯色']
            result[f'pol_{cat_name}_交替通过'] = pass_counts['交替']
            result[f'pol_{cat_name}_直向通过'] = straight_counts['交替']  # 交替版直向(0-2)
            result[f'pol_{cat_name}_斜向通过'] = diag_counts['交替']      # 交替版斜向(3-5)
            result[f'pol_{cat_name}_n'] = n_probes
            result[f'pol_{cat_name}_纯色ok'] = pass_counts['纯色'] >= max(2, n_probes // 2)
            result[f'pol_{cat_name}_交替ok'] = pass_counts['交替'] >= max(2, n_probes // 2)
    # 全局 policy 判定: 四类交替版至少 2 类通过 且 白守(防守) 通过
    pol_alt_oks = [result.get(f'pol_{c}_交替ok', False) for c in ['黑攻', '白攻', '黑守', '白守']]
    result['pol_全局ok'] = bool(sum(pol_alt_oks) >= 2 and result.get('pol_白守_交替ok', False))

    # ── 1d. 堵端敏感度探针（2026-08-16 新增: 死四/冲三盲区监控——用户摆局面实测
    #      黑死四判 +0.56、白活三被对手四连压成 -0.999 = 只认连子不认两端被堵, 与 v8 死四盲区同型）
    # 方法: 同一连子棋型 堵一端(对手子) vs 不堵, |value差| 应显著 >0（堵端威胁降级）
    # 判定: 活四堵端差 >0.5 达标; 活三堵端差 >0.2 达标
    def _blocked_probe(rows, cols, color, block_pos):
        b = np.zeros((15, 15), dtype=np.float32)
        for r, c in zip(rows, cols):
            b[r, c] = color
        b[block_pos] = -color
        return b

    block_probes = {
        '黑活四堵端': (([5, 6, 7, 8], [5, 6, 7, 8]), (4, 4), 0.5),  # 黑四连斜, 白堵(4,4)=死四
        '白活四堵端': (([5, 6, 7, 8], [5, 6, 7, 8]), (4, 4), 0.5),  # 白四连斜, 黑堵(4,4)=死四
        '黑活三堵端': (([5, 6, 7], [5, 6, 7]), (4, 4), 0.2),        # 黑三连斜, 白堵=冲三
        '白活三堵端': (([5, 6, 7], [5, 6, 7]), (4, 4), 0.2),        # 白三连斜, 黑堵=冲三
    }
    with torch.no_grad():
        for name, ((rows, cols), bp, thresh) in block_probes.items():
            main_color = 1 if name.startswith('黑') else -1
            b0 = _probe_board(rows, cols, main_color)
            # 🚨 2026-08-23 视角修复: canonical = main_color * 棋盘 (堵端探针持子方=评估视角)
            _, v0 = net(_to_planes((main_color * b0)[None, :, :], players=main_color))
            b1 = _blocked_probe(rows, cols, main_color, bp)
            _, v1 = net(_to_planes((main_color * b1)[None, :, :], players=main_color))
            v0f = float(v0.reshape(-1)[0]); v1f = float(v1.reshape(-1)[0])
            diff = abs(v0f - v1f)
            result[f'block_{name}_未堵'] = round(v0f, 4)
            result[f'block_{name}_堵端'] = round(v1f, 4)
            result[f'block_{name}_差'] = round(diff, 4)
            result[f'block_{name}_ok'] = bool(diff > thresh)

    # ── 1e. 相对价值探针（2026-08-16 新增: 白棋"不会相对价值比较"——用户局面2实测
    #      白先手活三只给 +0.375（该 +0.7 必胜势）= 双方都有威胁时不会按紧迫性排序）
    # 白视角(当前玩家=白) 期望排序: 己活四 > 己活三vs他活三 > 己活三vs他冲四 > 己活三vs他活四
    def _dual_probe(white_cells, black_cells):
        b = np.zeros((15, 15), dtype=np.float32)
        for r, c in white_cells:
            b[r, c] = 1
        for r, c in black_cells:
            b[r, c] = -1
        return b

    rel_probes = {
        '白活三_对黑活三': (_dual_probe([(5,5),(6,6),(7,7)], [(3,10),(4,11),(5,12)]), 0.5),       # 白先手成活四→白优
        '白活三_对黑冲四': (_dual_probe([(5,5),(6,6),(7,7)], [(2,10),(3,11),(4,12),(6,14)]), -0.3),  # 黑冲四更急,白必须堵→黑优
        '白活四_对黑活三': (_dual_probe([(5,5),(6,6),(7,7),(8,8)], [(3,10),(4,11),(5,12)]), 0.8),   # 白活四必胜
        '白活三_对黑活四': (_dual_probe([(5,5),(6,6),(7,7)], [(3,10),(4,11),(5,12),(6,13)]), -0.8),  # 黑活四必胜
    }
    with torch.no_grad():
        for name, (b_r, expect) in rel_probes.items():
            # v18: rel 探针构造是白视角 (白=+1 黑=-1) → 必须传 players=-1, 否则 ch2 默认黑视角方向全反
            _, v = net(_to_planes(b_r[None, :, :], players=-1))
            vf = float(v.reshape(-1)[0])
            ok = (vf > 0) == (expect > 0) and abs(vf) > 0.3
            result[f'rel_{name}'] = round(vf, 4)
            result[f'rel_{name}_ok'] = bool(ok)
    v_h4 = result.get('rel_白活四_对黑活三', 0)
    v_h3h3 = result.get('rel_白活三_对黑活三', 0)
    v_h3f4 = result.get('rel_白活三_对黑冲四', 0)
    v_h3h4 = result.get('rel_白活三_对黑活四', 0)
    result['rel_排序ok'] = bool(v_h4 > v_h3h3 > v_h3f4 > v_h3h4)

    # ── 2. 激活率（4个代表层） ──
    act_layers = {}
    hooks = []
    target_names = ['res_blocks.3.bn2', 'res_blocks.4.bn2',
                    'res_blocks.5.bn1', 'res_blocks.5.bn2']
    for name, module in net.named_modules():
        if name in target_names:
            holder = {}
            def make_hook(h):
                def hook_fn(mod, inp, out):
                    h['act'] = (out.detach() > 0).float().mean().item()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(holder)))
            act_layers[name] = holder

    probe_boards = [probes['黑活四'][0], probes['空盘'][0],
                    _probe_board([3, 5, 9], [3, 8, 12], 1)]
    with torch.no_grad():
        for b in probe_boards:
            net(_to_planes(b[None, :, :]))

    act_vals = []
    for name, holder in act_layers.items():
        val = holder.get('act', 0.0)
        act_vals.append(val)
        result['act_' + name.split('.')[-2] + '.' + name.split('.')[-1]] = round(val * 100, 1)
    if act_vals:
        result['act_mean'] = round(np.mean(act_vals) * 100, 1)
    for h in hooks:
        h.remove()

    # ── 3. BN 死神经元 ──
    dead = 0
    total = 0
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            g = module.weight.detach().cpu().numpy()
            total += len(g)
            dead += int((g < 0.01).sum())
    result['bn_dead'] = dead
    result['bn_total'] = total

    # ── 写入日志 ──
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    history = []
    if os.path.exists(HEALTH_LOG_PATH):
        try:
            with open(HEALTH_LOG_PATH) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(result)
    with open(HEALTH_LOG_PATH, 'w') as f:
        json.dump(history, f, indent=1)

    pol_summary = 'policy探针: 黑攻[{}/{} 交替{}] 白攻[{}/{} 交替{}] 黑守[{}/{} 交替{}] 白守[{}/{} 交替{}] 全局{}'.format(
        result.get('pol_黑攻_纯色通过', 0), result.get('pol_黑攻_n', 4),
        '✅' if result.get('pol_黑攻_交替ok') else '❌',
        result.get('pol_白攻_纯色通过', 0), result.get('pol_白攻_n', 4),
        '✅' if result.get('pol_白攻_交替ok') else '❌',
        result.get('pol_黑守_纯色通过', 0), result.get('pol_黑守_n', 4),
        '✅' if result.get('pol_黑守_交替ok') else '❌',
        result.get('pol_白守_纯色通过', 0), result.get('pol_白守_n', 4),
        '✅' if result.get('pol_白守_交替ok') else '❌',
        '✅' if result.get('pol_全局ok') else '❌')
    logging.info('健康体检: 活四[黑={} {} 白={} {}] 活三[黑={} {} 白={} {}] 冲四[黑={} {} 白={} {}] 五连[黑={} {} 白={} {}] 空盘={} | 激活率={}% 死神经元={}/{} | 交替版: 活四[黑={}({}) 白={}({})] 冲四[黑={}({}) 白={}({})] 活三[黑={}({}) 白={}({})] 五连[黑={}({}) 白={}({})] | {}'.format(
        result.get('value_黑活四'), '✅' if result.get('value_黑活四_ok') else '❌',
        result.get('value_白活四'), '✅' if result.get('value_白活四_ok') else '❌',
        result.get('value_黑活三'), '✅' if result.get('value_黑活三_ok') else '❌',
        result.get('value_白活三'), '✅' if result.get('value_白活三_ok') else '❌',
        result.get('value_黑冲四'), '✅' if result.get('value_黑冲四_ok') else '❌',
        result.get('value_白冲四'), '✅' if result.get('value_白冲四_ok') else '❌',
        result.get('value_黑五连'), '✅' if result.get('value_黑五连_ok') else '❌',
        result.get('value_白五连'), '✅' if result.get('value_白五连_ok') else '❌',
        result.get('value_空盘'),
        result.get('act_mean'), dead, total,
        result.get('alt_黑活四'), '✅' if result.get('alt_黑活四_ok') else '❌',
        result.get('alt_白活四'), '✅' if result.get('alt_白活四_ok') else '❌',
        result.get('alt_黑冲四'), '✅' if result.get('alt_黑冲四_ok') else '❌',
        result.get('alt_白冲四'), '✅' if result.get('alt_白冲四_ok') else '❌',
        result.get('alt_黑活三'), '✅' if result.get('alt_黑活三_ok') else '❌',
        result.get('alt_白活三'), '✅' if result.get('alt_白活三_ok') else '❌',
        result.get('alt_黑五连'), '✅' if result.get('alt_黑五连_ok') else '❌',
        result.get('alt_白五连'), '✅' if result.get('alt_白五连_ok') else '❌',
        pol_summary))
    return result


# ─────────────────────────────────────────────
# 主训练
# ─────────────────────────────────────────────
def train(config=None):
    if config is None:
        config = {
            'train_iterations': 500,   # 2026-08-11 续跑: 100→500 (判断用代码默认值, 不是metadata)
            'train_episodes_per_iteration': 10,
            'batches': 32,
            'batch_size': 512,
            'sim_count': 800,       # v6: 固定800模拟，全程不变
            'net_kwargs': {
                'conv_filters': [128],
                'residual_filters': [[128, 128]] * 8,
                'policy_filters': [128],
            },
            'lr': 0.001,
            'worker_count': 9,
            'worker_games_per_worker': 10,  # 9工人 × 10 = 90局
            # v19 (2026-08-26, Kimi 三步走): 阶段1 前 N 代冻结主干+Policy 只训 Value 头,
            # N 代后解冻全网络。value 头学会读取主干特征后再让主干适应新监督。
            # 08-26 用户拍板: 冻结 5 → 10 代 (新 value 头容量小, 多给几代稳定学特征读取)
            'freeze_policy_iters': 0,   # v24: 无冻结, 标准一起训 (ch2 恒1 已灭捷径, 无需防抢跑手术)
            'freeze_value_iters': 0,    # v24 (2026-08-30, 从零+ch2恒1): 不冻结, 全网络一起训
        }

    env = gym.make('Gomuku-v0')
    agent = AlphaZeroAgent(env=env, kwargs=config['net_kwargs'],
                           batches=config['batches'],
                           batch_size=config['batch_size'],
                           sim_count=config['sim_count'],
                           lr=config.get('lr', 0.001),
                           disable_compile=True)
    logging.info('v2 训练启动, device={}'.format(device))

    train_iterations = config['train_iterations']
    train_episodes_per_iteration = config['train_episodes_per_iteration']

    resumed = load_checkpoint(agent, config)
    start_iteration = 0
    if resumed is not None:
        start_iteration = resumed

    if start_iteration >= train_iterations:
        logging.info('训练已完成, 如需重跑请删除 {} 目录'.format(CHECKPOINT_DIR))
        return

    GAME_DATA_DIR = os.path.join(CHECKPOINT_DIR, 'game_data')
    os.makedirs(GAME_DATA_DIR, exist_ok=True)
    os.makedirs(REPLAY_DIR, exist_ok=True)

    for iteration in range(start_iteration, train_iterations):
        # ── v24 (2026-08-30, 从零+ch2恒1): 无冻结, 标准 AlphaZero 全网络一起训 ──
        # (ch2 恒1 已从输入层灭空盘捷径, value 无捷径可抢跑, 冻结/λ 等手术刀不再需要)
        # ── v2: 模拟量固定 400（全程不变，对比实验） ──
        # ── v6: 固定 800 模拟（全程不变） ──
        target = config.get('sim_count', 800)
        if agent.sim_count != target:
            agent.sim_count = target
            logging.info('v8 固定sim: 设置至 {} (代{})'.format(target, iteration))

        # ── 开局约束：不强制，让网络自由学习开局（v2 不启用） ──
        force_center = False

        dfs_trajectory = []
        saved_games = sorted(glob.glob(
            os.path.join(GAME_DATA_DIR, f'iter{iteration}_game_*.pkl')))
        if saved_games:
            logging.info('检测到 {} 场已保存的对局数据，正在恢复...'.format(len(saved_games)))
            for gf in saved_games:
                with open(gf, 'rb') as f:
                    dfs_trajectory.append(pickle.load(f))
            completed = len(dfs_trajectory)
        else:
            completed = 0
        iter_start = time.time()

        # ── 主进程自对弈 ──
        for episode in range(completed, train_episodes_per_iteration):
            df_trajectory = self_play(env, agent,
                                      return_trajectory=True, verbose=False,
                                      force_center=force_center)
            winner = df_trajectory['winner'].iloc[0]
            winner_str = '黑胜' if winner == 1 else '白胜' if winner == -1 else '平局'
            elapsed = time.time() - iter_start
            remaining = elapsed / (episode + 1) * (
                train_episodes_per_iteration - episode - 1)
            logging.info('代{} 局{}/{}: {} 经验{}条 | sim={} | {:.0f}s/局 | 已过{:.0f}s 预计剩余{:.0f}s'.format(
                iteration, episode + 1, train_episodes_per_iteration,
                winner_str, len(df_trajectory), agent.sim_count,
                elapsed / (episode + 1), elapsed, remaining))
            dfs_trajectory.append(df_trajectory)
            game_path = os.path.join(GAME_DATA_DIR,
                                     f'iter{iteration}_game_{episode}.pkl')
            with open(game_path, 'wb') as f:
                pickle.dump(df_trajectory, f)
        main_games = len(dfs_trajectory)
        logging.info('自对弈完成: {}局 (主进程)'.format(len(dfs_trajectory)))

        # ── 收集工兵经验 ──
        worker_games = 0
        os.makedirs(WORKER_DIR, exist_ok=True)
        worker_game_pattern = os.path.join(
            WORKER_DIR, 'iter{}_*_game_*.pkl'.format(iteration))
        expected = config.get('worker_count', 9) * config.get('worker_games_per_worker', 10)
        # v2 fix: 首次迭代无已存权重时跳过等待工兵（工兵需 weights.pt 才能启动）
        if os.path.exists(WEIGHTS_PATH):
            while True:
                done = sorted(glob.glob(worker_game_pattern))
                if len(done) >= expected:
                    break
                logging.info('等待工兵: {}/{}局已到'.format(len(done), expected))
                time.sleep(10)
        else:
            logging.info('首次迭代：无已存权重，跳过等待工兵，仅用主进程数据')
        worker_files = sorted(glob.glob(worker_game_pattern))
        for gf in worker_files:
            with open(gf, 'rb') as f:
                dfs_trajectory.append(pickle.load(f))
            worker_games += 1
        logging.info('收集工兵: {}局, 总计 {}局'.format(worker_games, len(dfs_trajectory)))

        # ── 杀工兵释放GPU ──
        import subprocess
        subprocess.run(['pkill', '-9', '-f', 'Gomoku-worker_v24.py'],
                       capture_output=True)
        time.sleep(5)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info('工兵已杀，GPU已释放')

        # ── v2: 经验回放池（保留最近 N 代数据） ──
        # 将当前代数据写入回放池
        pool_file = os.path.join(REPLAY_DIR, f'iter{iteration}.pkl')
        with open(pool_file, 'wb') as f:
            pickle.dump(dfs_trajectory, f)
        # 收集最近 N 代（⚠️ 必须按迭代号数值排序！glob 返回字典序，iter10 < iter2，
        # sorted() 会永远取到 iter5-9 而非最新代——2026-08-01 实踩的致命 bug）
        pool_files = sorted(glob.glob(os.path.join(REPLAY_DIR, 'iter*.pkl')),
                            key=lambda p: int(os.path.basename(p)[4:-4]))[-REPLAY_POOL_SIZE:]
        replay_dfs = []
        pool_detail = []
        import time as _t
        for pf in pool_files:
            with open(pf, 'rb') as f:
                games = pickle.load(f)
            mtime = _t.strftime('%m-%d %H:%M', _t.localtime(os.path.getmtime(pf)))
            pool_detail.append('{} ({}局, {})'.format(os.path.basename(pf), len(games), mtime))
            replay_dfs.extend(games)
        # 清晰输出回放池明细：每代文件名/局数/写入时间（便于发现新旧数据混入）
        logging.info('经验回放池: 取最近 {} 代 = [{}], 共 {} 局'.format(
            len(pool_files), ', '.join(pool_detail), len(replay_dfs)))

        # ── 学习 ──
        total_exp = len(replay_dfs)
        logging.info('训练 {}: 经验={}条(回放池), batches={}, batch_size={}'.format(
            iteration, total_exp, agent.batches, agent.batch_size))
        loss_info = agent.learn(replay_dfs)
        policy_loss = loss_info.get('policy_loss', 0)
        value_loss = loss_info.get('value_loss', 0)
        total_loss = loss_info.get('total_loss', 0)
        logging.info('训练 {}: 学习完成  policy_loss={:.4f}  value_loss={:.4f}  total={:.4f}'.format(
            iteration, policy_loss, value_loss, total_loss))

        # 存档
        config['losses'] = config.get('losses', {})
        config['losses'][str(iteration)] = {
            'policy_loss': round(policy_loss, 4),
            'value_loss': round(value_loss, 4),
            'total_loss': round(total_loss, 4),
        }
        save_checkpoint(agent, iteration, config)

        # ── v2: 每代健康体检 ──
        result = None
        try:
            result = health_check(agent, iteration)
        except Exception as e:
            logging.warning('健康体检失败: {}'.format(e))

        # ── TensorBoard 权重监控（2026-08-01 新增：可视化权重更新） ──
        try:
            import torch.utils.tensorboard as _tb
            _net = agent.raw_net if (hasattr(agent, 'raw_net') and agent.raw_net is not None) else agent.net
            _tb_dir = os.path.join(CHECKPOINT_DIR, 'tb_runs')
            _writer = _tb.SummaryWriter(log_dir=_tb_dir)
            # loss 标量
            _writer.add_scalar('loss/policy', policy_loss, iteration)
            _writer.add_scalar('loss/value', value_loss, iteration)
            _writer.add_scalar('loss/total', total_loss, iteration)
            # 权重统计（每层 weight 的 mean/std + 全网络 L2 范数）
            total_l2 = 0.0
            for name, module in _net.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    w = module.weight.detach().float()
                    total_l2 += float((w ** 2).sum())
                    _writer.add_scalar('weight_mean/{}'.format(name), float(w.mean()), iteration)
                    _writer.add_scalar('weight_std/{}'.format(name), float(w.std()), iteration)
                if hasattr(module, 'bias') and module.bias is not None:
                    b = module.bias.detach().float()
                    _writer.add_scalar('bias_mean/{}'.format(name), float(b.mean()), iteration)
            _writer.add_scalar('weight/l2_total', total_l2 ** 0.5, iteration)
            # BN gamma 统计（检测 gamma 萎缩）
            bn_gammas = []
            for name, module in _net.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_gammas.append(float(module.weight.detach().mean()))
            if bn_gammas:
                _writer.add_scalar('bn/gamma_mean', sum(bn_gammas) / len(bn_gammas), iteration)
            # 激活率（若体检已算）
            if result is not None and result.get('act_mean') is not None:
                _writer.add_scalar('health/activation_rate', result['act_mean'] / 100.0, iteration)
            if result is not None and result.get('bn_dead') is not None:
                _writer.add_scalar('health/dead_neurons', result['bn_dead'], iteration)
            _writer.close()
            logging.info('TensorBoard 已写入: {} (代{})'.format(_tb_dir, iteration))
        except Exception as e:
            logging.warning('TensorBoard 写入失败: {}'.format(e))

        # ── 重启工兵 ──
        for wid in range(config.get('worker_count', 9)):
            # 用 >> 追加日志，避免覆盖旧日志（历史教训: > 会清空旧日志）
            cmd = 'screen -dmS gomoku_w{} bash -c "{} -u {} {} >> {} 2>&1"'.format(
                wid, sys.executable,
                os.path.join(SCRIPT_DIR, 'Gomoku-worker_v24.py'),
                wid, os.path.join(SCRIPT_DIR, 'worker_v24_{}.log'.format(wid)))
            os.system(cmd)
        logging.info('{}个工兵(v24)已重启'.format(config.get('worker_count', 9)))


if __name__ == '__main__':
    import hashlib
    train()
