# 文件名: env/etf_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import baselinefin # 我们需要它来加载V_train和V_test

# [核心修正] 1. 创建一个包含通用逻辑的基类
class MultiComponentETFEnvBase(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, initial_action_budget: float, **kwargs):
        super().__init__()
        
        self.V_data = None # 数据将在子类中被加载
        self.n_assets = baselinefin.n
        self.T = 0 # 将在子类中被设定

        self.initial_risk_capital = 100.0
        self.initial_action_budget = initial_action_budget
        self.costs = {
            0: kwargs.get('inaction_cost', 0.2),
            1: kwargs.get('inspection_cost', 0.5),
            2: kwargs.get('recapitalization_cost', 10.0)
        }

        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        obs_shape = self.n_assets * 2 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        self.d = 0
        self.risk_capitals = np.zeros(self.n_assets, dtype=np.float32)
        self.action_budget = 0.0
        self._are_capitals_visible = np.zeros(self.n_assets, dtype=bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.d = 0
        self.risk_capitals.fill(self.initial_risk_capital)
        self.action_budget = self.initial_action_budget
        self._are_capitals_visible.fill(False)
        return self._get_full_observation(), {}

    def _get_full_observation(self, mode='agent'):
        # [核心修正] 使用通用的 self.V_data
        price_obs = self.V_data[self.d].astype(np.float32)
        budget_obs = np.array([self.action_budget], dtype=np.float32)
        visible_capitals = np.zeros(self.n_assets, dtype=np.float32)
        for i in range(self.n_assets):
            if mode == 'oracle' or self._are_capitals_visible[i]:
                visible_capitals[i] = self.risk_capitals[i]
            else: visible_capitals[i] = -1.0
        if mode == 'agent':
            self._are_capitals_visible.fill(False)
        return np.concatenate([price_obs, budget_obs, visible_capitals])

    def step(self, actions):
        if self.d >= self.T - 2:
            obs = self._get_full_observation()
            reward = np.sum(self.risk_capitals > 0)
            return obs, float(reward), True, False, {}

        total_cost = sum(self.costs[a] for a in actions)
        if self.action_budget < total_cost:
            obs = self._get_full_observation()
            return obs, 0.0, True, False, {"failure_reason": "insufficient_total_budget"}
        self.action_budget -= total_cost

        # [核心修正] 使用通用的 self.V_data
        price_prev = self.V_data[self.d]
        self.d += 1
        price_curr = self.V_data[self.d]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.where(price_prev > 0, (price_curr / price_prev) - 1.0, 0)

        for i in range(self.n_assets):
            if self.risk_capitals[i] <= 0: continue
            action_i = actions[i]
            if action_i == 1: self._are_capitals_visible[i] = True
            elif action_i == 2: self.risk_capitals[i] = self.initial_risk_capital
            return_i = returns[i]
            return_i = returns[i]
            if return_i < 0:
                # [核心修正] 从线性惩罚，升级为带有凸性的非线性惩罚
                
                # ------ 旧的线性方式 START ------
                # self.risk_capitals[i] -= abs(return_i) * 20
                # ------ 旧的线性方式 END ------

                # ------ 新的非线性方式 START ------
                loss_pct = abs(return_i) * 100 # 将回报率转为百分比, e.g., 0.01 -> 1.0
                
                # 定义一个基础消耗，再加上一个与亏损平方成正比的额外惩罚
                # 这里的系数 (如0.5, 0.2) 是可以调整的超参数
                base_depletion = loss_pct * 0.2
                convex_penalty = (loss_pct ** 2) * 0.1
                
                total_depletion = base_depletion + convex_penalty
                
                self.risk_capitals[i] -= total_depletion

        reward = np.sum(self.risk_capitals > 0)
        obs = self._get_full_observation()
        done = False
        info = {"risk_capitals": self.risk_capitals.copy()}
        return obs, float(reward), done, False, info

# [核心修正] 2. 创建两个明确的子类，用于训练和测试

class MultiComponentETFEnvTrain(MultiComponentETFEnvBase):
    """用于训练的环境，硬编码加载V_train.npy"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 硬编码加载训练数据
        self.V_data = baselinefin.V_train
        self.T = self.V_data.shape[0]
        print(f"训练环境已初始化，加载数据: V_train.npy (形状: {self.V_data.shape})")

class MultiComponentETFEnvTest(MultiComponentETFEnvBase):
    """用于测试的环境，硬编码加载V_test.npy"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 硬编码加载测试数据
        self.V_data = baselinefin.V_test
        self.T = self.V_data.shape[0]
        print(f"测试环境已初始化，加载数据: V_test.npy (形状: {self.V_data.shape})")