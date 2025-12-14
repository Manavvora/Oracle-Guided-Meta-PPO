# 文件名: env/sub_etf_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
# [核心修正] 导入基类，而不是已经不存在的MultiComponentETFEnv
from env.etf_env import MultiComponentETFEnvBase 

class SingleComponentETFEnv(gym.Env):
    """
    [最终修正版] 单组件训练/测试环境
    """
    metadata = {'render_modes': []}

    def __init__(self, 
                 base_env: MultiComponentETFEnvBase, # [核心修正] 类型提示改为基类
                 asset_index: int, 
                 allocated_budget: float, 
                 mode: str = 'agent',
                 history_window: int = 20):
        super().__init__()
        
        self.base_env = base_env
        self.asset_index = asset_index
        self.initial_allocated_budget = allocated_budget
        self.allocated_budget = allocated_budget
        self.mode = mode
        self.history_window = history_window
        self.price_history = deque(maxlen=self.history_window)
        self.action_space = spaces.Discrete(3) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _get_sub_observation(self):
        # ... (此方法内部逻辑无需修改，它不直接访问V_data) ...
        history = np.array(self.price_history)
        if len(history) == 0: return np.zeros(self.observation_space.shape, dtype=np.float32)
        sma5 = np.mean(history[-5:]) if len(history) >= 5 else history[-1]
        feat_price_vs_sma5 = history[-1] / sma5 if sma5 > 0 else 1.0
        sma20 = np.mean(history) if len(history) >= self.history_window else sma5
        feat_sma5_vs_sma20 = sma5 / sma20 if sma20 > 0 else 1.0
        if len(history) > 1:
            log_returns = np.diff(np.log(history))
            feat_volatility = np.std(log_returns)
        else: feat_volatility = 0.0
        feat_budget_ratio = self.allocated_budget / self.initial_allocated_budget if self.initial_allocated_budget > 0 else 0.0
        all_risk_capitals = self.base_env.risk_capitals
        all_visibilities = self.base_env._are_capitals_visible
        feat_visible_capital = -1.0
        if self.mode == 'oracle' or all_visibilities[self.asset_index]:
            feat_visible_capital = all_risk_capitals[self.asset_index]
        return np.array([feat_price_vs_sma5, feat_sma5_vs_sma20, feat_volatility, feat_budget_ratio, feat_visible_capital], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.allocated_budget = self.initial_allocated_budget
        self.price_history.clear()
        start_day = self.base_env.d
        start_slice = max(0, start_day - self.history_window + 1)
        # [核心修正] 从通用的 V_data 属性中读取价格
        initial_prices = self.base_env.V_data[start_slice:start_day+1, self.asset_index]
        for p in initial_prices:
            self.price_history.append(p)
        return self._get_sub_observation(), {}

    def step(self, action: int):
        action_cost = self.base_env.costs[action]
        if self.allocated_budget < action_cost:
            obs = self._get_sub_observation()
            return obs, 0.0, True, False, {"failure_reason": "insufficient_allocated_budget"}
        self.allocated_budget -= action_cost

        full_actions = np.zeros(self.base_env.n_assets, dtype=int)
        full_actions[self.asset_index] = action
        _, _, global_done, _, info = self.base_env.step(full_actions)

        # [核心修正] 从通用的 V_data 属性中读取价格
        current_price = self.base_env.V_data[self.base_env.d, self.asset_index]
        self.price_history.append(current_price)

        my_current_risk_capital = self.base_env.risk_capitals[self.asset_index]
        done = (my_current_risk_capital <= 0) or (self.allocated_budget <=0) or global_done
        reward = 1.0 if not done else 0.0
        sub_obs = self._get_sub_observation()
        return sub_obs, reward, done, False, {"my_risk_capital": my_current_risk_capital, **info}