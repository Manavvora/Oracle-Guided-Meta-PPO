#!/usr/bin/env python3
# file name: meta_ppo_test.py

import os
import argparse
import numpy as np
import json
from tqdm import tqdm
from stable_baselines3 import PPO
import gymnasium as gym

from env.etf_env import MultiComponentETFEnvTest
from env.sub_etf_env import SingleComponentETFEnv
from models.random_forest import BudgetSplitModel

# OracleGuidedEnv 封装器和 make_env_func 辅助函数保持不变
class OracleGuidedEnv(gym.Wrapper):
    def __init__(self, env: SingleComponentETFEnv, oracle_policy: np.ndarray, oracle_params: dict):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(2)
        self.oracle_policy = oracle_policy
        self.oracle_budget_step = oracle_params["budget_step"]
        self.oracle_total_budget = oracle_params["total_budget"]

    def step(self, action: int):
        if action == 1:
            return self.env.step(1)
        else: # action == 0
            true_k = self.env.base_env.risk_capitals[self.env.asset_index]
            true_c = self.env.allocated_budget
            k_idx = int(np.round(true_k))
            c_idx = int(np.round((true_c / self.oracle_total_budget) * (self.oracle_policy.shape[1] - 1)))
            k_idx = np.clip(k_idx, 0, self.oracle_policy.shape[0] - 1)
            c_idx = np.clip(c_idx, 0, self.oracle_policy.shape[1] - 1)
            oracle_action = self.oracle_policy[k_idx, c_idx]
            return self.env.step(oracle_action)

def make_env_func(total_budget, asset_index, budget, history_window, oracle_policy, oracle_params):
    """一个辅助函数，用于创建单个Oracle-Guided评估环境实例。"""
    def _init():
        # 在每个子进程中独立创建base_env，以保证安全
        base_env = MultiComponentETFEnvTest(initial_action_budget=total_budget)
        sub_env = SingleComponentETFEnv(base_env=base_env, asset_index=asset_index, allocated_budget=budget, history_window=history_window)
        guided_env = OracleGuidedEnv(sub_env, oracle_policy, oracle_params)
        return guided_env
    return _init

def evaluate_meta_ppo_pipeline(args):
    """完整的Oracle-Guided Meta-PPO评估流程"""
    print("--- 启动 Oracle-Guided Meta-PPO 评估流程 ---")

    # 步骤 1: 预算分配
    print("\nStep 1: Allocating Budgets using Random Forest...")
    allocator = BudgetSplitModel.load(args.budget_model_path)
    static_features = np.load(os.path.join(args.feature_dir, "features_X.npy"))
    proportions = allocator.predict(static_features[:args.n_components])
    allocated_budgets = proportions * args.total_budget
    
    # 步骤 2: 加载神谕
    print("\nStep 2: Loading Oracle Policy...")
    oracle_policy = np.load(os.path.join(args.oracle_dir, "oracle_policy.npy"))
    with open(os.path.join(args.oracle_dir, "oracle_params.json"), 'r') as f:
        oracle_params = json.load(f)

    # 步骤 3: 加载训练好的模型
    print(f"\nStep 3: Loading Pre-trained Meta-PPO model from {args.model_path}...")
    model = PPO.load(args.model_path)
    
    # [核心修正] 此处 base_env 的创建仅用于获取参数，实际环境在循环中创建
    # 这一步其实可以省略，但保留也无妨
    base_env_template = MultiComponentETFEnvTest(initial_action_budget=args.total_budget)

    all_component_survival_times = []

    # --- 5. 在每个组件上独立运行评估 ---
    for i in tqdm(range(args.n_components), desc="评估组件进度"):
        # 为每个组件创建独立的评估环境
        eval_env = make_env_func(args.total_budget, i, allocated_budgets[i], args.history_window, oracle_policy, oracle_params)()

        run_survival_times = []
        for _ in range(args.num_runs):
            # 每次运行前都重置世界状态
            eval_env.env.base_env.reset() # 注意，需要重置封装最深处的base_env
            obs, _ = eval_env.reset()
            done = False
            t = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                t += 1
            run_survival_times.append(t)
        
        avg_survival_time_for_comp = np.mean(run_survival_times)
        all_component_survival_times.append(avg_survival_time_for_comp)

    # --- 6. 汇总并保存结果 ---
    final_avg_survival_time = np.mean(all_component_survival_times)
    print(f"\n--- Meta-PPO 评估结果 ---")
    print(f"每个组件的平均生存时间: {final_avg_survival_time:.2f} 天")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, np.array(all_component_survival_times))
    print(f"✅ Meta-PPO 评估结果已保存于: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained Oracle-Guided Meta-PPO model.")
    # [核心修正] 移除不再需要的test_data_path参数
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument('--total_budget', type=float, default=20.0)
    parser.add_argument('--history_window', type=int, default=20)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--budget_model_path", type=str, required=True)
    parser.add_argument("--oracle_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    evaluate_meta_ppo_pipeline(args)