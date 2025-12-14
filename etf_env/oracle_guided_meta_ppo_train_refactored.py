#!/usr/bin/env python3
# file name: oracle_guided_meta_ppo_train_refactored.py

import os
import argparse
import numpy as np
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym

from env.etf_env import MultiComponentETFEnvTrain
from env.sub_etf_env import SingleComponentETFEnv
from models.random_forest import BudgetSplitModel

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
    def _init():
        base_env = MultiComponentETFEnvTrain(initial_action_budget=total_budget)
        sub_env = SingleComponentETFEnv(base_env=base_env, asset_index=asset_index, allocated_budget=budget, history_window=history_window)
        guided_env = OracleGuidedEnv(sub_env, oracle_policy, oracle_params)
        return guided_env
    return _init

def train_meta_ppo_pipeline(args):
    print("--- 启动 Oracle-Guided Meta-PPO 训练流程 ---")

    print("\nStep 1: Allocating Budgets...")
    allocator = BudgetSplitModel.load(args.budget_model_path)
    static_features = np.load(os.path.join(args.feature_dir, "features_X.npy"))
    proportions = allocator.predict(static_features[:args.n_components])
    allocated_budgets = proportions * args.total_budget
    
    print("\nStep 2: Loading Oracle Policy...")
    oracle_policy = np.load(os.path.join(args.oracle_dir, "oracle_policy.npy"))
    with open(os.path.join(args.oracle_dir, "oracle_params.json"), 'r') as f:
        oracle_params = json.load(f)

    print("\nStep 3: Creating Vectorized Training Environments...")
    env_funcs = [
        make_env_func(args.total_budget, i, allocated_budgets[i], args.history_window, oracle_policy, oracle_params)
        for i in range(args.n_components)
    ]
    VecEnv = SubprocVecEnv if args.num_cpu > 1 and args.n_components > 1 else DummyVecEnv
    train_env = VecEnv(env_funcs)
    
    print("\nStep 4: Training the Unified Meta-PPO Model...")
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "meta_ppo_model.zip")
    
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=os.path.join(args.output_dir, "tb_logs"))
    model.learn(total_timesteps=args.timesteps)
    model.save(model_path)
    
    print(f"\n🎉 save: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Oracle-Guided Meta-PPO model.")
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--total_budget", type=float, default=20.0)
    parser.add_argument("--history_window", type=int, default=20)
    parser.add_argument("--num_cpu", type=int, default=4)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--budget_model_path", type=str, required=True)
    parser.add_argument("--oracle_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    train_meta_ppo_pipeline(args)