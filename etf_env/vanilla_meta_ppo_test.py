#!/usr/bin/env python3
# 文件名: vanilla_ppo_test.py

import os
import argparse
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

from env.etf_env import MultiComponentETFEnvTest
from env.sub_etf_env import SingleComponentETFEnv

def evaluate_vanilla_agent(
    model_path: str,
    n_components: int,
    total_budget: float,
    num_runs: int,
    output_path: str,
    history_window: int
):

    print(f"---  {model_path} ---")
    model = PPO.load(model_path)
    base_env = MultiComponentETFEnvTest(initial_action_budget=total_budget)
    budget_per_component = total_budget / n_components if n_components > 0 else 0
    print(f"budget: {budget_per_component:.2f}")

    all_component_survival_times = []

    for i in tqdm(range(n_components), desc="evaul"):
        eval_env = SingleComponentETFEnv(base_env, i, budget_per_component, history_window=history_window)
        run_survival_times = []
        for _ in range(num_runs):
            base_env.reset()
            obs, _ = eval_env.reset()
            done = False
            t = 0
            while not done:
                action_array, _ = model.predict(obs, deterministic=True)
                
                action_int = action_array.item()
                
                obs, reward, done, truncated, info = eval_env.step(action_int)
                t += 1
            run_survival_times.append(t)

        avg_survival_time = np.mean(run_survival_times)
        all_component_survival_times.append(avg_survival_time)

    final_avg_survival_time = np.mean(all_component_survival_times)
    print(f"\n--- Vanilla PPO ---")
    print(f"每个组件的平均生存时间: {final_avg_survival_time:.2f} days")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(all_component_survival_times))
    print(f"✅{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a unified Vanilla PPO model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output_path", type=str, default="results_local_test/test_outputs/vanilla_ppo_survival_times.npy")
    parser.add_argument('--total_budget', type=float, default=20.0)
    parser.add_argument('--history_window', type=int, default=20)
    args = parser.parse_args()

    evaluate_vanilla_agent(
        model_path=args.model_path,
        n_components=args.n_components,
        total_budget=args.total_budget,
        num_runs=args.num_runs,
        output_path=args.output_path,
        history_window=args.history_window
    )