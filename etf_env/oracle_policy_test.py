#!/usr/bin/env python3
# file name: oracle_policy_test.py

import os
import argparse
import numpy as np
import json
from tqdm import tqdm


from env.etf_env import MultiComponentETFEnvTest
from env.sub_etf_env import SingleComponentETFEnv

def evaluate_oracle_policy(
    test_data_path: str,
    params_path: str,
    num_runs: int,
    output_path: str,
    total_budget: float,
    repair_threshold: float,
    n_components: int = None,
    history_window: int = 20
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        if repair_threshold is None:
            repair_threshold = params["best_repair_threshold"]
    except FileNotFoundError:
        print(f" {params_path}。")
        if repair_threshold is None:
             raise ValueError("warning")
        
    print(f"--- {repair_threshold}) ---")

    base_env = MultiComponentETFEnvTest(initial_action_budget=total_budget)
    n_assets_total = base_env.n_assets
    num_components_to_run = n_components if n_components is not None else n_assets_total
    
    budget_per_component = total_budget / num_components_to_run if num_components_to_run > 0 else 0

    all_component_survival_times = []

    for i in tqdm(range(num_components_to_run), desc="processing"):
        sub_env = SingleComponentETFEnv(
            base_env=base_env, asset_index=i,
            allocated_budget=budget_per_component, mode='oracle'
        )

        run_survival_times = []
        for run in range(num_runs):
            base_env.reset()
            obs, _ = sub_env.reset()
            done = False
            t = 0
            while not done:
                true_risk_capital = obs[-1]
                action = 2 if true_risk_capital < repair_threshold else 0
                obs, reward, done, truncated, info = sub_env.step(action)
                t += 1
            run_survival_times.append(t)
            
        avg_survival_time = np.mean(run_survival_times)
        all_component_survival_times.append(avg_survival_time)

    final_avg_survival_time = np.mean(all_component_survival_times)
    print(f"\n--- result ---")
    print(f"time: {final_avg_survival_time:.2f} days")
    
    np.save(output_path, np.array(all_component_survival_times))
    print(f"✅ save: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Oracle (Perfect Information) baseline policy.")
    

    parser.add_argument("--params_path", type=str, required=True, help="预计算的神谕参数文件路径(.json)。")
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output_path", type=str, default="results_local_test/test_outputs/oracle_survival_times.npy")
    parser.add_argument('--total_budget', type=float, default=20.0)
    parser.add_argument('--repair_threshold', type=float, default=None, help="save")

    args = parser.parse_args()

    evaluate_oracle_policy(
        test_data_path=None, 
        params_path=args.params_path,
        num_runs=args.num_runs,
        output_path=args.output_path,
        total_budget=args.total_budget,
        repair_threshold=args.repair_threshold,
        n_components=args.n_components
    )