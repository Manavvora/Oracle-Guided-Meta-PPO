#!/usr/bin/env python3
# 文件名: etf_realistic_baseline.py

import os
import argparse
import numpy as np
from tqdm import tqdm


from env.etf_env import MultiComponentETFEnvTest
from env.sub_etf_env import SingleComponentETFEnv

def run_naive_heuristic_baseline(
    num_runs: int,
    output_path: str,
    total_budget: float,
    inspection_interval: int,
    repair_threshold: float,
    n_components: int = None
):
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("--- check ---")



    base_env = MultiComponentETFEnvTest(initial_action_budget=total_budget)
    n_assets_total = base_env.n_assets
    
    num_components_to_run = n_components if n_components is not None else n_assets_total
    print(f" {num_components_to_run} / {n_assets_total} ")

    budget_per_component = total_budget / num_components_to_run if num_components_to_run > 0 else 0

    all_component_survival_times = []

 
    for i in tqdm(range(num_components_to_run), desc="processing"):
        

        sub_env = SingleComponentETFEnv(
            base_env=base_env,
            asset_index=i,
            allocated_budget=budget_per_component,
            mode='agent' 
        )

        run_survival_times = []
        for run in range(num_runs):

            base_env.reset()
            obs, _ = sub_env.reset()
            
            done = False
            t = 0
            days_since_last_inspection = inspection_interval 
            needs_repair = False

            while not done:

                action = 0 
                if needs_repair:
                    action = 2 
                    needs_repair = False
                elif days_since_last_inspection >= inspection_interval:
                    action = 1 
                    days_since_last_inspection = 0
                
                obs, reward, done, truncated, info = sub_env.step(action)
                
                visible_capital = obs[4] 
                if visible_capital != -1.0 and visible_capital < repair_threshold:
                    needs_repair = True

                t += 1
                days_since_last_inspection += 1
            
            run_survival_times.append(t)
            
        avg_survival_time = np.mean(run_survival_times)
        all_component_survival_times.append(avg_survival_time)


    final_avg_survival_time = np.mean(all_component_survival_times)
    print(f"\n--- results ---")
    print(f"每个组件的平均生存时间: {final_avg_survival_time:.2f} days")
    
    np.save(output_path, np.array(all_component_survival_times))
    print(f"✅save to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Naive Heuristic baseline policy.")
    

    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output_path", type=str, default="results_local_test/test_outputs/baseline_survival_times.npy")
    parser.add_argument('--total_budget', type=float, default=100.0)
    parser.add_argument('--inspection_interval', type=int, default=30)
    parser.add_argument('--repair_threshold', type=float, default=5.0)

    args = parser.parse_args()

    run_naive_heuristic_baseline(
        num_runs=args.num_runs,
        output_path=args.output_path,
        total_budget=args.total_budget,
        inspection_interval=args.inspection_interval,
        repair_threshold=args.repair_threshold,
        n_components=args.n_components
    )