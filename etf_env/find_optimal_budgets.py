# 文件名: find_optimal_budgets.py
import os
import argparse
import numpy as np
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp

from env.etf_env import MultiComponentETFEnvTrain
from env.sub_etf_env import SingleComponentETFEnv


try:
    if mp.get_start_method(allow_none=True) != 'fork':
        mp.set_start_method('fork', force=True)
except RuntimeError:
    print("waring。")

def run_simulation_for_budget(
    asset_index: int,
    budget_level: float,
    total_budget: float,
    num_runs: int,
    repair_threshold: float,
    history_window: int
) -> float:

    run_survival_times = []

    base_env = MultiComponentETFEnvTrain(initial_action_budget=total_budget)
    
    for _ in range(num_runs):
        sub_env = SingleComponentETFEnv(
            base_env=base_env, asset_index=asset_index,
            allocated_budget=budget_level, mode='oracle',
            history_window=history_window
        )
        base_env.reset()
        obs, _ = sub_env.reset()
        done = False
        t = 0
        while not done:

            true_risk_capital = obs[-1]
            action = 2 if true_risk_capital < repair_threshold else 0
            obs, reward, done, _, info = sub_env.step(action)
            t += 1
        run_survival_times.append(t)
    
    return np.mean(run_survival_times)

def find_optimal_budget_for_component(
    asset_index: int,
    total_budget: float,
    budget_test_levels: np.ndarray,
    num_runs_per_level: int,
    oracle_repair_threshold: float,
    history_window: int
) -> float:

    best_budget = -1
    max_survival_time = -1

    for budget in budget_test_levels:
        avg_survival = run_simulation_for_budget(
            asset_index, budget, total_budget, num_runs_per_level,
            oracle_repair_threshold, history_window
        )
        if avg_survival > max_survival_time:
            max_survival_time = avg_survival
            best_budget = budget
    

    return (asset_index, best_budget, max_survival_time)

def generate_optimal_budget_dataset(args):

    print(''traing'')


    component_indices = np.random.choice(range(args.n_total_components), size=args.sample_size, replace=False)
    

    budget_per_comp_avg = args.total_budget / args.n_total_components
    budget_test_levels = np.linspace(budget_per_comp_avg * 0.1, budget_per_comp_avg * 5, 10)
    print(f"level:\n{budget_test_levels}")


    tasks = [delayed(find_optimal_budget_for_component)(
                i, args.total_budget, budget_test_levels, 
                args.num_runs_per_level, args.oracle_repair_threshold,
                args.history_window
            ) for i in component_indices]
    
    results = Parallel(n_jobs=args.num_cpu)(
        tqdm(tasks, desc="optimal")
    )


    results.sort(key=lambda x: x[0])
    

    final_indices = [res[0] for res in results]
    optimal_budgets = [res[1] for res in results]
    

    all_features_X = np.load(os.path.join(args.feature_dir, "features_X.npy"))
    

    new_X = all_features_X[final_indices]
    new_y = np.array(optimal_budgets)
    

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "features_X_optimal.npy"), new_X)
    np.save(os.path.join(args.output_dir, "target_y_optimal.npy"), new_y)
    
    print("\n---finish---")
    print(f"{len(new_y)} ")
    print(f"{args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optimal budget dataset for Random Forest training.")

    parser.add_argument('--feature_dir', type=str, default="results_local_test/features")
    parser.add_argument('--output_dir', type=str, default="results_local_test/optimal_budget_data")
    parser.add_argument('--n_total_components', type=int, default=471)
    parser.add_argument('--sample_size', type=int, default=100, help="agents")
    parser.add_argument('--num_runs_per_level', type=int, default=10, help="episode")
    parser.add_argument('--total_budget', type=float, default=50000.0)
    parser.add_argument('--oracle_repair_threshold', type=float, default=20.0)
    parser.add_argument('--history_window', type=int, default=20)
    parser.add_argument('--num_cpu', type=int, default=4)
    args = parser.parse_args()
    
    generate_optimal_budget_dataset(args)