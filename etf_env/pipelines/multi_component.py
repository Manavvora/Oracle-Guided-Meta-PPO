import os
import numpy as np
import argparse
from models.random_forest import BudgetSplitModel
from models.random_forest_budget_split import load_data
from env.sub_etf_env import SubETFEnv
from etf_oracle_policy import generate_oracle_policy
from etf_oracle_guided_meta_ppo import train_oracle_guided_ppo


def allocate_budget_rf(feature_csv, model_path):
    rf = BudgetSplitModel()
    if os.path.exists(model_path):
        rf.load(model_path)
    else:
        X, y = load_data(feature_csv)
        rf.train(X, y)
        rf.save(model_path)
    X, _ = load_data(feature_csv)
    alloc = rf.predict(X)
    alloc = np.clip(alloc, 0, None)
    alloc = alloc / alloc.sum()
    return alloc


def multi_component_pipeline(feature_csv, rf_model_path,
                             oracle_output_dir, ppo_output_dir,
                             timesteps=50000):
    budgets = allocate_budget_rf(feature_csv, rf_model_path)
    n = len(budgets)

    oracle_paths = []
    for i in range(n):
        comp_env = SubETFEnv(i)
        comp_dir = os.path.join(oracle_output_dir, f"oracle_comp_{i}")
        os.makedirs(comp_dir, exist_ok=True)
        path = os.path.join(comp_dir, "oracle_policy.npy")
        generate_oracle_policy(env_id="ETF-v0", output_path=path,
                               component=i, budget_frac=budgets[i])
        oracle_paths.append(path)

    for i, oracle_path in enumerate(oracle_paths):
        comp_dir = os.path.join(ppo_output_dir, f"ppo_comp_{i}")
        os.makedirs(comp_dir, exist_ok=True)
        train_oracle_guided_ppo(
            env_id="ETF-v0",
            oracle_policy_path=oracle_path,
            total_timesteps=int(timesteps * budgets[i]),
            k_scale=5.0,
            p_hit=0.9,
            lambda_=0.3,
            output_dir=comp_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='data/features.csv')
    parser.add_argument('--budget_model', type=str, default='models/rf_budget.pkl')
    parser.add_argument('--oracle_out', type=str, default='results/oracle_policies')
    parser.add_argument('--ppo_out', type=str, default='results/ppo_models')
    parser.add_argument('--timesteps', type=int, default=100000)
    args = parser.parse_args()

    multi_component_pipeline(
        feature_csv=args.features,
        rf_model_path=args.budget_model,
        oracle_output_dir=args.oracle_out,
        ppo_output_dir=args.ppo_out,
        timesteps=args.timesteps
    )
