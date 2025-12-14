from stable_baselines3 import PPO
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_dir = os.path.join(parent_dir, 'env')
sys.path.append(env_dir)

from meta_ppo_env import MetaPPOEnv
from component_pomdp_repair import Component

def meta_ppo_test(num_components):
    components_data_path = os.path.join(env_dir, "1000_components_data.csv")
    components_df = pd.read_csv(components_data_path)
    component_ids = list(components_df['component_id'])
    component_ids = component_ids[:num_components]
    replacement_costs = list(components_df['replacement_cost'])
    replacement_costs = replacement_costs[:num_components]
    inspection_costs = list(components_df['inspection_cost'])
    inspection_costs = inspection_costs[:num_components]
    component_type_names = list(components_df['component_type_name'])
    component_type_names = component_type_names[:num_components]
    shape_factors = list(components_df['shape'])
    shape_factors = shape_factors[:num_components]
    scale_factors = list(components_df['scale'])
    scale_factors = scale_factors[:num_components]

    budget_vals = np.load(f'results/baseline_budget_split.npy')
    num_runs = 100
    avg_ttfs = np.zeros((len(component_ids),num_runs))
    replace_actions_data = np.zeros((len(component_ids),num_runs))
    inspect_actions_data = np.zeros((len(component_ids),num_runs))
    cost_incurred_data = np.zeros((len(component_ids),num_runs))
    for i in range(len(component_ids)):
        component = Component(name=component_type_names[i], initial_health=100, initial_cost_incurred=0, inspect_cost=inspection_costs[i], replace_cost=replacement_costs[i], importance_score=1.0, dynamics_scale=scale_factors[i], dynamics_shape=shape_factors[i], budget=budget_vals[i], max_steps=100, component_id=component_ids[i])
        mdp_policy_path = f'data/oracle_policies_baseline_budget_split/Component_{component_ids[i]}/value_iteration_policy_budget_{budget_vals[i]}.npy'
        mdp_policy = np.load(mdp_policy_path)
        env = MetaPPOEnv(component=component, mdp_policy=mdp_policy)

        model = PPO.load(f'results/oracle_guided_meta_ppo/oracle_guided_meta_ppo')
        
        for k in range(num_runs):
            _,_ = env.reset()
            terminated = False
            truncated = False
            while not terminated or not truncated:
                action, _ = model.predict(env.true_state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                if env.true_state[0] == 0 or truncated:
                    break

            avg_ttfs[i,k] = env.component.num_steps
            replace_actions_data[i,k] = len(np.where(np.array(env.action_history) == 2)[0])
            inspect_actions_data[i,k] = len(np.where(np.array(env.action_history) == 1)[0])
            cost_incurred_data[i,k] = env.true_state[1]

    np.save(f'results/oracle_guided_meta_ppo_baseline_budget_split/ppo_avg_ttfs_{num_components}_components_optimal_split.npy', avg_ttfs)
    np.save(f'results/oracle_guided_meta_ppo_baseline_budget_split/ppo_replace_actions_data_{num_components}_components_optimal_split.npy', replace_actions_data)
    np.save(f'results/oracle_guided_meta_ppo_baseline_budget_split/ppo_inspect_actions_data_{num_components}_components_optimal_split.npy', inspect_actions_data)
    np.save(f'results/oracle_guided_meta_ppo_baseline_budget_split/ppo_cost_incurred_data_{num_components}_components_optimal_split.npy', cost_incurred_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_components', type=int, default=1000)
    args = parser.parse_args()
    meta_ppo_test(args.num_components)