from stable_baselines3 import PPO
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_dir = os.path.join(parent_dir, 'env')
sys.path.append(env_dir)

from meta_ppo_env import MetaPPOEnv
from component_pomdp_repair import Component

components_data_path = os.path.join(env_dir, "1000_components_data.csv")
components_df = pd.read_csv(components_data_path)
component_ids = list(components_df['component_id'])
replacement_costs = list(components_df['replacement_cost'])
inspection_costs = list(components_df['inspection_cost'])
component_type_names = list(components_df['component_type_name'])
shape_factors = list(components_df['shape'])
scale_factors = list(components_df['scale'])

budget_vals = np.arange(0, 5500, 500)
num_runs = 100
avg_ttfs = np.zeros((len(component_ids),len(budget_vals),num_runs))
replace_actions_data = np.zeros((len(component_ids),len(budget_vals),num_runs))
inspect_actions_data = np.zeros((len(component_ids),len(budget_vals),num_runs))
cost_incurred_data = np.zeros((len(component_ids),len(budget_vals),num_runs))
for i in range(len(component_ids)):
    print(f'Component {component_ids[i]}/{len(component_ids)}')
    for j in range(len(budget_vals)):
        component = Component(name=component_type_names[i], initial_health=100, initial_cost_incurred=0, inspect_cost=inspection_costs[i], replace_cost=replacement_costs[i], importance_score=1.0, dynamics_scale=scale_factors[i], dynamics_shape=shape_factors[i], budget=budget_vals[j], max_steps=100, component_id=component_ids[i])
        oracle_policy_path = f'data/oracle_policies/Component_{component_ids[i]}/value_iteration_policy_budget_{budget_vals[j]}.npy'
        mdp_policy = np.load(oracle_policy_path)
        env = MetaPPOEnv(component=component, mdp_policy=mdp_policy)

        model = PPO.load(f'results/oracle_guided_meta_ppo/oracle_guided_meta_ppo')

        ttfs = []
        cost_incurred = []
        replacements = []
        inspections = []
        for k in range(num_runs):
            _,_ = env.reset()
            terminated = False
            truncated = False
            while not terminated or not truncated:
                # print(f'True State: {env.true_state}')
                action, _ = model.predict(env.true_state, deterministic=True)
                # print(f'Action: {action}')
                state, reward, terminated, truncated, info = env.step(action)
                if env.true_state[0] == 0 or truncated:
                    break

            avg_ttfs[i,j,k] = env.component.num_steps
            replace_actions_data[i,j,k] = len(np.where(np.array(env.action_history) == 2)[0])
            inspect_actions_data[i,j,k] = len(np.where(np.array(env.action_history) == 1)[0])
            cost_incurred_data[i,j,k] = env.true_state[1]

np.save(f'results/oracle_guided_meta_ppo/ppo_avg_ttfs_all_components.npy', avg_ttfs)
np.save(f'results/oracle_guided_meta_ppo/ppo_replace_actions_data_all_components.npy', replace_actions_data)
np.save(f'results/oracle_guided_meta_ppo/ppo_inspect_actions_data_all_components.npy', inspect_actions_data)
np.save(f'results/oracle_guided_meta_ppo/ppo_cost_incurred_data_all_components.npy', cost_incurred_data)