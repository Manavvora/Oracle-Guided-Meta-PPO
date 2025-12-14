from stable_baselines3 import PPO
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import seaborn as sns
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_dir = os.path.join(parent_dir, 'env')
sys.path.append(env_dir)

from baseline_env import BaselineEnv
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

inspect_interval = 5
repair_threshold = 15

for i in range(len(component_ids)):
    print(f'Component: {component_ids[i]}')
    for j in range(len(budget_vals)):

        component = Component(name=component_type_names[i], initial_health=100, initial_cost_incurred=0, inspect_cost=inspection_costs[i], replace_cost=replacement_costs[i], importance_score=1.0, dynamics_scale=scale_factors[i], dynamics_shape=shape_factors[i], budget=budget_vals[j], max_steps=100, component_id=component_ids[i])
        
        env = BaselineEnv(component=component)

        for k in range(num_runs):
            b,_ = env.reset()
            terminated = False
            truncated = False
            num_steps = 1

            while not terminated or not truncated:
                # print(f'Step: {num_steps}')
                # b[1] += random.uniform(2,5)
                if component.budget - b[1] > env.component.replace_cost and env.true_state[0] > env.component.failure_condition and b[0] < repair_threshold:
                    action = 2
                elif component.budget - b[1] > env.component.inspect_cost and env.true_state[0] > env.component.failure_condition and num_steps % inspect_interval == 0:
                    action = 1
                else:
                    action = 0
                next_state = env.component.trans_prob[int(b[0]), action, :]*env.component.healths
                next_state = np.sum(next_state)
                b, reward, terminated, truncated, info = env.step(action)
                b[0] = next_state
                num_steps += 1
                if env.true_state[0] == 0 or truncated:
                    break
                
            avg_ttfs[i,j,k] = env.component.num_steps
            replace_actions_data[i,j,k] = len(np.where(np.array(env.action_history) == 2)[0])
            inspect_actions_data[i,j,k] = len(np.where(np.array(env.action_history) == 1)[0])
            cost_incurred_data[i,j,k] = env.true_state[1]

np.save(f'results/realistic_baseline/realistic_baseline_avg_ttfs_all_components.npy', avg_ttfs)
np.save(f'results/realistic_baseline/realistic_baseline_replace_actions_data_all_components.npy', replace_actions_data)
np.save(f'results/realistic_baseline/realistic_baseline_inspect_actions_data_all_components.npy', inspect_actions_data)
np.save(f'results/realistic_baseline/realistic_baseline_cost_incurred_data_all_components.npy', cost_incurred_data)