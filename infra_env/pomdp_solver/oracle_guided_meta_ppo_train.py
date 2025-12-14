from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import numpy as np
import os
import sys
import pandas as pd

# Setup paths and load your environment configuration as before
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_dir = os.path.join(parent_dir, 'env')
sys.path.append(env_dir)

from meta_ppo_env import MetaPPOEnv
from component_pomdp_repair import Component

components_data_path = os.path.join(env_dir, "1000_components_data.csv")
components_df = pd.read_csv(components_data_path)
component_ids = list(components_df['component_id'])
train_indices = np.random.choice(len(component_ids), 10, replace=False)
replacement_costs = list(components_df['replacement_cost'])
inspection_costs = list(components_df['inspection_cost'])
component_type_names = list(components_df['component_type_name'])
shape_factors = list(components_df['shape'])
scale_factors = list(components_df['scale'])

budgets = np.array([0,500,1500,2500,300,700,4000]) #Randomly selected budgets
first_time = True
previous_model_path = None
for index in train_indices:
    for budget in budgets:
        print(f"Training PPO for Component {component_ids[index]} with budget {budget}")
        component = Component(name=component_type_names[index], initial_health=100, initial_cost_incurred=0,
                            inspect_cost=inspection_costs[index], replace_cost=replacement_costs[index], importance_score=1.0,
                            dynamics_scale=scale_factors[index], dynamics_shape=shape_factors[index], budget=budget, max_steps=100,
                            component_id=component_ids[index])

        oracle_policy_path = os.path.join(f'data/oracle_policies/Component_{component_ids[index]}/value_iteration_policy_budget_{budget}.npy')
        mdp_policy = np.load(oracle_policy_path)
        env = MetaPPOEnv(component=component, mdp_policy=mdp_policy)

        if first_time:
        # Initialize the PPO model
            model = PPO(MlpPolicy, 
                        env, 
                        verbose=0, 
                        tensorboard_log="meta_latest/log",
                        learning_rate=1e-4, 
                        n_steps=4096, 
                        batch_size=128, 
                        n_epochs=10, 
                        gamma=0.95,
                        gae_lambda=0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                        clip_range=0.2,  # Clip parameter for PPO, defines the cutoff for probability ratios
                        ent_coef=0.01,  # Entropy coefficient for the loss calculation
                        vf_coef=1)  # Value function coefficient in the loss calculation
            first_time = False
        else:
            print(previous_model_path)
            model = PPO.load(previous_model_path, env=env, tensorboard_log="meta_latest/log")

        # Train the model
        model.learn(total_timesteps=200000)

        model_path = f"meta_latest/ppo_{component_ids[index]}_{budget}"
        model.save(model_path)
        previous_model_path = model_path

model.save(f"oracle_guided_meta_ppo/oracle_guided_meta_ppo")  # Save the final model
np.save("data/train_indices_meta_ppo.npy", train_indices)