#file name: vanilla_meta_ppo_train.py
import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env.etf_env import MultiComponentETFEnvTrain
from env.sub_etf_env import SingleComponentETFEnv

def make_env_func(total_budget, asset_index, budget_per_comp, history_window):

    def _init():
        base_env = MultiComponentETFEnvTrain(initial_action_budget=total_budget)
        
        return SingleComponentETFEnv(
            base_env=base_env,
            asset_index=asset_index,
            allocated_budget=budget_per_comp,
            history_window=history_window,
            mode='agent'
        )
    return _init

def train_vanilla_agent(
    n_components: int,
    total_budget: float,
    timesteps: int,
    output_dir: str,
    history_window: int,
    num_cpu: int
):
    print(f"---({n_components} ---")
    
    budget_per_component = total_budget / n_components if n_components > 0 else 0
    print(f"budget {budget_per_component:.2f}")

    env_funcs = [
        make_env_func(total_budget, i, budget_per_component, history_window)
        for i in range(n_components)
    ]
    
    if num_cpu > 1 and n_components > 1:
        train_env = SubprocVecEnv(env_funcs, start_method='fork')
    else:
        train_env = DummyVecEnv(env_funcs)
    
    model_path = os.path.join(output_dir, "vanilla_ppo_unified.zip")
    os.makedirs(output_dir, exist_ok=True)
    
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=os.path.join(output_dir, "tb_logs"))
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    
    print(f"\n🎉 save: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a unified Vanilla PPO model.")
    
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="results_local_test/vanilla_models")
    parser.add_argument('--total_budget', type=float, default=20.0)
    parser.add_argument('--history_window', type=int, default=20)
    parser.add_argument('--num_cpu', type=int, default=4)
    
    args = parser.parse_args()

    train_vanilla_agent(
        n_components=args.n_components,
        total_budget=args.total_budget,
        timesteps=args.timesteps,
        output_dir=args.output_dir,
        history_window=args.history_window,
        num_cpu=args.num_cpu
    )