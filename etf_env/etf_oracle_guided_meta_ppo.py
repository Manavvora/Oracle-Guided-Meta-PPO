#!/usr/bin/env python3
import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

import env.etf_env  

def train_oracle_guided_ppo(
    env_id: str,
    oracle_policy_path: str,
    total_timesteps: int,
    k_scale: float,
    p_hit: float,
    lambda_: float,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(env_id)
    env.k_scale = k_scale
    env.p_hit   = p_hit
    env.lambda_ = lambda_

    oracle_pi = np.load(oracle_policy_path)
    

    env.unwrapped.set_oracle_policy(oracle_pi)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb_logs")
    )

    model.learn(total_timesteps=total_timesteps)

    model_path = os.path.join(output_dir, "oracle_guided_meta_ppo")
    model.save(model_path)
    print(f"🔖 Saved trained model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Oracle-Guided Meta-PPO on ETF-v0"
    )
    parser.add_argument("--env_id", default="ETF-v0",
                        help="Gym environment ID")
    parser.add_argument("--oracle_policy", default="results/etf_oracle_policies/etf_oracle_policy.npy",
                        help="Path to one-step oracle policy (.npy)")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total PPO training timesteps")
    parser.add_argument("--k_scale", type=float, default=5.0,
                        help="tanh scale k")
    parser.add_argument("--p_hit", type=float, default=0.9,
                        help="observation probability")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.3,
                        help="EMA smoothing λ")
    parser.add_argument("--output_dir", default="results/etf_oracle_guided_meta_ppo",
                        help="Where to save model & logs")
    args = parser.parse_args()

    train_etf_oracle_guided(
        env_id=args.env_id,
        oracle_policy_path=args.oracle_policy,
        total_timesteps=args.timesteps,
        k_scale=args.k_scale,
        p_hit=args.p_hit,
        lambda_=args.lambda_,
        output_dir=args.output_dir
    )
