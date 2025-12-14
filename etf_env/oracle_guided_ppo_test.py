#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from env.sub_etf_env import SubETFEnv

def test_oracle_guided(oracle_model_dir: str, num_runs: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(100):
        model_path = os.path.join(oracle_model_dir, f"oracle_comp_{i}", "oracle_guided_meta_ppo_final.zip")
        if not os.path.exists(model_path):
            break

        model = PPO.load(model_path)
        env = SubETFEnv(i)
        T = env.unwrapped.T
        returns = np.zeros((num_runs, T), dtype=np.float32)

        for run in tqdm(range(num_runs), desc=f"Oracle Comp {i}"):
            obs, _ = env.reset()
            done = False
            t = 0
            last_nv = 1.0
            while not done and t < T:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
                nv = info.get("net_value", last_nv)
                returns[run, t] = nv
                last_nv = nv
                t += 1
            if t < T:
                returns[run, t:] = last_nv

        out_path = os.path.join(output_dir, f"oracle_comp_{i}", "oracle_guided_meta_ppo_final.npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, returns)
        print(f"✅ Saved → {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    test_oracle_guided(args.model_dir, args.num_runs, args.output_dir)
