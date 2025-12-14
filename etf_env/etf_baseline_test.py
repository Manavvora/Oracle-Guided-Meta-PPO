#!/usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm
from baselineenv import ETFPOMDPEnv

def run_baseline(
    num_particles: int = 500,
    output_dir: str = "results/etf_baseline_test"
):
    os.makedirs(output_dir, exist_ok=True)

    env = ETFPOMDPEnv(num_particles=num_particles)
    T = env.T
    n = env.n

    w_trace = np.zeros((T, n), dtype=np.float32)
    cum_returns = np.zeros(T, dtype=np.float32)

    obs, _ = env.reset()
    w_trace[0] = env.w.copy()

    for t in tqdm(range(T-1), desc="Baseline POMDP Test"):
        price_now = env.V_test[t]
        price_next = env.V_test[t+1]
        gains = env.w * (price_next - price_now)
        j_star = int(np.argmax(gains))


        obs, reward, done, truncated, info = env.step(j_star)

        w_trace[t+1] = env.w.copy()
        cum_returns[t+1] = cum_returns[t] + reward
        if done or truncated:
            break

    w_path = os.path.join(output_dir, "baseline_w_trace.npy")
    r_path = os.path.join(output_dir, "baseline_cum_returns.npy")
    np.save(w_path, w_trace)
    np.save(r_path, cum_returns)
    print(f"Saved weight trace → {w_path}")
    print(f"Saved cumulative returns → {r_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("ETF Baseline POMDP Test")
    parser.add_argument(
        "--num_particles", type=int, default=500,
        help="Number of particles for belief filtering"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/etf_baseline_test",
        help="Directory to save baseline test results"
    )
    args = parser.parse_args()
    run_baseline(
        num_particles=args.num_particles,
        output_dir=args.output_dir
    )
