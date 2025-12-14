# etf_oracle_policy.py (Final Corrected Version)

import os
import numpy as np
import pandas as pd
import argparse

from env.etf_env import ETFEnv 

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:

    delta = prices.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()


    rs = gain / loss
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
    rs.fillna(method='ffill', inplace=True)
    
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_oracle_policy(
    output_dir: str,
    asset_index: int,
    initial_budget: float,
    inspection_cost: float,
    max_drawdown_threshold: float,
    inaction_cost: float,
    belief_penalty_factor: float, # [新] 增加 belief_penalty_factor 参数
    # 策略本身的参数
    long_ma_window: int = 40,
    rsi_period: int = 14,
    rsi_threshold: float = 50.0
):

    print(f"--- Generating Upgraded Oracle Policy for component {asset_index} ---")
    

    env = ETFEnv(
        initial_budget=initial_budget,
        inspection_cost=inspection_cost,
        max_drawdown_threshold=max_drawdown_threshold,
        inaction_cost=inaction_cost,
        belief_penalty_factor=belief_penalty_factor
    )

    V = env.V_test
    T = env.T
    n = env.n

    prices = V[:, asset_index]
    price_series = pd.Series(prices)


    ma_long = price_series.rolling(window=long_ma_window).mean()
    
    rsi = calculate_rsi(price_series, period=rsi_period)

    policy = np.zeros((T, n), dtype=np.float32)


    start_index = max(long_ma_window, rsi_period)
    for t in range(start_index, T):
        

        is_uptrend = prices[t] > ma_long[t]
        has_momentum = rsi[t] > rsi_threshold
        
        if is_uptrend and has_momentum:
            policy[t, asset_index] = 1.0

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "oracle_policy.npy")
    np.save(output_path, policy)
    print(f"✅ Saved upgraded oracle policy for asset {asset_index} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an upgraded, heuristic-based Oracle policy.")
    parser.add_argument("--asset_index", type=int, required=True, help="The index of the asset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the oracle_policy.npy file.")


    parser.add_argument('--initial_budget', type=float, default=20.0)
    parser.add_argument('--inspection_cost', type=float, default=0.2)
    parser.add_argument('--max_drawdown', type=float, default=0.20)
    parser.add_argument('--inaction_cost', type=float, default=0.05)
    parser.add_argument('--belief_penalty_factor', type=float, default=0.01) 
    args = parser.parse_args()


    generate_oracle_policy(
        output_dir=args.output_dir,
        asset_index=args.asset_index,
        initial_budget=args.initial_budget,
        inspection_cost=args.inspection_cost,
        max_drawdown_threshold=args.max_drawdown,
        inaction_cost=args.inaction_cost,
        belief_penalty_factor=args.belief_penalty_factor
    )