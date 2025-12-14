#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import argparse
from scipy.stats import skew, kurtosis

def generate_features_from_npy(input_npy_path: str, output_dir: str):

    price_matrix = np.load(input_npy_path)
    print(f"Loaded price matrix from {input_npy_path} with shape: {price_matrix.shape}")
    
    df_close = pd.DataFrame(price_matrix)
    

    print("Cleaning data: Replacing non-positive values with NaN, then filling...")

    df_close[df_close <= 0] = np.nan
    

    df_close.ffill(inplace=True)
    df_close.bfill(inplace=True)
    # ------------------------------------


    rets = df_close.pct_change()
    tickers = df_close.columns.tolist()

    records = []
    for tk in tickers:
        r = rets[tk].dropna().values
        
        if len(r) < 63:
            print(f"⚠️ Warning: Skipping asset {tk} due to insufficient valid return data after cleaning ({len(r)} points).")
            continue

        mean_ret = np.mean(r) * 252
        vol = np.std(r) * np.sqrt(252)
        sharpe = mean_ret / vol if vol > 1e-8 else 0.0
        
        mom3 = np.prod(1 + r[-63:]) - 1
        mom6 = np.prod(1 + r[-126:]) - 1 if len(r) >= 126 else np.nan
        mom12 = np.prod(1 + r[-252:]) - 1 if len(r) >= 252 else np.nan
        
        sk = skew(r)
        kt = kurtosis(r)
        
        ps = np.cumprod(1 + r)
        peak = np.maximum.accumulate(ps)
        drawdown = (ps - peak) / peak
        max_dd = drawdown.min()
        
        records.append({
            "ticker": tk, "mean_ret": mean_ret, "volatility": vol, "sharpe": sharpe,
            "momentum_3m": mom3, "momentum_6m": mom6, "momentum_12m": mom12,
            "skew": sk, "kurtosis": kt, "max_drawdown": max_dd,
        })
        
    if not records:
        raise ValueError("No valid assets found after processing. Please check your input data for widespread issues.")

    feat_df = pd.DataFrame(records).set_index("ticker")
    feat_df.dropna(inplace=True)

    volatilities = feat_df["volatility"].values
    budget_proportions = np.clip(volatilities, 0, None)
    budget_proportions /= budget_proportions.sum()
    feat_df["budget"] = budget_proportions

    os.makedirs(output_dir, exist_ok=True)
    y = feat_df['budget'].values
    X_df = feat_df.drop(columns=['budget'])
    X = X_df.values
    feature_names = X_df.columns.values

    np.save(os.path.join(output_dir, "features_X.npy"), X)
    np.save(os.path.join(output_dir, "target_y.npy"), y)
    np.save(os.path.join(output_dir, "feature_names.npy"), feature_names)

    print(f"✅ Data generated for {len(X)} assets and saved to directory: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate features from .npy price data.")

    parser.add_argument('--input_npy', type=str, default="data/V_train.npy", help='Path to input .npy file (e.g., V_train.npy)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output .npy files')
    args = parser.parse_args()
    
    generate_features_from_npy(args.input_npy, args.output_dir)