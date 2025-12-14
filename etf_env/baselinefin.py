# baselinefin.py (Final version with robust ticker handling)

import os
import numpy as np

DATA_DIR = 'data/'
V_TRAIN_PATH = os.path.join(DATA_DIR, 'V_train.npy')
V_TEST_PATH  = os.path.join(DATA_DIR, 'V_test.npy')
TICKERS_PATH = os.path.join(DATA_DIR, 'tickers.npy')

V_train = np.load(V_TRAIN_PATH)
V_test = np.load(V_TEST_PATH)
n = V_train.shape[1]
T_test, _ = V_test.shape

try:
    tickers = np.load(TICKERS_PATH, allow_pickle=True).tolist()
    print(f"Successfully loaded {len(tickers)} tickers from {TICKERS_PATH}")
    if len(tickers) != n:
        raise ValueError("Number of tickers does not match number of assets in V_train.npy")
except FileNotFoundError:
    print(f"Warning: '{TICKERS_PATH}' not found.")
    print("Generating placeholder tickers (e.g., 'ETF_0', 'ETF_1')...")
    tickers = [f'ETF_{i}' for i in range(n)]

print(f"baselinefin.py configured for {n} assets.")
print(f"  - V_train shape for model parameters: {V_train.shape}")
print(f"  - V_test shape for environment execution: {V_test.shape}")



print("Calculating T_mats from training data...")
T_mats = {}
for i in range(n):
    tkr = tickers[i]
    price_series = V_train[:, i]
    returns = np.sign(np.diff(price_series))
    states = (returns + 1).astype(int)
    
    counts = np.zeros((3, 3), dtype=float)
    for s, s_next in zip(states[:-1], states[1:]):
        counts[s, s_next] += 1
    
    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row_sums, where=row_sums != 0)
    P[row_sums.flatten() == 0, :] = 1.0 / 3.0
    T_mats[tkr] = P

print("Calculating A_asset from training data...")
signs_train = np.sign(V_train[1:] - V_train[:-1])
P_asset = np.zeros((n, n), dtype=float)
N_asset = np.zeros((n, n), dtype=float)
for diff_row in signs_train:
    same = diff_row[:, None] * diff_row[None, :] >= 0
    P_asset[same] += 1
    N_asset[~same] += 1
den = P_asset + N_asset
R_asset = np.divide(P_asset, den, out=np.zeros_like(P_asset), where=den != 0)
sum_R = R_asset.sum(axis=1, keepdims=True)
A_asset = R_asset - (1 + (sum_R - R_asset)) / (n - 1)
np.fill_diagonal(A_asset, 1.0)

p_hit = 0.9
O_obs_model = np.full((3, 3), (1 - p_hit) / 2.0)
np.fill_diagonal(O_obs_model, p_hit)

b_belief = np.ones(3) / 3.0
w_weights_current = np.ones(n) / n

k_tanh  = 5.0
alpha_D = 0.05

__all__ = [
    'tickers', 'n', 'T_mats', 'A_asset', 'V_test', 'O_obs_model',
    'b_belief', 'w_weights_current', 'T_test', 'k_tanh', 'alpha_D',
]

print("✅ baselinefin.py finished execution and all variables are ready for import.")