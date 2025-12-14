import pandas as pd
import yfinance as yf
import numpy as np
import os

print("--- [1/5]  ---")
# Step 1: Fetch S&P 500 tickers from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df = pd.read_html(url, header=0)[0]
tickers = df["Symbol"].tolist()

tickers = [t.replace(".", "-") for t in tickers]

print(f"✅  {len(tickers)} ")


print("\n--- [2/5]")
# Step 2: Download historical Adjusted Close prices for the last 3 years

data = yf.download(tickers, start="2017-01-01", end="2021-01-01", auto_adjust=True)['Close']
print("✅ 。")



print("\n--- [3/5] --")

initial_count = len(data.columns)


min_valid_points = int(len(data) * 0.90)
data.dropna(axis='columns', thresh=min_valid_points, inplace=True)


data.ffill(inplace=True)
data.bfill(inplace=True)

final_count = len(data.columns)
print(f"✅ {final_count} ")


print("\n--- [ 4/5] --")
V = data.values  # shape = (T_total, n_valid_tickers)

# Step 4: Split into train and test sets
T_test_horizon = 120  
V_train = V[:-T_test_horizon]
V_test = V[-T_test_horizon:]

final_tickers = data.columns.tolist()

print("✅ 。")


print("\n--- [5/5] ---")
# Step 5: Save to .npy files
os.makedirs("data", exist_ok=True)
np.save("data/V_train.npy", V_train)
np.save("data/V_test.npy", V_test)
np.save("data/tickers.npy", np.array(final_tickers)) 

print("\n🎉 finish！")
print(f"  - data/V_train.npy with shape: {V_train.shape}")
print(f"  - data/V_test.npy  with shape: {V_test.shape}")
print(f"  - data/tickers.npy with shape: ({len(final_tickers)},)")