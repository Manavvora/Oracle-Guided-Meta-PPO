# aggregate_generalizability.py
import re, os, glob, numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "results_120d"
RUN_RE = re.compile(r"N(\d+)__([^/]+)__E__([^/]+)$")  # N{N}__{train_tag}__E__{eval_tag}

def load_metric(dirpath, algo_prefix):
    arrs = []
    for f in glob.glob(os.path.join(dirpath, "test_outputs", f"{algo_prefix}_*.npy")):
        try:
            x = np.load(f)
            arrs.append(np.nanmean(x))
        except Exception:
            pass
    return np.nanmean(arrs) if arrs else np.nan

def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "N*__*__E__*"))):
        m = RUN_RE.search(os.path.basename(d))
        if not m: 
            continue
        N = int(m.group(1))
        train_tag = m.group(2)
        eval_tag = m.group(3)

        train_metric = load_metric(d, "meta")     
        eval_metric  = load_metric(d, "meta")     

        rows.append(dict(N=N, split="Train", metric=train_metric, train_tag=train_tag, eval_tag=eval_tag))
        rows.append(dict(N=N, split="Eval",  metric=eval_metric,  train_tag=train_tag, eval_tag=eval_tag))

    df = pd.DataFrame(rows).dropna(subset=["metric"])
    g = df.groupby(["N","split"]).agg(mean=("metric","mean"), std=("metric","std"), count=("metric","size")).reset_index()
    g = g.sort_values("N")
    g.to_csv("generalizability_summary.csv", index=False)
    print("saved generalizability_summary.csv")
    print(g)

    Ns = sorted(g["N"].unique())
    train_mean = [float(g[(g.N==n)&(g.split=="Train")]["mean"]) for n in Ns]
    train_std  = [float(g[(g.N==n)&(g.split=="Train")]["std"])  for n in Ns]
    eval_mean  = [float(g[(g.N==n)&(g.split=="Eval")]["mean"])  for n in Ns]
    eval_std   = [float(g[(g.N==n)&(g.split=="Eval")]["std"])   for n in Ns]

    plt.figure(figsize=(5.0,3.6), dpi=200)
    plt.plot(Ns, train_mean, marker="o", label="Train window")
    if not np.all(np.isnan(train_std)):
        plt.fill_between(Ns, np.array(train_mean)-np.array(train_std), np.array(train_mean)+np.array(train_std), alpha=0.15)
    plt.plot(Ns, eval_mean, marker="s", linestyle="--", label="Eval window")
    if not np.all(np.isnan(eval_std)):
        plt.fill_between(Ns, np.array(eval_mean)-np.array(eval_std), np.array(eval_mean)+np.array(eval_std), alpha=0.15)

    plt.xlabel("Number of ETFs (N)")
    plt.ylabel("Performance (↑)")  
    plt.title("Generalizability / Robustness across N and windows")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_generalizability.pdf")
    print("saved fig_generalizability.pdf")

if __name__ == "__main__":
    main()
