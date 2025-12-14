import numpy as np
import pandas as pd
import cvxpy as cp
import os
import joblib
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_dir = os.path.join(parent_dir, 'env')

def budget_split(num_components):
    # Load data
    components_data_path = os.path.join(env_dir, "1000_components_data.csv")
    components_df = pd.read_csv(components_data_path)
    component_ids = list(components_df['component_id'])
    component_ids = component_ids[:num_components]

    # Load random forest model
    model = joblib.load('results/random_forest_regressor.joblib')
    component_info = np.load('data/1000_components_info.npy')
    component_info = component_info[:num_components]

    expected_ttfs_path = os.path.join(env_dir, 'expected_ttfs.npy')
    expected_ttfs = np.load(expected_ttfs_path)
    expected_ttfs_over_runs = np.mean(expected_ttfs, axis=1)
    expected_ttfs_over_runs = expected_ttfs_over_runs[:num_components]

    c_vals = 100 * np.ones(len(component_ids))
    a_vals = expected_ttfs_over_runs - c_vals
    b_vals = model.predict(component_info)

    x = cp.Variable(len(a_vals))
    t = cp.Variable()

    objective = cp.Maximize(t)
    constraints = [cp.sum(x) <= 500 * num_components, x >= 0]

    for i in range(len(component_ids)):
        constraints.append(t <= a_vals[i] * cp.exp(b_vals[i] * x[i]) + c_vals[i])

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)

    np.save(f'results/budget_split_maxmin_{num_components}_components.npy', x.value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_components', type=int, default=1000)
    args = parser.parse_args()

    budget_split(args.num_components)
