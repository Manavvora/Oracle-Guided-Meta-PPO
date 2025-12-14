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
    components_data_path = os.path.join(env_dir, "benchmark_components.csv")
    components_df = pd.read_csv(components_data_path)
    component_ids = list(components_df['component_id'])
    component_ids = component_ids[:num_components]

    #Load random forest model
    model = joblib.load('results/random_forest_regressor.joblib')
    component_info = np.load('benchmark_data/benchmark_components_info.npy')
    component_info = component_info[:num_components]

    expected_ttfs_path = os.path.join(env_dir, 'benchmark_expected_ttfs.npy')
    expected_ttfs = np.load(expected_ttfs_path)
    expected_ttfs = expected_ttfs[:num_components]

    c_vals = 100*np.ones(len(component_ids))
    a_vals = expected_ttfs - c_vals
    b_vals = model.predict(component_info) 

    x = cp.Variable(len(a_vals))
    objective = cp.Maximize(cp.sum([a_vals[i]*cp.exp(b_vals[i]*x[i]) + c_vals[i]
                                    for i in range(len(component_ids))]))
    constraint = [cp.sum(x) <= 200000,x>=0]
    prob = cp.Problem(objective,constraint)
    result = prob.solve(verbose=False)
    np.save(f'benchmark_results/budget_split_{num_components}_components.npy', x.value)

    repeated_split = np.tile(x.value, 100)
    np.save(f'benchmark_results/budget_split_100000_components.npy', repeated_split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_components', type=int, default=1000)
    args = parser.parse_args()

    budget_split(args.num_components)