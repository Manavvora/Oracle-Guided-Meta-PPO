import os
import numpy as np
import pandas as pd

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_dir = os.path.join(parent_dir, 'env')
components_data_path = os.path.join(env_dir, "10000_components_data.csv")

# Load data
components_df = pd.read_csv(components_data_path)
component_ids = list(components_df['component_id'])[:100]
replacement_costs = np.array(components_df['replacement_cost'])[:100]
inspection_costs = np.array(components_df['inspection_cost'])[:100]
component_type_names = list(components_df['component_type_name'])[:100]
shape_factors = np.array(components_df['shape'])[:100]
scale_factors = np.array(components_df['scale'])[:100]
expected_ttfs_path = os.path.join(env_dir, 'expected_ttfs.npy')
expected_ttfs = np.load(expected_ttfs_path)[:100]

# Calculate mean expected TTFs over runs
expected_ttfs_over_runs = np.mean(expected_ttfs, axis=1)

# Total budget
budget = 50000

# Calculate the budget allocation ratio for each component
budget_allocation_ratios = replacement_costs / expected_ttfs_over_runs

# Normalize the ratios so they sum to 1
normalized_ratios = budget_allocation_ratios / np.sum(budget_allocation_ratios)

# Allocate the budget based on the normalized ratios
allocated_budgets = normalized_ratios * budget
np.save('results/baseline_budget_split_100_components.npy', allocated_budgets)