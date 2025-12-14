import numpy as np
import pandas as pd
from component_mdp_repair import Component

#Load components data
components_data_path = "benchmark_components.csv"
components_df = pd.read_csv(components_data_path)
component_ids = list(components_df['component_id'])
replacement_costs = list(components_df['replacement_cost'])
inspection_costs = list(components_df['inspection_cost'])
component_type_names = list(components_df['component_type_name'])
shape_factors = list(components_df['shape'])
scale_factors = list(components_df['scale'])


num_runs = 10
ttfs = np.zeros((len(component_ids),num_runs))
expected_ttfs = np.zeros(len(component_ids))
variance_ttfs = np.zeros(len(component_ids))
                         


for index in range(len(component_ids)):
    print(f'Gathering transition statistics for component {component_ids[index]}')
    env = Component(name=component_type_names[index], initial_health=100, initial_cost_incurred=0, inspect_cost=inspection_costs[index], replace_cost=replacement_costs[index], importance_score=1.0, dynamics_scale=scale_factors[index], dynamics_shape=shape_factors[index], budget=500, max_steps=100, component_id=component_ids[index])

    for k in range(num_runs):
        s = env.reset()
        done = False
        run_duration = 0
        while not done:
            run_duration += 1
            probs = np.ones(2)
            a = 0
            s, r, done = env.step(a)
        ttfs[index,k] = run_duration
    expected_ttfs[index] = np.mean(ttfs[index,:])
    variance_ttfs[index] = np.var(ttfs[index,:])

np.save('benchmark_expected_ttfs.npy', expected_ttfs)
np.save('benchmark_variance_ttfs.npy', variance_ttfs)
