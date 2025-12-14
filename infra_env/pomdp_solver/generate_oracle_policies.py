import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
import time
import sys
import os
from multiprocessing import Pool

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_dir = os.path.join(parent_dir, 'env')
sys.path.append(env_dir)

from component_mdp_repair import Component

class ValueIteration:

    def __init__(self, env, Vfunc, theta=5e-3, gamma=0.95):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.probs = self.env.trans_prob
        self.Vfunc = Vfunc

    def value_iteration(self, plots = True):
        V = self.Vfunc

        pi = np.ones([self.env.num_healths,self.env.num_costs+1,self.env.num_actions])/self.env.num_actions
        delta = np.inf
        iters = 0
        while delta > self.theta:
            delta = 0
            iters += 1
            count = 0
            for s in range(self.env.num_healths):
                for c in range(self.env.num_costs-1):
                    count += 1
                    v = V[s,c]
                    V_temp = np.zeros(self.env.num_actions)
                    for a in range(self.env.num_actions):
                        for s_new in range(self.env.num_healths):
                            if a == 0:
                                c_new = c
                            else:
                                c_new = c + 1
                            V_temp[a] +=  self.probs[s,a,s_new]*(self.env.reward_func([s,self.env.cost_array[c]]) + self.gamma*V[s_new,c_new])
                    V[s,c] = np.max(V_temp)
                    pi[s,c] = np.eye(self.env.num_actions)[np.argmax(V_temp)]
                    delta = max(delta, np.abs(v-V[s,c]))

        return V,pi
    
def process_component(component_id):

    replacement_cost = components_df.loc[components_df['component_id'] == component_id, 'replacement_cost'].values[0]
    component_type_name = components_df.loc[components_df['component_id'] == component_id, 'component_type_name'].values[0]
    inspection_cost = components_df.loc[components_df['component_id'] == component_id, 'inspection_cost'].values[0]
    shape_factor = components_df.loc[components_df['component_id'] == component_id, 'shape'].values[0]
    scale_factor = components_df.loc[components_df['component_id'] == component_id, 'scale'].values[0]
    print(f'Value Iteration for Component {component_id} with replacement cost {replacement_cost} and inspection cost {inspection_cost}')
    Vfunc = np.zeros((101, 101))
    Vfunc[:, 100] = -10
    for budget in budget_vals:
        # print(f'Value Iteration for Component {component_id} with budget {budget}')
        if not os.path.exists(f'data/oracle_policies/Component_{component_id}'):
            os.makedirs(f'data/oracle_policies/Component_{component_id}')
        # if not os.path.exists(f'./value_iter_traj/Component_{component_id}'):
        #     os.makedirs(f'./value_iter_traj/Component_{component_id}')
        env = Component(name=component_type_name, initial_health=100, initial_cost_incurred=0, inspect_cost=inspection_cost, 
                                  replace_cost=replacement_cost, importance_score=1.0, dynamics_scale=scale_factor, dynamics_shape=shape_factor, 
                                  budget=budget, max_steps=100, component_id=component_id)
        V, pi = ValueIteration(env, Vfunc).value_iteration(plots=False)
        Vfunc = V
        np.save(f'data/oracle_policies/Component_{component_id}/value_iteration_policy_budget_{budget}.npy', pi)
        print(f'Value Iteration for Component {component_id} with budget {budget} completed')

components_data_path = os.path.join(env_dir, '1000_components_data.csv')
components_df = pd.read_csv(components_data_path)
components_df = components_df[components_df['component_id']]
components_df = components_df[components_df['component_id']]
budget_vals = np.arange(0,5500,500)

if __name__ == '__main__':

    component_ids = list(components_df['component_id'])
    with Pool(processes=8) as pool:  
        pool.map(process_component, component_ids)