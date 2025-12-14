import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import h5py

class Component:

    def __init__(self, name, initial_health, initial_cost_incurred, inspect_cost, replace_cost, importance_score, dynamics_scale, dynamics_shape, budget, max_steps, component_id):

        # component parameters
        self.name = name
        self.budget = budget
        self.budget_finished = False
        self.component_died = False
        self.budget_index = budget//1000
        self.inspect_cost = inspect_cost
        self.replace_cost = replace_cost
        # print(self.budget_index)
        self.cost_incurred = initial_cost_incurred
        self.num_healths = 101
        self.states = np.arange(0, self.num_healths, 1)
        self.cost_array = np.arange(0, 101*int(self.replace_cost), int(self.replace_cost))
        self.num_costs = len(self.cost_array)
        self.num_states = 2
        self.num_health_obs = self.num_healths + 1
        self.num_cost_obs = self.num_costs + 1
        self.num_actions = 2
        self.initial_health = initial_health  # Initial health of the component
        self.healths = np.arange(0, self.num_healths, 1, dtype=int)
        self.actions = np.arange(0, self.num_actions, 1, dtype=int)
        self.failure_condition = 0
        self.trans_prob = np.zeros((self.num_healths, self.num_actions, self.num_healths))
        self.i_score = importance_score

        # component logging
        self.initial_state = [self.initial_health, self.cost_incurred]
        self.current_state = self.initial_state
        self.time_to_failure = 1
        self.num_steps = 0
        self.check_first_zero = 0
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.offline_count = 0
        self.offline_steps = []
        self.cost_history = []
        self.max_steps = max_steps
        self.component_id = component_id
        self.tolerance = 2

        self.dynamics_scale = dynamics_scale
        self.dynamics_shape = dynamics_shape
        self.trans_prob = self.gen_trans_prob()

    def reset(self):
        """
        Reset the component state and histories
        """

        self.current_state = [self.initial_health, self.cost_incurred]
        self.num_steps = 0
        self.time_to_failure = 1
        self.check_first_zero = 0
        self.budget_finished = False
        self.component_died = False
        self.state_history = [self.current_state.copy()]
        self.action_history = []
        self.reward_history = []
        self.cost_history = []
        self.offline_count = 0
        self.offline_steps = []
        return self.current_state.copy()
    
    def synthesize_dynamics(self, health):
        """
        Synthesize transition dynamics from the given state in the absence of maintenance action
        """
        probs = np.zeros(self.num_healths)

        for next_health in range(health,-1,-1):
            # weibull distribution local
            probs[next_health] =  stats.weibull_min.pdf(101-next_health+1, self.dynamics_shape , scale=self.dynamics_scale)
        probs = probs/np.sum(probs)
        return probs
    
    def synthesize_dynamics_repair(self, health):
        """
        Synthesize transition dynamics from the given state when repair action is performed
        """
        probs = np.zeros(self.num_healths)

        for next_health in range(health, self.num_healths):
            # weibull distribution local
            probs[next_health] =  stats.weibull_min.pdf(next_health-health+1, self.dynamics_shape , scale=self.dynamics_scale)
        # if health != 100:
        #     probs = np.flip(probs)
        probs = probs/np.sum(probs)
        return probs

    def gen_trans_prob(self):
        """
        A function that generates transition probability for states of the component
        """

        trans_prob = np.zeros((self.num_healths, self.num_actions, self.num_healths))
        for health in self.healths:
            for action in self.actions:

                # no reload action or inspection action
                if health <= self.failure_condition:
                    # print("Here")
                    # print(f'Health is {health}')
                    trans_prob[health, action, :] = self.synthesize_dynamics(health)

                elif health > self.failure_condition:
                    # no action
                    if action == 0:
                        # if health > 2:
                        #     trans_prob[health, action, health-1] = 0.5
                        #     trans_prob[health, action, 1] = 0.5
                        # elif health == 2:
                        #     trans_prob[health, action, 1] = 1.0
                        # else:
                        #     trans_prob[health, action, 0] = 1.0
                        trans_prob[health, action, :] = self.synthesize_dynamics(health)

                    # replace action
                    elif action == 1:
                        # if self.current_state[0] == 0:
                        #     trans_prob[health, action, self.healths[0]] = 1.0
                        # else:
                        trans_prob[health, action, :] = self.synthesize_dynamics_repair(health)
                        # trans_prob[health, action, self.healths[85]] = 0.3
                        # trans_prob[health, action, self.healths[70]] = 0.3

                    
        return trans_prob
    
    def reward_func(self, state):
        """
        Calculate the reward obtained by the agent 
        """
        reward = 0.0
        if state[1] > self.budget:
            reward -= 10
        else:
            if state[0] <= self.failure_condition:
                reward -= 1*(100-self.num_steps)/100
            if state[0] > self.failure_condition:
                reward += 1*(self.num_steps)/100

        return reward

    def step(self, action):
        """
        Update the state of the component and log data
        """
        self.time_to_failure += 1   
        self.current_state[0] = np.random.choice(self.num_healths, p=self.trans_prob[self.current_state[0], action, :])

        if action == 0:
            self.cost_history.append(0)
            self.current_state[1] += 0
        # elif action == 1:
        #     self.cost_history.append(self.inspect_cost)
        #     self.current_state[1] += self.inspect_cost
        elif action == 1:
            self.cost_history.append(self.replace_cost)
            self.current_state[1] += self.replace_cost
        
        reward = 0
        if self.current_state[1] > self.budget:
            reward -= 10
        else:
            if self.current_state[0] <= self.failure_condition:
                reward -= 1*(100-self.num_steps)/100
            if self.current_state[0] > self.failure_condition:
                reward += 1*(self.num_steps)/100

        self.num_steps += 1
        done = False
        
        if self.num_steps >= self.max_steps-1 or self.current_state[0] <= self.failure_condition:
            done = True
        
        self.state_history.append(self.current_state.copy())
        self.action_history.append(action)
        return self.current_state.copy(), reward, done

    def visualize_history(self):
        """
        Visualize the historic state of the component and the action taken
        """
        # get_inspect_indices = [i for i in range(1,len(self.action_history)) if self.action_history[i] == 1]
        get_replace_indices = [i for i in range(1,len(self.action_history)) if self.action_history[i] == 1]
        # get_inspect_states = [self.state_history[i][0] for i in get_inspect_indices]
        get_replace_states = [self.state_history[i][0] for i in get_replace_indices]
        plt.figure(figsize=(15,5))
        plt.plot([self.state_history[i][0] for i in range(len(self.state_history))], '.-', color='gray', linewidth=2, alpha=1.0,  label='component CI')
        # plt.scatter(get_inspect_indices, get_inspect_states, marker='o', color='green', s=70, alpha=0.8, label='inspection')
        plt.scatter(get_replace_indices, get_replace_states, marker='o', color='blue', s=70, alpha=0.8, label='replacement')
        plt.plot(self.failure_condition*np.ones(self.max_steps), 'k--', label='failure condition', linewidth=2.5, alpha=0.6)
        plt.scatter(len(self.state_history), self.state_history[-1][0], marker = 'x', color='red', linewidths=5, s=70, alpha=1, label='final state')
        plt.legend()
        plt.xlabel('time step', fontsize=18)
        plt.xticks(fontsize=14)
        plt.ylabel('condition index', fontsize=18)
        plt.yticks(fontsize=14)
        plt.legend(prop={'size': 14}, frameon=False)
        sns.despine(right=True, top=True)
        plt.title(f'CI History for {self.name} with $\lambda$: {self.i_score}, Replace Cost: {self.replace_cost}', fontsize=16)
        plt.tight_layout()
        plt.show()

    def visualize_trajectory_samples(self, num_samples=10):
        """
        Visualize a few samples of the trajectory of the component
        """
        trajectory_samples = []
        for i in range(num_samples):
            state_history = []
            action_history = []
            current_state = self.current_state
            while current_state > 0:
                action = 0
                state_history.append(current_state)
                action_history.append(action)
                current_state = np.random.choice(self.states, p=self.trans_prob[current_state, action, :])
            trajectory_samples.append(state_history)

        plt.figure(figsize=(15,5))
        for i in range(num_samples):
            plt.plot(trajectory_samples[i], linewidth=2, alpha=0.8)
        plt.plot(self.failure_condition*np.ones(len(self.state_history)), 'k--', label='failure condition', linewidth=2.5)
        plt.xlabel('time step', fontsize=18)
        plt.xticks(fontsize=14)
        plt.ylabel('Condition Index', fontsize=18)
        plt.yticks(fontsize=14)
        
    def visualize_transition_prob(self, state):
        """
        Visualize the transition probability for a given state for different actions
        """

        actions = [0,1,2]
        colors = ['blue', 'green', 'red']
        
        # figure with three subplots for three actions
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        fig.suptitle(f'Transition Probability for {self.name} at state {state}', fontsize=16)
        for i, action in enumerate(actions):
            dist = self.trans_prob[state, action, :]
            ax[i].bar(self.states, dist, width=0.8, color=colors[i], alpha=0.8)
            ax[i].set_title(f'Action: {action}', fontsize=14)
            ax[i].set_xlabel('State')
            ax[i].set_ylabel('Probability')