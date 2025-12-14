import numpy as np
from gymnasium import spaces, Env

class MetaPPOEnv(Env):
    def __init__(self, component, mdp_policy) -> None:
        self.component = component
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.zeros(3), high=np.array([100, np.inf, np.inf]), dtype=np.float32)
        self.belief_var = np.var(self.component.particles, axis=0)
        self.true_state = [self.component.current_state[0], self.component.current_state[1], self.belief_var]
        self.belief_state = [self.component.current_health_belief_state[0], self.component.current_health_belief_state[1], self.belief_var]
        self.mdp_policy = mdp_policy
        self.true_state_history = [self.true_state]
        self.belief_state_history = [self.belief_state]
        self.action_history = []

    def reset(self, **kwargs):
        _,_, = self.component.reset()
        self.belief_var = np.var(self.component.particles, axis=0)
        self.true_state = [self.component.current_state[0], self.component.current_state[1], self.belief_var]
        self.belief_state = [self.component.current_health_belief_state[0], self.component.current_health_belief_state[1], self.belief_var]
        info = {'true_state': self.true_state, 'belief_state': self.belief_state}
        self.true_state_history = [self.true_state]
        self.belief_state_history = [self.belief_state]
        self.action_history = []
        return np.array(self.belief_state), info
    
    def reward(self, action):
        r = 0
        remaining_budget = self.component.budget - self.belief_state[1]
        if remaining_budget < 0:
            r -= 100
        else:
            if self.belief_state[0] <= self.component.failure_condition:
                r -= 20
            else:
                r += self.component.num_steps/10
                r -= abs(self.belief_state[0] - self.true_state[0])/1000
                # r -= self.belief_state[2]/100
                # if action == 1:
                #     if self.belief_var == 0:
                #         r -= 10
        if self.component.num_steps == self.component.max_steps or self.component.component_died:
            # print(f'Step: {self.component.num_steps}')
            # print(f'Component Died: {self.component.component_died}')
            if remaining_budget > self.component.inspect_cost and remaining_budget < self.component.replace_cost:
                # print('Budget completely used!!')
                r += 50
            elif remaining_budget >= self.component.replace_cost:
                # print('Could have used more of the budget!!')
                r -= 100
        return r
    
    def step(self, action):
        if action == 1:
            _,_,_,_ = self.component.step(action)
        else:
            a = np.argmax(self.mdp_policy[int(self.component.current_health_belief_state[0]), int(self.component.current_health_belief_state[1]//int(self.component.replace_cost))])
            if a == 1:
                a = 2
            _,_,_,_ = self.component.step(a)
            action = a

        self.belief_var = np.var(self.component.particles, axis=0)
        self.true_state = [self.component.current_state[0], self.component.current_state[1], self.belief_var]
        self.belief_state = [self.component.current_health_belief_state[0], self.component.current_health_belief_state[1], self.belief_var]
        reward = self.reward(action)
        terminated = False
        truncated = False

        self.true_state_history.append(self.true_state)
        self.belief_state_history.append(self.belief_state)
        # print(f'Action in step: {action}')
        self.action_history.append(action)
        if self.component.num_steps >= self.component.max_steps:
            truncated = True
        
        if self.belief_state[0] <= self.component.failure_condition:
            terminated = True
        info = {'true_state': self.true_state, 'belief_state': self.belief_state}

        return np.array(self.belief_state), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass