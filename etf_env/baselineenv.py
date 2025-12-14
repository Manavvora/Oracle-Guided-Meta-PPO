import numpy as np
import baselinefin
import gymnasium as gym
from gymnasium import spaces, Env

class ETFPOMDPEnv(Env):
    metadata = {'render_modes': []}

    def __init__(self, num_particles=500):
        self.tickers = baselinefin.tickers
        self.n = baselinefin.n
        self.T_mats = baselinefin.T_mats
        self.A_asset = baselinefin.A_asset
        self.V_test = baselinefin.V_test
        self.O_obs_model = baselinefin.O_obs_model
        self.k_tanh = baselinefin.k_tanh
        self.alpha_D = baselinefin.alpha_D
        self.dates = baselinefin.dates_test
        self.num_states = baselinefin.b_belief.shape[0]
        self.num_particles = num_particles
        self.T = self.V_test.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n + 1)

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.d = 0

        prior = baselinefin.b_belief.copy()
        self.particles = np.random.choice(
            self.num_states, size=self.num_particles, p=prior
        )
        self.particle_weights = np.ones(self.num_particles) / self.num_particles
        self.b_belief = prior.copy()

        self.w = baselinefin.w_weights_current.copy()

        obs = self._get_observation().astype(np.float32)
        return obs, {}

    def step(self, action: int):
        INSPECTION_COST = 0.01
        assert 0 <= action <= self.n, "Action out of range"
        assert self.d < self.T - 1, "Episode already done"

        done = False
        most_likely_state = int(np.argmax(self.b_belief))

        if action < self.n:
            j_star = action
            price_now = self.V_test[self.d]
            price_next = self.V_test[self.d + 1]

            P_mat = self.T_mats[self.tickers[j_star]]
            preds = np.array([np.random.choice(self.num_states, p=P_mat[s]) for s in self.particles])
            self.particles = preds

            obs_state = int(np.sign(price_next[j_star] - price_now[j_star]) + 1)
            weights = self.O_obs_model[obs_state, self.particles]
            if weights.sum() > 0:
                weights /= weights.sum()
            else:
                weights = np.ones_like(weights) / self.num_particles

            idx = np.random.choice(self.num_particles, size=self.num_particles, p=weights)
            self.particles = self.particles[idx]

            counts = np.bincount(self.particles, minlength=self.num_states)
            self.b_belief = counts / self.num_particles
            most_likely_state = int(np.argmax(self.b_belief))

            if most_likely_state == 0:
                reward = 0.0
                done = True
            else:
                reward = 1.0 - INSPECTION_COST
                delta = self.b_belief.max() - self.b_belief.min()
                D = self.alpha_D * np.tanh(self.k_tanh * delta)
                self.w = np.clip(self.w + D * self.A_asset[:, j_star], 0, None)
                self.w = self.w / self.w.sum()

        else: # No-inspect action
            j_star = -1
            reward = 1.0

        self.d += 1
        if self.d >= self.T - 1:
            done = True

        truncated = False
        obs = self._get_observation().astype(np.float32)
        info = {
            'j_star': j_star,
            'weights': self.w.copy(),
            'belief': self.b_belief.copy(),
            'most_likely_state': most_likely_state
        }
        return obs, reward, done, truncated, info

    def _get_observation(self):
        return self.V_test[self.d]

    def render(self, mode='human'):
        pass

    def close(self):
        pass

from gymnasium.envs.registration import register
register(
    id='ETF-v0',
    entry_point='__main__:ETFPOMDPEnv',
)