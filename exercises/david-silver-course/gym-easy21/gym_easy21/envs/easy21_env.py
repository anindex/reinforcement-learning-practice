import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import numpy as np

class Easy21Env(gym.Env):
    metadata = {'render.modes': ['game_state']}
    deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #deck values

    def __init__(self):
        self.action_space = spaces.Discrete(2) # two action for player: stick(0) or hit(1)
        self.observation_space = spaces.Box(low=-10, high=30, dtype=np.int32, shape=(2,1)) # each state represents the score of dealer and player
        self.player = 0 # player and dealer score
        self.dealer = 0

        self.seed()
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        done = False

        if action == 1: # hit
            card_val, card_type = self._draw_card()
            self.player += card_val if card_type == 0 else -card_val
            if self.player < 1 or self.player > 21: # player burst
                self.player = -np.inf
                done = True

        else: #stick
            done = True
            while self.dealer < 17:
                card_val, card_type = self._draw_card()
                self.dealer += card_val if card_type == 0 else -card_val
                if self.dealer < 1 or self.dealer > 21: #dealer burst
                    self.dealer = -np.inf
                    break

        reward = (int(self.player > self.dealer) - int(self.player < self.dealer)) if done else 0
        return self._get_obs(), reward, done, {}

    def seed(self, seed=None):
        self.random_num, seed_num = seeding.np_random(seed)
        self.random_type, seed_type = seeding.np_random(seed)

        return [seed_num, seed_type]

    def reset(self):
        self.player = self.random_num.choice(Easy21Env.deck)
        self.dealer = self.random_num.choice(Easy21Env.deck)

        return self._get_obs()

    def _get_obs(self):
        return (self.player, self.dealer)

    def _draw_card(self):
        card_type = 1 if self.random_type.randint(3) < 1 else 0
        return self.random_num.choice(Easy21Env.deck), card_type


    def render(self, mode='game_state', close=False):
        print("Observation: Player score: {0}, Dealer card: {1}".format(self.player, self.dealer))
