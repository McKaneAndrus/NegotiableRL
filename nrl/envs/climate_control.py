import numpy as np
from .discrete_env import DiscreteEnv
from itertools import product
import random

#Only small rooms are affected by adjacent rooms, rooms enact low self-change
MIN_CHANGE_DYNAMICS = np.array([
    [1,1,1,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0],
    [0,1,0,1,0,0],
    [0,0,1,0,1,0],
    [0,0,0,0,0,1]
])

#All rooms affect all adjacent rooms, rooms enact large self-change
MAX_CHANGE_DYNAMICS = np.array([
    [2,1,1,0,0,0],
    [1,2,1,1,0,0],
    [1,1,2,0,1,0],
    [0,1,0,2,1,1],
    [0,0,1,1,2,1],
    [0,0,0,1,1,2]
])
# Transfer only occurs from left to right
LEFT_DOMINANT_DYNAMICS = np.array([
    [1,1,1,0,0,0],
    [0,1,0,1,0,0],
    [0,0,1,0,1,0],
    [0,0,0,1,0,1],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1]
])
# Transfer only occurs from right to left
RIGHT_DOMINANT_DYNAMICS = np.array([
    [1,0,0,0,0,0],
    [1,1,0,0,0,0],
    [1,0,1,0,0,0],
    [0,1,0,1,0,0],
    [0,0,1,0,1,0],
    [0,0,0,1,0,1]
])


DEFAULT_DPS = [0.5,0.25,0.0,0.25]


def generate_random_preferences():
    samples = np.random.rand(len(DEFAULT_DPS))
    return list(samples/sum(samples))

RANDOM_DPS = generate_random_preferences

DYNAMICS = [MIN_CHANGE_DYNAMICS, MAX_CHANGE_DYNAMICS, LEFT_DOMINANT_DYNAMICS, RIGHT_DOMINANT_DYNAMICS]

class ClimateControlEnv(DiscreteEnv):
    """
    You have just rented out your first smart home. This smart home has built in
    climate control that attempts to suit your preferences.
          ___________
    _____| 2| 4 | 6 |
    |  1 |__|___|___|
    |____|_3|___5___|

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,  num_rooms=2,num_temps=3, prefs=None, alt_dynamics_probs = {}, dynamics_probs=DEFAULT_DPS,
                sparsity = None, max_steps=100, cold_discount=0, hot_discount=0, max_utility=10):
        # prefs : array of tuples, where 1st element is the desired temperature, 2nd is max utility from room
        # cooling_dynamics and heating_dynamics are nested dictionaries, with the outer dictionary
        # indexed by temperature (state) tuples and the inner dictionaries indexed by action tuples,
        # returning a set of dynamics_weights
        self.nR, self.nT = num_rooms, num_temps
        self.cd, self.hd = cold_discount, hot_discount
        if prefs is None:
            prefs = [(int(random.random() * num_temps),random.random() * max_utility/num_rooms)
                     for _ in range(num_rooms)]
        self.prefs = prefs

        nS = num_temps**num_rooms
        # Turn temp up or down in each room
        nA = num_rooms * 2
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        D = [dyn[:num_rooms,:num_rooms] for dyn in DYNAMICS]
        isd = np.zeros(nS)
        isd[0] = 1.0

        if sparsity is not None:
            alt_dynamics_probs = {}
            diff_num = round(sparsity * nS) + 1
            diff_sa_pairs = random.sample([tup for tup in product(range(nS), range(nA))], diff_num)
            for sa in diff_sa_pairs:
                alt_dynamics_probs[sa] = RANDOM_DPS()


        def to_s(temps):
            return sum([temps[i] * num_temps ** i for i in range(num_rooms)])

        def calc_reward(temps):
            rew = 0
            for i,pref in enumerate(prefs[:num_rooms]):
                diff = temps[i] - pref[0]
                r = pref[1] * abs(abs(diff) - (num_temps-1))/(num_temps-1)
                if diff < 0:
                    r *= cold_discount
                elif diff > 0:
                    r *= hot_discount
                rew += r
            return rew

        self.rewards = {temps:calc_reward(temps) for temps in product(range(num_temps), repeat=num_rooms)}


        def inc(temps, a, dynamics):

            def change_temp(temp, delta):
                if delta < 0:
                    new_temp = max(temp + delta, 0)
                else:
                    new_temp = min(temp + delta, num_temps - 1)
                return new_temp

            new_temps = [0] * num_rooms
            temp_change = (2 * (a % 2) - 1)
            action_room = int(a / 2)
            for room in range(num_rooms):
                new_temps[room] = change_temp(temps[room], temp_change * dynamics[action_room, room])
            return tuple(new_temps)

        for temps in product(range(num_temps), repeat=num_rooms):
            s = to_s(temps)
            for a in range(nA):
                li = P[s][a]
                if (s,a) in alt_dynamics_probs:
                    d_probs = alt_dynamics_probs[(s,a)]
                else:
                    d_probs = dynamics_probs
                raw_tups = {}
                for dyn,p in zip(D,d_probs):
                    if p > 0.0:
                        new_temps = inc(temps, a, dyn)
                        newstate = to_s(new_temps)
                        if newstate in raw_tups:
                            raw_tups[newstate][0] += p
                        else:
                            rew = self.rewards[new_temps]
                            raw_tups[newstate] = [p,rew]
                for newstate in raw_tups.keys():
                    p, rew = raw_tups[newstate]
                    li.append((p, newstate, rew))


        super(ClimateControlEnv, self).__init__(nS, nA, P, isd, max_steps)

    def s_to_temps(self,s):
        temps = []
        for i in range(self.nR):
            temps += [s % self.nT]
            s = int(s / self.nT)
        return tuple(temps)

    def temps_to_s(self,temps):
        return sum([temp * self.nT ** i for i,temp in enumerate(temps)])

    def get_reward(self,s,a=None, sprime=None):
        if sprime == None:
            temps = self.s_to_temps(s)
        else:
            temps = self.s_to_temps(sprime)
        return self.rewards[(temps)]

