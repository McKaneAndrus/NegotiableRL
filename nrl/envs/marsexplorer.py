import numpy as np
import sys
from six import StringIO, b
import itertools as it

from gym import utils
from .discrete_env import categorical_sample, DiscreteEnv
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cs


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
NOOP = 4
SAMPLE = 5

MAPS = {
    "9x9_multigoal": [
        "F2FFUFF1F",
        "FFFBFBFFF",
        "FFFFFFFFF",
        "BFBBBBBFB",
        "FFFFSFFFF",
        "FFFFFFFFF",
        "FFBBBBBFF",
        "FUFB3BFUF",
        "1FFFFFFF2"
    ] ,
        "5x5_multigoal": [
        "1FFF2",
        "FBFUF",
        "1B2BF",
        "FFFUF",
        "SFFB3"
    ],
    "4x4_multigoal": [
        "FFFF",
        "2B1U",
        "FUFF",
        "SFB3"
    ],
}

class MarsExplorerEnv(DiscreteEnv):
    """
    NASA's most recent Mars Rover has finally landed. In the surrounding area there
    are locations that are to unique interest to either the Astrobiology or Geology.
    In order
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """


    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, soil_uncertainty, preference_ordering, desc=None, map_name="9x9_multigoal",
                 uncertain_paths=True, path_uncertainty=(0.075,0.1,0.05), preference_scale=2,
                 true_preference=1, seed=None):

        assert type(preference_ordering) == list or type(preference_ordering) == tuple
        assert type(soil_uncertainty) == list or type(soil_uncertainty) == tuple

        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.scale = render_scale
        self.num_goals = num_goals = len(np.where(np.logical_or(desc == b'1', desc == b'2',desc == b'3'))[0])
        self.max_reward = preference_scale**(len(preference_ordering)-1)
        self.step_num = 0
        nA = 5
        nS = (2 ** num_goals * nrow * ncol)
        path_uncertainty = list(path_uncertainty)
        path_uncertainty += [1 - path_uncertainty[0]*2 + path_uncertainty[1] + path_uncertainty[2]]

        multiple_soils = type(soil_uncertainty[0]) == tuple
        if multiple_soils:
            new_su = []
            for soil in soil_uncertainty:
                soil = list(soil)
                soil += [1 - soil[0] * 2 + soil[1] + soil[2]]
                new_su += [soil]
            soil_uncertainty = new_su
        else:
            soil_uncertainty = list(soil_uncertainty)
            soil_uncertainty += [1 - soil_uncertainty[0] * 2 + soil_uncertainty[1] + soil_uncertainty[2]]

        def goal_states_to_int(goals):
            assert len(goals) == num_goals
            assert np.all([g in [0, 1] for g in goals])
            total = 0
            for i in range(num_goals):
                total += 2**i * goals[i]
            return total

        def to_s(row, col, goals):
            return int(((row*ncol + col) << num_goals) + goal_states_to_int(goals))


        if seed is not None:
            self._seed(seed)

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            return (row, col)


        isd = np.zeros(nS)
        goal_inds = {}
        goal_count = 0
        for row in range(nrow):
            for col in range(ncol):
                if desc[row, col] == b'S':
                    isd[to_s(row,col,[0]*num_goals)] = 1.0
                if desc[row, col] in b'123':
                    goal_inds[(row, col)] = goal_count
                    goal_count += 1

        isd /= np.sum(isd)

        true_goals = np.where(np.logical_or(desc == str.encode(str(true_preference)),desc == b'3'))[0]
        self.true_goal_transitions = []

        for row in range(nrow):
            for col in range(ncol):
                for goals in map(list, it.product([0, 1], repeat=num_goals)):
                    s = to_s(row, col, goals)
                    for a in range(nA):
                        li = P[s][a]
                        letter = desc[row, col]
                        rew = 0
                        if a == SAMPLE and letter in b'123':
                            goal_already_active = bool(goals[goal_inds[(newrow, newcol)]])
                            if not goal_already_active:
                                newgoals = goals.copy()
                                newgoals[goal_inds[(newrow, newcol)]] = 1.0
                                newstate = to_s(newrow, newcol, newgoals)
                                assert((row,col) in preference_ordering)
                                rew = preference_scale ** preference_ordering.index((row,col))
                                done = goal_states_to_int(newgoals) == 2**(self.num_goals+1) - 1
                                if (row,col) in true_goals:
                                    self.true_goal_transitions += [(s,newstate)]
                            else:
                                newstate = s
                                done = False
                            li.append((1.0, newstate, rew, done))
                        elif a == NOOP:
                            li.append((1.0, s, rew, False))
                        elif letter in b'U':
                            su = soil_uncertainty.pop(0) if multiple_soils else soil_uncertainty
                            probs = [(su[0],(a-1)%nA-1),
                                     (su[0],(a+1)%nA-1),
                                     (su[1], NOOP),
                                     (su[2],(a+2)%nA-1),
                                     (su[3], a)]
                            for p,b in probs:
                                newrow, newcol = inc(row, col, b)
                                newletter = desc[newrow, newcol]
                                newstate = to_s(newrow, newcol, goals) if newletter not in b'B' else s
                                li.append((p, newstate, rew, False))
                        elif uncertain_paths:
                            probs = [(path_uncertainty[0],(a-1)%nA-1),
                                     (path_uncertainty[0],(a+1)%nA-1),
                                     (path_uncertainty[1], NOOP),
                                     (path_uncertainty[2],(a+2)%nA-1),
                                     (path_uncertainty[3], a)]
                            for p,b in probs:
                                newrow, newcol = inc(row, col, b)
                                newletter = desc[newrow, newcol]
                                newstate = to_s(newrow, newcol, goals) if newletter not in b'B' else s
                                li.append((p, newstate, rew, False))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newletter = desc[newrow, newcol]
                            newstate = to_s(newrow, newcol, goals) if newletter not in b'B' else s
                            li.append((1.0, newstate, rew, False))

        super(MarsExplorerEnv, self).__init__(nS, nA, P, isd)


    def _step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        self.step_num += 1
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob": p})

    def get_reward(self, s, a, sprime=None):
        r = 0
        if sprime is None:
            for t in self.P[s][a]:
                r += t[0] * t[2]
        else:
            for t in self.P[s][a]:
                if t[1] == sprime:
                    r = t[2]
        return r
