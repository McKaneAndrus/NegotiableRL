import numpy as np
import sys
from six import StringIO, b
import itertools as it

from gym import utils
from .discrete_env import DiscreteEnv


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "4x4_multigoal": [
        "SFFF",
        "FHF1",
        "FFFH",
        "H2FF"
    ],
    "8x8_multigoal": [
        "FFFFFFF2",
        "FFFFFFFF",
        "FFFHFFFF",
        "SFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFF1"
    ],
    "7x7_balanced": [
        "HFHFFF2",
        "FFFFHFF",
        "FHFFFHF",
        "SFFFFFF",
        "FHFFFHF",
        "FFFFHFF",
        "HFHFFF1"
    ],
    "3x3_balanced": [
        "HF2",
        "SFF",
        "HF1",
    ],
}


class RockSampleEnv(DiscreteEnv):
    """
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

    def __init__(self, diffs=1, desc=None, map_name="7x7_balanced",is_slippery=True,partial_slip=0.2,rewards=[0,1], seed=None):
        assert type(rewards) == list
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.sinks = []
        self.max_reward = max(rewards)
        nA = 4
        nS = nrow * ncol

        if seed is not None:
            self._seed(seed)

        if type(diffs) == float:
            if diffs > 1 or diffs < 0:
                print("Sparsity out of range, please choose float between 0 and 1.")
                return
            nF = dict(zip(*np.unique(self.desc, return_counts=True)))[b'F']
            diff_select = np.array(([True] * int(round(nF * 4 * diffs))) + ([False] * int(round(nF * 4 * (1-diffs)))))
            self.np_random.shuffle(diff_select)
            diff_select = list(diff_select)
        else:
            if diffs.shape != (nS,nA):
                print("Incorrect shape of difference matrix.")
                return
            Fs = np.where(desc.flatten()  == b'F')[0]
            diff_select = list(diffs[Fs].flatten())

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        # Add rewards for non-reward states to construct reward dict
        rewards += [0,0,0,0]
        grid_rewards = {char:rewards[i] for i,char in enumerate([b'1',b'2',b'S',b'H',b'F',b'Q'])}

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                letter = desc[row, col]
                for a in range(4):
                    li = P[s][a]
                    if letter in b'12GH':
                        li.append((1.0, s, 0, True))
                        self.sinks += [s]
                    elif letter in b'QF' and diff_select.pop(0):
                        desc[row,col] = b'Q'
                        for b in [(a-1)%4, a, (a+1)%4]:
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'12GH'
                            rew = grid_rewards[newletter]
                            li.append((1 - 2 * partial_slip if b==a else partial_slip, newstate, rew, done))
                    elif is_slippery:
                        for b in [(a-1)%4, a, (a+1)%4]:
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'12GH'
                            rew = grid_rewards[newletter]
                            li.append((0.8 if b==a else 0.1, newstate, rew, done))
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        done = bytes(newletter) in b'12GH'
                        rew = grid_rewards[newletter]
                        li.append((1.0, newstate, rew, done))

        super(RockSampleEnv, self).__init__(nS, nA, P, isd)



    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        return outfile