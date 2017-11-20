import numpy as np
from itertools import product


class SparseNocAlg:

    def __init__(self, game, T, obs_horizon=3, updates=False, max_iter=100, keep_values=False):
        self.game = game
        self.T = T
        self.obs_horizon = obs_horizon
        self.updates = updates
        self.max_iter = max_iter
        self.keep_values = keep_values
        self.q_tuples, self.O, self.obs_dicts = self.find_diffs(self.game.mdps)
        self.plan = self.approximate_NOC()
        self.temp_plan = self.plan

    def getFirstAction(self, s):
        self.temp_plan = self.plan
        self.policy = self.temp_plan[0]
        if self.keep_values:
            self.values = self.weight_mix(self.temp_plan[2], self.temp_plan[3])
        return self.policy[s]

    def getNextAction(self, s,a,sprime):
        if (s, a, sprime) in self.temp_plan[1]:
            if self.updates:
                self.plan = self.approximate_NOC()
                self.temp_plan = self.plan
            else:
                self.temp_plan = self.temp_plan[1][(s, a, sprime)]
            self.policy = self.temp_plan[0]
            if self.keep_values:
                self.values = self.weight_mix(self.temp_plan[2], self.temp_plan[3])
        return self.policy[sprime]


    def approximate_NOC(self):
        gamma = self.game.gamma
        mdps = self.game.mdps
        ws = self.game.getBelief()
        horizon = self.obs_horizon
        S = mdps[0].nS
        I = len(mdps)
        # Construct uninitialized Q-value backups for each agent for each combination of observations for each time-step
        # Ex: backup of shape DxDxIxQ for time-step 2
        # D=number of different transitions between mdps, Q=number of different Q-values, I=number of agents
        q_backups = {i: np.zeros(tuple([len(self.O) for _ in range(i)] + [I, len(self.q_tuples)])) for i in range(horizon - 1)}
        conditional_plan = [None, {}] if not self.keep_values else [None, {}, None, ws]

        def reweight(history):
            ps = ws.copy()
            for i in range(I):
                for obs in history:
                    ps[i] *= self.obs_dicts[i][obs][0]
            return ps/sum(ps)

        def compute_val(mdp_index, pol, dist=None, prevQ=None):
            T_tens, R_mat = self.game.Ts[mdp_index], self.game.Rs[mdp_index]
            nS = R_mat.shape[1]
            if prevQ is None:
                T, R = self.construct_TpiRpi(T_tens, R_mat, pol)
                V = np.linalg.solve((np.eye(nS) - gamma * T), R)
            else:
                aug_T, aug_R = self.construct_Qbackup_TR(T_tens, R_mat, prevQ)
                T, R = self.construct_TpiRpi(aug_T, aug_R, np.append(pol,[0]))
                V = np.linalg.solve((np.eye(nS + 1) - gamma * T), R)[:nS]
            if dist is None:
                return V
            else:
                return V.dot(dist)

        for h in reversed(range(horizon)):
            for history in product(range(len(self.O)), repeat=h):
                temp_plan = conditional_plan
                if h == 0:
                    tempws = ws
                    Qs = q_backups[h]
                    pol = self.mixed_policy_iteration(tempws, Qs)
                else:
                    obs_history = [self.O[i] for i in history]
                    obs = obs_history[-1]
                    tempws = reweight(obs_history)
                    baseQ = (obs[0], obs[1])
                    initial_dist = np.zeros((S))
                    initial_dist[obs[2]] = 1.0
                    if h != horizon - 1:
                        Qs = q_backups[h][history]
                        pol = self.mixed_policy_iteration(tempws, Qs)
                    else:
                        Qs = [None for _ in range(len(mdps))]
                        pol = self.mixed_policy_iteration(tempws)
                    for i in range(I):
                        p, rew = self.obs_dicts[i][obs]
                        V = compute_val(i,pol, initial_dist, Qs[i])
                        q_backups[h - 1][tuple(list(history[:-1]) + [i, self.q_tuples.index(baseQ)])] += \
                            p * (rew + gamma * V)
                    for ob in obs_history:
                        if ob not in temp_plan[1]:
                            temp_plan[1][ob] = [None, {}] if not self.keep_values else [None, {}, None, None]
                        temp_plan = temp_plan[1][ob]

                temp_plan[0] = pol
                if self.keep_values:
                    temp_plan[2] = [compute_val(i,pol,prevQ=Qs[i]) for i in range(I)]
                    temp_plan[3] = tempws

        return conditional_plan

    # Helper Methods


    def construct_TpiRpi(self, T, R, pi):
        """
        Based on the MDP_Toolbox function
        Compute the transition matrix and the reward matrix for a policy.

        :param T: Transition tensor of shape (A,S,S)
        :param R: Reward tensor of shape (A,S,S)
        :param pi: the policy of shape (S)
        :return: Tpi, the transition matrix of the policy of shape (S,S)
            Rpi, the reward vector of the policy of shape (S)
        """
        nS = T.shape[1]
        nA = T.shape[0]
        Tpi = np.empty((nS, nS))
        Rpi = np.zeros(nS)
        pi = np.array(pi)
        for a in range(nA):  # avoid looping over S
            # the rows that use action a.
            ind = (pi == a).nonzero()[0]
            if ind.size > 0:
                Tpi[ind, :] = T[a][ind, :]
                Rpi[ind] = R[a][ind]
        return Tpi,Rpi



    def construct_Qbackup_TR(self, raw_T, raw_R, Qs):
        # Include sink state in transitions for contested state-action pairs
        S = self.game.mdps[0].nS
        A = self.game.mdps[0].nA
        T = np.zeros((A, S + 1, S + 1))
        T[:, S, S] = 1
        R = np.zeros((A, S + 1))
        T[:, :S, :S] = raw_T
        R[:, :S] = raw_R
        # Transition all contested state-action pairs into the sink state
        # Set reward for those state-action pairs to be the weighted average q-backups
        for i, q in enumerate(self.q_tuples):
            s, a = q
            T[a, s, :] = np.zeros(S + 1)
            T[a, s, S] = 1
            R[a, s] = Qs[i]
        return T,R


    # amortized compute time per action
    def mixed_policy_iteration(self, belief, Qbackups=None):

        gamma = self.game.gamma
        mdps = self.game.mdps
        S = mdps[0].nS
        A = mdps[0].nA

        def eval_pol(pi):
            Tpi, Rpi = self.construct_TpiRpi(T,R,pi)
            V = np.linalg.solve(np.eye(S) - gamma * Tpi, Rpi)
            return V

        def update_pol(V):
            # Looping through each action the Q-value matrix is calculated.
            Q = np.empty((A, S))
            for a in range(A):
                Q[a] = (R[a] + gamma * T[a].dot(V))
            return Q.argmax(axis=0)


        T = self.weight_mix(self.game.Ts, belief)
        R = self.weight_mix(self.game.Rs, belief)

        if Qbackups is not None:
            Qs = self.weight_mix(Qbackups, belief)
            T,R = self.construct_Qbackup_TR(T,R,Qs)
            S += 1


        V = np.zeros(S)
        prev_pol, pol = np.empty(S), np.zeros(S)
        for i in range(self.max_iter):
            pol = update_pol(V)
            V = eval_pol(pol)
            if np.array_equal(pol, prev_pol):
                break
            prev_pol = pol

        if Qbackups is None:
            return pol
        else:
            return pol[:S-1]


    @staticmethod
    def weight_mix(arrays, weights):
        return sum([arrays[i] * weights[i] for i in range(len(weights))])


    #TODO Matrixfy
    @staticmethod
    def find_diffs(mdps):
        assert (len(set([mdp.nS for mdp in mdps])) == 1 and len(set([mdp.nA for mdp in mdps])) == 1)
        I = len(mdps)
        Ps = [mdp.P for mdp in mdps]
        nS, nA = mdps[0].nS, mdps[0].nA
        q_tuples = []
        obs_tuples = []
        obs_dicts = [{} for _ in range(I)]
        max_rewards = [0] * I
        for s in range(nS):
            for a in range(nA):
                q_added = False
                sp_dicts = [{tup[1]: tup for tup in P[s][a]} for P in Ps]
                sprimes = set([sprime for sp_dict in sp_dicts for sprime in sp_dict])
                for sprime in sprimes:
                    tups = [sp_dict[sprime] if sprime in sp_dict else (1e-16, sprime, 0.0) for sp_dict in sp_dicts]
                    diff = len(set([(tup[0], tup[1]) for tup in tups])) != 1
                    if diff:
                        if not q_added:
                            q_tuples += [(s, a)]
                            q_added = True
                        obs_tuples += [(s, a, sprime)]
                        for i in range(I):
                            if tups[i] == (0.0, 0.0):
                                reward_func = getattr(mdps[i], "reward", lambda *_: 0.0)
                                p = 0.0
                                r = reward_func(s, a, sprime)
                            else:
                                p, r = tups[i][0], tups[i][2]
                            obs_dicts[i][(s, a, sprime)] = (p, r)
                    for i in range(I):
                        max_rewards[i] = max(max_rewards[i], tups[i][2])
        return q_tuples, obs_tuples, obs_dicts #, max_rewards