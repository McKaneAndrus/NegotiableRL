import numpy as np
from itertools import product


class POMDPlite:

# Does not handle sink state rewards well, or just requires sub ~0.1 betas.
    def __init__(self,game, T, beta=1.0, max_iter=100, keep_values=False):
        self.game = game
        self.T = T
        self.beta = beta
        self.max_iter = max_iter
        self.policy = None
        self.prev_belief = self.game.b.copy()
        self.keep_values = keep_values
        if keep_values:
           self.values = None

    def getFirstAction(self,s):
        self.POMDP_lite_policy_iteration()
        return self.policy[s]

    def getNextAction(self,s,a,sprime):
        if not np.array_equal(self.game.b, self.prev_belief):
            self.POMDP_lite_policy_iteration()
            self.prev_belief = self.game.b.copy()
        return self.policy[sprime]




    # Helper Methods




    def reward_bonus(self,joint_T):
        """
        Calculates the matrix of reward bonuses for state action pairs.
        Reward bonuses as defined in (Chen et. al 2016) are a weighted sum
        of the L1 divergences between current beliefs and beliefs after a
        transition.
        :param joint_T: The weighted combination of transition tensors
        :return: (AxS) matrix of reward bonuses
        """
        belief = self.game.b
        Ts = self.game.Ts
        divergences = sum([abs(np.nan_to_num(np.divide(belief[i] * Ts[i], joint_T)) - belief[i]) for i in range(len(Ts))])
        bonus = self.beta * np.multiply(joint_T, divergences).sum(2)
        return bonus


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


    def POMDP_lite_policy_iteration(self):

        gamma = self.game.gamma
        mdps = self.game.mdps
        belief = self.game.b

        I = len(mdps)
        S = mdps[0].nS
        A = mdps[0].nA
        T = sum([belief[i] * self.game.Ts[i] for i in range(I)])
        R = sum([belief[i] * self.game.Rs[i] for i in range(I)]) + self.reward_bonus(T)

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


        V = np.zeros(S)
        prev_pol = np.empty(S)
        for i in range(self.max_iter):
            pol = update_pol(V)
            V = eval_pol(pol)
            if np.array_equal(pol, prev_pol):
                break
            prev_pol = pol

        self.policy = pol
        if self.keep_values:
            self.values = V



# def calculate_transitions(self, belief, s, a):
    #     belief = np.array(belief)
    #     mdps = self.game.mdps
    #     I = len(mdps)
    #     sprimes = self.game.observationMap[(s, a)]
    #     # Vectors of size theta, P(s'|theta,s,a)
    #     trans_probs = {sprime: np.zeros(I) for sprime in sprimes}
    #     for i in range(I):
    #         for t in mdps[i].P[s][a]:
    #             trans_probs[t[1]][i] += t[0]
    #     # P(s'|b,s,a)
    #     joint_trans = {s: trans_probs[s].dot(belief) for s in sprimes}
    #     return trans_probs, joint_trans

    # def construct_TR(self, belief, pi):
    #     mdps = self.game.mdps
    #     nS = mdps[0].nS
    #     T = np.zeros([nS, nS])
    #     R = np.zeros([nS])
    #     for s in range(nS):
    #         ts = [mdp.P[s][pi[s]] for mdp in mdps]
    #         for t,w in zip(ts,belief):
    #             for ret in t:
    #                 T[s][ret[1]] += w * ret[0]
    #                 R[s] +=  w * ret[0] * ret[2]
    #         R[s] += self.reward_bonus(belief, s, pi[s])
    #     return T, R

    # def construct_TR(self, belief, pi):
    #     trans_tensors = self.game.trans_tensors
    #     reward_vectors = self.game.reward_vectors
    #     I = len(trans_tensors)
    #     nS = trans_tensors[0].shape[0]
    #     T = np.zeros([nS, nS])
    #     R = np.zeros([nS])
    #     for s in range(nS):
    #         a = pi[s]
    #         for i in range(I):
    #             T[s] += belief[i] * trans_tensors[i][s][a]
    #             R[s] +=  belief[i] * trans_tensors[i][s][a].dot(reward_vectors[i])
    #         R[s] += self.reward_bonus(belief, s, pi[s])
    #     return T, R

    #
    # def compute_Q(s, V):
    #     raw_Q = {}
    #     obsMap = self.game.observationMap
    #     actions = self.game.getAllActions()
    #     for a in actions:
    #         ind_trans, joint_trans = self.calculate_transitions(s, a)
    #         for i, mdp in enumerate(mdps):
    #             for t in mdp.P[s][a]:
    #                 Q[a] += belief[i] * t[0] * t[2]
    #     return Q

    # pi = np.zeros(S)
    # for s in range(S):
    #     Qs_s = compute_Q([mdp.P[s] for mdp in mdps], V)
    #     pi[s] = max(Qs_s, key=lambda x: Qs_s[x])
    # return pi
