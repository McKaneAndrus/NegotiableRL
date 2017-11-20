import itertools
import numpy as np



class NegotiableGame:
    """
    A model of the negotiable MDP game.
    Throughout, world_state denotes the true world state.
    State denotes the state in the POMDP, a tuple of the form
    (world_state, mdp_index).
    Currently only works with environments that are defined by P-dicts.
    """

    def __init__(self, mdps, true_mdp=None, ws=None, gamma=0.95, matrix_form=True):
        """
        Initializes an instance of a negotiable game.
        :param true_mdp: The index of the true mdp, or a separate mdp object.
        :param human_policy: the policy that the human follows.
        :param initial_world_state: the initial world state.
        :param num_theta: the number of possible theta values.
        :param num_ingredients: the number of possible recipes in the game; note
            the number of human and robot actions is num_ingredients - 1.
        :param reward_set: the set of states in which
        """
        self.mdps = mdps
        self.I = len(mdps)
        self.gamma = gamma
        self.theta_set = list(range(self.I))
        self.allStates = self.getAllStates()
        if ws == None:
            self.b = np.array([1/self.I for _ in mdps])
        else:
            self.b = np.array(ws)
        self.ws = self.b.copy()
        if true_mdp is None:
            self.true_mdp = self.sampleMDPs()
            self.random_true_mdp = True
        else:
            self.true_mdp = true_mdp
            self.random_true_mdp = False
        self.world_state = self.true_mdp.reset()
        # Separate_rewards be used for analysis, but has no impact on the game.
        self.separate_rewards = np.zeros(self.I)
        self.true_reward, self.discounted_reward = 0,0
        self.time_step = 0
        self.observationMap = self.buildObservationMap()
        if matrix_form:
            self.Ts, self.Rs = [], []
            for mdp in mdps:
                T,R = self.buildTRs(mdp)
                self.Ts += [T]
                self.Rs += [R]


    # Methods to interface with the game


    def act(self, action):
        """
        Implements an action in the game. Updates the world_state, belief, and
        true/individual rewards.
        :param action: the robot's action.
        """
        s = self.world_state
        sprime, r, done, pdict = self.true_mdp.step(action)
        self.world_state = sprime
        self.updateBelief(s, action, sprime)
        self.true_reward += r
        self.discounted_reward += r * self.gamma ** self.time_step
        self.separate_rewards += np.array([self.getReward((s,theta),action,(sprime, theta)) for theta in self.getAllTheta()])
        self.time_step += 1
        return sprime, done, pdict['prob']

    def getBelief(self):
        """
        Returns the belief over ϴ as a |ϴ|-dimensional vector.
        """
        return self.b

    def getReward(self, state, action, end_state=None, world_states=False):
        """
        Returns the reward when the game is in a particular state.
        :param state:
        """
        if world_states:
            if end_state is None:
                return sum([self.mdps[i].get_reward(state, action) * self.b[i] for i in range(self.I)])
            return sum([self.mdps[i].get_reward(state, action, end_state) * self.b[i] for i in range(self.I)])
        if end_state is None:
            return self.mdps[state[1]].get_reward(state[0],action)
        return self.mdps[state[1]].get_reward(state[0], action, end_state[0])

    def getAllWorldStates(self):
        """
        Returns all possible world states in the game as a Python list.
        """
        return list(range(self.mdps[0].nS))

    def getAllTheta(self):
        """
        Returns all possible values of theta as a Python list.
        """
        return self.theta_set

    def getAllStates(self):
        """
        Returns all possible states in the POMDP game as a Python list.
        """
        return list(itertools.product(self.getAllWorldStates(), self.getAllTheta()))

    def getAllActions(self):
        """
        Returns all possible actions of the negotiator.
        """
        return list(range(self.mdps[0].nA))

    def reset(self):
        """
        Resets the game by reverting to the initial belief and rewards, resampling the true_mdp
        if necessary, and resetting the world state.
        """
        self.b = self.ws.copy()
        self.separate_rewards = np.zeros(self.I)
        self.true_reward, self.discounted_reward = 0,0
        self.time_step = 0
        self.true_mdp.reset()
        if self.random_true_mdp:
            self.true_mdp = self.sampleMDPs()
        self.world_state = self.true_mdp.reset()
        return self.world_state




    ############ Helper methods #############


    def sampleMDPs(self):
        """
        Returns an mdp to use as the true mdp.
        """
        return np.random.choice(self.mdps)

    def buildTRs(self, mdp):
        """
        Builds a transition tensor of shape (A,S,S) and a reward matrix of shape (A,S)
        from the P-dict of mdp.
        :param mdp: the mdp the transition tensor and reward matrix are generated from.
        """
        S = self.getAllWorldStates()
        A = self.getAllActions()
        trans_tensor = np.zeros((len(A), len(S), len(S)))
        reward_tensor = np.zeros((len(A), len(S), len(S)))
        for a in A:
            for s in S:
                for t in mdp.P[s][a]:
                    trans_tensor[(a,s,t[1])] = t[0]
                    reward_tensor[(a,s,t[1])] = t[2]
        reward_mat = np.array([np.multiply(trans_tensor[a], reward_tensor[a]).sum(1).reshape(len(S)) for a in A])
        return (trans_tensor, reward_mat)

    def transition(self, initial_state, action, final_state):
        """
        Returns the probability of transitioning from initial_state to
        final_state given an initial state and action.
        :param initial_state: a (world_state, ϴ) tuple.
        :param action: the robot's action.
        :param final_state: a (world_state, ϴ) tuple.
        """
        if initial_state[1] != final_state[1]:
            return 0
        p = 0
        mdp = self.mdps[initial_state[1]]
        for t in mdp.P[initial_state[0]][action]:
            if t[1] == final_state[0]:
                p += t[0]
        return p

    def updateBelief(self, s,a,sprime):
        # self.b += np.ones(self.I) * 1e-12
        raw = [b * self.transition((s,i),a,(sprime,i)) for i,b in enumerate(self.b)]
        self.b = [b /sum(raw) for b in raw]
        return self.b


    def buildObservationMap(self):
        obs = {}
        for mdp in self.mdps:
            for s in range(mdp.nS):
                for a in range(mdp.nA):
                    for t in mdp.P[s][a]:
                        if (s,a) in obs:
                            obs[(s,a)].add(t[1])
                        else:
                            obs[(s,a)] = set([t[1]])
        return obs