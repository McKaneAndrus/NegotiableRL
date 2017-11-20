import numpy as np


class ValueIteration:
    """
    An implementation of Value Iteration.
    """
    def __init__(self, game, T):
        """
        Initializes an instance of the Value Iteration solver.

        :param game: the game to be solved.
        :param T: the time horizon of the game.
        """
        self.game = game
        self.T = T
        self.alpha_dict = None

    def initialize(self):
        self.valueIteration()

    def getFirstAction(self,s):
        """
        Run value iteration to accrue alpha-vectors, then return best action
        :param s: the initial state
        :return: best action
        """
        if self.alpha_dict is None:
            self.initialize()
        return self.getNextAction(None,None,s)

    def getFirstActionValue(self,s):
        if self.alpha_dict is None:
            self.initialize()
        return self.getNextActionValue(None,None,s)


    def getNextAction(self,s,a,sprime):
        """
        Use computed alpha vectors to get the best action from sprime
        :param s: the previous state
        :param a: the previous action
        :param sprime: the current state
        :return: best action
        """
        belief = self.game.getBelief()
        alphacts = self.alpha_dict[sprime]
        values = np.zeros(len(alphacts))
        for i, alphact in enumerate(alphacts):
            values[i] = alphact.evaluateBelief(belief)
        optimal_action = alphacts[np.argmax(values)].action
        return optimal_action

    def getNextActionValue(self,s,a,sprime):
        """
        Use computed alpha vectors to get the best action from sprime
        :param s: the previous state
        :param a: the previous action
        :param sprime: the current state
        :return: value of best action
        """
        belief = self.game.getBelief()
        alphacts = self.alpha_dict[sprime]
        values = np.zeros(len(alphacts))
        for i, alphact in enumerate(alphacts):
            values[i] = alphact.evaluateBelief(belief)
        return np.argmax(values)



    # HELPER FUNCTIONS

    def valueIteration(self):
        """
        Returns a set of height self.T computed through value iteration.

        :param initial_plans: the set of initial conditional plans.
        """
        states = self.game.getAllWorldStates()
        t = self.T
        print("Beginning Value Iteration with Horizon " + str(self.T))
        while t > 0:
            print("\nTime : " + str(t))
            print("Generating New Alphas with " + str(sum([len(self.alpha_dict[s]) for s in states])) + " Alphas")
            self.alpha_dict = self.backup(self.alpha_dict)
            t -= 1
        print("Generated " + str(sum([len(self.alpha_dict[s]) for s in states])) + " Alphas")

    def initialAlphaDict(self):
        """
        Creates the initial set of plans i.e. plans at time T for the robot in
        the specified game.

        Since the reward is defined over states, all plans here will have the
        same alpha vector and hence we only need to define a single plan in the
        initial set.

        """
        states = self.game.getAllWorldStates()
        base_action = self.game.getAllActions()[0]
        initial_alpha = np.zeros(len(self.game.getAllTheta()))
        return {s:[AlphaAct(base_action,initial_alpha)] for s in states}

    def backup(self, Vs):
        U = {}
        V = {}
        for s in self.game.getAllWorldStates():
            V[s] = {}
            for a in self.game.getAllActions():
                Vsases = []
                for sprime in self.game.observationMap[(s,a)]:
                    Vsas = []
                    T_vec = np.array([self.game.transition((s, theta), a, (sprime, theta)) for theta in self.game.getAllTheta()])
                    R_vec = np.array([self.game.getReward((s, theta), a, (sprime, theta)) for theta in self.game.getAllTheta()])
                    for alphact in Vs[sprime]:
                        Vsas += [T_vec * (R_vec + alphact.alpha)]
                    Vsases += [Vsas]
                V[s][a] = [AlphaAct(a,alpha) for alpha in Vsases[0]]
                for i in range(1,len(Vsases)):
                    temp_alphacts = []
                    for alphact in V[s][a]:
                        for beta in Vsases[i]:
                            temp_alphacts += [AlphaAct(a, alphact.alpha+beta)]
                    V[s][a] = self.prune(temp_alphacts)
            Vsa_s = [V[s][a] for a in self.game.getAllActions()]
            U[s] = self.prune([alphact for Vsa in Vsa_s for alphact in Vsa])
        return U


    def prune(self, alphacts):
        """
        Returns a set of non-dominated plans after pruning the provided set
            of plans.

        :param plans: the set of plans to be pruned.
        """
        if len(alphacts) <= 1:
            return alphacts
        alphas = np.array([alphact.alpha for alphact in alphacts])
        to_keep = []
        for i in range(len(alphacts)):
            alphas[i,:] = alphacts[i].alpha
        for i in range(len(alphacts)):
            if not self.is_dominated(alphas, i):
                to_keep.append(alphacts[i])
        return to_keep


    def is_dominated(self, array, i):
        """
        Given a numpy array, returns false if the i^th row is entirely
        dominated by another row (i.e. every entry is less than or equal
        to some other row, but the two rows are not equal) in the array
        and true otherwise.

        :param array: the numpy array
        :param i: the index to check
        """
        row_length = array.shape[1]
        for j in range(array.shape[0]):
            if j == i or np.array_equal(array[i,:], array[j,:]):
                continue
            num_leq = sum(array[i,:] <= array[j,:])
            if num_leq == row_length:
                return True
        return False

class AlphaAct:

    def __init__(self, action, alpha):
        self.alpha = alpha
        self.action = action

    def __copy__(self):
        return AlphaAct(self.action,self.alpha.copy())

    def evaluateBelief(self, belief):
        return np.dot(self.alpha, belief)



    # def prune(self, alphacts):
    #     """
    #     Remove dominated alpha-vectors using Lark's filtering algorithm
    #     """
    #     n_thetas = len(self.game.getAllTheta())
    #     # parameters for linear program
    #     delta = 0.0000000001
    #     # equality constraints on the belief states
    #     A_eq = np.array([np.append(np.ones(n_thetas), [0.])])
    #     b_eq = np.array([1.])
    #
    #     # dirty set
    #     F = alphacts.copy()
    #     # clean set
    #     Q = set()
    #     bests = set()
    #     for i in range(n_thetas):
    #         max_i = -np.inf
    #         best = None
    #         for alphact in F:
    #             if alphact.alpha[i] > max_i:
    #                 max_i = alphact.alpha[i]
    #                 best = alphact
    #         bests.add(best)
    #     for best in bests:
    #         Q.add(best)
    #         F.remove(best)
    #     while F:
    #         alphact_i = F[-1]  # get a reference to alphact_i
    #         # F.add(alphact_i)alpha # don't want to remove it yet from F
    #         dominated = False
    #         for alphact_j in Q:
    #             c = np.append(np.zeros(n_thetas), [1.])
    #             A_ub = np.array([np.append(-(alphact_i.alpha - alphact_j.alpha), [-1.])])
    #             b_ub = np.array([-delta])
    #
    #             res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
    #             if res.x[n_thetas] > 0.0:
    #                 # this one is dominated
    #                 dominated = True
    #                 F.pop()
    #                 break
    #
    #         if not dominated:
    #             max_k = -np.inf
    #             best = None
    #             for alphact_k in F:
    #                 b = res.x[0:n_thetas]
    #                 v = np.dot(alphact_k.alpha, b)
    #                 if v > max_k:
    #                     max_k = v
    #                     best = alphact_k
    #             F.remove(best)
    #             if not self.check_duplicate(Q, best):
    #                 Q.update({best})
    #     return Q
    #
    # @staticmethod
    # def check_duplicate(alphacts, alphact):
    #     """
    #     Check whether alpha vector av is already in set a
    #     :param a:
    #     :param av:
    #     :return:
    #     """
    #     for alphact_i in alphacts:
    #         if np.allclose(alphact_i.alpha, alphact.alpha):
    #             return True


    #
    # def prune_vecs(self, alphas):
    #     """
    #     Returns a set of non-dominated plans after pruning the provided set
    #         of plans.
    #
    #     :param plans: the set of plans to be pruned.
    #     """
    #     if len(alphas) <= 1:
    #         return alphas
    #     alpha_mat = np.array(alphas)
    #     to_keep = []
    #     for i in range(len(alphas)):
    #         if not self.is_dominated(alpha_mat, i):
    #             to_keep.append(alphas[i])
    #     return to_keep
    #

    # def findBestAction(self):
    #     """
    #     Returns the best plan to follow given a set of plans (with the alpha
    #     vectors computed).
    #
    #     :param plans:
    #     """
    #     # states = self.game.allStates
    #     belief = self.game.getBelief()
    #     alphacts = self.alpha_dict[s]
    #     values = np.zeros(len(alphacts))
    #     for i, alphact in enumerate(alphacts):
    #         values[i] = alphact.evaluateBelief(belief)
    #     optimal_action = alphacts[np.argmax(values)].action
    #     value = max(values)
    #     return optimal_action, value