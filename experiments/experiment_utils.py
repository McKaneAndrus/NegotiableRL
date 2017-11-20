import numpy as np
import time

def run_simulation(algorithm, vis_history=False):
    game = algorithm.game
    s, done = game.reset(), False
    a = algorithm.getFirstAction(s)
    vals = hasattr(algorithm, 'values')
    if vis_history:
      # Add probabilities?
      history = {'rewards': [game.separate_rewards.copy()],
                 'beliefs': [game.b.copy()],
                 'states': [s],
                 'actions': [a],
                 'reward': [game.true_reward],
                 'policies': [algorithm.policy]}
      if vals:
        history['values'] = [algorithm.values]
    while not done:
      sprime, done, p = game.act(a)
      a = algorithm.getNextAction(s, a, sprime)
      s = sprime
      if vis_history:
        history['rewards'] += [game.separate_rewards.copy()]
        history['beliefs'] += [game.b.copy()]
        history['states'] += [s]
        history['actions'] += [a]
        history['reward'] += [game.true_reward]
        history['policies'] += [algorithm.policy]
        if vals:
          history['values'] += [algorithm.values]
    if vis_history:
      return game.discounted_reward, history
    else:
      return game.discounted_reward

# TODO: need to include value returns for 
def test_alg_settings(alg, n, save_times=False):
    n = int(n)
    rewards = np.zeros(n)
    if save_times:
        times = np.zeros(n)
    for i in range(n):
        if save_times:
            start = time.time()
            rewards[i] = run_simulation(alg)
            times[i] = time.time() - start
        else:
            rewards[i] = run_simulation(alg)
    # print("Mean: " + str(rewards.mean()) + " Std: " + str(rewards.std()))
    if save_times:
        return rewards, times
    else:
        return rewards
