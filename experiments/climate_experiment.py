from nrl.envs.climate_control import *
from nrl.negotiable_envs.negotiablegame import NegotiableGame
from nrl.algorithms import pomdplite, sparsenoc, valueIteration
plite = pomdplite.POMDPlite
snoc = sparsenoc.SparseNocAlg
vi = valueIteration.ValueIteration
from experiments.experiment_utils import *
import numpy as np
from itertools import product
from timeit import default_timer as timer
import time
import signal
import multiprocessing as mp
import os
import pickle as pkl
import traceback


class TimeoutError(Exception):
    pass

def null_handler(signum, frame):
    raise TimeoutError("Timeout")

data_dir = 'data/games'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Static Parameters
max_obs_horizon = 6
max_time=2000
breadth_tests = 1000
full_tests_mult = 5
noc_runtime_tests = 20
betas = np.linspace(0, 15, 20)
experiment_name = "R+S_Variations"

# Independent Variables
time_horizons = [75]
gammas = [0.9]
obs_range = range(2, max_obs_horizon)
agents = [2]
rooms = reversed(range(2,7))
temps = [3]
sparsities = reversed([0.01,0.02,0.03,0.04,0.05])


# max_obs_horizon = 7
# max_time=1
# breadth_tests = 10
# noc_runtime_tests = 20
# full_tests_mult = 5
# betas = np.linspace(0, 15, 20)
# experiment_name = "Testytest"
#
#
# time_horizons = [50]
# gammas = [0.9]
# obs_range = range(2, max_obs_horizon)
# agents = [2]
# rooms = [2]
# temps = [2]
# sparsities = [0.01,0.02]
#

header_list = ["Num Rooms", "Num Temps", "Sparsity", "Num Agents", "Time Horizon", "Discount"] +\
        ["OH {} time".format(i) for i in obs_range] +\
        ["OH {} Exp Rew".format(i) for i in obs_range] +\
        ["OH {} Av RT".format(i) for i in obs_range] +\
        ["OH {} Std RT".format(i) for i in obs_range] +\
        ["P-Lite RO Mean", "P-Lite RO stderr", "P-lite RO iters",
         "Max P-Lite Early Term Error", "P-Lite Av RT", "P-Lite Std RT", "P-Lite Best Beta"]
header = ",".join(header_list)

# all_data = []
# @profile
def run_experiment(args):
    try:
        nR, nT, sparsity, num_agents, ms, gamma = args
        pid = '[' + str(os.getpid()) + '] '
        print(pid + "nR: {} nT: {} sparsity: {}  # agents: {}  max time: {}  gamma: {}".format(
            nR, nT, sparsity, num_agents, ms, gamma))
        data = [nR, nT, sparsity, num_agents, ms, gamma]
        # Guarantee that mdp combination results in desired sparsity
        diffs = 0
        while diffs < num_agents * (round(nT ** nR * sparsity) + 1):
            mdps = [ClimateControlEnv(num_rooms=nR, num_temps=nT, sparsity=sparsity, max_steps=ms) for _ in
                    range(num_agents)]
            q_tuples,_,_ = snoc.find_diffs(mdps)
            diffs = len(q_tuples)
        for mdp in mdps:
            mdp.seed(0)
        climate_game = NegotiableGame(mdps, gamma=gamma)
        print(pid + "Found suitable environment.")
        max_plite_error = gamma ** ms * max(sum([pref[1] for pref in mdp.prefs]) for mdp in mdps)
        sparse_nocs = []
        for i in obs_range:
            signal.signal(signal.SIGALRM, null_handler)
            signal.alarm(max_time)
            try:
                start = timer()
                sparse_nocs += [snoc(climate_game, ms, obs_horizon=i, keep_values=True)]
                end = timer()
            except Exception as ex:
                print(pid + "Plan with obs horizon {} took too long".format(i))
                data += [np.nan for _ in range(i, max_obs_horizon)]
                break
            finally:
                signal.alarm(0)
            # Look ahead to cut off evaluations that will definitely timeout
            data += [end - start]
            print(pid + "Plan with obs horizon {} took {}s".format(i, end - start))
            if (end - start) * 8 > max_time and i < max_obs_horizon - 1:
                print(pid + "Plan with obs horizon {} will take too long".format(i+1))
                data += [np.nan for _ in range(i+1, max_obs_horizon)]
                break

        if len(sparse_nocs) == 0:
            return data + [np.nan for _ in range(len(data),len(header_list))]
        exp_rews, run_times = [], []
        s = climate_game.reset()
        for noc in sparse_nocs:
            noc.getFirstAction(s)
            exp_rews += [noc.values[s]]
            _, rts = test_alg_settings(noc, noc_runtime_tests, save_times=True)
            run_times += [rts]
        high_noc = exp_rews[-1]
        if len(sparse_nocs) != len(obs_range):
            exp_rews += [np.nan for _ in range(2 + len(sparse_nocs), max_obs_horizon)]
            run_times += [np.array([np.nan]) for _ in range(2 + len(sparse_nocs), max_obs_horizon)]
        data += exp_rews + [times.mean() for times in run_times] +\
                [times.std()/np.sqrt(noc_runtime_tests) for times in run_times]


        best = (0, float("-inf"))
        _, ts = test_alg_settings(plite(climate_game, ms, beta=max(betas), keep_values=True), 2, save_times=True)
        plite_runtime = ts.mean()
        plite_breadth = max(min(round(max_time/(plite_runtime *len(betas))), breadth_tests),1)
        plite_full =  plite_breadth * full_tests_mult
        print(pid + "Finding good beta for {}s with breadth {}".format(
            plite_runtime * (plite_breadth * len(betas) + plite_full),plite_breadth))
        for beta in betas:
            score = test_alg_settings(plite(climate_game, ms, beta=beta, keep_values=True), plite_breadth).mean()
            if score > best[0]:
                best = (score, beta)
        full_plite_test, full_plite_times = test_alg_settings(plite(climate_game, ms, beta=best[1], keep_values=True),
                                                              plite_full, save_times=True)
        print(pid + "NOC: {} P-lite {}: {}".format(round(high_noc,5), best[1], round(full_plite_test.mean(),5)))
        data += [full_plite_test.mean(),
                 full_plite_test.std() / np.sqrt(plite_full),
                 plite_full,
                 max_plite_error,
                 full_plite_times.mean(),
                 full_plite_times.std() / np.sqrt(plite_full),
                 best[1]]
        print(pid + "Completed experiment.")
        return data, climate_game, args
    except Exception:
        print("Error handled, but:")
        print(traceback.print_exc())
        return ([np.nan] * len(header_list), None, args)



def handle_output(data):
    global results_file, games
    f = open(results_file,'a')
    str_data = [str(datum) for datum in data[0]]
    f.write("\n" + ",".join(str_data))
    f.close()
    games[data[2]] = data[1]
    pkl.dump(games, open("data/games/{}.pkl".format(experiment_name), "wb"))


if __name__ == "__main__":
    games = {}
    results_file = "data/{}.csv".format(experiment_name)
    if os.path.isfile(results_file):
        results_file = "data/{}-{}.csv".format(experiment_name,time.strftime("%H-%M-%S"))
    f = open(results_file,'w')
    f.write(header)
    f.close()
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)
    for args in product(rooms,temps, sparsities,agents,time_horizons,gammas):
        res = pool.apply_async(run_experiment, (args,), callback=handle_output)
    pool.close()
    pool.join()

# try:
# num_workers = mp.cpu_count()
# with mp.Pool(num_workers) as p:
#     all_data = p.map(run_experiment, product(rooms,temps, sparsities,agents,time_horizons,gammas))
# all_data = []
# for args in product(rooms,temps, sparsities,agents,time_horizons,gammas):
#     all_data += run_experiment(args)
# np.savetxt("data/climate{}.csv".format(time.strftime("%d.%m_%H-%M-%S")), all_data, delimiter=",", header=header)
# except KeyboardInterrupt:
#     np.savetxt("data/climate{}.csv".format(time.strftime("%d.%m_%H-%M-%S")), all_data, delimiter=",", header=header)

