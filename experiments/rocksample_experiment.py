from nrl.envs.rock_sample import *
from nrl.negotiable_envs.negotiablegame import NegotiableGame
from nrl.algorithms import pomdplite, sparsenoc, valueIteration
POMDPlite = pomdplite.POMDPlite
SparseNocAlg = sparsenoc.SparseNocAlg
ValueIteration = valueIteration.ValueIteration
from experiments.experiment_utils import *
import numpy as np
from itertools import product
from timeit import default_timer as timer
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

# # Static Parameters
max_obs_horizon = 6
obs_range = range(2, max_obs_horizon)
max_time=2000
breadth_tests = 1000
full_tests_mult = 5
noc_runtime_tests = 50
betas = np.linspace(0, 0.5, 20)
prefs = ([1,0],[0,1])
agents = 2
max_step = 1000
exp_map="7x7_balanced"
experiment_name = "Rock_Sample_2_Agent_Standard_Prefs"

# Independent Variables
# Max difference between expected action success for questionable squares, where max is always 1
slip_rate_ranges = [(0.05,0.1),(0.1,0.2),(0.05,0.2),(0.1,0.3)]
sparsities = reversed([0.01,0.02,0.03,0.04,0.05])
slippery = [0,1]
gammas = [0.9,0.95]

# Static Parameters
# max_obs_horizon = 7
# obs_range = range(2, max_obs_horizon)
# max_time=10
# breadth_tests = 1
# full_tests_mult = 5
# noc_runtime_tests = 20
# betas = np.linspace(0, 0.5, 20)
# prefs = ([1,0],[0,1])
# agents = 2
# max_step = 1000
# exp_map="7x7_balanced"
# experiment_name = "Rock_Sample_2_Agent_Standard_Prefs"

# Independent Variables
# Max difference between expected action success for questionable squares, where max is always 1
# slip_rate_ranges = [(0.1,0.2)]
# sparsities = [0.01,0.02,0.03,0.04,0.05]
# slippery = [0,1]
# gammas = [0.9]

header_list = ["#Lowest Slip Rate", "Highest Slip Rate", "Sparsity", "Slippery", "Discount"] +\
         ["OH {} time".format(i) for i in obs_range] +\
         ["OH {} Exp Rew".format(i) for i in obs_range] +\
         ["OH {} Av RT".format(i) for i in obs_range] +\
         ["OH {} Std RT".format(i) for i in obs_range] +\
         ["P-Lite RO Mean", "P-Lite RO stderr", "P-lite RO iters",
          "P-Lite Av RT",  "P-Lite Std RT", "P-Lite Best Beta"]
header = ",".join(header_list)

# all_data = []
# @profile
def run_experiment(args):
    try:
        sparsity,slip_rates, slip, gamma = args
        lsr, hsr = slip_rates
        pid = '[' + str(os.getpid()) + '] '
        print(pid + "BSR: {} SRM: {} sparsity: {}  slippery: {}  gamma: {}".format(lsr, hsr, sparsity, slip, gamma))
        data = [lsr, hsr, sparsity, slip, gamma]
        # Guarantee that mdp combination results in desired sparsity
        srs = np.linspace(lsr,hsr,agents,endpoint=True)
        seed = int(timer())
        mdps = [RockSampleEnv(diffs=sparsity, map_name=exp_map, is_slippery=slip, partial_slip=sr, seed=seed) for sr in srs]
        for mdp in mdps:
            mdp.seed(0)
        rocksample_game = NegotiableGame(mdps, gamma=gamma)
        print(pid + "Found suitable environment.")
        sparse_nocs = []
        for i in obs_range:
            signal.signal(signal.SIGALRM, null_handler)
            signal.alarm(max_time)
            try:
                start = timer()
                sparse_nocs += [SparseNocAlg(rocksample_game, max_step, obs_horizon=i, keep_values=True)]
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
            if (end - start) * 6 > max_time and i < max_obs_horizon - 1:
                print(pid + "Plan with obs horizon {} will take too long".format(i+1))
                data += [np.nan for _ in range(i+1, max_obs_horizon)]
                break

        if len(sparse_nocs) == 0:
            return data + [np.nan for _ in range(len(data),len(header_list))]
        exp_rews, run_times = [], []
        s = rocksample_game.reset()
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
        _, ts = test_alg_settings(POMDPlite(rocksample_game, max_step, beta=max(betas), keep_values=True), 2, save_times=True)
        plite_runtime = ts.mean()
        plite_breadth = max(min(round(max_time/(plite_runtime *len(betas))), breadth_tests),1)
        plite_full =  plite_breadth * full_tests_mult
        print(pid + "Finding good beta for {}s with breadth {}".format(
            plite_runtime * (plite_breadth * len(betas) + plite_full),plite_breadth))
        for beta in betas:
            score = test_alg_settings(POMDPlite(rocksample_game, max_step, beta=beta, keep_values=True), plite_breadth).mean()
            if score > best[0]:
                best = (score, beta)
        full_plite_test, full_plite_times = test_alg_settings(
                                        POMDPlite(rocksample_game, max_step, beta=best[1],keep_values=True),
                                        plite_full, save_times=True)
        print(pid + "NOC: {} P-lite {}: {}".format(round(high_noc,5), best[1], round(full_plite_test.mean(),5)))
        data += [full_plite_test.mean(),
                 full_plite_test.std() / np.sqrt(plite_full),
                 plite_full,
                 full_plite_times.mean(),
                 full_plite_times.std() / np.sqrt(plite_full),
                 best[1]]
        return data, rocksample_game, args
    except Exception:
        print("Error handled, but:")
        print(traceback.print_exc())
        return [np.nan] * len(header_list), None, args

def handle_output(data):
    global results_file, games
    f = open(results_file,'a')
    str_data = [str(datum) for datum in data[0]]
    f.write("\n" + ",".join(str_data))
    f.close()
    games[data[2]] = data[1]
    pkl.dump(games, open("data/games/{}.pkl".format(experiment_name), "wb"))


if __name__ == "main":
    games={}
    results_file = "data/{}.csv".format(experiment_name)
    if os.path.isfile(results_file):
        results_file = "data/{}-{}.csv".format(experiment_name,time.strftime("%H-%M-%S"))
    f = open(results_file,'w')
    f.write(header)
    f.close()
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)
    for args in product(sparsities,slip_rate_ranges,slippery,gammas):
        res = pool.apply_async(run_experiment, (args,), callback=handle_output)
    pool.close()
    pool.join()




# num_workers = mp.cpu_count()
# with mp.Pool(num_workers) as p:
    # all_data = p.map(run_experiment, product(sparsities,slip_rate_ranges,slippery,gammas))
# all_data = []
# for args in product(slip_rate_ranges,sparsities, slippery,gammas):
#     all_data += [run_experiment(args)]
# np.savetxt("data/{}.csv".format(experiment_name), all_data, delimiter=",", header=header)

