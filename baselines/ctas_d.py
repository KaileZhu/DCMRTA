"""CTAS-D baseline solver integration.

Reads CTAS-D solver results from YAML files and executes the routes on the
DCMRTA environment.
"""

import copy
import glob
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from natsort import natsorted

from dcmrta.environment import TaskEnv


def get_agent_route(param_file_path, result_file_path):
    """Parse agent routes from CTAS-D output YAML files."""
    with open(param_file_path, 'r') as f:
        param_data = yaml.safe_load(f)
    if param_data['flagSolver'] == 'TEAMPLANNER_DET':
        num_veh = param_data['vehNum']
    else:
        num_veh = param_data['vehNumPerType'][0]

    with open(result_file_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'vehicle' not in data:
        return None

    nodes = []
    for i in range(num_veh):
        key = f'vv{i + 1}'
        if key not in data['vehicle']:
            continue
        nodes.append(data['vehicle'][key]['node'])
    return nodes


def ctasd_read_results(env, path):
    """Load CTAS-D results and pre-set routes on the environment."""
    if os.path.exists(path + 'results.yaml'):
        routes = get_agent_route(path + 'planner_param.yaml', path + 'results.yaml')
        if routes is None:
            return None
        for i in range(len(routes)):
            if routes[i] != [0]:
                env.pre_set_route(copy.copy(routes[i])[1:], i)
        return True
    return None


def run_ctasd_evaluation(folder='data/testSet_20A_50T_CONDET'):
    """Run CTAS-D baseline evaluation on a test set and save results CSV."""
    method = 'CTAS-D'
    env_dir = os.path.join(folder, 'environments')
    config_dir = os.path.join(folder, 'configs')
    results_dir = os.path.join(folder, 'results')
    os.makedirs(results_dir, exist_ok=True)

    files = natsorted(glob.glob(os.path.join(env_dir, 'env_*.pkl')), key=lambda y: y.lower())
    perf_metrics = {
        'success_rate': [], 'makespan': [], 'time_cost': [],
        'waiting_time': [], 'travel_dist': [], 'efficiency': [],
    }

    for pkl_path in files:
        env = pickle.load(open(pkl_path, 'rb'))
        agents = env.agent_dic
        tasks = env.task_dic
        depot = env.depot
        env.reactive_planning = False
        test_env = (tasks, agents, depot)
        env.reset(test_env)
        env.clear_decisions()

        env_num = os.path.basename(pkl_path).replace('.pkl', '')
        cfg_path = os.path.join(config_dir, env_num, '')

        result = ctasd_read_results(env, cfg_path)
        if result is None:
            perf_metrics['success_rate'].append(0)
            for key in ['makespan', 'time_cost', 'waiting_time', 'travel_dist', 'efficiency']:
                perf_metrics[key].append(np.nan)
            continue

        env.force_wait = True
        env.execute_by_route(cfg_path, method, False)
        reward, finished_tasks = env.get_episode_reward(100)

        if np.sum(finished_tasks) / len(finished_tasks) < 1:
            perf_metrics['success_rate'].append(np.sum(finished_tasks) / len(finished_tasks))
            for key in ['makespan', 'time_cost', 'waiting_time', 'travel_dist', 'efficiency']:
                perf_metrics[key].append(np.nan)
        else:
            perf_metrics['success_rate'].append(np.sum(finished_tasks) / len(finished_tasks))
            perf_metrics['makespan'].append(env.current_time)
            perf_metrics['time_cost'].append(
                np.sum(np.nan_to_num(env.get_matrix(env.task_dic, 'time_start'), nan=100))
            )
            perf_metrics['waiting_time'].append(
                np.mean(env.get_matrix(env.agent_dic, 'sum_waiting_time'))
            )
            perf_metrics['travel_dist'].append(
                np.sum(env.get_matrix(env.agent_dic, 'travel_dist'))
            )
            perf_metrics['efficiency'].append(
                np.mean(env.get_matrix(env.task_dic, 'sum_waiting_time'))
            )
        print(pkl_path)

    df = pd.DataFrame(perf_metrics)
    df.to_csv(os.path.join(results_dir, f'{method}.csv'))
    print(f"Results saved to {results_dir}/{method}.csv")


if __name__ == '__main__':
    run_ctasd_evaluation()
