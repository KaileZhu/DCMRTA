#!/usr/bin/env python
"""Generate a test set of DCMRTA environments and their CTAS-D solver configs.

Usage:
    python scripts/generate_testset.py

Configuration:
    Edit TEST_SET, NUM_INSTANCES, AGENTS_RANGE, TASKS_RANGE below.
"""

import math
import os
import pickle
from itertools import permutations

import yaml

from dcmrta.environment import TaskEnv

# ── Configuration ─────────────────────────────────────────────────────────────
TEST_SET = 'data/testSet_20A_50T_CONDET'
NUM_INSTANCES = 50
AGENTS_RANGE = (20, 20)
TASKS_RANGE = (50, 50)
PLANNER = 'TEAMPLANNER_CONDET'  # or 'TEAMPLANNER_DET'
SOLVER_TIME = 300.0
# ──────────────────────────────────────────────────────────────────────────────


def compute_euclidean_distance_matrix(locations):
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = math.hypot(
                    from_node[0] - to_node[0], from_node[1] - to_node[1],
                )
    return distances


def main():
    env_dir = os.path.join(TEST_SET, 'environments')
    os.makedirs(env_dir, exist_ok=True)

    agent_yaml = {}
    task_yaml = {}
    planner_param = {}
    graph_yaml = {}

    for i in range(NUM_INSTANCES):
        env = TaskEnv(AGENTS_RANGE, TASKS_RANGE, traits_dim=1, max_coalition_size=5, seed=i)
        pickle.dump(env, open(os.path.join(env_dir, f'env_{i}.pkl'), 'wb'))

        config_dir = os.path.join(TEST_SET, 'configs', f'env_{i}')
        os.makedirs(config_dir, exist_ok=True)

        env.force_wait = True
        coords = env.get_matrix(env.task_dic, 'location')
        dist_matrix = compute_euclidean_distance_matrix(coords)
        depot = env.depot['location']
        depot_distance = [
            math.hypot(depot[0] - coords[i][0], depot[1] - coords[i][1])
            for i in range(len(coords))
        ]
        p = list(permutations(range(len(env.task_dic)), 2))

        if PLANNER == 'TEAMPLANNER_CONDET':
            agent_yaml.update({
                'vehicle0': {
                    'engCap': 1e6, 'engCost': 0., 'capVector': [1.0], 'capVar': [0.],
                },
            })
            graph_yaml.update({
                'vehicle0': {
                    f'edge{i}': [
                        t[0], t[1], 0,
                        dist_matrix[t[0]][t[1]], 0,
                        float(dist_matrix[t[0]][t[1]] / 0.2),
                    ]
                    for i, t in enumerate(p)
                },
            })
            for j in range(len(env.task_dic)):
                graph_yaml['vehicle0'][f'edge{2*j + len(p)}'] = [
                    int(env.tasks_num), j, 0, depot_distance[j], 0,
                    depot_distance[j] / 0.2,
                ]
                graph_yaml['vehicle0'][f'edge{2*j + len(p) + 1}'] = [
                    j, int(env.tasks_num) + 1, 0, depot_distance[j], 0,
                    depot_distance[j] / 0.2,
                ]
            for tid, task in env.task_dic.items():
                graph_yaml['vehicle0'][f'node{tid}'] = float(task['time'])

        elif PLANNER == 'TEAMPLANNER_DET':
            for a in range(len(env.agent_dic)):
                agent_yaml.update({
                    f'vehicle{a}': {
                        'engCap': 1e6, 'engCost': 1.,
                        'capVector': [1.0], 'capVar': [0.],
                    },
                })
                graph_yaml.update({
                    f'vehicle{a}': {
                        f'edge{i}': [
                            t[0], t[1], 0,
                            dist_matrix[t[0]][t[1]], 0,
                            float(dist_matrix[t[0]][t[1]] / 0.2),
                        ]
                        for i, t in enumerate(p)
                    },
                })
                for j in range(len(env.task_dic)):
                    graph_yaml[f'vehicle{a}'][f'edge{2*j + len(p)}'] = [
                        int(env.tasks_num) + a, j, 0, depot_distance[j], 0,
                        depot_distance[j] / 0.2,
                    ]
                    graph_yaml[f'vehicle{a}'][f'edge{2*j + len(p) + 1}'] = [
                        j, int(env.tasks_num) + int(env.agents_num) + a, 0,
                        depot_distance[j], 0, depot_distance[j] / 0.2,
                    ]
                for tid, task in env.task_dic.items():
                    graph_yaml[f'vehicle{a}'][f'node{tid}'] = float(task['time'])

        for task in env.task_dic.items():
            task_yaml.update({
                f'task{task[0]}': {
                    'and0': {
                        'or0': {
                            'geq': True, 'capId': 0,
                            'capReq': float(task[1]['requirements']),
                            'capVar': 0.,
                        },
                    },
                },
            })

        with open(os.path.join(config_dir, 'vehicle_param.yaml'), 'w') as f:
            yaml.dump(agent_yaml, f, sort_keys=False)
        with open(os.path.join(config_dir, 'task_param.yaml'), 'w') as f:
            yaml.dump(task_yaml, f, sort_keys=False)

        planner_param = {
            'flagOptimizeCost': True,
            'flagTaskComplete': True,
            'flagSprAddCutToSameType': True,
            'taskCompleteReward': 10000,
            'timePenalty': 100,
            'recoursePenalty': 1.0,
            'taskRiskPenalty': 0.0,
            'LARGETIME': 10000.0,
            'MAXTIME': 1000.0,
            'MAXENG': 1E8,
            'flagSolver': PLANNER,
            'CcpBeta': 0.95,
            'taskBeta': 0.95,
            'solverMaxTime': SOLVER_TIME,
            'solverIterMaxTime': 50.0,
            'flagNotUseUnralavant': True,
            'MAXALPHA': 20.0,
            'taskNum': int(env.tasks_num),
            'vehNum': 1 if PLANNER == 'TEAMPLANNER_CONDET' else int(env.agents_num),
            'capNum': 1,
            'vehTypeNum': 1,
            'vehNumPerType': (
                [int(env.agents_num)]
                if PLANNER == 'TEAMPLANNER_CONDET'
                else [1] * int(env.agents_num)
            ),
            'sampleNum': 500,
            'randomType': 0,
            'capType': [0],
            'vehicleParamFile': f'./{config_dir}/vehicle_param.yaml',
            'taskParamFile': f'./{config_dir}/task_param.yaml',
            'graphFile': f'./{config_dir}/graph.yaml',
        }
        with open(os.path.join(config_dir, 'planner_param.yaml'), 'w') as f:
            yaml.dump(planner_param, f, sort_keys=False)
        with open(os.path.join(config_dir, 'graph.yaml'), 'w') as f:
            yaml.dump(graph_yaml, f, sort_keys=False)

    print(f"Test set generated at {TEST_SET}")


if __name__ == '__main__':
    main()
