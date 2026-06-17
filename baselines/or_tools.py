"""OR-Tools VRP baseline solver for DCMRTA.

Uses Google OR-Tools routing library to solve a Vehicle Routing Problem (VRP)
approximation of the MRTA task allocation problem.
"""

import copy
import glob
import math
import os
import pickle

import numpy as np
import pandas as pd
from natsort import natsorted
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from dcmrta.environment import TaskEnv


def compute_euclidean_distance_matrix(locations):
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = int(
                    math.hypot(from_node[0] - to_node[0], from_node[1] - to_node[1]) * 5
                    + to_node[2]
                )
    return distances


def routes2id(routes, task_dict):
    tasks = []
    for i in routes:
        if i == 0:
            tasks.append(0)
        else:
            tasks.append(task_dict[i - 1]['ID'] + 1)
    return tasks


class TSPSolver:
    def __init__(self):
        self.magnify = 1000
        self.coords = None

    def create_data_model(self, coords, num_vehicles=1, depot=0):
        data = {
            'locations': np.array(coords) * self.magnify,
            'num_vehicles': num_vehicles,
            'depot': depot,
        }
        return data

    @staticmethod
    def print_solution(data, manager, routing, solution):
        routes = {}
        max_route_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id,
                )
                routes[vehicle_id] = routes.get(vehicle_id, []) + [manager.IndexToNode(index)]
            max_route_distance = max(route_distance, max_route_distance)
        return routes, max_route_distance

    def run_solver(self, coords, num_vehicles=1, depot=0):
        data = self.create_data_model(coords, num_vehicles)
        distance_matrix = compute_euclidean_distance_matrix(data['locations'])
        manager = pywrapcp.RoutingIndexManager(
            len(data['locations']), data['num_vehicles'], data['depot'],
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index, 0, 200000, True, dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.time_limit.seconds = 10

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            route, route_distance = self.print_solution(data, manager, routing, solution)
            return route, route_distance
        return None

    def vrp(self, env):
        """Split agents into groups and solve mTSP for each group."""
        task_groups, agent_groups = env.get_grouped_tasks()
        routes = {}
        agent_id = 0
        for cat, tasks in task_groups.items():
            coords = env.get_matrix(tasks, 'location')
            time_ = env.get_matrix(tasks, 'time')
            coords = np.hstack([coords, np.array(time_).reshape(len(time_), -1)])
            coords = np.vstack([env.depot['location'].tolist() + [0], coords])
            routes, _ = self.run_solver(coords, agent_groups[cat])
            for i in range(agent_groups[cat]):
                routes[i] = routes2id(routes[i], tasks)
                if routes[i] == [0]:
                    continue
                for _ in range(cat):
                    env.pre_set_route(copy.copy(routes[i])[:-1], agent_id)
                    agent_id += 1
                    if agent_id >= env.agents_num:
                        agent_id -= env.agents_num
        return routes


def run_ortools_evaluation(folder='data/testSet_20A_50T_CONDET'):
    """Run OR-Tools baseline evaluation on a test set and save results CSV."""
    solver = TSPSolver()
    method = 'OR-Tools'
    env_dir = os.path.join(folder, 'environments')
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

        solver.vrp(env)
        env.force_wait = True
        env.execute_by_route(
            os.path.join(os.path.dirname(pkl_path), ''), method, False,
        )
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
    run_ortools_evaluation()
