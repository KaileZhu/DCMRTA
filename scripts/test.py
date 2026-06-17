#!/usr/bin/env python
"""Evaluate a trained DCMRTA model on a test set.

Usage:
    python scripts/test.py

Configuration:
    Edit METHOD ('LF' for coalition-based, 'IA' for independent agent),
    TEST_SET (path to test set directory), and FOLDER_NAME below.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

from dcmrta.attention import AttentionNet
from dcmrta.config import AGENT_INPUT_DIM, EMBEDDING_DIM, FOLDER_NAME, TASK_INPUT_DIM
from dcmrta.worker import Worker

# ── Configuration ─────────────────────────────────────────────────────────────
METHOD = 'LF'       # 'LF' = coalition-based (leader-follower), 'IA' = independent agent
TEST_SET = 'data/testSet_20A_50T_CONDET'
DEVICE = 'cpu'
# ──────────────────────────────────────────────────────────────────────────────

model_path = f'checkpoints/{FOLDER_NAME}'

device = torch.device(DEVICE)
global_network = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM).to(device)
checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location=device)
global_network.load_state_dict(checkpoint['model'])
worker = Worker(0, global_network, global_network, 0, device)

perf_metrics = {
    'success_rate': [], 'makespan': [], 'time_cost': [],
    'waiting_time': [], 'travel_dist': [], 'efficiency': [],
}
df = pd.DataFrame(perf_metrics)

for i in range(50):
    env = pickle.load(open(f'{TEST_SET}/environments/env_{i}.pkl', 'rb'))
    agents = env.agent_dic
    tasks = env.task_dic
    depot = env.depot
    env.max_waiting_time = 10
    env.reactive_planning = False
    test_env = (tasks, agents, depot)
    env.reset(test_env)
    env.clear_decisions()
    env.force_wait = True

    if METHOD == 'IA':
        results = worker.run_episode(i, mode='independent', test_env=test_env)
    else:
        results = worker.run_episode(i, mode='evaluation', test_env=test_env)

    df_ = pd.DataFrame(results, index=[i])
    df = pd.concat([df, df_])

os.makedirs(f'{TEST_SET}/results', exist_ok=True)
df.to_csv(f'{TEST_SET}/results/REINFORCE_{METHOD}.csv')
print(f"Results saved to {TEST_SET}/results/REINFORCE_{METHOD}.csv")
