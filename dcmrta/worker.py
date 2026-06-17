"""Worker that executes episodes and tests for DCMRTA.

Provides a unified inference method that covers the four previously duplicated
paths: training rollouts, evaluation, independent sampling, and baseline.
"""

import copy

import numpy as np
import scipy.signal as signal
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from dcmrta.attention import AttentionNet
from dcmrta.config import (
    AGENTS_RANGE, AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM,
    COALITION_SIZE, GAMMA, MAX_TIME, TASKS_RANGE, gifs_path, model_path,
)
from dcmrta.environment import TaskEnv


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker:
    """Runs episodes using a given policy network and collects results.

    Supports three execution modes via ``run_episode``:
    - ``training`` (default): sample actions via Categorical distribution
    - ``evaluation``: greedy argmax action selection
    - ``independent``: greedy argmax, no coalition grouping (agent-level decisions)
    """

    def __init__(self, meta_agent_id, local_network, local_baseline, global_step,
                 device='cuda', save_image=False, agents_num=AGENTS_RANGE,
                 tasks_num=TASKS_RANGE, seed=None):
        self.device = device
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image

        self.env = TaskEnv(
            agents_num, tasks_num, traits_dim=1, max_coalition_size=COALITION_SIZE,
            seed=seed, plot_figure=save_image,
        )
        self.baseline_env = copy.deepcopy(self.env)
        self.local_net = local_network
        self.local_baseline = local_baseline
        self.experience = None
        self.episode_number = None
        self.perf_metrics = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def work(self, episode_number):
        """Training episode: collect experience buffer and metrics."""
        self.episode_number = episode_number
        self.perf_metrics = self.run_episode(episode_number, mode='training')

    def run_episode(self, episode_number, mode='training', test_env=None, image_path=None):
        """Unified episode runner.

        :param episode_number: episode index for logging / image naming
        :param mode: 'training', 'evaluation', or 'independent'
        :param test_env: if provided, use this env tuple instead of self.env
        :param image_path: if provided, save animation to this path
        :return: perf_metrics dict
        """
        env = self._prepare_env(mode, test_env)
        episode_buffer = [[] for _ in range(9)]
        perf_metrics = {}

        while not env.finished and env.current_time < MAX_TIME:
            with torch.no_grad():
                decision_agents, current_time = env.next_decision()
                groups = env.get_unique_group(decision_agents)
                env.current_time = current_time
                env.task_update()
                env.agent_update()

                if mode == 'independent':
                    # Agent-level greedy decisions, no coalition grouping
                    for agent_id in decision_agents:
                        self._make_decision(
                            env, [agent_id], mode, episode_buffer, training=(mode == 'training'),
                        )
                else:
                    for group in groups:
                        while len(group) > 0:
                            self._make_decision(
                                env, group, mode, episode_buffer, training=(mode == 'training'),
                            )
                env.finished = env.check_finished()

        reward, finished_tasks = env.get_episode_reward(MAX_TIME)
        perf_metrics = self._collect_metrics(env, finished_tasks)

        if mode == 'training':
            self._compute_advantages(episode_buffer, reward)
        if self.save_image and mode in ('training', 'evaluation'):
            env.plot_animation(gifs_path, episode_number)
        if image_path is not None:
            env.plot_animation(image_path, 'RL')

        if mode == 'training':
            self.experience = episode_buffer

        return perf_metrics

    def baseline_test(self):
        """Run baseline (greedy) policy on baseline_env and return reward."""
        self.baseline_env.plot_figure = False
        self.run_episode(0, mode='evaluation',
                         test_env=(self.baseline_env.task_dic,
                                   self.baseline_env.agent_dic,
                                   self.baseline_env.depot))
        reward, _ = self.baseline_env.get_episode_reward(MAX_TIME)
        return reward

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prepare_env(self, mode, test_env):
        """Set up the environment for the given mode."""
        if test_env is not None:
            # Use externally provided environment (for testing / baseline)
            tasks, agents, depot = test_env
            self.baseline_env.task_dic = tasks
            self.baseline_env.agent_dic = agents
            self.baseline_env.depot = depot
            self.baseline_env.tasks_num = len(tasks)
            self.baseline_env.agents_num = len(agents)
            self.baseline_env.plot_figure = False
            self.baseline_env.finished = False
            self.baseline_env.current_time = 0
            return self.baseline_env
        elif mode == 'training':
            return self.env
        else:
            return self.baseline_env

    def _make_decision(self, env, group, mode, buffer, training=True):
        """Make a single decision for the leader of a group."""
        leader_id = np.random.choice(group)
        agent = env.agent_dic[leader_id]
        if agent['returned']:
            return

        mask = env.get_unfinished_task_mask()
        if np.sum(mask) == env.tasks_num:
            mask = np.insert(mask, 0, False)  # No depot when all tasks unfinished
        else:
            mask = np.insert(mask, 0, True)

        total_agents = torch.FloatTensor(
            env.get_current_agent_status(agent)
        ).unsqueeze(0).to(self.device)
        task_info = torch.FloatTensor(
            env.get_current_task_status(agent)
        ).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(mask).unsqueeze(0).to(self.device)
        agent_id_tensor = torch.tensor([[[agent['ID']]]]).to(self.device)

        # Determine which network to use
        net = self.local_baseline if not training else self.local_net
        logp_list = net(task_info, total_agents, mask_tensor)

        if training:
            # Sample from distribution
            action = Categorical(logp_list.exp()).sample()
            while action.item() > env.tasks_num:
                action = Categorical(logp_list.exp()).sample()
        else:
            # Greedy argmax
            action = torch.argmax(logp_list.exp() * ~mask_tensor, dim=1)

        current_action_index = agent.get('current_action_index', 0)

        # Execute: return to depot (action==0) or form coalition (action>0)
        if action.item() == 0:
            group.remove(leader_id)
            env.agent_step(leader_id, action.item())
        else:
            group, _ = env.step(group, leader_id, action.item(), current_action_index)

        env.task_update()
        agent['current_action_index'] = current_action_index
        env.agent_update()

        # Buffer collection (training mode)
        if training:
            buffer[0] += [total_agents]
            buffer[1] += [task_info]
            buffer[2] += [action.unsqueeze(0)]
            buffer[3] += [mask_tensor]
            buffer[4] += [torch.FloatTensor([[0]]).to(self.device)]  # reward placeholder
            buffer[5] += [agent_id_tensor]
            buffer[6] += [torch.FloatTensor([[0]]).to(self.device)]  # advantage placeholder

    def _compute_advantages(self, episode_buffer, reward):
        """Compute discounted advantages for a training episode."""
        episode_buffer[4][-1] += reward
        self.baseline_test()
        greedy_reward, _ = self.baseline_env.get_episode_reward(MAX_TIME)
        adv = reward - greedy_reward
        episode_buffer[6][-1] += adv

        advantages = [a.cpu().numpy() for a in episode_buffer[6]]
        discounted = discount(np.array(advantages).reshape(-1), GAMMA).tolist()
        discounted = torch.FloatTensor(discounted).unsqueeze(1).to(self.device)
        for i in range(len(advantages)):
            episode_buffer[6][i] = discounted[i, :]

    @staticmethod
    def _collect_metrics(env, finished_tasks):
        """Extract standard performance metrics from the environment."""
        return {
            'success_rate': np.sum(finished_tasks) / len(finished_tasks),
            'makespan': env.current_time,
            'time_cost': np.nanmean(env.get_matrix(env.task_dic, 'time_start')),
            'waiting_time': np.mean(env.get_matrix(env.agent_dic, 'sum_waiting_time')),
            'travel_dist': np.sum(env.get_matrix(env.agent_dic, 'travel_dist')),
            'efficiency': np.mean(env.get_matrix(env.task_dic, 'sum_waiting_time')),
        }

    # ── Padding utilities (static) ────────────────────────────────────────────

    @staticmethod
    def zero_padding(a, max_len=AGENTS_RANGE[1]):
        return F.pad(a, (0, 0, 0, max_len - a.shape[1]), 'constant', -1)

    @staticmethod
    def true_padding(a, max_len=TASKS_RANGE[1]):
        return F.pad(a, (0, max_len - a.shape[1]), 'constant', True)


# ═══════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = torch.device('cpu')
    local_network = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM).to(device)
    for i in range(10):
        worker = Worker(1, local_network, local_network, 0, device=device, seed=i, save_image=False)
        worker.run_episode(i, mode='training')
        print(i)
