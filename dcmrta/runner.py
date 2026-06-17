"""Distributed runner for DCMRTA training and evaluation.

Runner: single-process wrapper around Worker.
RLRunner: Ray remote actor for distributed training (requires ``ray``).
"""

import numpy as np
import torch

from dcmrta.attention import AttentionNet
from dcmrta.config import (
    AGENTS_RANGE, AGENT_INPUT_DIM, EMBEDDING_DIM, TASK_INPUT_DIM,
    NUM_GPU, NUM_META_AGENT, SAVE_IMG, SAVE_IMG_GAP, TASKS_RANGE, USE_GPU,
)
from dcmrta.environment import TaskEnv
from dcmrta.worker import Worker


class Runner:
    """Actor object to start running simulation on workers.

    Gradient computation is also executed on this object.
    """

    def __init__(self, meta_agent_id):
        self.metaAgentID = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)
        self.local_baseline = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM)
        self.local_baseline.to(self.device)

    def get_weights(self):
        return self.local_network.state_dict()

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def set_baseline_weights(self, weights):
        self.local_baseline.load_state_dict(weights)

    def single_threaded_job(self, episode_number, agents_num, tasks_num):
        save_img = SAVE_IMG and (episode_number % SAVE_IMG_GAP == 0)
        worker = Worker(
            self.metaAgentID, self.local_network, self.local_baseline,
            episode_number, self.device, save_img, agents_num, tasks_num,
        )
        worker.work(episode_number)
        return worker.experience, worker.perf_metrics

    def testing(self, agents_range=AGENTS_RANGE, tasks_range=TASKS_RANGE, seed=None):
        worker = Worker(
            self.metaAgentID, self.local_network, self.local_baseline,
            0, self.device, False, agents_num=agents_range,
            tasks_num=tasks_range, seed=seed,
        )
        return worker.baseline_test()

    def comparison(self, test_episode, sample, sample_number, env_params):
        worker = Worker(
            self.metaAgentID, self.local_network, self.local_baseline,
            0, self.device, False,
        )
        env = TaskEnv(*env_params)
        perf_metrics = worker.run_episode(
            test_episode, mode='evaluation', test_env=(
                env.task_dic, env.agent_dic, env.depot,
            ),
        )
        return perf_metrics, self.metaAgentID

    def job(self, global_weights, baseline_weights, episode_number, agents_num, tasks_num):
        print(f"starting episode {episode_number} on metaAgent {self.metaAgentID}")
        self.set_weights(global_weights)
        self.set_baseline_weights(baseline_weights)

        jobResults, metrics = self.single_threaded_job(
            episode_number, agents_num, tasks_num,
        )
        info = {
            "id": self.metaAgentID,
            "episode_number": episode_number,
        }
        return jobResults, metrics, info


# ── Ray remote actor (lazy import - ray is only required for distributed mode) ─

def _get_rlrunner():
    """Factory that returns the RLRunner class, importing ray on demand."""
    import ray  # noqa: F811
    @ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT)
    class _RLRunner(Runner):
        def __init__(self, meta_agent_id):
            super().__init__(meta_agent_id)
    return _RLRunner


# Re-export for backward compatibility — only used in training.py (which imports ray itself)
# Users importing from dcmrta get Runner (works without ray).
# training.py constructs RLRunner via _get_rlrunner() or imports ray directly.


if __name__ == '__main__':
    import ray
    ray.init()
    RLRunner = _get_rlrunner()
    runner = RLRunner.remote(0)
    job_id = runner.single_threaded_job.remote(1, AGENTS_RANGE, TASKS_RANGE)
    out = ray.get(job_id)
    print(out[1])
