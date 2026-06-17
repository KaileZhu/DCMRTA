"""Distributed REINFORCE training loop for DCMRTA using Ray.

Usage:
    python -m dcmrta.training
    # or from scripts/train.py
"""

import copy
import os
import random

import numpy as np
import ray
import torch
import torch.optim as optim
from scipy.stats import ttest_rel
from torch.utils.tensorboard import SummaryWriter

from dcmrta.attention import AttentionNet
from dcmrta.config import (
    AGENTS_RANGE, AGENT_INPUT_DIM, BATCH_SIZE, DECAY_STEP, EMBEDDING_DIM,
    EVALUATE, GAMMA, LOAD_MODEL, LR, MAX_TIME, NUM_GPU, NUM_META_AGENT,
    RESET_OPT, SUMMARY_WINDOW, TASK_INPUT_DIM, TASKS_RANGE, USE_GPU,
    USE_GPU_GLOBAL, WANDB_LOG, gifs_path, model_path, train_path,
)
from dcmrta.runner import Runner


# Ray remote actor — defined here because training.py already requires ray.
@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)

if WANDB_LOG:
    import wandb


def write_to_tensorboard(writer, tensorboard_data, curr_episode, plot_means=True):
    """Log metrics to TensorBoard and optionally W&B."""
    if plot_means:
        tensorboard_data = np.array(tensorboard_data)
        tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    (reward, value_loss, policy_loss, entropy, grad_norm,
     success_rate, time_, time_cost, waiting, distance, effi) = tensorboard_data

    writer.add_scalar('Losses/Policy Loss', policy_loss, curr_episode)
    writer.add_scalar('Losses/Entropy', entropy, curr_episode)
    writer.add_scalar('Losses/Grad Norm', grad_norm, curr_episode)
    writer.add_scalar('Losses/Value Loss', value_loss, curr_episode)

    writer.add_scalar('Perf/Reward', reward, curr_episode)
    writer.add_scalar('Perf/Makespan', time_, curr_episode)
    writer.add_scalar('Perf/Success rate', success_rate, curr_episode)
    writer.add_scalar('Perf/Time cost', time_cost, curr_episode)
    writer.add_scalar('Perf/Waiting time', waiting, curr_episode)
    writer.add_scalar('Perf/Traveling distance', distance, curr_episode)
    writer.add_scalar('Perf/Waiting Efficiency', effi, curr_episode)

    if WANDB_LOG:
        wandb.log({
            "Losses": {
                "Grad Norm": grad_norm, "Policy Loss": policy_loss, "Entropy": entropy,
            },
            "Perf": {
                "Reward": reward, "Time": time_, "Success Rate": success_rate,
                "Waiting Time": waiting, "Traveling Distance": distance,
                "Waiting Efficiency": effi,
            },
        }, step=curr_episode)


def main():
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    global_network = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM).to(device)
    baseline_network = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.98)

    ray.init()
    writer = SummaryWriter(train_path)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(gifs_path, exist_ok=True)

    if WANDB_LOG:
        wandb.init(project="CF")
        wandb.watch(global_network)

    curr_episode = 0
    best_perf = -100
    curr_level = 0

    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth')
        global_network.load_state_dict(checkpoint['model'])
        baseline_network.load_state_dict(checkpoint['model'])
        global_optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        curr_level = checkpoint['level']
        print("curr_episode set to ", curr_episode)

        best_path = model_path + '/best_model_checkpoint.pth'
        if os.path.exists(best_path):
            best_model_checkpoint = torch.load(best_path)
            best_perf = best_model_checkpoint['best_perf']
            baseline_network.load_state_dict(best_model_checkpoint['model'])
            print('best performance so far:', best_perf)
        print(global_optimizer.state_dict()['param_groups'][0]['lr'])
        if RESET_OPT:
            global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
            lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.98)
            curr_episode = 0

    # Launch meta-agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # Get initial weights
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        baseline_weights = baseline_network.to(local_device).state_dict()
        global_network.to(device)
        baseline_network.to(device)
    else:
        weights = global_network.state_dict()
        baseline_weights = baseline_network.state_dict()

    # Launch first batch of jobs
    jobList = []
    agents_num = np.random.randint(AGENTS_RANGE[0], AGENTS_RANGE[1] + 1)
    tasks_num = np.random.randint(TASKS_RANGE[0], TASKS_RANGE[1] + 1)
    for meta_agent in meta_agents:
        jobList.append(meta_agent.job.remote(weights, baseline_weights, curr_episode, agents_num, tasks_num))
        curr_episode += 1

    metric_names = ['success_rate', 'makespan', 'time_cost', 'waiting_time', 'travel_dist', 'efficiency']
    tensorboard_data = []
    training_data = []
    experience_buffer = [[] for _ in range(9)]
    test_set = np.random.randint(low=0, high=1e8, size=[256 // NUM_META_AGENT, NUM_META_AGENT])
    baseline_value = None

    try:
        while True:
            # Wait for any job to complete
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            done_jobs = ray.get(done_id)
            random.shuffle(done_jobs)

            perf_metrics = {n: [] for n in metric_names}
            for job in done_jobs:
                jobResults, metrics, info = job
                for i in range(9):
                    experience_buffer[i] += jobResults[i]
                for n in metric_names:
                    perf_metrics[n].append(metrics[n])

            update_done = False
            while len(experience_buffer[0]) >= BATCH_SIZE:
                agents_num = np.random.randint(AGENTS_RANGE[0], AGENTS_RANGE[1] + 1)
                tasks_num = np.random.randint(TASKS_RANGE[0], TASKS_RANGE[1] + 1)

                # Extract minibatch
                rollouts = [buf[:BATCH_SIZE] for buf in experience_buffer]
                for i in range(len(experience_buffer)):
                    experience_buffer[i] = experience_buffer[i][BATCH_SIZE:]

                if len(experience_buffer[0]) < BATCH_SIZE:
                    update_done = True
                    experience_buffer = [[] for _ in range(9)]

                # Prepare tensors
                agent_inputs = torch.stack(rollouts[0], dim=0)
                task_inputs = torch.stack(rollouts[1], dim=0)
                action_batch = torch.stack(rollouts[2], dim=0)
                mask_batch = torch.stack(rollouts[3], dim=0)
                advantage_batch = torch.stack(rollouts[6], dim=0)
                reward_batch = torch.stack(rollouts[4], dim=0)
                index = torch.stack(rollouts[5])

                if device != local_device:
                    agent_inputs = agent_inputs.to(device)
                    task_inputs = task_inputs.to(device)
                    action_batch = action_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    reward_batch = reward_batch.to(device)
                    advantage_batch = advantage_batch.to(device)
                    index = index.to(device)

                logp_list = global_network(task_inputs, agent_inputs, mask_batch)
                logp = torch.gather(logp_list, 1, action_batch)
                entropy = (logp_list * logp_list.exp()).nansum(dim=-1).mean()
                policy_loss = -logp * advantage_batch.detach()
                policy_loss = policy_loss.mean()

                loss = policy_loss
                global_optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    global_network.parameters(), max_norm=10, norm_type=2,
                )
                global_optimizer.step()
                lr_decay.step()

                perf_data = [np.nanmean(perf_metrics[n]) for n in metric_names]
                data = [
                    reward_batch.mean().item(), 0, policy_loss.item(),
                    entropy.item(), grad_norm.item(), *perf_data,
                ]
                training_data.append(data)

            for meta_agent in meta_agents:
                jobList.append(meta_agent.job.remote(
                    weights, baseline_weights, curr_episode, agents_num, tasks_num,
                ))
                curr_episode += 1

            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensorboard(writer, training_data, curr_episode)
                training_data = []

            if update_done:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    baseline_weights = baseline_network.to(local_device).state_dict()
                    global_network.to(device)
                    baseline_network.to(device)
                else:
                    weights = global_network.state_dict()
                    baseline_weights = baseline_network.state_dict()

            # Periodic checkpoint
            if curr_episode % 512 == 0:
                print('Saving model...')
                checkpoint = {
                    "model": global_network.state_dict(),
                    "optimizer": global_optimizer.state_dict(),
                    "episode": curr_episode,
                    "lr_decay": lr_decay.state_dict(),
                    "level": curr_level,
                    "best_perf": best_perf,
                }
                torch.save(checkpoint, model_path + "/checkpoint.pth")
                print('Saved model')

            # Evaluate and possibly update baseline
            if EVALUATE and curr_episode % 1024 == 0:
                ray.wait(jobList, num_returns=NUM_META_AGENT)
                for a in meta_agents:
                    ray.kill(a)
                torch.cuda.empty_cache()
                print('Evaluate baseline model at ', curr_episode)

                # Test baseline model
                if baseline_value is None:
                    baseline_value = _eval_agents(baseline_weights, test_set)

                # Test current model
                test_value = _eval_agents(weights, test_set)

                meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

                print(f'test value: {test_value.mean():.4f}')
                print(f'baseline value: {baseline_value.mean():.4f}')

                if test_value.mean() > baseline_value.mean():
                    _, p = ttest_rel(test_value, baseline_value)
                    print(f'p value: {p:.4f}')
                    if p < 0.05:
                        print('Updating baseline model at', curr_episode)
                        if device != local_device:
                            weights = global_network.to(local_device).state_dict()
                            global_network.to(device)
                        else:
                            weights = global_network.state_dict()
                        baseline_weights = copy.deepcopy(weights)
                        baseline_network.load_state_dict(baseline_weights)
                        test_set = np.random.randint(
                            low=0, high=1e8,
                            size=[256 // NUM_META_AGENT, NUM_META_AGENT],
                        )
                        baseline_value = None
                        best_perf = test_value.mean()
                        print('Saving best model...')
                        checkpoint = {
                            "model": global_network.state_dict(),
                            "optimizer": global_optimizer.state_dict(),
                            "episode": curr_episode,
                            "lr_decay": lr_decay.state_dict(),
                            "best_perf": best_perf,
                        }
                        torch.save(checkpoint, model_path + "/best_model_checkpoint.pth")
                        print('Saved best model')

                jobList = []
                for meta_agent in meta_agents:
                    jobList.append(meta_agent.job.remote(
                        weights, baseline_weights, curr_episode, agents_num, tasks_num,
                    ))
                    curr_episode += 1

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        if WANDB_LOG:
            wandb.finish()
        for a in meta_agents:
            ray.kill(a)


def _eval_agents(weights, test_set):
    """Evaluate a set of weights across the test set, returning reward array."""
    test_agent_list = [RLRunner.remote(meta_agent_id=i) for i in range(NUM_META_AGENT)]
    for test_agent in test_agent_list:
        ray.get(test_agent.set_baseline_weights.remote(weights))
    rewards = []
    for i in range(256 // NUM_META_AGENT):
        sample_job_list = []
        for j, test_agent in enumerate(test_agent_list):
            sample_job_list.append(test_agent.testing.remote(seed=test_set[i][j]))
        sample_done_id, _ = ray.wait(sample_job_list, num_returns=NUM_META_AGENT)
        reward = ray.get(sample_done_id)
        rewards.extend(reward)
    for a in test_agent_list:
        ray.kill(a)
    return np.stack(rewards)


if __name__ == "__main__":
    main()
