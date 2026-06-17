"""
DCMRTA: Dynamic Coalition Formation and Routing for Multirobot Task Allocation.

Reference: ICRA 2024 Paper - "Dynamic Coalition Formation and Routing for
Multirobot Task Allocation via Reinforcement Learning"
"""

from dcmrta.config import (
    USE_GPU,
    USE_GPU_GLOBAL,
    NUM_GPU,
    NUM_META_AGENT,
    LR,
    GAMMA,
    DECAY_STEP,
    RESET_OPT,
    EVALUATE,
    SUMMARY_WINDOW,
    AGENTS_RANGE,
    TASKS_RANGE,
    COALITION_SIZE,
    MAX_TIME,
    TRAIT_DIM,
    FOLDER_NAME,
    model_path,
    train_path,
    gifs_path,
    LOAD_MODEL,
    SAVE_IMG,
    SAVE_IMG_GAP,
    WANDB_LOG,
    BATCH_SIZE,
    AGENT_INPUT_DIM,
    TASK_INPUT_DIM,
    EMBEDDING_DIM,
    SAMPLE_SIZE,
    PADDING_SIZE,
)
from dcmrta.attention import AttentionNet
from dcmrta.environment import TaskEnv
from dcmrta.worker import Worker
from dcmrta.runner import Runner
