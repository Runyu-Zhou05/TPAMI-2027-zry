"""Lossless per-response reward tracing for SSL4RL experiments."""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _as_string(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class RewardTraceRecorder:
    """Write one atomic Parquet file containing every raw reward in a step."""

    def __init__(self, root_dir, task_name):
        self.root_dir = Path(os.path.expanduser(root_dir)).resolve()
        self.task_name = str(task_name)
        self.task_dir = self.root_dir / self.task_name
        self.task_dir.mkdir(parents=True, exist_ok=True)

    def write_step(self, batch, reward_tensor, global_step, epoch):
        raw_rewards = reward_tensor.sum(-1).detach().float().cpu().numpy()
        count = len(raw_rewards)

        non_tensor = batch.non_tensor_batch
        uids = non_tensor.get("uid", np.array([None] * count, dtype=object))
        data_sources = non_tensor.get("data_source", np.array([None] * count, dtype=object))
        extra_infos = non_tensor.get("extra_info", np.array([None] * count, dtype=object))
        reward_models = non_tensor.get("reward_model", np.array([None] * count, dtype=object))

        rollout_counts = defaultdict(int)
        rollout_ids = []
        dataset_indices = []
        ground_truths = []
        for uid, extra_info, reward_model in zip(uids, extra_infos, reward_models, strict=True):
            uid_string = _as_string(uid)
            rollout_ids.append(rollout_counts[uid_string])
            rollout_counts[uid_string] += 1
            dataset_indices.append(_as_string(extra_info.get("index") if isinstance(extra_info, dict) else None))
            ground_truths.append(
                _as_string(reward_model.get("ground_truth") if isinstance(reward_model, dict) else None)
            )

        table = pa.table(
            {
                "task": [self.task_name] * count,
                "global_step": np.full(count, int(global_step), dtype=np.int64),
                "epoch": np.full(count, int(epoch), dtype=np.int32),
                "uid": [_as_string(value) for value in uids],
                "rollout_id": np.asarray(rollout_ids, dtype=np.int16),
                "dataset_index": dataset_indices,
                "data_source": [_as_string(value) for value in data_sources],
                "ground_truth": ground_truths,
                "raw_reward": raw_rewards,
            }
        )

        output = self.task_dir / f"step_{int(global_step):08d}.parquet"
        temporary = output.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, output)

        grouped = defaultdict(list)
        for uid, reward in zip(uids, raw_rewards, strict=True):
            grouped[_as_string(uid)].append(float(reward))
        all_fail_rate = float(np.mean([max(values) <= 0.0 for values in grouped.values()]))
        return {
            "difficulty/raw_reward_mean": float(np.mean(raw_rewards)),
            "difficulty/raw_reward_std": float(np.std(raw_rewards)),
            "difficulty/zero_rate": float(np.mean(raw_rewards <= 0.0)),
            "difficulty/all_rollouts_fail_rate": all_fail_rate,
            "difficulty/pass_at_n": 1.0 - all_fail_rate,
            "difficulty/reward_trace_count": int(count),
        }
