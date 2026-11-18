"""Lossless per-response reward tracing for SSL4RL experiments."""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from verl.utils.reward_difficulty import aggregate_group_statistics, group_reward_statistics


def _as_string(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class RewardTraceRecorder:
    """Write one atomic Parquet file containing every raw reward in a step."""

    def __init__(self, root_dir, task_name, run_id):
        self.root_dir = Path(os.path.expanduser(root_dir)).resolve()
        self.task_name = str(task_name)
        self.run_id = str(run_id)
        if not self.run_id:
            raise ValueError("Reward trace run_id must be non-empty")
        self.task_dir = self.root_dir / self.task_name
        self.task_dir.mkdir(parents=True, exist_ok=True)

    def reconcile_to_checkpoint(self, committed_step):
        committed_step = int(committed_step)
        rolled_back = []
        pattern = re.compile(r"step_(\d{8})\.parquet(?:\.tmp)?$")
        for path in sorted(self.task_dir.glob("step_*.parquet*")):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            step = int(match.group(1))
            if step > committed_step:
                rolled_back.append((step, path))
        if not rolled_back:
            return None

        quarantine = self.task_dir / "rolled_back" / f"checkpoint_{committed_step:08d}_{time.time_ns()}"
        quarantine.mkdir(parents=True, exist_ok=False)
        moved = []
        for step, source in rolled_back:
            destination = quarantine / source.name
            os.replace(source, destination)
            moved.append({"step": step, "source": str(source), "quarantined_as": str(destination)})
        manifest = {
            "committed_step": committed_step,
            "reconciled_at_utc_ns": time.time_ns(),
            "run_id": self.run_id,
            "task": self.task_name,
            "traces": moved,
        }
        (quarantine / "rollback_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Quarantined {len(moved)} reward traces beyond committed step {committed_step}: {quarantine}")
        return quarantine

    def write_step(self, batch, reward_tensor, global_step, epoch, responses, reward_computed_at_utc_ns):
        raw_rewards = reward_tensor.sum(-1).detach().float().cpu().numpy()
        count = len(raw_rewards)
        responses = list(responses)
        if len(responses) != count:
            raise ValueError(f"Response count {len(responses)} does not match reward count {count}")
        if reward_computed_at_utc_ns is None:
            raise ValueError("reward_computed_at_utc_ns is required")
        reward_computed_at_utc_ns = int(reward_computed_at_utc_ns)
        trace_written_at_utc_ns = time.time_ns()
        if trace_written_at_utc_ns < reward_computed_at_utc_ns:
            raise ValueError("Trace write timestamp precedes reward computation timestamp")

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
            if uid_string in (None, "", "None"):
                raise ValueError("Reward tracing requires a non-empty prompt UID for every rollout")
            rollout_ids.append(rollout_counts[uid_string])
            rollout_counts[uid_string] += 1
            dataset_indices.append(_as_string(extra_info.get("index") if isinstance(extra_info, dict) else None))
            ground_truths.append(
                _as_string(reward_model.get("ground_truth") if isinstance(reward_model, dict) else None)
            )
        group_sizes = set(rollout_counts.values())
        if len(group_sizes) != 1:
            raise ValueError(f"Reward tracing found inconsistent rollout group sizes: {sorted(group_sizes)}")

        table = pa.table(
            {
                "run_id": [self.run_id] * count,
                "task": [self.task_name] * count,
                "global_step": np.full(count, int(global_step), dtype=np.int64),
                "epoch": np.full(count, int(epoch), dtype=np.int32),
                "reward_computed_at_utc_ns": np.full(count, reward_computed_at_utc_ns, dtype=np.int64),
                "trace_written_at_utc_ns": np.full(count, trace_written_at_utc_ns, dtype=np.int64),
                "uid": [_as_string(value) for value in uids],
                "rollout_id": np.asarray(rollout_ids, dtype=np.int16),
                "dataset_index": dataset_indices,
                "data_source": [_as_string(value) for value in data_sources],
                "ground_truth": ground_truths,
                "response": [_as_string(value) for value in responses],
                "raw_reward": raw_rewards,
            }
        )

        output = self.task_dir / f"step_{int(global_step):08d}.parquet"
        temporary = output.with_suffix(".parquet.tmp")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing reward trace: {output}")
        if temporary.exists():
            raise FileExistsError(f"Incomplete reward trace requires manual review: {temporary}")
        pq.write_table(table, temporary, compression="zstd")
        try:
            # Hard-link publication is atomic and refuses to overwrite a file
            # created by another process between the checks above.
            os.link(temporary, output)
        finally:
            if output.exists():
                temporary.unlink(missing_ok=True)

        grouped = defaultdict(list)
        for uid, reward in zip(uids, raw_rewards, strict=True):
            grouped[_as_string(uid)].append(float(reward))
        group_metrics = [group_reward_statistics(values) for values in grouped.values()]
        difficulty = aggregate_group_statistics(group_metrics)
        return {
            "difficulty/raw_reward_mean": float(np.mean(raw_rewards)),
            "difficulty/raw_reward_std": float(np.std(raw_rewards)),
            "difficulty/zero_rate": float(np.mean(raw_rewards <= 0.0)),
            "difficulty/all_rollouts_fail_rate": difficulty["all_fail_group_fraction"],
            "difficulty/all_rollouts_success_rate": difficulty["all_success_group_fraction"],
            "difficulty/informative_group_fraction": difficulty["informative_group_fraction"],
            "difficulty/mean_group_reward_entropy_bits": difficulty["mean_group_reward_entropy_bits"],
            "difficulty/mean_group_reward_variance": difficulty["mean_group_reward_variance"],
            "difficulty/mixed_success_group_fraction": difficulty["mixed_success_group_fraction"],
            "difficulty/pass_at_n": 1.0 - difficulty["all_fail_group_fraction"],
            "difficulty/reward_trace_count": int(count),
        }
