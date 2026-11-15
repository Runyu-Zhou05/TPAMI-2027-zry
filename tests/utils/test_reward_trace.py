from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

from verl.utils.reward_trace import RewardTraceRecorder


def test_reward_trace_records_every_rollout(tmp_path):
    batch = SimpleNamespace(
        non_tensor_batch={
            "uid": np.array(["a", "a", "b", "b"], dtype=object),
            "data_source": np.array(["rotation"] * 4, dtype=object),
            "extra_info": np.array([{"index": 10}, {"index": 10}, {"index": 11}, {"index": 11}], dtype=object),
            "reward_model": np.array(
                [
                    {"ground_truth": "top-left"},
                    {"ground_truth": "top-left"},
                    {"ground_truth": "bottom-right"},
                    {"ground_truth": "bottom-right"},
                ],
                dtype=object,
            ),
        }
    )
    reward = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.0]])
    recorder = RewardTraceRecorder(tmp_path, "rotation")
    metrics = recorder.write_step(batch, reward, global_step=3, epoch=1)

    output = tmp_path / "rotation/step_00000003.parquet"
    table = pq.read_table(output)
    assert table.num_rows == 4
    assert table["raw_reward"].to_pylist() == [1.0, 0.0, 1.0, 0.0]
    assert table["rollout_id"].to_pylist() == [0, 1, 0, 1]
    assert metrics["difficulty/raw_reward_mean"] == 0.5
    assert metrics["difficulty/all_rollouts_fail_rate"] == 0.0
